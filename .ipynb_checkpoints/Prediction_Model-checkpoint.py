import ray
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, infer_auto_device_map
import warnings
from torch.cuda.amp import autocast, GradScaler
import gc
import captum
import copy
from captum_helper import LLMGradientAttribution_Features
# Suppress specific warnings that contain certain text
warnings.filterwarnings(
    "ignore",
    message=r"for.*: copying from a non-meta parameter",
    category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message=r"torch\.cuda\.amp\.GradScaler is enabled, but CUDA is not available",
    category=UserWarning
)

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.logits = None  # To store logits during forward pass

    def __call__(self, input_ids, attention_mask=None):
        self.forward(input_ids, attention_mask=None)

    def forward(self, input_ids, attention_mask=None):
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.logits = outputs.logits  # Capture logits for later use
        return outputs.logits

class LLM_remote:
    def __init__(self,model_id,GPU_Savings,max_memory,tokenizer,Relevance_Map_Method,Baselines,SG_iterations=10):
        self.model_id = model_id
        self.GPU_Savings=GPU_Savings
        self.max_memory=max_memory #Example: {0: "14GiB", "cpu": "30GiB"}
        self.tokenizer = tokenizer
        self.captum_tokenizer = None
        self.extracted_outputs={}
        self.layer_of_interest={}
        self.SG_iterations=SG_iterations
        self.Relevance_Map_Method=Relevance_Map_Method
        self.Baselines=Baselines

        # Step 1: Use infer_auto_device_map to get the device map for the model
        with init_empty_weights():
            my_model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        if not isinstance(self.max_memory, str):
            self.device_map = infer_auto_device_map(my_model, max_memory=self.max_memory)
        else:
            self.device_map = self.max_memory
        # Step 2: Initialize Model
        self.model = AutoModelForCausalLM.from_pretrained(model_id,device_map=self.device_map)
        if self.GPU_Savings:
            self.model.gradient_checkpointing_enable()
        self.model_device=next(self.model.parameters()).device
        self.model_captum=None
        self.model_wrapped=None
        self.needs_baseline=False
        # Step 3: Find Layers to place a hook

        self.layers_to_hook_layer_list=[]
        self.layers_to_hook_name_list=[]

        self.layers_to_hook = {}
        self.layers_to_hook["embed_tokens"]=[self.model.model.embed_tokens]
        self.layers_to_hook_layer_list.append(self.model.model.embed_tokens)
        self.layers_to_hook_name_list.append(["embed_tokens",0])

        for i_p,i in enumerate(self.model.model.layers):

            if "q_proj" not in self.layers_to_hook:
                self.layers_to_hook["q_proj"]=[]
            self.layers_to_hook["q_proj"].append(i.self_attn.q_proj)
            self.layers_to_hook_layer_list.append(i.self_attn.q_proj)
            self.layers_to_hook_name_list.append(["q_proj",i_p])

            if "k_proj" not in self.layers_to_hook:
                self.layers_to_hook["k_proj"]=[]
            self.layers_to_hook["k_proj"].append(i.self_attn.k_proj)
            self.layers_to_hook_layer_list.append(i.self_attn.k_proj)
            self.layers_to_hook_name_list.append(["k_proj",i_p])

            if "v_proj" not in self.layers_to_hook:
                self.layers_to_hook["v_proj"]=[]
            self.layers_to_hook["v_proj"].append(i.self_attn.v_proj)
            self.layers_to_hook_layer_list.append(i.self_attn.v_proj)
            self.layers_to_hook_name_list.append(["v_proj",i_p])

            if "o_proj" not in self.layers_to_hook:
                self.layers_to_hook["o_proj"]=[]
            self.layers_to_hook["o_proj"].append(i.self_attn.o_proj)
            self.layers_to_hook_layer_list.append(i.self_attn.o_proj)
            self.layers_to_hook_name_list.append(["o_proj",i_p])

            #Is not used in this setting (no gradients found)
            #if "rotary_emb" not in layers_to_hook:
            #    layers_to_hook["rotary_emb"]=[]
            #layers_to_hook["rotary_emb"].append(i.self_attn.rotary_emb)

            if "mlp" not in self.layers_to_hook:
                self.layers_to_hook["mlp"]=[]
            self.layers_to_hook["mlp"].append(i.mlp)
            self.layers_to_hook_layer_list.append(i.mlp)
            self.layers_to_hook_name_list.append(["mlp",i_p])

        self.layers_to_hook["rotary_emb_end"]=[self.model.model.rotary_emb]
        self.layers_to_hook_layer_list.append(self.model.model.rotary_emb)
        self.layers_to_hook_name_list.append(["rotary_emb_end",0])


        # Step 4: Initialize hook
        self.hooks=[]

        if self.Relevance_Map_Method=='vanilla_gradient':
            for layer_name, layer_arr in self.layers_to_hook.items():
                for layer_pos,layer in enumerate(layer_arr):
                    hook = layer.register_forward_hook(self.create_hook_fn_vanilla_gradient(layer_name,layer_pos))
                    self.hooks.append(hook)

        elif self.Relevance_Map_Method=='smoothGrad':
            for layer_name, layer_arr in self.layers_to_hook.items():
                self.layer_of_interest[layer_name]={}
                for layer_pos,layer in enumerate(layer_arr):
                    self.layer_of_interest[layer_name][layer_pos]=False
                    hook = layer.register_forward_hook(self.create_hook_fn_smoothGrad(layer_name,layer_pos))
                    self.hooks.append(hook)

        elif self.Relevance_Map_Method=="captum-GradientXActivation":
            #self.model_wrapped=ModelWrapper(self.model)
            self.captum_tokenizer=copy.deepcopy(self.tokenizer)
            self.captum_tokenizer.pad_token = self.captum_tokenizer.eos_token
            fa = captum.attr.LayerGradientXActivation(self.model,self.layers_to_hook_layer_list)
            self.model_captum = LLMGradientAttribution_Features(fa, self.captum_tokenizer)

        
        elif self.Relevance_Map_Method=="captum-GradientShap":
            self.captum_tokenizer=copy.deepcopy(self.tokenizer)
            self.captum_tokenizer.pad_token = self.captum_tokenizer.eos_token
            fa = captum.attr.LayerGradientShap(self.model,self.layers_to_hook_layer_list)
            self.model_captum = LLMGradientAttribution_Features(fa, self.captum_tokenizer)
            self.needs_baseline=True

        elif self.Relevance_Map_Method=="captum-IntegratedGradients":
            self.captum_tokenizer=copy.deepcopy(self.tokenizer)
            self.captum_tokenizer.pad_token = self.captum_tokenizer.eos_token
            fa = captum.attr.LayerIntegratedGradients(self.model,self.layers_to_hook_layer_list)
            self.model_captum = LLMGradientAttribution_Features(fa, self.captum_tokenizer)
            self.needs_baseline=True

        elif self.Relevance_Map_Method=='hiddenFeatures':
            for layer_name, layer_arr in self.layers_to_hook.items():
                for layer_pos,layer in enumerate(layer_arr):
                    hook = layer.register_forward_hook(self.create_hook_fn_hidden_features(layer_name,layer_pos))
                    self.hooks.append(hook)


    def forward_pass(self,Task_Text_Tokens,Result_Target=None,Use_Captum=False):
        #Forward Pass
        if self.GPU_Savings:
            with torch.autograd.graph.save_on_cpu(): #Optimized for GPU Memory savings
                with autocast():
                    if Use_Captum:
                        if self.needs_baseline:
                            M_outputs = self.model_captum.attribute(Task_Text_Tokens, baselines=self.Baselines ,target=Result_Target)
                        else:
                            M_outputs = self.model_captum.attribute(Task_Text_Tokens, target=Result_Target)
                    else:
                        M_outputs = self.model(**Task_Text_Tokens)
        else:
            if Use_Captum:
                if self.needs_baseline:
                    M_outputs = self.model_captum.attribute(Task_Text_Tokens, baselines=self.Baselines ,target=Result_Target)
                else:
                    M_outputs = self.model_captum.attribute(Task_Text_Tokens, target=Result_Target)
            else:
                M_outputs = self.model(**Task_Text_Tokens)

        return M_outputs


    def backward_pass(self,loss):
        self.model.zero_grad()
        if self.GPU_Savings or (isinstance(self.max_memory, str) and self.device_map!="cpu"):
            GradScaler().scale(loss).backward()
        else:
            loss.backward()


    def tensors_to_lists(self,data,Gradient_Ex=True): #Helper function which ensures the detachment of the gradient
        if isinstance(data, torch.Tensor):  # Check if it's a tensor
            if Gradient_Ex:
                return data.grad.cpu().numpy()
            else:
                return data.cpu().detach().numpy()
        elif isinstance(data, dict):  # If it's a dictionary, recursively check its values
            return {key: self.tensors_to_lists(value,Gradient_Ex) for key, value in data.items()}
        elif isinstance(data, list):  # If it's a list, recursively check each element
            return [self.tensors_to_lists(item,Gradient_Ex) for item in data]
        elif isinstance(data, tuple):  # If it's a tuple, recursively check each element
            return tuple(self.tensors_to_lists(item,Gradient_Ex) for item in data)
        else:
            return data  # If it's not a tensor, return it as-is


    def helper_grad(self,data): #Helper function which ensures that gradients are saved
        if isinstance(data, torch.Tensor):  # Check if it's a tensor
            data.requires_grad_(True)
            data.retain_grad()
            return data
        elif isinstance(data, dict):  # If it's a dictionary, recursively check its values
            return {key: self.helper_grad(value) for key, value in data.items()}
        elif isinstance(data, list):  # If it's a list, recursively check each element
            return [self.helper_grad(item) for item in data]
        elif isinstance(data, tuple):  # If it's a tuple, recursively check each element
            return tuple(self.helper_grad(item) for item in data)
        else:
            return data  # If it's not a tensor, return it as-is

    def helper_hiden_rep(self,data): #Helper function which ensures that gradients are saved
        if isinstance(data, torch.Tensor):  # Check if it's a tensor
            return data.cpu().numpy()
        elif isinstance(data, dict):  # If it's a dictionary, recursively check its values
            return {key: self.helper_grad(value) for key, value in data.items()}
        elif isinstance(data, list):  # If it's a list, recursively check each element
            return [self.helper_grad(item) for item in data]
        elif isinstance(data, tuple):  # If it's a tuple, recursively check each element
            return tuple(self.helper_grad(item) for item in data)
        else:
            return data  # If it's not a tensor, return it as-is


    def create_hook_fn_hidden_features(self,layer_name,layer_index):
        def hook_fn(module, input, output):
            if layer_name not in self.extracted_outputs:
                self.extracted_outputs[layer_name] = {}
            self.extracted_outputs[layer_name][layer_index]=self.helper_hiden_rep(output)
        return hook_fn


    def create_hook_fn_vanilla_gradient(self,layer_name,layer_index):
        def hook_fn(module, input, output):
            if layer_name not in self.extracted_outputs:
                self.extracted_outputs[layer_name] = {}

            self.extracted_outputs[layer_name][layer_index]=self.helper_grad(output)
        return hook_fn


    def create_hook_fn_smoothGrad(self,layer_name,layer_index):

        def hook_fn(module, input, output):
            if self.layer_of_interest[layer_name][layer_index]:

                if isinstance(output, torch.Tensor):
                    std_dev = 0.1 * (output.max() - output.min()).item()  # 10% of the range
                    output+= torch.randn_like(output) * std_dev
                    output.requires_grad_(True)
                    output.retain_grad()
                    self.extracted_outputs[layer_name][layer_index]=output
                else:
                    self.extracted_outputs[layer_name][layer_index]=[]
                    for aop in range(len(output)):
                        std_dev = 0.1 * (output[aop].max() - output[aop].min()).item()  # 10% of the range
                        self.extracted_outputs[layer_name][layer_index].append(output[aop]+(torch.randn_like(output[aop]) * std_dev))
                        self.extracted_outputs[layer_name][layer_index][aop].requires_grad_(True)
                        self.extracted_outputs[layer_name][layer_index][aop].retain_grad()
                    self.extracted_outputs[layer_name][layer_index]=tuple(self.extracted_outputs[layer_name][layer_index])
                #print(self.extracted_outputs[layer_name][layer_index])
                output=None
                return self.extracted_outputs[layer_name][layer_index]

        return hook_fn

    def Prepare_Input(self,Task_Text,Task_Result):
        Task_Result_Token=self.tokenizer(Task_Result, return_tensors="pt").input_ids[0][1].item()
        Task_Text_Tokens = self.tokenizer(Task_Text, return_tensors="pt")
        Task_Text_Tokens = {key: value.to(self.model_device) for key, value in Task_Text_Tokens.items()}
        return Task_Result_Token,Task_Text_Tokens

    def get_results_vanilla_gradient(self,Task_Text,Task_Result):
        # Prepare Input
        Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)

        # Forward and backward pass
        Model_output=self.forward_pass(Task_Text_Tokens)
        loss = Model_output.logits[0][-1][Task_Result_Token]
        self.backward_pass(loss)

        # Prepare outputs
        if Task_Result_Token==torch.argmax(Model_output.logits[0][-1]).item():
            Correctly_classified=1
        else:
            Correctly_classified=0
        Gradients=self.tensors_to_lists(self.extracted_outputs)

        return Gradients,Correctly_classified


    def helper_grad_smoothGrad(self,combined_gradients,Layer_Name,Ac_Layer_Index): #Helper function to prepare gradient output for smoothGrad
        if not  isinstance(self.extracted_outputs[Layer_Name][Ac_Layer_Index], torch.Tensor):
            if Ac_Layer_Index not in combined_gradients[Layer_Name]:
                combined_gradients[Layer_Name][Ac_Layer_Index]=[]
                for aet in self.extracted_outputs[Layer_Name][Ac_Layer_Index]:
                    combined_gradients[Layer_Name][Ac_Layer_Index].append(aet.grad.cpu().detach().numpy())
            else:
                for p_aet,aet in enumerate(self.extracted_outputs[Layer_Name][Ac_Layer_Index]):
                    combined_gradients[Layer_Name][Ac_Layer_Index][p_aet]+=aet.grad.cpu().detach().numpy()

        else:
            if Ac_Layer_Index not in combined_gradients[Layer_Name]:
                combined_gradients[Layer_Name][Ac_Layer_Index]=self.extracted_outputs[Layer_Name][Ac_Layer_Index].grad.cpu().detach().numpy()
            else:
                combined_gradients[Layer_Name][Ac_Layer_Index]+=self.extracted_outputs[Layer_Name][Ac_Layer_Index].grad.cpu().detach().numpy()

        return combined_gradients

    def list_of_attributions_to_dict(self,attributions):
        for ac_a_p,ac_a in enumerate(attributions):
            pass

    def get_results_smoothGrad(self,Task_Text,Task_Result):
        # Prepare Input
        Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)

        #Get and average over Gradients:
        combined_gradients={}
        Correctly_classified=0
        Total_classified=0
        for Layer_Name in self.layers_to_hook:
            if Layer_Name not in self.extracted_outputs:
                self.extracted_outputs[Layer_Name]={}
            combined_gradients[Layer_Name]={}
            for Ac_Layer_Index,Ac_Layer in enumerate(self.layers_to_hook[Layer_Name]):
                self.layer_of_interest[Layer_Name][Ac_Layer_Index]=True
                if Ac_Layer_Index not in self.extracted_outputs[Layer_Name]:
                    self.extracted_outputs[Layer_Name][Ac_Layer_Index]=None

                for _ in range(self.SG_iterations):

                    Model_output=self.forward_pass(Task_Text_Tokens)
                    loss = Model_output.logits[0][-1][Task_Result_Token]
                    self.backward_pass(loss)

                    if Task_Result_Token==torch.argmax(Model_output.logits[0][-1]).item():
                        Correctly_classified+=1
                    Total_classified+=1

                    combined_gradients=self.helper_grad_smoothGrad(combined_gradients,Layer_Name,Ac_Layer_Index)
                    self.extracted_outputs[Layer_Name][Ac_Layer_Index]=None

                self.layer_of_interest[Layer_Name][Ac_Layer_Index]=False
                gc.collect()

                #Averaging of gradients
                if not isinstance(combined_gradients[Layer_Name][Ac_Layer_Index], torch.Tensor):
                    for pa in range(len(combined_gradients[Layer_Name][Ac_Layer_Index])):
                        combined_gradients[Layer_Name][Ac_Layer_Index][pa]=combined_gradients[Layer_Name][Ac_Layer_Index][pa]/self.SG_iterations
                else:
                    combined_gradients[Layer_Name][Ac_Layer_Index]=combined_gradients[Layer_Name][Ac_Layer_Index]/self.SG_iterations
        Correctly_classified=Correctly_classified/Total_classified
        return combined_gradients,Correctly_classified

    def get_results_vanilla_gradient(self,Task_Text,Task_Result):
        # Prepare Input
        Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)

        # Forward and backward pass
        Model_output=self.forward_pass(Task_Text_Tokens)
        loss = Model_output.logits[0][-1][Task_Result_Token]
        self.backward_pass(loss)

        # Prepare outputs
        if Task_Result_Token==torch.argmax(Model_output.logits[0][-1]).item():
            Correctly_classified=1
        else:
            Correctly_classified=0
        Gradients=self.tensors_to_lists(self.extracted_outputs)

        return Gradients,Correctly_classified



    def get_results_hidden_Representation(self,Task_Text,Task_Result):
        # Prepare Input
        with torch.no_grad():
            Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)

            # Forward and backward pass
            Model_output=self.forward_pass(Task_Text_Tokens)

            # Prepare outputs
            if Task_Result_Token==torch.argmax(Model_output.logits[0][-1]).item():
                Correctly_classified=1
            else:
                Correctly_classified=0

        return self.tensors_to_lists(self.extracted_outputs,Gradient_Ex=False),Correctly_classified

    def create_nested_dict(self,attributions, layers_to_hook_name_list):
        """
        Creates a nested dictionary from `attributions` and `layers_to_hook_name_list`.

        Args:
            attributions (list): A list where each element is either a tensor or a list/tuple of tensors.
            layers_to_hook_name_list (list): A list of two-element lists specifying the key hierarchy.

        Returns:
            dict: A nested dictionary where keys are specified by `layers_to_hook_name_list` and
                  values are the tensors from `attributions` converted to NumPy arrays.
        """
        nested_dict = {}

        for attr, hook_names in zip(attributions, layers_to_hook_name_list):
            # Ensure the list of hook names has exactly two elements
            if len(hook_names) != 2:
                raise ValueError("Each element of layers_to_hook_name_list must have exactly two elements.")

            layer_key, hook_key = hook_names

            # Convert tensor or list/tuple of tensors to numpy
            if isinstance(attr, (list, tuple)):
                transformed_attr = [tensor[0].detach().numpy() for tensor in attr]
            else:
                transformed_attr = attr[0].detach().numpy()

            # Create nested structure
            if layer_key not in nested_dict:
                nested_dict[layer_key] = {}

            nested_dict[layer_key][hook_key] = transformed_attr

        return nested_dict


    def get_results_captum_GradientXActivationt(self,Task_Text,Task_Result):
        # Prepare Input
        #Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)
        inp = captum.attr.TextTokenInput(
            Task_Text,
            self.captum_tokenizer
            )

        # Forward and backward pass
        attributions=self.forward_pass(inp,Result_Target=Task_Result,Use_Captum=True)
        #print(attributions[-1])
        #print(attributions[0])
        #print("*"*100)
        #print(len(attributions))
        #print(len(attributions[0]))
        #print(len(attributions[0][-1]))
        #print(attributions[0][-1][0].shape)
        #print(len(self.layers_to_hook_name_list))
        #print(len(attributions[-1]))
        #print(attributions[-1][0].shape())
        #if Task_Result_Token==torch.argmax(self.model_wrapped.logits[0][-1]).item():
        #    Correctly_classified=1
        #else:
        #    Correctly_classified=0
        #print("end")
        #exit()

        att_dict=self.create_nested_dict(attributions[0], self.layers_to_hook_name_list)

        return att_dict,0

    def get_results_captum(self,Task_Text,Task_Result,needs_baseline):
        # Prepare Input
        #Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)
        inp = captum.attr.TextTokenInput(
            Task_Text,
            self.captum_tokenizer
            )

        # Forward and backward pass
        attributions=self.forward_pass(inp,Result_Target=Task_Result,Use_Captum=True)

        att_dict=self.create_nested_dict(attributions[0], self.layers_to_hook_name_list)

        return att_dict,0

    
    def Cleanup(self):
        gc.collect()
        self.extracted_outputs={}


    def get_results(self,Task_Text,Task_Result):
        if self.Relevance_Map_Method=='vanilla_gradient':
            result = self.get_results_vanilla_gradient(Task_Text,Task_Result)
        elif self.Relevance_Map_Method=='smoothGrad':
            result = self.get_results_smoothGrad(Task_Text,Task_Result)
        elif self.Relevance_Map_Method=='hiddenFeatures':
            result = self.get_results_hidden_Representation(Task_Text,Task_Result)
        elif self.Relevance_Map_Method=="captum-GradientXActivation":
            result=self.get_results_captum(Task_Text,Task_Result)
        elif self.Relevance_Map_Method=="captum-GradientShap":
            result=self.get_results_captum(Task_Text,Task_Result)
        elif self.Relevance_Map_Method=="captum-IntegratedGradients":
            result=self.get_results_captum(Task_Text,Task_Result)
        self.Cleanup()
        return result


# Instantiate with dynamic num_gpus setting
def create_llm_remote(model_id, GPU_Savings, max_memory, tokenizer,Relevance_Map_Method,Baselines,num_gpus=0,num_cpus=1):
    return ray.remote(LLM_remote).options(num_gpus=num_gpus,num_cpus=num_cpus).remote(model_id, GPU_Savings, max_memory,tokenizer,Relevance_Map_Method,Baselines)