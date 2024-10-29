import ray
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, infer_auto_device_map
import warnings
from torch.cuda.amp import autocast, GradScaler
import gc

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

class LLM_remote:
    def __init__(self,model_id,GPU_Savings,max_memory,tokenizer,Relevance_Map_Method,SG_iterations=10):
        self.model_id = model_id
        self.GPU_Savings=GPU_Savings
        self.max_memory=max_memory #Example: {0: "14GiB", "cpu": "30GiB"}
        self.tokenizer = tokenizer
        self.extracted_outputs={}
        self.layer_of_interest={}
        self.SG_iterations=SG_iterations
        self.Relevance_Map_Method=Relevance_Map_Method
        
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
        
        # Step 3: Find Layers to place a hook
        self.layers_to_hook = {}
        self.layers_to_hook["embed_tokens"]=[self.model.model.embed_tokens]
        for i in self.model.model.layers:
            
            if "q_proj" not in self.layers_to_hook:
                self.layers_to_hook["q_proj"]=[]
            self.layers_to_hook["q_proj"].append(i.self_attn.q_proj)
            
            if "k_proj" not in self.layers_to_hook:
                self.layers_to_hook["k_proj"]=[]
            self.layers_to_hook["k_proj"].append(i.self_attn.k_proj)
            
            if "v_proj" not in self.layers_to_hook:
                self.layers_to_hook["v_proj"]=[]
            self.layers_to_hook["v_proj"].append(i.self_attn.v_proj)
            
            if "o_proj" not in self.layers_to_hook:
                self.layers_to_hook["o_proj"]=[]
            self.layers_to_hook["o_proj"].append(i.self_attn.o_proj)
            
            #Is not used in this setting (no gradients found)
            #if "rotary_emb" not in layers_to_hook:
            #    layers_to_hook["rotary_emb"]=[]
            #layers_to_hook["rotary_emb"].append(i.self_attn.rotary_emb)
            
            if "mlp" not in self.layers_to_hook:
                self.layers_to_hook["mlp"]=[]
            self.layers_to_hook["mlp"].append(i.mlp)

        self.layers_to_hook["rotary_emb_end"]=[self.model.model.rotary_emb]
        
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
        
        
    def forward_pass(self,Task_Text_Tokens):
        #Forward Pass
        if self.GPU_Savings:
            with torch.autograd.graph.save_on_cpu(): #Optimized for GPU Memory savings
                with autocast():
                    M_outputs = self.model(**Task_Text_Tokens)  
        else:
            M_outputs = self.model(**Task_Text_Tokens)
        
        return M_outputs
        
        
    def backward_pass(self,loss):
        self.model.zero_grad()
        if self.GPU_Savings or (isinstance(self.max_memory, str) and self.device_map!="cpu"):
            GradScaler().scale(loss).backward()
        else:
            loss.backward()
            
            
    def tensors_to_lists(self,data): #Helper function which ensures the detachment of the gradient 
        if isinstance(data, torch.Tensor):  # Check if it's a tensor
            return data.grad.numpy()
        elif isinstance(data, dict):  # If it's a dictionary, recursively check its values
            return {key: self.tensors_to_lists(value) for key, value in data.items()}
        elif isinstance(data, list):  # If it's a list, recursively check each element
            return [self.tensors_to_lists(item) for item in data]
        elif isinstance(data, tuple):  # If it's a tuple, recursively check each element
            return tuple(self.tensors_to_lists(item) for item in data)
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
                    combined_gradients[Layer_Name][Ac_Layer_Index].append(aet.grad.detach().numpy())
            else:
                for p_aet,aet in enumerate(self.extracted_outputs[Layer_Name][Ac_Layer_Index]):
                    combined_gradients[Layer_Name][Ac_Layer_Index][p_aet]+=aet.grad.detach().numpy()

        else:
            if Ac_Layer_Index not in combined_gradients[Layer_Name]:
                combined_gradients[Layer_Name][Ac_Layer_Index]=self.extracted_outputs[Layer_Name][Ac_Layer_Index].grad.detach().numpy()
            else:
                combined_gradients[Layer_Name][Ac_Layer_Index]+=self.extracted_outputs[Layer_Name][Ac_Layer_Index].grad.detach().numpy()
        
        return combined_gradients
        
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
        
        
    def Cleanup(self):
        gc.collect()
        self.extracted_outputs={}
        
        
    def get_results(self,Task_Text,Task_Result):
        if self.Relevance_Map_Method=='vanilla_gradient':
            result = self.get_results_vanilla_gradient(Task_Text,Task_Result)
        elif self.Relevance_Map_Method=='smoothGrad':
            result = self.get_results_smoothGrad(Task_Text,Task_Result)
        self.Cleanup()
        return result
       

# Instantiate with dynamic num_gpus setting
def create_llm_remote(model_id, GPU_Savings, max_memory, tokenizer,Relevance_Map_Method,num_gpus=1):
    return ray.remote(LLM_remote).options(num_gpus=num_gpus).remote(model_id, GPU_Savings, max_memory,tokenizer,Relevance_Map_Method)

