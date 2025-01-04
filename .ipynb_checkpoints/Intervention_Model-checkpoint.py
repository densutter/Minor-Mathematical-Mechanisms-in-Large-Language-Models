import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights, infer_auto_device_map
import warnings
from torch.cuda.amp import autocast, GradScaler
import gc
import captum
import copy
from captum_helper import LLMGradientAttribution_Features
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
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
    def __init__(self,model_id,max_memory,tokenizer,masking_dictionary,change_arr):
        self.model_id = model_id
        self.max_memory=max_memory #Example: {0: "14GiB", "cpu": "30GiB"}
        self.tokenizer = tokenizer
        self.masking_dictionary=masking_dictionary
        self.captum_tokenizer = None
        self.extracted_outputs={}
        self.change_arr=change_arr
        #self.task_1=task_1
        #self.task_2=task_2

        # Step 1: Use infer_auto_device_map to get the device map for the model
        self.device_map = self.max_memory


        # Step 2: Initialize Model
        self.model = AutoModelForCausalLM.from_pretrained(model_id,device_map=self.device_map)
        self.model_device=next(self.model.parameters()).device
        

        self.layers_to_hook_layer_list=[]
        self.layers_to_hook_name_list=[]

        self.layers_to_hook = {}
        self.layers_to_hook["embed_tokens"]=[self.model.model.embed_tokens]
        self.layers_to_hook_layer_list.append(self.model.model.embed_tokens)
        self.layers_to_hook_name_list.append(["embed_tokens",0])

        self.number_tokens=[]
        for ac_num in range(1001):
            self.number_tokens.append(self.tokenizer(str(ac_num), return_tensors="pt").input_ids[0][1].item())

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

        #self.layers_to_hook["rotary_emb_end"]=[self.model.model.rotary_emb]
        #self.layers_to_hook_layer_list.append(self.model.model.rotary_emb)
        #self.layers_to_hook_name_list.append(["rotary_emb_end",0])


        # Step 4: Initialize hook
        self.hooks=[]
        self.Mode="Features" #["Features","Interception","Nothing"]
        self.Interception_Layer=None
        for layer_name, layer_arr in self.layers_to_hook.items():
            for layer_pos,layer in enumerate(layer_arr):
                hook = layer.register_forward_hook(self.create_hook_fn_hidden_features(layer_name,layer_pos))
                self.hooks.append(hook)



    def Get_Data(self,Task_Text,Task_Result):
        # Prepare Input
        with torch.no_grad():
            Task_Result_Token,Task_Text_Tokens=self.Prepare_Input(Task_Text,Task_Result)

            # Forward pass
            Model_output=self.model(**Task_Text_Tokens)
            #print("mout", self.model.generate(**Task_Text_Tokens, max_new_tokens=1))
            #print("mout", self.tokenizer.decode(self.model.generate(**Task_Text_Tokens, max_new_tokens=1)[0]))
            # Prepare outputs
            logits=Model_output.logits[0][-1][self.number_tokens]
        return logits,Task_Result_Token

    

    def selective_exchange(self, input_tensor):
        # Extract relevant tensors from your attributes
        intervention_tensor = self.extracted_outputs[self.Interception_Layer[0]][self.Interception_Layer[1]]
        masking_array = self.masking_dictionary[self.Interception_Layer[0]][self.Interception_Layer[1]]
        
        # Step 1: Sort the input tensor and determine the sorted indices
        sorted_indices = np.argsort(masking_array)
        
        # Step 2: Calculate the number of features to change based on change_array
        n = intervention_tensor.shape[2]  # n is the number of features (last dimension)
        start_percentile = int(self.change_arr[0] * n)  # Starting index for top 50% (for example)
        end_percentile = int(self.change_arr[1] * n)  # Ending index for top 75% (for example)
        
        # Step 3: Identify the features that need to be changed (between the start and end percentiles)
        change_indices = sorted_indices[start_percentile:end_percentile]
        
        #print(intervention_tensor.shape)
        #print(input_tensor.shape)
        #print(change_indices)
        #print(len(change_indices))
        
        # Step 4: Intervene by setting the values of the selected features to the corresponding values from the intervention tensor
        # Loop through each batch and each word in the sequence
        for acb in range(input_tensor.shape[0]):  # Iterate over the batch dimension
            for aw in range(input_tensor.shape[1]):  # Iterate over the word dimension
                # Change the values in the selected features for each word and batch element
                #print(intervention_tensor.shape)
                input_tensor[acb, aw, change_indices] = intervention_tensor[acb, aw, change_indices]
    
        # Return the modified input tensor
        return input_tensor.detach()


    def create_hook_fn_hidden_features(self,layer_name,layer_index):
        def hook_fn(module, input, output):
            if self.Mode=="Features":
                if layer_name not in self.extracted_outputs:
                    self.extracted_outputs[layer_name] = {}
                self.extracted_outputs[layer_name][layer_index]=output
            elif self.Mode=="Interception":
                #print("h1")
                if layer_name==self.Interception_Layer[0] and layer_index==self.Interception_Layer[1]:
                    #print("h2")
                    return self.selective_exchange(output)
        return hook_fn




    def Prepare_Input(self,Task_Text,Task_Result):
        #TODO
        #print(Task_Result)
        Task_Result_Token=self.tokenizer(Task_Result, return_tensors="pt").input_ids[0][1].item()
        Task_Text_Tokens = self.tokenizer(Task_Text, return_tensors="pt")
        Task_Text_Tokens = {key: value.to(self.model_device) for key, value in Task_Text_Tokens.items()}
        return Task_Result_Token,Task_Text_Tokens


    def Cleanup(self):
        gc.collect()
        self.extracted_outputs={}
        
    def jensen_shannon_divergence(self,P, Q):
        M = 0.5 * (P + Q)
        return 0.5 * (F.kl_div(M.log(), P, reduction='batchmean') + F.kl_div(M.log(), Q, reduction='batchmean'))

    def make_scores(self,
                    Logits_Base_Task,
                    Result_Token_Base_Task,
                    Logits_Inter_Task,
                    Result_Token_Inter_Task,
                    Logits_Intervention,
                    Result_Token_Intervention):
        
        Gold_Result_Base_Task=int(self.tokenizer.decode(Result_Token_Base_Task))
        Gold_Result_Inter_Task=int(self.tokenizer.decode(Result_Token_Inter_Task))
        #print(Gold_Result_Base_Task,Gold_Result_Inter_Task)
        #print(Logits_Base_Task)
        max_Base_Task=torch.argmax(Logits_Base_Task).item()
        max_Inter_Task=torch.argmax(Logits_Inter_Task).item()
        max_Intervention=torch.argmax(Logits_Intervention).item()

        Pred_Result_Base_Task=int(self.tokenizer.decode(self.number_tokens[max_Base_Task]))
        Pred_Result_Inter_Task=int(self.tokenizer.decode(self.number_tokens[max_Inter_Task]))
        Pred_Result_Intervention=int(self.tokenizer.decode(self.number_tokens[max_Intervention]))
        #print(Pred_Result_Base_Task,Pred_Result_Inter_Task,Pred_Result_Intervention)

        #print(Logits_Base_Task)
        Prob_Base_Task = F.softmax(Logits_Base_Task,dim=0)
        Prob_Inter_Task = F.softmax(Logits_Inter_Task,dim=0)
        Prob_Intervention = F.softmax(Logits_Intervention,dim=0)

        result={}
        result["correctly_classified"]={}
        result["correctly_classified"]["base"]=0
        #print(Gold_Result_Base_Task,Pred_Result_Base_Task)
        if Gold_Result_Base_Task==Pred_Result_Base_Task:
            result["correctly_classified"]["base"]=1
        result["correctly_classified"]["intervention"]=0
        if Gold_Result_Inter_Task==Pred_Result_Inter_Task:
            result["correctly_classified"]["intervention"]=1

        
        result["correctly_classified_intervent"]={}
        result["correctly_classified_intervent"]["base"]=0
        if Gold_Result_Base_Task==Pred_Result_Intervention:
            result["correctly_classified_intervent"]["base"]=1
        result["correctly_classified_intervent"]["intervention"]=0
        if Gold_Result_Inter_Task==Pred_Result_Intervention:
            result["correctly_classified_intervent"]["intervention"]=1

        result["similar_classified"]={}
        result["similar_classified"]["base"]=0
        if Pred_Result_Base_Task==Pred_Result_Intervention:
            result["similar_classified"]["base"]=1
        result["similar_classified"]["intervention"]=0
        if Pred_Result_Inter_Task==Pred_Result_Intervention:
            result["similar_classified"]["intervention"]=1

        result["abs_difference_gold"]={}
        result["abs_difference_gold"]["base"]=abs(Gold_Result_Base_Task-Pred_Result_Intervention)
        result["abs_difference_gold"]["intervention"]=abs(Gold_Result_Inter_Task-Pred_Result_Intervention)
        
        result["abs_difference_pred"]={}
        result["abs_difference_pred"]["base"]=abs(Pred_Result_Base_Task-Pred_Result_Intervention)
        result["abs_difference_pred"]["intervention"]=abs(Pred_Result_Inter_Task-Pred_Result_Intervention)

        result["JS_Divergence"]={}
        result["JS_Divergence"]["base"] = float(self.jensen_shannon_divergence(Prob_Base_Task,Prob_Intervention))
        result["JS_Divergence"]["intervention"] = float(self.jensen_shannon_divergence(Prob_Inter_Task,Prob_Intervention))
        
        return result
        
    def get_results(self,Task_base_Text,Task_base_Result,Task_Inter_Text,Task_Inter_Result):
        #TODO
        self.Mode="Nothing"
        Logits_Base_Task, Result_Token_Base_Task = self.Get_Data(Task_base_Text,Task_base_Result)
        #Extract features and output
        self.Mode="Features"
        Logits_Inter_Task, Result_Token_Inter_Task = self.Get_Data(Task_Inter_Text,Task_Inter_Result)
        #Extract features and output
        self.Mode="Interception"
        result={}
        #print("Interception")
        for ac_Layer in self.layers_to_hook_name_list:
            if ac_Layer[0] not in result:
                result[ac_Layer[0]]={}
            self.Interception_Layer=ac_Layer
            Logits_Intervention, Result_Token_Intervention=self.Get_Data(Task_base_Text,Task_base_Result)
            result[ac_Layer[0]][str(ac_Layer[1])]=self.make_scores(
                Logits_Base_Task,
                Result_Token_Base_Task,
                Logits_Inter_Task,
                Result_Token_Inter_Task,
                Logits_Intervention,
                Result_Token_Intervention,
            )
        return result


# Instantiate with dynamic num_gpus setting
def create_llm_remote(model_id, max_memory, tokenizer,masking_dictionary,change_arr):
    return LLM_remote(model_id, max_memory,tokenizer,masking_dictionary,change_arr)
