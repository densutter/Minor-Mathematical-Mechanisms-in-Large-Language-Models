#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
import random
from os import listdir
import zipfile
import pickle

model_id = "Local-Meta-Llama-3.2-1B"
random.seed(13)




#Helper Functions
tokenizer = AutoTokenizer.from_pretrained(model_id)
def MakeRegressionTask(tokenizer,Context=False,max_examples_token_length=200):
    td=''

    weights=[random.randint(0, 100),random.randint(0, 100),random.randint(0, 100)]

    while True:
        inN=[random.randint(0, 100),random.randint(0, 100),random.randint(0, 100)]
        res=weights[0]*inN[0]+weights[1]*inN[1]+weights[2]*inN[2]
        td_new=td+'input = ( '+str(inN[0])+' , '+str(inN[1])+' , '+str(inN[2])+' ) ; output = '+str(res)+' \n'
        if tokenizer(td, return_tensors="pt").input_ids.shape[1]>max_examples_token_length:
            break
        else:
            td=td_new

    inN=[random.randint(0, 100),random.randint(0, 100),random.randint(0, 100)]
    res=weights[0]*inN[0]+weights[1]*inN[1]+weights[2]*inN[2]
    td=td+'input = ( '+str(inN[0])+' , '+str(inN[1])+' , '+str(inN[2])+' ) ; output = '
    if Context:
        td='The output represents the result of this linear equation given the input as the 3 input numbers: \n\n'+td
    return td,str(res)

f=open('./Datasets/positive-words.txt',"r")
pos_words=f.read().split('\n')[:-1]
f.close()
f=open('./Datasets/negative-words.txt',"r",encoding="ISO-8859-1")
neg_words=f.read().split('\n')[:-1]
f.close()

def MakeClassificationTask(tokenizer,Context=False,max_examples_token_length=200):
    td=''
    
    while True:

        if random.randint(0, 1)==1:
            td_new=td+'input = '+pos_words[random.randint(0, len(pos_words)-1)]+' ; output = positiv \n'
        else:
            td_new=td+'input = '+neg_words[random.randint(0, len(neg_words)-1)]+' ; output = negativ \n'
        if tokenizer(td, return_tensors="pt").input_ids.shape[1]>max_examples_token_length:
            break
        else:
            td=td_new

    res=None
    if random.randint(0, 1)==1:
        td=td+'input = '+pos_words[random.randint(0, len(pos_words)-1)]+' ; output = '
        res='positiv'
    else:
        td=td+'input = '+neg_words[random.randint(0, len(neg_words)-1)]+' ; output = '
        res='negativ'
    if Context:
        td='The following words are classified by the sentiment they imply: \n\n'+td
    return td,res


def move_to_cpu_with_grad(data):
    if isinstance(data, torch.Tensor):  # Check if it's a tensor
        # Move to CPU, ensure requires_grad is True, and retain gradients
        #data = data.to('cpu')
        return data
    elif isinstance(data, dict):  # If it's a dictionary, recursively check its values
        return {key: move_to_cpu_with_grad(value) for key, value in data.items()}
    elif isinstance(data, list):  # If it's a list, recursively check each element
        return [move_to_cpu_with_grad(item) for item in data]
    elif isinstance(data, tuple):  # If it's a tuple, recursively check each element
        return tuple(move_to_cpu_with_grad(item) for item in data)
    else:
        return data  # If it's not a tensor, return it as-is


"""
def tensors_to_lists(data):
    if isinstance(data, torch.Tensor):  # Check if it's a tensor
        return data.grad.tolist()
    elif isinstance(data, dict):  # If it's a dictionary, recursively check its values
        return {key: tensors_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):  # If it's a list, recursively check each element
        return [tensors_to_lists(item) for item in data]
    elif isinstance(data, tuple):  # If it's a tuple, recursively check each element
        return tuple(tensors_to_lists(item) for item in data)
    else:
        return data  # If it's not a tensor, return it as-is
"""

# Hook function factory that returns a hook function for each layer

extracted_Tensor = None
def hook_fn(module, input, output):
    global extracted_Tensor
    if isinstance(output, torch.Tensor):    
        
        std_dev = 0.1 * (output.max() - output.min()).item()  # 10% of the range
        noise = torch.randn_like(output) * std_dev 
        output+=noise 
        
        output.requires_grad_(True)
        output.retain_grad() 

    
    else:

        new_output=[]
        for aop in range(len(output)): 
            
            std_dev = 0.1 * (output[aop].max() - output[aop].min()).item()  # 10% of the range
            noise = torch.randn_like(output[aop]) * std_dev 
            new_output.append(output[aop]+noise)
            
            new_output[aop].requires_grad_(True)
            new_output[aop].retain_grad() 
        output=tuple(new_output)
    extracted_Tensor=output

    return output



#initialization Model

with init_empty_weights():
    my_model = AutoModelForCausalLM.from_pretrained(model_id)
device_map = 'cpu' #infer_auto_device_map(my_model, max_memory={0: "6GiB", "cpu": "30GiB"})

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map=device_map)
model.gradient_checkpointing_enable()

def move_grad_to_cpu_hook(grad):
    return grad.cpu()  # Move the gradient to CPU (RAM)

# Register the hook for all parameters in the model
"""
for param in model.parameters():
     param.requires_grad = False
for param in model.lm_head.parameters():  # Unfreeze the output layer
    param.requires_grad = True
"""


#Create hooks for layers under examination

layers_to_hook = {}

layers_to_hook["embed_tokens"]=[model.model.embed_tokens]

for i in model.model.layers:
    
    if "q_proj" not in layers_to_hook:
        layers_to_hook["q_proj"]=[]
    layers_to_hook["q_proj"].append(i.self_attn.q_proj)
    
    if "k_proj" not in layers_to_hook:
        layers_to_hook["k_proj"]=[]
    layers_to_hook["k_proj"].append(i.self_attn.k_proj)
    
    if "v_proj" not in layers_to_hook:
        layers_to_hook["v_proj"]=[]
    layers_to_hook["v_proj"].append(i.self_attn.v_proj)

    if "o_proj" not in layers_to_hook:
        layers_to_hook["o_proj"]=[]
    layers_to_hook["o_proj"].append(i.self_attn.o_proj)
    
    if "rotary_emb" not in layers_to_hook:
        layers_to_hook["rotary_emb"]=[]
    layers_to_hook["rotary_emb"].append(i.self_attn.rotary_emb)
    
    if "mlp" not in layers_to_hook:
        layers_to_hook["mlp"]=[]
    layers_to_hook["mlp"].append(i.mlp)


layers_to_hook["rotary_emb"]=[model.model.rotary_emb]


#hooks = []
#for layer_name, layer_arr in layers_to_hook.items():
#    for layer_pos,layer in enumerate(layer_arr):
#        hook = layer.register_forward_hook(create_hook_fn(layer_name,layer_pos))
#        hooks.append(hook)


#Get the Gradients already stored 
actual_gradient_file_num=[len(listdir('./Raw_Gradients/0')),len(listdir('./Raw_Gradients/1'))]
#print(actual_gradient_file_num)




# In[2]:


#Evaluation Loop

Actual_Task=0

while True:
    

    extracted_outputs={}
    print('Processed:',actual_gradient_file_num,end='\r')
    
    Task_Text=None
    Task_Result=None
    if Actual_Task==0:
        Task_Text,Task_Result=MakeRegressionTask(tokenizer,Context=True,max_examples_token_length=200)
    else:
        Task_Text,Task_Result=MakeClassificationTask(tokenizer,Context=True,max_examples_token_length=200)
    Task_Result_Token=tokenizer(Task_Result, return_tensors="pt").input_ids[0][1].item()
    
    inputs = tokenizer(Task_Text, return_tensors="pt")
    
    for Layer_Name in layers_to_hook:
        for Ac_Layer_Index,Ac_Layer in enumerate(layers_to_hook[Layer_Name]):
            
            hook = Ac_Layer.register_forward_hook(hook_fn)

            for _ in range(10):
                
                extracted_Tensor = None
                
                
                with torch.autograd.graph.save_on_cpu():
                    with autocast():
                        M_outputs = model(**inputs)  
            
                model.zero_grad()
                loss = M_outputs.logits[0][-1][Task_Result_Token]
                GradScaler().scale(loss).backward()
                """
                if not  isinstance(extracted_Tensor, torch.Tensor):
                    new_extracted_Gradient=[]
                    for aet in extracted_Tensor:
                        new_extracted_Gradient.append(aet.grad.detach().numpy())
                    extracted_Gradient=new_extracted_Gradient
                else:
                    extracted_Gradient=extracted_Tensor.grad.detach().numpy()

                if Layer_Name not in extracted_outputs:
                    extracted_outputs[Layer_Name]={}
                if Ac_Layer_Index not in extracted_outputs[Layer_Name]:
                    extracted_outputs[Layer_Name][Ac_Layer_Index]=extracted_Gradient
                else:
                    if not isinstance(extracted_Gradient, torch.Tensor):
                        for paet,aet in enumerate(extracted_Gradient):
                            extracted_outputs[Layer_Name][Ac_Layer_Index][paet]+=aet
                    else:
                        extracted_outputs[Layer_Name][Ac_Layer_Index]+=extracted_Gradient
            if not isinstance(extracted_Gradient, torch.Tensor):
                for pa in range(len(extracted_Gradient)):
                    extracted_outputs[Layer_Name][Ac_Layer_Index][pa]=extracted_outputs[Layer_Name][Ac_Layer_Index][pa]/10
            else:
                extracted_outputs[Layer_Name][Ac_Layer_Index]=extracted_outputs[Layer_Name][Ac_Layer_Index]/10
                
            """
            hook.remove()
            
    #extracted_outputs=tensors_to_lists(extracted_outputs)

    """
    with zipfile.ZipFile('./Raw_Gradients/'+str(Actual_Task)+'/'+str(actual_gradient_file_num[Actual_Task])+'.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Save the string into the zip as a file
        with zipf.open(str(actual_gradient_file_num[Actual_Task])+'.json', 'w') as f:
            f.write(json.dumps(extracted_outputs).encode('utf-8'))
    """
    with open('./Raw_Gradients/'+str(Actual_Task)+'/'+str(actual_gradient_file_num[Actual_Task])+'.pkl', 'wb') as f:  # 'wb' mode for writing binary
        pickle.dump(extracted_outputs, f)
    
    actual_gradient_file_num[Actual_Task]+=1
    #with open('./Raw_Gradients/0/'+str(actual_gradient_file_num[Actual_Task])+'.json', 'w') as json_file:
    #    json.dump(extracted_outputs, json_file)  
    
        
            
    Actual_Task=1-Actual_Task





# In[ ]:



