import LLM_Tasks
import Relevance_Maps
import Probing
import Interventions
from transformers import AutoTokenizer
import torch
import numpy as np
from collections import OrderedDict
import gc

#Hyperparameters:
Relevance_Steps=2#100
Probing_Steps=2#2000
Testing_Samples=2#100
Intervention_Steps=2#100
#torch.set_num_threads(6)

max_examples_token_length=300
Use_Context=True
Relevance_Map_Method="vanilla_gradient" 
Probing_Method='Probing'

"""
model_id = "meta-llama/Llama-3.2-1B" 
max_memory=OrderedDict([
    ('model.embed_tokens', 0), 
    ('model.layers.0', 0), 
    ('model.layers.1', 1), 
    ('model.layers.2', 1), 
    ('model.layers.3', 1), 
    ('model.layers.4', 1), 
    ('model.layers.5', 1),
    ('model.layers.6', 1), 
    ('model.layers.7', 1), 
    ('model.layers.8', 1), 
    ('model.layers.9', 1), 
    ('model.layers.10', 1), 
    ('model.layers.11', 1), 
    ('model.layers.12', 0), 
    ('model.layers.13', 0),
    ('model.layers.14', 0), 
    ('model.layers.15', 0), 
    ('model.norm', 0), 
    ('model.rotary_emb', 0), 
    ('lm_head', 0), 
    ('model.layers.13.mlp', 0)])
"""

#"""
model_id="Local-Meta-Llama-3.2-3B"
max_memory=OrderedDict([('model.embed_tokens', 0), 
                        ('model.layers.0', 1), 
                        ('model.layers.1', 1), 
                        ('model.layers.2', 1), 
                        ('model.layers.3', 1), 
                        ('model.layers.4', 1), 
                        ('model.layers.5', 1), 
                        ('model.layers.6', 1), 
                        ('model.layers.7', 1), 
                        ('model.layers.8', 1), 
                        ('model.layers.9', 1), 
                        ('model.layers.10', 2), 
                        ('model.layers.11', 2), 
                        ('model.layers.12', 2), 
                        ('model.layers.13', 2), 
                        ('model.layers.14', 2), 
                        ('model.layers.15', 2), 
                        ('model.layers.16', 2), 
                        ('model.layers.17', 2), 
                        ('model.layers.18', 2), 
                        ('model.layers.19', 2), 
                        ('model.layers.20', 3), 
                        ('model.layers.21', 3), 
                        ('model.layers.22', 3), 
                        ('model.layers.23', 3), 
                        ('model.layers.24', 3), 
                        ('model.layers.25', 3), 
                        ('model.layers.26', 3), 
                        ('model.layers.27', 3), 
                        ('model.norm', 0), 
                        ('model.rotary_emb', 0), 
                        ('lm_head', 0)])
#"""


tokenizer = AutoTokenizer.from_pretrained(model_id)




#Step 1: Select Tasks
print("[INFO] Step 1: Prepare Tasks")

"""
original_list = list(range(23))

# Shuffle the list randomly
np.random.shuffle(original_list)

# If the length is odd, ignore the last element
if len(original_list) % 2 != 0:
    original_list = original_list[:-1]

# Split into two equal parts
half_size = len(original_list) // 2
array1 = original_list[:half_size]
array2 = original_list[half_size:]
"""

Task_1=LLM_Tasks.Regression_Task_Int(
    tokenizer,
    Display_Context=Use_Context,
    max_examples_token_length=max_examples_token_length,
    #number_range_weights=array1,
    #Task_Name="Task1"
)
Task_2=LLM_Tasks.Multiclass_Logistic_Regression_Task(
    tokenizer,
    Display_Context=Use_Context,
    max_examples_token_length=max_examples_token_length,
    #number_range_weights=array2,
    #Task_Name="Task2"
)

Task_1_Test=LLM_Tasks.Regression_Task_Int(
    tokenizer,
    Display_Context=Use_Context,
    Testing_samples_num=Testing_Samples,
    max_examples_token_length=max_examples_token_length,
    #number_range_weights=array1,
    #Task_Name="Task1"
)
Task_2_Test=LLM_Tasks.Multiclass_Logistic_Regression_Task(
    tokenizer,
    Display_Context=Use_Context,
    Testing_samples_num=Testing_Samples,
    max_examples_token_length=max_examples_token_length,
    #number_range_weights=array2,
    #Task_Name="Task2"
)
print("[INFO] Step 1: Finsished")


#Step 2: Use different methods to compute relevance maps
print("[INFO] Step 2: Create Relevance Maps")
RelMap=Relevance_Maps.Relevance_Map(
    model_id,
    Relevance_Map_Method,
    Task_1,
    Task_2,
    tokenizer,
    max_memory=max_memory
)
Rel_Map_Result=RelMap.Get_Relevance_Map(Number_of_samples=Relevance_Steps)
print("[INFO] Step 2: Finished")



#Step 3: Use relevance map for probing
print("[INFO] Step 3: Probing with 1 Layer")
MyProbing=Probing.Probing(
    model_id,
    Probing_Method,
    Task_1_Test,
    Task_2_Test,
    tokenizer,
    Rel_Map_Result,
    Testing_Samples,
    probing_layers=1,
    max_memory=max_memory
)
MyProbing.Get_Probing_Results(Number_of_samples=Probing_Steps)
print("[INFO] Step 3: Finished")


"""
#Step 4: Use relevance map for probing
print("[INFO] Step 4: Probing with 2 Layers")
MyProbing=Probing.Probing(
    model_id,
    Probing_Method,
    Task_1_Test,
    Task_2_Test,
    tokenizer,
    Rel_Map_Result,
    Testing_Samples,
    probing_layers=2,
    Allowed_Model_Usage_Before_Refresh=200,
    max_memory=max_memory,
    num_gpus=1,
    num_cpus=10
)
MyProbing.Get_Probing_Results(Number_of_samples=2000)
print("[INFO] Step 4: Finished")
"""


#Step 2: Make experiments with Intervention
print("[INFO] Step 5: Intervention")
tasksettings=[]
tasksettings.append([Task_1,Task_1])
tasksettings.append([Task_1,Task_2])
tasksettings.append([Task_2,Task_1])
tasksettings.append([Task_2,Task_2])

rangesettings=[]
rangesettings.append([0,0.5])
rangesettings.append([0.5,1])

for actasks in tasksettings:
    for acrange in rangesettings:
        print("[INFO] Step 5: Substep:",actasks[0].Task_Name,";",actasks[1].Task_Name,";",str(acrange))
        Inti=Interventions.Intervention(
            model_id,
            acrange,
            actasks[0],
            actasks[1],
            tokenizer,
            Rel_Map_Result,
            max_memory=max_memory,
        )
        Inti.Get_Intervention_Results(Number_of_samples=Intervention_Steps)
print("[INFO] Step 5: Intervention")

