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
Relevance_Steps=1000 
Probing_Steps=20000 
Testing_Samples=40 
Num_testing_iterations=100 
Layers_per_run=10000
Intervention_Steps=500 
Probing_learning_rate=0.0001

Max_examples_token_length=600
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


model_id="meta-llama/Llama-3.2-3B"

max_memory=OrderedDict([('model.embed_tokens', 7), 
                        ('model.layers.0', 6), 
                        ('model.layers.1', 6), 
                        ('model.layers.2', 6), 
                        ('model.layers.3', 6), 
                        ('model.layers.4', 6), 
                        ('model.layers.5', 6), 
                        ('model.layers.6', 6), 
                        ('model.layers.7', 6), 
                        ('model.layers.8', 6), 
                        ('model.layers.9', 6), 
                        ('model.layers.10', 6), 
                        ('model.layers.11', 6), 
                        ('model.layers.12', 6), 
                        ('model.layers.13', 6), 
                        ('model.layers.14', 5), 
                        ('model.layers.15', 5), 
                        ('model.layers.16', 5), 
                        ('model.layers.17', 5), 
                        ('model.layers.18', 5), 
                        ('model.layers.19', 5), 
                        ('model.layers.20', 5), 
                        ('model.layers.21', 5), 
                        ('model.layers.22', 5), 
                        ('model.layers.23', 5), 
                        ('model.layers.24', 5), 
                        ('model.layers.25', 5), 
                        ('model.layers.26', 5), 
                        ('model.layers.27', 5), 
                        ('model.norm', 7), 
                        ('model.rotary_emb', 7), 
                        ('lm_head', 7)])
"""


model_id="meta-llama/Llama-3.1-8B"
max_memory=OrderedDict([('model.embed_tokens', 0), 
                        ('model.layers.0', 1), 
                        ('model.layers.1', 1), 
                        ('model.layers.2', 1), 
                        ('model.layers.3', 1), 
                        ('model.layers.4', 1), 
                        ('model.layers.5', 1), 
                        ('model.layers.6', 2), 
                        ('model.layers.7', 2), 
                        ('model.layers.8', 2), 
                        ('model.layers.9', 2), 
                        ('model.layers.10', 2), 
                        ('model.layers.11', 3), 
                        ('model.layers.12', 3), 
                        ('model.layers.13', 3), 
                        ('model.layers.14', 3), 
                        ('model.layers.15', 4), 
                        ('model.layers.16', 4), 
                        ('model.layers.17', 4), 
                        ('model.layers.18', 4), 
                        ('model.layers.19', 5), 
                        ('model.layers.20', 5), 
                        ('model.layers.21', 5), 
                        ('model.layers.22', 5), 
                        ('model.layers.23', 6), 
                        ('model.layers.24', 6), 
                        ('model.layers.25', 6), 
                        ('model.layers.26', 6), 
                        ('model.layers.27', 6), 
                        ('model.layers.28', 7), 
                        ('model.layers.29', 7), 
                        ('model.layers.30', 7), 
                        ('model.layers.31', 7), 
                        ('model.norm', 0), 
                        ('model.rotary_emb', 0), 
                        ('lm_head', 0)])
"""

tokenizer = AutoTokenizer.from_pretrained(model_id)


#Step 1: Select Tasks
print("[INFO] Step 1: Prepare Tasks")



Task_1=LLM_Tasks.Regression_Task_Int(
    tokenizer,
    Display_Context=Use_Context,
    max_examples_token_length=Max_examples_token_length
)
Task_2=LLM_Tasks.Manhattan_Distance_Problem_Int(
    tokenizer,
    Display_Context=Use_Context,
    max_examples_token_length=Max_examples_token_length
)

Task_1_Test=LLM_Tasks.Regression_Task_Int(
    tokenizer,
    Display_Context=Use_Context,
    Testing_samples_num=Testing_Samples,
    max_examples_token_length=Max_examples_token_length
)
Task_2_Test=LLM_Tasks.Manhattan_Distance_Problem_Int(
    tokenizer,
    Display_Context=Use_Context,
    Testing_samples_num=Testing_Samples,
    max_examples_token_length=Max_examples_token_length
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
    max_memory=max_memory,
    layers_per_run=Layers_per_run,
    num_testing_iterations=Num_testing_iterations,
    learning_rate=Probing_learning_rate,
    Max_tokens=Max_examples_token_length+100

)
MyProbing.Get_Probing_Results(Number_of_samples=Probing_Steps)
print("[INFO] Step 3: Finished")

"""
#Possible code for probing with a 2 layer MLP:

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
    max_memory=max_memory,
    layers_per_run=Layers_per_run,
    num_testing_iterations=Num_testing_iterations,
    learning_rate=Probing_learning_rate,
    Max_tokens=Max_examples_token_length+100
)
MyProbing.Get_Probing_Results(Number_of_samples=Probing_Steps)
print("[INFO] Step 4: Finished")


"""
#Step 5: Make experiments with Intervention
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

