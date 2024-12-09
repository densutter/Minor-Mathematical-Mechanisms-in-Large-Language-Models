import LLM_Tasks
import Relevance_Maps
import Probing
from transformers import AutoTokenizer
import torch
torch.set_num_threads(6)

#Hyperparameters:
model_id = "meta-llama/Llama-3.2-1B" #"Local-Meta-Llama-3.2-1B"
Relevance_Map_Method='captum-GradientXActivation' #'smoothGrad'
Probing_Method='Probing'
tokenizer = AutoTokenizer.from_pretrained(model_id)
Use_Context=True
Testing_Samples=100


#Step 1: Select Tasks
print("[INFO] Step 1: Prepare Tasks")
Task_1=LLM_Tasks.Regression_Task_Int(tokenizer,Display_Context=Use_Context)
Task_2=LLM_Tasks.Multiclass_Logistic_Regression_Task(tokenizer,Display_Context=Use_Context)

Task_1_Test=LLM_Tasks.Regression_Task_Int(tokenizer,Display_Context=Use_Context,Testing_samples_num=Testing_Samples)
Task_2_Test=LLM_Tasks.Multiclass_Logistic_Regression_Task(tokenizer,Display_Context=Use_Context,Testing_samples_num=Testing_Samples)
print("[INFO] Step 1: Finsished")


#Step 2: Use different methods to compute relevance maps
print("[INFO] Step 2: Create Relevance Maps")
RelMap=Relevance_Maps.Relevance_Map(
    model_id,
    Relevance_Map_Method,
    Task_1,
    Task_2,
    tokenizer,
    Allowed_Model_Usage_Before_Refresh=10,
    max_memory='cpu',
    num_gpus=0,
    num_cpus=6
)
Rel_Map_Result=RelMap.Get_Relevance_Map(Number_of_samples=100)
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
    Allowed_Model_Usage_Before_Refresh=200,
    max_memory={0: "10GiB", "cpu": "30GiB"},
    num_gpus=1,
    num_cpus=10
)
MyProbing.Get_Probing_Results(Number_of_samples=2000)
print("[INFO] Step 3: Finished")


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
    max_memory={0: "10GiB", "cpu": "30GiB"},
    num_gpus=1,
    num_cpus=10
)
MyProbing.Get_Probing_Results(Number_of_samples=2000)
print("[INFO] Step 4: Finished")