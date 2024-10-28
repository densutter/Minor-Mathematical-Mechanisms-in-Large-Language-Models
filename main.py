import LLM_Tasks
import Relevance_Maps
from transformers import AutoTokenizer


#Hyperparameters:
model_id = "Local-Meta-Llama-3.2-1B"
Relevance_Map_Method='vanilla_gradient' #'smoothGrad' 

tokenizer = AutoTokenizer.from_pretrained(model_id)


#Step 1: Select Tasks

Task_1=LLM_Tasks.Regression_Task_Int(tokenizer,Display_Context=True)
Task_2=LLM_Tasks.Word_Sentiment_Task(tokenizer,Display_Context=True)


#Step 2: Use different methods to compute relevance maps

RelMap=Relevance_Maps.Relevance_Map(model_id,Relevance_Map_Method,Task_1,Task_2,tokenizer)
RelMap.Get_Relevance_Map(Number_of_samples=10)
    

