#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
import time
import os
import transformers
import torch
import stanza
import json

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
import numpy as np

from tqdm import tqdm
from datetime import datetime
import gc
from numba import cuda
import ray
from accelerate import infer_auto_device_map, init_empty_weights

ray.init()


# In[2]:


stanza.download('en')       
nlp = stanza.Pipeline('en') 


# In[3]:


@ray.remote(num_gpus=1)
class LLM_remote:
    def __init__(self):
        model_id = "Local-Meta-Llama-3.1-8B-Instruct"
        
        # Step 2: Use infer_auto_device_map to get the device map for the model
        
        
        with init_empty_weights():
            my_model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
        device_map = infer_auto_device_map(my_model, max_memory={0: "14GiB", "cpu": "30GiB"})
        
        # Step 3: Create the pipeline using the model and tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map=device_map,
        )

    def generate(self,messages):
        with torch.no_grad():
            outputs = self.pipeline(
                messages,
                max_new_tokens=512,
            )
        return outputs



# In[4]:


global_LLM_Reuses=50

global_LLM_usage_counter=0
global_LLM_handler=None
def LLM(messsages):
    global global_LLM_usage_counter
    global global_LLM_handler
    global global_LLM_Reuses
    torch.cuda.empty_cache()
    gc.collect()
    if global_LLM_handler is None:
        global_LLM_handler=LLM_remote.remote()
        global_LLM_usage_counter=0
    elif global_LLM_usage_counter>=global_LLM_Reuses:
        ray.kill(global_LLM_handler)
        global_LLM_handler=LLM_remote.remote()
        global_LLM_usage_counter=0
    global_LLM_usage_counter+=1
    return_value=ray.get(global_LLM_handler.generate.remote(messsages)) 
    return return_value


# In[5]:


if False:
    ia=0
    while True:
        ia+=1
        messages = [
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": Prompt_Warmstart},
            ]
        print(LLM(messages))
        break


# In[6]:


Prompt_Warmstart="""You are assisting me with automated machine learning optimizing a function.
    This function is a nonlinear black-box problem involving three variables using polynomial functions. 
    I’m exploring a subset of hyperparameters detailed as: Variable x which is an float between -1000 and 1000, Variable y which is an float between -1000 and 1000 and Variable z which is an float between -1000 and 1000.
    Please suggest 5 diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning.
    You mustn't include ‘None’ in the configurations.
    Your response should include only a list of dictionaries, where each dictionary describes one recommended configuration. Do not enumerate the dictionaries. 
    The dictionaries are in form of {"Variable x": float, "Variable y": float, "Variable z": float} by exchanging "float" with the float values.
    The final output therefore should look like "## [{...},{...},{...}] ##" where "{...}" are exchanged with the dictionaries.
    """


# In[7]:


Prompt_Context="""The following are examples of the performance of a function and the corresponding model hyperparameter configurations.
    This function is a nonlinear black-box problem involving three variables using polynomial functions.
    The input includes 3 features (numerical). """
Prompt_Performance_Prediction='Your response should only contain the predicted accuracy in the format "## performance ##".'

""" Hyperparameter configuration: {configuration 1}
    Performance: {performance 1}
    ...
    Hyperparameter configuration: {configuration n}
    Performance: {performance n}
    Hyperparameter configuration: {configuration to predict performance}
    Performance:
"""
pass


# In[8]:


Prompt_Candidate_Suggestion="""The allowable ranges for the hyperparameters are: Variable x with an float between -1000 and 1000, Variable y with an float between -1000 and 1000 and Variable z with an float between -1000 and 1000. 
    Recommend a configuration that can achieve the target performance of [target score]. 
    Your response must only contain the predicted configuration, in the format ## {"Variable x": float, "Variable y": float, "Variable z": float} ## exchanging "float" with the float values. 
    Please ensure to suggest a new hyperparameter configuration not used and evaluated before.
    """

"""
    Performance: {performance 1}
    Hyperparameter configuration: configuration 1
    ...
    Performance: {performance n}
    Hyperparameter configuration: {configuration n}
    Performance: {performance used to sample configuration}
    Hyperparameter configuration:
    """
pass


# In[9]:

"""
def Blackbox_Function_2(x,y,z):
    return (x+((abs(z)+1)**(y/1000)))-(abs(y)+(z**2+1)**(x/1000))-(abs(z)+1)**(y/1000)
"""
"""
Blackbox_Function_1_Config={}
Blackbox_Function_1_Config["Blackbox"]=Blackbox_Function_1_Config
Blackbox_Function_1_Config["Description"]="a function having the form  of (x+(z**y))/y*x**z-z**y"
Blackbox_Function_1_Config["Hyperparameter"]=[]
Blackbox_Function_1_Config["Hyperparameter"].append(["Variable x","Float between -1000 and 1000"])
"""
pass

def Blackbox_Function_1(x, y, z):
    # Function with a global minimum at x=500, y=-400, z=234
    return (x - 500)**2 + (y + 400)**2 + (z - 234)**2

# In[10]:


Variation={}
Variation["Sugestion_Algorithm"]=["LLM","Bayesian","Best"]
Variation["Use_Context"]=[True,False]
Variation["Warm_Start"]=[True,False]


# In[11]:


def Is_IFD(inp):
    return (not (isinstance(ac_sample["Variable x"], float) or isinstance(ac_sample["Variable x"], int) ))


# In[12]:


def Check_Dictionary(ac_sample):
    if "Variable x" not in ac_sample:
        raise ValueError('Variable x not available')
    elif Is_IFD(ac_sample["Variable x"]):
        raise ValueError('Variable x is not float')
    elif ac_sample["Variable x"]<-1000 or ac_sample["Variable x"]>1000:
        raise ValueError('Variable x not in range')
        
    if "Variable y" not in ac_sample:
        raise ValueError('Variable y not available')
    elif Is_IFD(ac_sample["Variable y"]):
        raise ValueError('Variable y is not float')
    elif ac_sample["Variable y"]<-1000 or ac_sample["Variable y"]>1000:
        raise ValueError('Variable y not in range')
        
    if "Variable z" not in ac_sample:
        raise ValueError('Variable z not available')
    elif Is_IFD(ac_sample["Variable z"]):
        raise ValueError('Variable z is not float')
    elif ac_sample["Variable z"]<-1000 or ac_sample["Variable z"]>1000:
        raise ValueError('Variable z not in range')



# In[13]:


train_X=[]
train_X_Dict=[]
train_Y=[]
train_Y_Minus=[]

Results={}

#Note LLM minimizes and Botorch maximizes
acround=0
totalrounds=len(Variation["Sugestion_Algorithm"])*len(Variation["Use_Context"])*len(Variation["Warm_Start"])
for SA in Variation["Sugestion_Algorithm"]:
    for UC in Variation["Use_Context"]:
        for WS in Variation["Warm_Start"]:
            print()
            print()
            acround+=1
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(acround,'/',totalrounds,'(',current_time,')',':')
            print('*'*100)
            print('Sugestion_Algorithm:',SA)
            print('Use_Context:',UC)
            print('Warm_Start:',WS)
            if SA not in Results:
                Results[SA]={}
            if UC not in Results[SA]:
                Results[SA][UC]={}
            if WS not in Results[SA][UC]:
                Results[SA][UC][WS]=[]
            #print(WS)
            if WS:
                #make Warmstart
                messages = [
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": Prompt_Warmstart},
                ]
                while True:
                    try:
                        outputs=LLM(messages)
                        output=outputs[0]['generated_text'][2]['content']
                        output=output.split("##")
                        if len(output)!=3 and len(output)!=2:
                            raise ValueError('Warmstart: "## ... ##" output formation not fulfilled')
                        output=json.loads(output[1].replace("'", '"'))
                        if len(output)!=5:
                            raise ValueError('Warmstart: Not 5 suggested examples')
                        for ac_sample in output:
                            Check_Dictionary(ac_sample)
                                
                            train_X.append([ac_sample["Variable x"],ac_sample["Variable y"],ac_sample["Variable z"]])
                            resulti=Blackbox_Function_1(ac_sample["Variable x"],ac_sample["Variable y"],ac_sample["Variable z"])
                            train_Y.append([resulti])
                            train_Y_Minus.append([-resulti])
                            train_X_Dict.append(ac_sample)
                        break
                    except Exception as e:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print('[ERROR]',current_time,'LLM warm start')
                        print(e)
                    

            for ac_iteration in tqdm(range(1000)):
                
                #Get suggestion of LLM
                ##Get candidates of LLM
                input_text_sampler=""
                input_text_predictor=""
                if UC:
                    input_text_sampler+=Prompt_Context+"\n"
                
                alpha=0.01
                smax=max(train_Y)[0]
                smin=min(train_Y)[0]
                snew=smin-(alpha*(smax-smin))
                Prompt_Candidate_Suggestion_c=Prompt_Candidate_Suggestion
                Prompt_Candidate_Suggestion_c=Prompt_Candidate_Suggestion_c.replace("[target score]", str(snew))
                input_text_sampler+=Prompt_Candidate_Suggestion_c

                for p_i,i in enumerate(train_X_Dict):
                    input_text_sampler+="\nPerformance: "+str(train_Y[p_i][0])
                    input_text_sampler+="\nHyperparameter configuration: ## "+str(i)+" ##"
                input_text_sampler+="\nPerformance: "+str(snew)
                input_text_sampler+="\nHyperparameter configuration:"
                
                #print(input_text_sampler)
                messages = [
                    {"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": input_text_sampler},
                ]
                Candidates=[]
                for _ in range(10): #Todo 10
                    while True:
                        try:
                            outputs=LLM(messages)
                            output=outputs[0]['generated_text'][2]['content']
                            output=output.split("##")
                            if len(output)!=3 and len(output)!=2:
                                raise ValueError('Sampler: "## ... ##" output formation not fulfilled')
                            #print(output)
                            ac_sample=json.loads(output[1].replace("'", '"'))
                            Check_Dictionary(ac_sample)
                            Candidates.append(ac_sample)
                            break
                        except Exception as e:
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")
                            print('[ERROR]',current_time,'LLM candidate generation')
                            print(e)
                #print(Candidates)

                ##Extract Best Sample
                best_val=None
                best_hypers=None
                if UC:
                    input_text_predictor=Prompt_Context+"\n"
                input_text_predictor+=Prompt_Performance_Prediction
                for p_i,i in enumerate(train_X_Dict):
                    input_text_predictor+="\nHyperparameter configuration: "+str(i)
                    input_text_predictor+="\nPerformance: ## "+str(train_Y[p_i][0])+" ##"
                    
                for j in Candidates:
                    input_text_predictor_c=input_text_predictor
                    input_text_predictor_c+="\nHyperparameter configuration: "+str(j)
                    input_text_predictor_c+="\nPerformance:"

                    messages = [
                        {"role": "system", "content": "You are an AI assistant that helps people find information."},
                        {"role": "user", "content": input_text_predictor_c},
                    ]
                    
                    #print('*'*100)
                    #print(messages)
                    while True:
                        try:
                            outputs=LLM(messages)
                            #print(outputs)
                            output=outputs[0]['generated_text'][2]['content'] 
                            output=output.split("##")
                            if len(output)!=3 and len(output)!=2:
                                raise ValueError('Sampler: "## ... ##" output formation not fulfilled')
                            output=float(output[1].replace(",","."))
                            if (best_val is None) or best_val>output:
                                best_val=output
                                best_hypers=j
                            break
                        except Exception as e:
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")
                            print('[ERROR]',current_time,'LLM performance prediction:')
                            print(e)
                LLM_Suggestion=best_hypers  

                
                #Get suggestion of Bayesian optimizer
                tensor_train_X=torch.tensor(train_X).to(torch.double)
                tensor_train_Y=torch.tensor(train_Y_Minus).to(torch.double)
                gp = SingleTaskGP(
                    train_X=tensor_train_X,
                    train_Y=tensor_train_Y,
                    input_transform=Normalize(d=3),
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                
                logEI = LogExpectedImprovement(model=gp, best_f=tensor_train_Y.max())
                
                bounds = torch.stack([torch.full((3,),-1000), torch.full((3,),1000)]).to(torch.double)
                candidate, acq_value = optimize_acqf(
                    logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
                )
                Bayes_sugg=candidate[0].tolist()

                #print(Bayes_sugg)
                #print(LLM_Suggestion)
                #evaluate
                ##LLM Perf
                LLM_Y=Blackbox_Function_1(LLM_Suggestion["Variable x"],LLM_Suggestion["Variable y"],LLM_Suggestion["Variable z"])

                ## Bayes Perf                
                Bay_Y=Blackbox_Function_1(Bayes_sugg[0],Bayes_sugg[1],Bayes_sugg[2])

                #update history accrding to SA
                Results[SA][UC][WS].append([Bay_Y-LLM_Y,Bay_Y,LLM_Y])
                with open("Results_LLMvsBay.json", 'w') as json_file:
                    json.dump(Results, json_file)
                with open("train_X_save.json", 'w') as json_file:
                    json.dump(train_X, json_file)
                with open("train_X_Dict_save.json", 'w') as json_file:
                    json.dump(train_X_Dict, json_file)
                with open("train_Y_save.json", 'w') as json_file:
                    json.dump(train_Y, json_file)
                with open("train_Y_Minus_save.json", 'w') as json_file:
                    json.dump(train_Y_Minus, json_file)
                    
                if SA=="LLM" or (SA=="Best" and Bay_Y>=LLM_Y):
                    train_X.append([LLM_Suggestion["Variable x"],LLM_Suggestion["Variable y"],LLM_Suggestion["Variable z"]])
                    train_X_Dict.append(LLM_Suggestion)
                    train_Y.append([LLM_Y])
                    train_Y_Minus.append([-LLM_Y])
                elif SA=="Bayesian" or (SA=="Best" and Bay_Y<=LLM_Y):    
                    acDict={}
                    acDict["Variable x"]=Bayes_sugg[0]
                    acDict["Variable y"]=Bayes_sugg[1]
                    acDict["Variable z"]=Bayes_sugg[2]                
                    train_X.append(Bayes_sugg)
                    train_X_Dict.append(acDict)
                    train_Y.append([Bay_Y])
                    train_Y_Minus.append([-Bay_Y])
                else:
                    raise ValueError('tack([torch.(x+(abs(z)**(y/1000)))-(abs(y)+(z**2+Unknown Keyword for SA')
                #print(Results)            
                #print(train_X)
                #print(train_X_Dict)
                #print(train_Y)
                #print(train_Y_Minus)
                
                    
                
            
            


# In[ ]:


train_X
train_X_Dict
train_Y
train_Y_Minus


# In[ ]:





# In[ ]:


input_text_predictor


# In[ ]:




