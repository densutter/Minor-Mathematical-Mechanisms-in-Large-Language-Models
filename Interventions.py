import Prediction_Helpers
import Intervention_Model
import torch.functional as F
import torch.nn as nn
import torch
import numpy as np
import json
import os
import gc
from TimeMeasurer import TimeMeasurer

class Intervention(Prediction_Helpers.Prediction_Helper):
    def __init__(
        self,
        model_id,
        Change_arr, #e.g. [0.5,1.0] 
        task_1,
        task_2,
        tokenizer,
        relevance_map,
        probing_layers=1,
        interim_results_folder='./interim_results/',
        verbose=True,
        Allowed_Model_Usage_Before_Refresh=10, 
        max_memory="cpu",
        num_gpus=0,
        num_cpus=1,
        Max_tokens=400
        ):
        self.model_id=model_id
        self.Change_arr=Change_arr
        self.task_1=task_1
        self.task_2=task_2
        self.tokenizer=tokenizer
        self.relevance_map=relevance_map
        self.probing_layers=probing_layers
        self.interim_results_folder=interim_results_folder
        self.verbose=verbose
        self.Allowed_Model_Usage_Before_Refresh=Allowed_Model_Usage_Before_Refresh
        self.max_memory=max_memory
        self.num_gpus=num_gpus
        self.num_cpus=num_cpus
        self.Max_tokens=Max_tokens
        self.Actual_Model_Usage=0
        self.Computed_Samples=0
        self.results=None
        self.interim_results_path=interim_results_folder
        self.model_handler=None
        self.TimeMeasurer=TimeMeasurer()
    
    def Get_Results(self,Task_base_Text,Task_base_Result,Task_Inter_Text,Task_Inter_Result):
        if self.model_handler is None:
            if self.verbose:
                print("[INFO] LLM generated")
            self.model_handler=Intervention_Model.create_llm_remote(self.model_id, self.max_memory , self.tokenizer, self.relevance_map, self.Change_arr)

        results=self.model_handler.get_results(Task_base_Text,Task_base_Result,Task_Inter_Text,Task_Inter_Result)

        return results
    
            
    def Load_Interim_Results(self):
        ac_url=self.interim_results_path+"Intervention_Base_"+self.task_1.Task_Name+"_Intervention_"+self.task_2.Task_Name
        ac_url=ac_url+"_Section_"+str(self.Change_arr[0])+"-"+str(self.Change_arr[1])
        all_files=self.safe_listdir(ac_url)
        if all_files is not None and 'Metadata.json' in all_files:
            with open(ac_url+'/Metadata.json', 'r') as file:
                Metadata = json.load(file)
                self.Computed_Samples=Metadata[0]
                self.results=Metadata[1]
        
       
    def Save_Interim_Results(self):
        ac_url=self.interim_results_path+"Intervention_Base_"+self.task_1.Task_Name+"_Intervention_"+self.task_2.Task_Name
        ac_url=ac_url+"_Section_"+str(self.Change_arr[0])+"-"+str(self.Change_arr[1])
        self.ensure_folder_exists(ac_url)
        with open(ac_url+'/Metadata.json', 'w') as file:
            json.dump([self.Computed_Samples,self.results], file)


    def add_nested_dicts(self,dict1, dict2):
        """
        Adds two nested dictionaries element-wise.
        Assumes that both dict1 and dict2 have the same structure.
        
        :param dict1: The first dictionary
        :param dict2: The second dictionary
        :return: A new dictionary with the same structure as dict1 and dict2, containing the summed values
        """
        result = {}
        
        # Iterate through keys in the first dictionary
        for key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # If both values are dictionaries, recurse into them
                result[key] = self.add_nested_dicts(dict1[key], dict2[key])
            else:
                # Otherwise, sum the values directly
                result[key] = dict1[key] + dict2[key]
        
        return result
    def Get_Intervention_Results(self,Number_of_samples):
        self.Load_Interim_Results()
        self.TimeMeasurer.start()
        while self.Computed_Samples<Number_of_samples:
            processed_percentage=self.Computed_Samples/Number_of_samples
            print('[INFO] Processing:',self.task_1.Task_Name,'and',self.task_2.Task_Name,'; Range: ',str(self.Change_arr),'; Computed Samples:',self.Computed_Samples,"Remaining:",self.TimeMeasurer.stop(processed_percentage),flush=True)
            #print(self.Computed_Samples)
            task_1_text,task_1_Result,_=self.task_1.Generate_Task()
            task_2_text,task_2_Result,_=self.task_2.Generate_Task()
            #print(task_1_Result,task_2_Result)
            #print(task_1_text,task_2_text)
            ac_result=self.Get_Results(task_1_text,task_1_Result,task_2_text,task_2_Result)
            if self.results is None:
                self.results=ac_result
            else:
                self.results=self.add_nested_dicts(self.results,ac_result)
            self.Computed_Samples+=1
            self.Save_Interim_Results()
            
        del self.model_handler
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        return self.results
        