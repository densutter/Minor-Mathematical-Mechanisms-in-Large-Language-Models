#Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
import random
import os
import zipfile
import pickle
import json
import Prediction_Model
import Prediction_Helpers
import numpy as np
import ray
import gc
import copy




class Relevance_Map(Prediction_Helpers.Prediction_Helper):
    def __init__(
        self,
        model_id,
        Relevance_Map_Method,
        task_1,
        task_2,
        tokenizer,
        interim_results_folder='./interim_results/',
        verbose=True,
        GPU_Savings=True,
        Allowed_Model_Usage_Before_Refresh=1,
        max_memory="cpu",
        num_gpus=0,
        num_cpus=1
        ):

        self.model_id=model_id

        self.Actually_Supported_Relevance_Map_Methods=['vanilla_gradient','smoothGrad']
        if Relevance_Map_Method not in self.Actually_Supported_Relevance_Map_Methods:
            raise Exception("Relevance_Map_Method "+str(Relevance_Map_Method)+" is not supported in this Version.")
        self.Relevance_Map_Method=Relevance_Map_Method

        self.task_1=task_1
        self.task_2=task_2
        self.interim_results_folder=interim_results_folder
        self.interim_results_path=self.interim_results_folder+'Relevance_'+Relevance_Map_Method+'/' #task_1.Task_Name+'-VS-'+task_2.Task_Name+

        self.Check_Interim_Results_Folder()
        self.Metadata=None
        self.Load_Interim_Results()
        self.verbose=verbose
        self.GPU_Savings=GPU_Savings
        self.tokenizer=tokenizer

        #It is likely that there is some memory leakage.Therefore the model gets deleted all Allowed_Model_Usage_Before_Refresh iterations to clean up the memory
        self.Allowed_Model_Usage_Before_Refresh=Allowed_Model_Usage_Before_Refresh
        self.Actual_Model_Usage=0
        self.model_handler=None
        self.max_memory=max_memory
        self.num_gpus=num_gpus
        self.num_cpus=num_cpus

        self.Random_Offset=20275043


    def Check_Interim_Results_Folder(self):

        if not os.path.exists(self.interim_results_path):
            os.makedirs(self.interim_results_path)



    def Load_Interim_Results(self):

        self.Metadata={}
        ac_url=self.interim_results_path+self.task_1.Task_Name
        all_files=self.safe_listdir(ac_url)
        if  all_files is not None and 'Metadata.json' in all_files:
            with open(ac_url+'/Metadata.json', 'r') as file:
                self.Metadata[self.task_1.Task_Name] = json.load(file)
        else:
            self.Metadata[self.task_1.Task_Name]={}
            self.Metadata[self.task_1.Task_Name]["Computed"]=0
            self.Metadata[self.task_1.Task_Name]["Accuracy"]=0

        ac_url=self.interim_results_path+self.task_2.Task_Name
        all_files=self.safe_listdir(ac_url)
        if all_files is not None and 'Metadata.json' in all_files:
            with open(ac_url+'/Metadata.json', 'r') as file:
                self.Metadata[self.task_2.Task_Name] = json.load(file)
        else:
            self.Metadata[self.task_2.Task_Name]={}
            self.Metadata[self.task_2.Task_Name]["Computed"]=0
            self.Metadata[self.task_2.Task_Name]["Accuracy"]=0


    def Save_Interim_Results(self):
        self.ensure_folder_exists(self.interim_results_path+self.task_1.Task_Name)
        ac_url=self.interim_results_path+self.task_1.Task_Name+'/Metadata.json'
        with open(ac_url, 'w') as file:
            json.dump(self.Metadata[self.task_1.Task_Name], file)

        self.ensure_folder_exists(self.interim_results_path+self.task_2.Task_Name)
        ac_url=self.interim_results_path+self.task_2.Task_Name+'/Metadata.json'
        with open(ac_url, 'w') as file:
            json.dump(self.Metadata[self.task_2.Task_Name], file)


    def Load_Gradient_Data(self,task_name,Addition):
        # Check if the file exists
        filename=self.interim_results_path+task_name+'/Interrim_Results'+Addition+'.pkl'
        if not os.path.exists(filename):
            return None

        # Load the dictionary from the pickle file if it exists
        with open(filename, 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict

    def Save_Gradient_Data(self,task_name,Addition,task_dict):
        self.ensure_folder_exists(self.interim_results_path+task_name)
        with open(self.interim_results_path+task_name+'/Interrim_Results'+Addition+'.pkl', 'wb') as file:
            pickle.dump(task_dict, file)



    def update_dicts(self, Average_Dict, New_Dict, mult_fact):
        # If Average_Dict is None, return New_Dict * mult_fact directly
        if Average_Dict is None:
            result_dict = {}
            for key1 in New_Dict:
                result_dict[key1] = {}
                for key2 in New_Dict[key1]:
                    B = New_Dict[key1][key2]

                    # Check if the value is a single numpy array
                    if isinstance(B, np.ndarray):
                        result_dict[key1][key2] = mult_fact * B
                        #print(B.shape)

                    # Otherwise, assume it's a list of numpy arrays
                    else:
                        result_dict[key1][key2] = [mult_fact * b for b in B]

            return result_dict

        # Proceed with the original computation if Average_Dict is not None
        result_dict = {}
        #print("yeiii")
        for key1 in Average_Dict:
            result_dict[key1] = {}
            for key2 in Average_Dict[key1]:
                A = Average_Dict[key1][key2]
                B = New_Dict[key1][key2]

                # Check if the value is a single numpy array
                if isinstance(A, np.ndarray):
                    result_dict[key1][key2] = A + mult_fact * B

                # Otherwise, assume it's a list of numpy arrays
                else:
                    result_dict[key1][key2] = [a + mult_fact * b for a, b in zip(A, B)]

        return result_dict

    def max_dicts(self, Average_Dict, New_Dict):
        # If Average_Dict is None, return New_Dict * mult_fact directly
        if Average_Dict is None:
            return copy.deepcopy(deep_copied_dict = copy.deepcopy(original_dict))

        # Proceed with the original computation if Average_Dict is not None
        result_dict = {}
        for key1 in Average_Dict:
            result_dict[key1] = {}
            for key2 in Average_Dict[key1]:
                A = Average_Dict[key1][key2]
                B = New_Dict[key1][key2]

                if isinstance(A, np.ndarray):
                    result_dict[key1][key2] = np.max(A,B)

                else:
                    result_dict[key1][key2] = [ np.max(a,b) for a, b in zip(A, B)]

        return result_dict


    def create_weighted_list(self, length, noise_factor=0.1):
        """
        Create a list of weights that sums up to 1,
        with earlier elements being smaller (less influence).

        Parameters:
        - length (int): Length of the list.
        - noise_factor (float): Controls how quickly weights increase. Smaller values lead to a sharper increase.

        Returns:
        - list: A list of weights that sum to 1.
        """
        if length <= 0:
            return []

        # Generate a list of exponentially increasing values
        values = np.linspace(0, 1, length) ** noise_factor

        # Normalize the values to sum to 1
        weights = values / np.sum(values)

        return weights.tolist()


    import numpy as np

    def squared_difference(self, mean_dict, new_dict):
        """
        Compute the squared difference between two dictionaries.

        Parameters:
        - mean_dict (dict): The first dictionary containing mean values.
        - new_dict (dict): The second dictionary containing new values.

        Returns:
        - dict: A new dictionary containing the squared differences.
        """
        # Create a new dictionary to store the results
        result_dict = {}

        # Iterate through the first level of keys
        for key1 in mean_dict:
            result_dict[key1] = {}

            # Iterate through the second level of keys
            for key2 in mean_dict[key1]:
                A = mean_dict[key1][key2]
                B = new_dict[key1][key2]

                # Check if the values are single numpy arrays
                if isinstance(A, np.ndarray):
                    result_dict[key1][key2] = (A - B) ** 2

                # Otherwise, assume they are lists of numpy arrays
                else:
                    result_dict[key1][key2] = [(a - b) ** 2 for a, b in zip(A, B)]

        return result_dict

    def Preprocess_Gradient(self,acGrad):
        acGrad=np.abs(acGrad)
        acGrad=np.mean(acGrad, axis=0)
        return acGrad

    def Preprocess_Gradient_Dicts(self,acGrad_Dict):
        result_dict={}

        for key1 in acGrad_Dict:
            result_dict[key1] = {}

            # Iterate through the second level of keys
            for key2 in acGrad_Dict[key1]:
                A = acGrad_Dict[key1][key2]

                # Check if the values are single numpy arrays
                if isinstance(A, np.ndarray):
                    if len(A.shape)==3:
                        A=A[0]
                    A=self.Preprocess_Gradient(A)

                # Otherwise, assume they are lists of numpy arrays
                else:
                    NA=[]
                    for AP in range(len(A)):
                        NA.append(A[AP])
                        if len(NA[AP].shape)==3:
                            NA[AP]=NA[AP][0]
                        NA[AP]=self.Preprocess_Gradient(NA[AP])
                    A=NA

                result_dict[key1][key2]=A

        return result_dict

    def Compute_Relevance_Map_Relation(self,mean_task1,mean_task2,var_task1,var_task2):
        mean_task1 = np.array(mean_task1)
        mean_task2 = np.array(mean_task2)
        var_task1  = np.array(var_task1)
        var_task2  = np.array(var_task2)

        normalizing = (var_task1+var_task2)/2
        dev_from_0_task1 = mean_task1/normalizing
        dev_from_0_task2 = mean_task2/normalizing

        diff_tasks = dev_from_0_task1-dev_from_0_task2

        return diff_tasks

    def Compute_Relevance_Map(self, ac_mean, ac_var):

        ret_val={}
        #print(ac_mean)
        for key_1 in ac_mean[self.task_1.Task_Name]:
            ret_val[key_1]={}
            for key_2 in ac_mean[self.task_1.Task_Name][key_1]:
                if isinstance(ac_mean[self.task_1.Task_Name][key_1][key_2], torch.Tensor):
                    ret_val[key_1][key_2]=self.Compute_Relevance_Map_Relation(
                        ac_mean[self.task_1.Task_Name][key_1][key_2],
                        ac_mean[self.task_2.Task_Name][key_1][key_2],
                        ac_var[self.task_1.Task_Name][key_1][key_2],
                        ac_var[self.task_2.Task_Name][key_1][key_2]
                    )
                else:
                    ret_val[key_1][key_2]=[]
                    for key_3 in range(len(ac_mean[self.task_1.Task_Name][key_1][key_2])):
                        ret_val[key_1][key_2].append(
                            self.Compute_Relevance_Map_Relation(
                                ac_mean[self.task_1.Task_Name][key_1][key_2][key_3],
                                ac_mean[self.task_2.Task_Name][key_1][key_2][key_3],
                                ac_var[self.task_1.Task_Name][key_1][key_2][key_3],
                                ac_var[self.task_2.Task_Name][key_1][key_2][key_3]
                            )
                        )

        return ret_val



    def Get_Relevance_Map(self, Number_of_samples):
        torch.cuda.empty_cache()
        self.Load_Interim_Results()

        #Model gets initialized automaically when self.Get_Results is called

        Gradient_Means={}
        Gradient_Means[self.task_1.Task_Name]=self.Load_Gradient_Data(self.task_1.Task_Name,"")
        Gradient_Means[self.task_2.Task_Name]=self.Load_Gradient_Data(self.task_2.Task_Name,"")
        Gradient_Variance={}
        Gradient_Variance[self.task_1.Task_Name]=self.Load_Gradient_Data(self.task_1.Task_Name,'_Variance')
        Gradient_Variance[self.task_2.Task_Name]=self.Load_Gradient_Data(self.task_2.Task_Name,'_Variance')
        noisy_averaging_weights=self.create_weighted_list(Number_of_samples, noise_factor=0.5)

        while self.Metadata[self.task_2.Task_Name]['Computed']<Number_of_samples or self.Metadata[self.task_1.Task_Name]['Computed']<Number_of_samples:

            actual_task=None
            if self.Metadata[self.task_1.Task_Name]['Computed']<=self.Metadata[self.task_2.Task_Name]['Computed']:
                actual_task=self.task_1
                actual_task_key=self.task_1.Task_Name
            else:
                actual_task=self.task_2
                actual_task_key=self.task_2.Task_Name

            if self.verbose:
                print('[INFO] Processing:',actual_task_key,'; Processed',self.task_1.Task_Name,':',self.Metadata[self.task_1.Task_Name]['Computed'],',', self.task_2.Task_Name,':',self.Metadata[self.task_2.Task_Name]['Computed'],flush=True)

            actual_task.Set_Random_Seed(self.Random_Offset+self.Metadata[actual_task_key]['Computed'])#Ensure that all tasks get the same samples at the same step
            actual_task.New_Task()
            Task_Text,Task_Result,_=actual_task.Generate_Task()
            self.Metadata[actual_task_key]['Computed']=self.Metadata[actual_task_key]['Computed']+1

            Gradients,Correctly_classified=self.Get_Results(Task_Text,Task_Result)

            self.Metadata[actual_task_key]['Accuracy']+=1
            #print("*"*100)
            #print(Correctly_classified)
            #print(Gradients)

            #Save Averaging of Gradients
            Gradients=self.Preprocess_Gradient_Dicts(Gradients)
            #For averaging
            Gradient_Means[actual_task.Task_Name]=self.update_dicts(Gradient_Means[actual_task.Task_Name],Gradients,1/Number_of_samples)
            #For max:
            #Gradient_Means[actual_task.Task_Name]=self.max_dicts(Gradient_Means[actual_task.Task_Name],Gradients)
            self.Save_Gradient_Data(actual_task.Task_Name,"",Gradient_Means[actual_task.Task_Name])

            #Save noisy Averaging of Variance
            #Actual mean
            inRes=self.update_dicts(None,Gradient_Means[actual_task.Task_Name],Number_of_samples/self.Metadata[actual_task_key]['Computed'])
            #Noisy squared error
            inRes=self.squared_difference(inRes,Gradients)
            #Noisy Variance weighted
            Gradient_Variance[actual_task.Task_Name]=self.update_dicts(
                Gradient_Variance[actual_task.Task_Name],
                inRes,
                noisy_averaging_weights[self.Metadata[actual_task_key]['Computed']-1])
            self.Save_Gradient_Data(actual_task.Task_Name,'_Variance',Gradient_Variance[actual_task.Task_Name])

            #Save Metadata
            self.Save_Interim_Results()

        ray.shutdown()
        return self.Compute_Relevance_Map(Gradient_Means,Gradient_Variance)

