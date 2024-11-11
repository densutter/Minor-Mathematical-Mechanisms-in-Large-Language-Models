import torch.functional as F
import torch.nn as nn
from torch import optim
import Prediction_Helpers
import Prediction_Model
import torch
import ray
import numpy as np
import json
import os

class Probing_Model_Attention(nn.Module):
    def __init__(self, input_dim, output_dim, max_tokens, projection_dim=512, hidden_dim=512, num_layers=1):
        """
        Initialize the CustomAttention module.

        Parameters:
        output_dim (int): Dimension of the input tensor per token.
        projection_dim (int): Dimension to project each token to (default is 512).
        max_tokens (int): Maximum number of tokens expected in the input.
        num_layers (int): Number of layers for the output MLP.
        """
        super(Probing_Model_Attention, self).__init__()

        self.input_dim=input_dim
        self.max_tokens=max_tokens
        # Learnable initial projection layer (output_dim -> projection_dim)
        self.initial_projection = nn.Linear(input_dim, projection_dim)

        # Learnable attention vector of length max_tokens
        self.attention_vector = nn.Parameter(torch.randn(max_tokens))

        # Output MLP layers
        self.output_layers = nn.ModuleList()
        last_dim = projection_dim
        for _ in range(num_layers - 1):
            new_dim=hidden_dim
            self.output_layers.append(nn.Linear(last_dim, new_dim))
            self.output_layers.append(nn.GELU())
            last_dim = new_dim
        self.output_layers.append(nn.Linear(last_dim, output_dim))


    def forward(self, values):
        """
        Forward pass through the CustomAttention module.

        Parameters:
        values (torch.Tensor): Input tensor of shape (tokens, output_dim).

        Returns:
        torch.Tensor: Output tensor of shape (projection_dim).
        """
        #print(values.dtype)
        tokens, in_dim = values.shape
        assert in_dim == self.input_dim, \
            f"Expected values to have shape (tokens, {self.input_dim}), but got {values.shape}"
        assert tokens <= self.max_tokens, \
            f"Number of tokens ({tokens}) exceeds max_tokens ({self.max_tokens})"
        #print(self.attention_vector)
        # Project each token from output_dim to projection_dim
        values = self.initial_projection(values)  # Shape: (tokens, projection_dim)

        # Apply softmax on the learnable attention vector (up to `tokens` entries)
        attention_weights = nn.functional.softmax(self.attention_vector[:tokens], dim=0)  # Shape: (tokens,)

        # Perform weighted sum across tokens
        weighted_sum = (attention_weights.unsqueeze(1) * values).sum(dim=0)  # Shape: (projection_dim)

        # Pass through the output layers
        output = weighted_sum
        for layer in self.output_layers:
            output = layer(output)

        return output




class Probing(Prediction_Helpers.Prediction_Helper):
    def __init__(
        self,
        model_id,
        Probing_Map_Method,
        task_1,
        task_2,
        tokenizer,
        relevance_map,
        testing_Samples,
        probing_layers=1,
        interim_results_folder='./interim_results/',
        verbose=True,
        GPU_Savings=True,
        Allowed_Model_Usage_Before_Refresh=10,
        max_memory="cpu",
        num_gpus=0,
        num_cpus=1,
        Max_tokens=400,
        learning_rate=0.001
        ):

        self.model_id=model_id
        self.task_1=task_1
        self.task_2=task_2
        self.interim_results_folder=interim_results_folder
        self.Probing_Map_Method=Probing_Map_Method
        self.Relevance_Map_Method="hiddenFeatures"
        self.Actually_Supported_Probing_Methods=['Probing'] #Circuit Probing
        if Probing_Map_Method not in self.Actually_Supported_Probing_Methods:
            raise Exception("Relevance_Map_Method "+str(Probing_Map_Method)+" is not supported in this Version.")
        self.interim_results_path=self.interim_results_folder+"Probing_"+Probing_Map_Method+'_Layer_'+str(probing_layers)+'/'#+task_1.Task_Name+'-VS-'+task_2.Task_Name+'/'

        self.verbose=verbose
        self.Allowed_Model_Usage_Before_Refresh=Allowed_Model_Usage_Before_Refresh
        self.Actual_Model_Usage=0
        self.model_handler=None
        self.max_memory=max_memory
        self.GPU_Savings=GPU_Savings
        self.num_gpus=num_gpus
        self.num_cpus=num_cpus

        self.tokenizer=tokenizer

        self.Metadata=None
        self.Models=None
        self.Models_Optimizer=None

        self.relevance_map=relevance_map

        self.Random_Offset=4987239478
        torch.manual_seed(self.Random_Offset-1)
        self.criterion=nn.MSELoss()
        self.Splittance=[1,0.5,0.25]
        self.Testing=False
        self.testing_Samples=testing_Samples
        self.Max_tokens=Max_tokens
        self.learning_rate=learning_rate
        self.Save_after_steps=self.Allowed_Model_Usage_Before_Refresh
        self.Unsaved_steps=0
        self.probing_layers=probing_layers


    def Load_Interim_Results(self):

        self.Metadata={}
        ac_url=self.interim_results_path+self.task_1.Task_Name
        all_files=self.safe_listdir(ac_url)
        if all_files is not None and 'Metadata.json' in all_files:
            with open(ac_url+'/Metadata.json', 'r') as file:
                self.Metadata[self.task_1.Task_Name] = json.load(file)
        else:
            self.Metadata[self.task_1.Task_Name]={}
            self.Metadata[self.task_1.Task_Name]["Computed"]=0
            self.Metadata[self.task_1.Task_Name]["Loss"]=[]

        ac_url=self.interim_results_path+self.task_2.Task_Name
        all_files=self.safe_listdir(ac_url)
        if all_files is not None and 'Metadata.json' in all_files:
            with open(ac_url+'/Metadata.json', 'r') as file:
                self.Metadata[self.task_2.Task_Name] = json.load(file)
        else:
            self.Metadata[self.task_2.Task_Name]={}
            self.Metadata[self.task_2.Task_Name]["Computed"]=0
            self.Metadata[self.task_2.Task_Name]["Loss"]=None


    def Save_Interim_Results(self):
        ac_url=self.interim_results_path+self.task_1.Task_Name
        self.ensure_folder_exists(ac_url)
        with open(ac_url+'/Metadata.json', 'w') as file:
            json.dump(self.Metadata[self.task_1.Task_Name], file)

        ac_url=self.interim_results_path+self.task_2.Task_Name
        self.ensure_folder_exists(ac_url)
        with open(ac_url+'/Metadata.json', 'w') as file:
            json.dump(self.Metadata[self.task_2.Task_Name], file)


    def save_data_structure(self,data_structure, optimizer_structure, save_path, self_learning_rate=0.001):
        """
        Recursively saves a nested data structure with models and optimizers as .pt and .opt files.

        Parameters:
        - data_structure: The structure containing models (can be a list, dictionary, or model).
        - optimizer_structure: The structure containing optimizers corresponding to the models.
        - save_path: The root path where the structure should be saved.
        - self_learning_rate: The learning rate for the Adam optimizer (used when saving the optimizer).
        """
        # Check if the data_structure is a model (i.e., a single model)
        if isinstance(data_structure, torch.nn.Module):
            # Save the model
            save_path = os.path.join(save_path, f"Save")
            model_path = f"{save_path}.mod"
            torch.save(data_structure, model_path)

            # Save the corresponding optimizer
            optimizer_path = f"{save_path}.opt"
            optimizer = optim.Adam(data_structure.parameters(), lr=self_learning_rate)
            torch.save(optimizer.state_dict(), optimizer_path)

        # If the data_structure is a list
        elif isinstance(data_structure, list):
            for idx, item in enumerate(data_structure):
                item_save_path = os.path.join(save_path, f"[item]_{idx}")
                os.makedirs(item_save_path, exist_ok=True)
                # Recursively save the model and optimizer structure for each item in the list
                self.save_data_structure(item, optimizer_structure[idx], item_save_path, self_learning_rate)

        # If the data_structure is a dictionary
        elif isinstance(data_structure, dict):
            for key, item in data_structure.items():
                if isinstance(key, int):
                    item_save_path =os.path.join(save_path, f"[int]_{key}")
                elif isinstance(key, float):
                    item_save_path = os.path.join(save_path, f"[float]_{key}")
                else:
                    item_save_path = os.path.join(save_path, key)
                os.makedirs(item_save_path, exist_ok=True)
                # Recursively save the model and optimizer structure for each key in the dictionary
                self.save_data_structure(item, optimizer_structure[key], item_save_path, self_learning_rate)




    def load_data_structure(self, load_path, self_learning_rate=0.001):
        """
        Recursively loads a nested data structure with models and optimizers saved as .mod and .opt files.

        Parameters:
        - load_path: The root path where the structure is saved.
        - self_learning_rate: The learning rate for the Adam optimizer (used when reinitializing the optimizer).

        Returns:
        - A tuple with two data structures:
            - model_structure: The original data structure with models loaded from .mod files.
            - optimizer_structure: The parallel structure with optimizers loaded from .opt files.
        """
        # Check if the path corresponds to a model (.mod) and an optimizer (.opt) file
        load_path_files=os.path.join(load_path, f"Save")
        model_path = f"{load_path_files}.mod"
        optimizer_path = f"{load_path_files}.opt"

        if os.path.isfile(model_path) and os.path.isfile(optimizer_path):
            # Load the entire model directly
            model = torch.load(model_path)

            # Initialize the optimizer and load its state_dict
            optimizer = optim.Adam(model.parameters(), lr=self_learning_rate)
            optimizer.load_state_dict(torch.load(optimizer_path))

            return model, optimizer

        # Check if the path is a directory and process recursively
        elif os.path.isdir(load_path):
            items = os.listdir(load_path)

            # Check if we have a list or dictionary structure
            if all(item.startswith("[item]_") for item in items):  # Looks like a list structure
                model_list = []
                optimizer_list = []
                for item in sorted(items, key=lambda x: int(x.split('_')[1])):
                    item_path = os.path.join(load_path, item)
                    model, optimizer = self.load_data_structure(item_path, self_learning_rate)
                    model_list.append(model)
                    optimizer_list.append(optimizer)
                return model_list, optimizer_list

            else:  # It should be a dictionary structure
                model_dict = {}
                optimizer_dict = {}
                for item in items:
                    item_path = os.path.join(load_path, item)
                    model, optimizer = self.load_data_structure(item_path, self_learning_rate)

                    # Handle case for integer or float keys
                    if item.startswith("[int]_"):
                        key = int(item.split("_")[1])
                    elif item.startswith("[float]_"):
                        key = float(item.split("_")[1])
                    else:
                        key = item  # Default to using the item name as the key

                    model_dict[key] = model
                    optimizer_dict[key] = optimizer

                return model_dict, optimizer_dict







    def selective_keep(self, Input_Arr, Masking_Arr, keep):
        # Get the number of dimensions
        #print(Input_Arr.shape)
        #print(Masking_Arr)
        Masking_Arr=np.array(Masking_Arr)
        Input_Arr=Input_Arr[0]

        #print( Input_Arr.shape,Masking_Arr.shape)
        if Input_Arr.shape[1] != Masking_Arr.shape[0]:
            raise ValueError("The second dimension of Input_Arr must match the size of Masking_Arr.")


        input_dim = Masking_Arr.shape[0]

        # Calculate the number of top/bottom indices to keep
        keep_count = int(np.ceil(keep * input_dim))

        # Get the indices of dimensions with highest and lowest values in Masking_Arr
        sorted_indices = np.argsort(Masking_Arr)  # Sort Masking_Arr to find the min and max indices
        #print(sorted_indices)
        top_indices = sorted_indices[-keep_count:]  # Top 'keep_count' indices (highest Masking_Arr values)
        bottom_indices = sorted_indices[:keep_count]  # Bottom 'keep_count' indices (lowest Masking_Arr values)

        # Select the top and bottom values from Input_Arr for each token
        top_values = Input_Arr[:, top_indices]
        bottom_values = Input_Arr[:, bottom_indices]

        return top_values, bottom_values

    def init_Models(self,Given_Input,Masking,keep,max_tokens,learning_rate=0.001):
        Output={}

        self.task_1.Set_Random_Seed(0)#Ensure that all tasks get the same samples at the same step
        self.task_1.New_Task()
        Task_Text,Task_Result,Intermediate_Variables=self.task_1.Generate_Task()
        Output[self.task_1.Task_Name]=Intermediate_Variables

        self.task_2.Set_Random_Seed(0)#Ensure that all tasks get the same samples at the same step
        self.task_2.New_Task()
        Task_Text,Task_Result,Intermediate_Variables=self.task_2.Generate_Task()
        Output[self.task_2.Task_Name]=Intermediate_Variables

        #print(Output.keys())
        new_Models={}
        new_Optimizers={}
        new_Models[self.task_1.Task_Name]={}
        new_Models[self.task_2.Task_Name]={}
        new_Optimizers[self.task_1.Task_Name]={}
        new_Optimizers[self.task_2.Task_Name]={}
        self.Metadata[self.task_1.Task_Name]["Loss"]={}
        self.Metadata[self.task_2.Task_Name]["Loss"]={}

        for key_2 in [self.task_1,self.task_2]:
            for key_1 in key_2.Intermediate_Results_Names:
                new_Models[key_2.Task_Name][key_1]={}
                new_Optimizers[key_2.Task_Name][key_1]={}
                self.Metadata[key_2.Task_Name]["Loss"][key_1]={}

        for key_1 in Given_Input:
            for aux_key_1 in [self.task_1,self.task_2]:
                for aux_key_2 in aux_key_1.Intermediate_Results_Names:
                    new_Models[aux_key_1.Task_Name][aux_key_2][key_1]={}
                    new_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1]={}
                    self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1]={}

            for key_2 in Given_Input[key_1]:
                for aux_key_1 in [self.task_1,self.task_2]:
                    for aux_key_2 in aux_key_1.Intermediate_Results_Names:
                        new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2]={}
                        new_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2]={}
                        self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2]={}


                #print(type(Given_Input[key_1][key_2]))
                #print(isinstance(Given_Input[key_1][key_2], np.ndarray))
                if isinstance(Given_Input[key_1][key_2], np.ndarray):
                    example_input=self.selective_keep(Given_Input[key_1][key_2], Masking[key_1][key_2], keep)
                    example_input=example_input[0].shape
                    for aux_key_1 in [self.task_1,self.task_2]:
                        for aux_key_2 in aux_key_1.Intermediate_Results_Names:
                            self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2]=[]
                            self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2].append({})
                            self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2].append({})
                            new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2]=[
                                Probing_Model_Attention(example_input[1], np.size(Output[aux_key_1.Task_Name][aux_key_2]), max_tokens, num_layers=self.probing_layers),
                                Probing_Model_Attention(example_input[1], np.size(Output[aux_key_1.Task_Name][aux_key_2]), max_tokens, num_layers=self.probing_layers)
                            ]
                            new_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2]= [
                                optim.Adam(new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][0].parameters(), lr=learning_rate),
                                optim.Adam(new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][1].parameters(), lr=learning_rate)
                            ]

                else:
                    new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2]=[None]*len(Given_Input[key_1][key_2])
                    for key_3 in range(len(Given_Input[key_1][key_2])):
                        #print(key_1,key_2)
                        #print("*"*100)
                        example_input=self.selective_keep(Given_Input[key_1][key_2][key_3], Masking[key_1][key_2][key_3], keep)
                        example_input=example_input[0].shape
                        for aux_key_1 in [self.task_1,self.task_2]:
                            for aux_key_2 in aux_key_1.Intermediate_Results_Names:
                                self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][key_3]=[]
                                self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][key_3].append({})
                                self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][key_3].append({})
                                new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3]=[
                                    Probing_Model_Attention(example_input[1], np.size(Output[aux_key_1.Task_Name][aux_key_2]), max_tokens, num_layers=self.probing_layers),
                                    Probing_Model_Attention(example_input[1], np.size(Output[aux_key_1.Task_Name][aux_key_2]), max_tokens, num_layers=self.probing_layers)
                                ]
                                new_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3]= [
                                    optim.Adam(new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3][0].parameters(), lr=learning_rate),
                                    optim.Adam(new_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3][1].parameters(), lr=learning_rate)
                                ]
        Output=None
        return new_Models,new_Optimizers

    def Train_Ac_Probing_Model(self,ac_model,optimizer,ac_input,ac_expected_output):

        if not self.Testing:
            outputs = ac_model(torch.from_numpy(ac_input).float()).unsqueeze(0)
        else:
            with torch.no_grad():
                outputs = ac_model(torch.from_numpy(ac_input).float()).unsqueeze(0)
        #print(optimizer)
        loss = self.criterion(outputs, torch.from_numpy(np.array(ac_expected_output).flatten()).unsqueeze(0).float())
        #print(loss.dtype)
        #exit()
        # Backward pass and optimization
        if not self.Testing:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def Add_Loss(self,dict,keep,result):
        if keep in dict:
            dict[keep].append(result)
        else:
            dict[keep]=[result]

    def Train_Probing_Models_keep_Cat(self,ac_Models,ac_Optimizers,Given_Input,Expected_Output,Masking,keep,aux_key_1):
        for key_1 in Given_Input:

            for key_2 in Given_Input[key_1]:

                if isinstance(Given_Input[key_1][key_2], np.ndarray):
                    top_input,bottom_input=self.selective_keep(Given_Input[key_1][key_2], Masking[key_1][key_2], keep)
                    for aux_key_2 in aux_key_1.Intermediate_Results_Names:
                        loss=self.Train_Ac_Probing_Model(
                            ac_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][0],
                            ac_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2][0],
                            top_input,
                            Expected_Output[aux_key_2]
                        )
                        self.Add_Loss(self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][0],keep,loss)

                        loss=self.Train_Ac_Probing_Model(
                            ac_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][1],
                            ac_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2][1],
                            bottom_input,
                            Expected_Output[aux_key_2]
                        )
                        self.Add_Loss(self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][1],keep,loss)


                else:
                    for key_3 in range(len(Given_Input[key_1][key_2])):
                        top_input,bottom_input=self.selective_keep(Given_Input[key_1][key_2][key_3], Masking[key_1][key_2][key_3], keep)
                        for aux_key_2 in aux_key_1.Intermediate_Results_Names:
                            loss=self.Train_Ac_Probing_Model(
                                ac_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3][0],
                                ac_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3][0],
                                top_input,
                                Expected_Output[aux_key_2]
                            )
                            self.Add_Loss(self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][key_3][0],keep,loss)

                            loss=self.Train_Ac_Probing_Model(
                                ac_Models[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3][1],
                                ac_Optimizers[aux_key_1.Task_Name][aux_key_2][key_1][key_2][key_3][1],
                                bottom_input,
                                Expected_Output[aux_key_2]
                            )
                            self.Add_Loss(self.Metadata[aux_key_1.Task_Name]["Loss"][aux_key_2][key_1][key_2][key_3][1],keep,loss)

    def Save_Models_Opims(self,Force_save=False,Is_Testing=False):
        self.Unsaved_steps+=1
        if Force_save or Is_Testing or self.Save_after_steps<self.Unsaved_steps:
            for acTask in [self.task_1.Task_Name,self.task_2.Task_Name]:
                self.Unsaved_steps=0
                self.Save_Interim_Results()
                if not Is_Testing:
                    Save_Models={}
                    Save_Optims={}
                    for ac_S in self.Splittance:
                        Save_Models[ac_S]=self.Models[ac_S][acTask]
                        Save_Optims[ac_S]=self.Models_Optimizer[ac_S][acTask]
                    self.save_data_structure(Save_Models, Save_Optims, self.interim_results_path+acTask+"/Model_Save_Data")

    def Load_Models_Opims(self):
        self.Models={}
        self.Models_Optimizer={}

        self.task_1.Set_Random_Seed(0)#Ensure that all tasks get the same samples at the same step
        self.task_1.New_Task()
        Task_Text,Task_Result,Intermediate_Variables=self.task_1.Generate_Task()
        Given_Input,_=self.Get_Results(Task_Text,Task_Result)

        for ac_keep in self.Splittance:
            ma,mb=self.init_Models(Given_Input,self.relevance_map,ac_keep,self.Max_tokens,self.learning_rate)
            self.Models[ac_keep]=ma
            self.Models_Optimizer[ac_keep]=mb

        for ac_task in [self.task_1,self.task_2]:
            loaded_data=self.load_data_structure(self.interim_results_path+ac_task.Task_Name+"/Model_Save_Data", self_learning_rate=self.learning_rate)
            if loaded_data is not None:
                print('*'*200)
                print('yes it worked')
                for ac_S in self.Splittance:
                    self.Models[ac_S][ac_task.Task_Name]=loaded_data[0][ac_S]
                    self.Models_Optimizer[ac_S][ac_task.Task_Name]=loaded_data[1][ac_S]

    def Train_Probing_Models(self,Given_Input,Expected_Output,ac_task,Masking,max_tokens,learning_rate=0.001):

        for ac_keep in self.Splittance:
            self.Train_Probing_Models_keep_Cat(self.Models[ac_keep],self.Models_Optimizer[ac_keep],Given_Input,Expected_Output,Masking,ac_keep,ac_task)


    def Get_Probing(self,Number_of_samples):
        torch.cuda.empty_cache()
        self.Load_Interim_Results()
        self.Load_Models_Opims()


        self.Testing=False
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
            Task_Text,Task_Result,Intermediate_Variables=actual_task.Generate_Task()
            self.Metadata[actual_task_key]['Computed']=self.Metadata[actual_task_key]['Computed']+1

            Hidden_Features,Correctly_classified=self.Get_Results(Task_Text,Task_Result)

            self.Train_Probing_Models(Hidden_Features,Intermediate_Variables,actual_task,self.relevance_map,self.Max_tokens,learning_rate=self.learning_rate)
            self.Save_Models_Opims()

        self.Save_Models_Opims(Force_save=True)
        self.Testing=True
        for test_idx in range(self.testing_Samples):
            for actual_task in [self.task_1,self.task_2]:

                print('[INFO] Processing:',actual_task.Task_Name,'; Processed',self.task_1.Task_Name,':',self.Metadata[self.task_1.Task_Name]['Computed'],',', self.task_2.Task_Name,':',self.Metadata[self.task_2.Task_Name]['Computed'],flush=True)
                actual_task_key=actual_task.Task_Name
                Task_Text,Task_Result,Intermediate_Variables=actual_task.get_Train_Sample(test_idx)
                self.Metadata[actual_task_key]['Computed']=self.Metadata[actual_task_key]['Computed']+1
                Hidden_Features,Correctly_classified=self.Get_Results(Task_Text,Task_Result)
                self.Train_Probing_Models(
                    Hidden_Features,Intermediate_Variables,actual_task,self.relevance_map,self.Max_tokens,learning_rate=self.learning_rate)
                self.Save_Models_Opims(Is_Testing=True)

        self.Save_Models_Opims(Force_save=True,Is_Testing=True)

        return self.Metadata[self.task_1.Task_Name]["Loss"],self.Metadata[self.task_2.Task_Name]["Loss"]

    def Get_Probing_Results(self,Number_of_samples):
        if self.Probing_Map_Method=="Probing":
            self.Get_Probing(Number_of_samples)
        ray.shutdown()