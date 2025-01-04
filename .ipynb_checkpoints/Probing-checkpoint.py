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
from TimeMeasurer import TimeMeasurer

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
        Allowed_Model_Usage_Before_Refresh=10, 
        max_memory="cpu", 
        probing_device="cpu",
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
        self.num_gpus=num_gpus
        self.num_cpus=num_cpus
        self.Baselines=None
        self.tokenizer=tokenizer

        self.Metadata=None

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
        self.ac_combi=None
        self.probing_device=probing_device

        self.Example_Input,_=self.Get_Results("test","0")
        self.TimeMeasurer=TimeMeasurer()


        
       

    def Save_Metadata(self):
        ac_url=self.interim_results_path+"Probing_"+self.task_1.Task_Name+"_to_"+self.task_2.Task_Name
        self.ensure_folder_exists(ac_url)
        with open(ac_url+'/Metadata.json', 'w') as file:
            json.dump(self.Metadata, file)

            
    def Load_Metadata(self): 
        ac_url=self.interim_results_path+"Probing_"+self.task_1.Task_Name+"_to_"+self.task_2.Task_Name
        all_files=self.safe_listdir(ac_url)
        if all_files is not None and 'Metadata.json' in all_files:
            with open(ac_url+'/Metadata.json', 'r') as file:
                self.Metadata = json.load(file)
        else:
            self.Metadata={}
            self.Metadata[self.task_1.Task_Name]={}
            self.Metadata[self.task_2.Task_Name]={}
            self.Metadata[self.task_1.Task_Name]["Loss"]={}
            self.Metadata[self.task_2.Task_Name]["Loss"]={}
            self.Metadata["progress"]=0
            self.Metadata["Number_of_samples"]=-1
            
                
        

    

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


    
    def Add_Loss(self,dict,keep,result):
        if keep in dict:
            dict[keep].append(result)
        else:
            dict[keep]=[result]

    
    def init_Models_aux(self,keep):
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

        for key_1 in [self.task_1,self.task_2]:
            new_Models[key_1.Task_Name]={}
            new_Optimizers[key_1.Task_Name]={}
            for key_2 in key_1.Intermediate_Results_Names:
                if key_2 not in self.Metadata[key_1.Task_Name]["Loss"]:
                    self.Metadata[key_1.Task_Name]["Loss"][key_2]={}
                if self.ac_combi[0] not in self.Metadata[key_1.Task_Name]["Loss"][key_2]:
                    self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]]={}
                if self.ac_combi[1] not in self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]]:
                    self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]]={}
                
                if isinstance(self.Example_Input[self.ac_combi[0]][self.ac_combi[1]], np.ndarray): 
                    
                    example_input=self.selective_keep(
                        self.Example_Input[self.ac_combi[0]][self.ac_combi[1]], 
                        self.relevance_map[self.ac_combi[0]][self.ac_combi[1]], 
                        keep)
                    example_input=example_input[0].shape

                    self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]]=[]
                    self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]].append({})
                    self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]].append({})
                    
                    new_Models[key_1.Task_Name][key_2]=[
                        Probing_Model_Attention(
                            example_input[1], 
                            np.size(Output[key_1.Task_Name][key_2]),
                            self.Max_tokens, 
                            num_layers=self.probing_layers
                            ).to(self.probing_device),
                        Probing_Model_Attention(
                            example_input[1], 
                            np.size(Output[key_1.Task_Name][key_2]), 
                            self.Max_tokens, 
                            num_layers=self.probing_layers
                        ).to(self.probing_device)
                    ]
                    
                    new_Optimizers[key_1.Task_Name][key_2] = [
                        optim.Adam(new_Models[key_1.Task_Name][key_2][0].parameters(), lr=self.learning_rate),
                        optim.Adam(new_Models[key_1.Task_Name][key_2][1].parameters(), lr=self.learning_rate)
                    ]
                             
                else:
                    new_Models[key_1.Task_Name][key_2]=[None]*len(self.Example_Input[self.ac_combi[0]][self.ac_combi[1]])
                    new_Optimizers[key_1.Task_Name][key_2]=[None]*len(self.Example_Input[self.ac_combi[0]][self.ac_combi[1]])
                    for key_3 in range(len(self.Example_Input[self.ac_combi[0]][self.ac_combi[1]])):
                        
                        example_input=self.selective_keep(
                            self.Example_Input[self.ac_combi[0]][self.ac_combi[1]][key_3],
                            self.relevance_map[self.ac_combi[0]][self.ac_combi[1]][key_3], 
                            keep
                        )
                        
                        example_input=example_input[0].shape   
                        self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]][key_3]=[]
                        self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]][key_3].append({})
                        self.Metadata[key_1.Task_Name]["Loss"][key_2][self.ac_combi[0]][self.ac_combi[1]][key_3].append({})
                        
                        new_Models[key_1.Task_Name][key_2][key_3]=[
                            Probing_Model_Attention(
                                example_input[1], 
                                np.size(Output[key_1.Task_Name][key_2]), 
                                self.Max_tokens, 
                                num_layers=self.probing_layers
                            ).to(self.probing_device),
                            Probing_Model_Attention(
                                example_input[1], 
                                np.size(Output[key_1.Task_Name][key_2]),
                                self.Max_tokens, 
                                num_layers=self.probing_layers
                            ).to(self.probing_device)
                        ]
                        new_Optimizers[key_1.Task_Name][key_2][key_3]= [
                            optim.Adam(new_Models[key_1.Task_Name][key_2][key_3][0].parameters(), lr=self.learning_rate),
                            optim.Adam(new_Models[key_1.Task_Name][key_2][key_3][1].parameters(), lr=self.learning_rate)
                        ]
        Output=None
        return new_Models,new_Optimizers

    def init_Models(self):
        new_Models={}
        new_Optimizers={}
        for ac_keep in self.Splittance:
            ac_new_Models,ac_new_Optimizers=self.init_Models_aux(ac_keep)
            new_Models[ac_keep]=ac_new_Models
            new_Optimizers[ac_keep]=ac_new_Optimizers
        return new_Models,new_Optimizers
            
    
    def Train_Ac_Probing_Model(self,ac_model,optimizer,ac_input,ac_expected_output):
        if not self.Testing:
            outputs = ac_model(torch.from_numpy(ac_input).float().to(self.probing_device)).unsqueeze(0)
        else:
            with torch.no_grad():
                outputs = ac_model(torch.from_numpy(ac_input).float().to(self.probing_device)).unsqueeze(0)
        #print(optimizer)
        loss = self.criterion(outputs, torch.from_numpy(np.array(ac_expected_output).flatten()).to(self.probing_device).unsqueeze(0).float())
        if not self.Testing:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss.item()

    
    def Train_Probing_Models_aux(self,ac_Models,ac_Optimizers,Given_Input,Expected_Output,keep,ac_task):
        
        key_1 = self.ac_combi[0]
        key_2 = self.ac_combi[1]

        if isinstance(Given_Input[key_1][key_2], np.ndarray): 
            top_input,bottom_input=self.selective_keep(Given_Input[key_1][key_2], self.relevance_map[key_1][key_2], keep)            
            for aux_key_1 in ac_task.Intermediate_Results_Names:
                loss=self.Train_Ac_Probing_Model(
                    ac_Models[ac_task.Task_Name][aux_key_1][0],
                    ac_Optimizers[ac_task.Task_Name][aux_key_1][0],
                    top_input,
                    Expected_Output[aux_key_1]    
                )
                self.Add_Loss(self.Metadata[ac_task.Task_Name]["Loss"][aux_key_1][key_1][key_2][0],keep,loss)
                
                loss=self.Train_Ac_Probing_Model(
                    ac_Models[ac_task.Task_Name][aux_key_1][1],
                    ac_Optimizers[ac_task.Task_Name][aux_key_1][1],
                    bottom_input,
                    Expected_Output[aux_key_1]    
                )
                self.Add_Loss(self.Metadata[ac_task.Task_Name]["Loss"][aux_key_1][key_1][key_2][1],keep,loss)
                    
            
        else:
            for key_3 in range(len(Given_Input[key_1][key_2])):
                top_input,bottom_input=self.selective_keep(Given_Input[key_1][key_2][key_3], self.relevance_map[key_1][key_2][key_3], keep)                
                for aux_key_1 in ac_task.Intermediate_Results_Names:
                    loss=self.Train_Ac_Probing_Model(
                        ac_Models[ac_task.Task_Name][aux_key_1][key_3][0],
                        ac_Optimizers[ac_task.Task_Name][aux_key_1][key_3][0],
                        top_input,
                        Expected_Output[aux_key_1]    
                    )
                    self.Add_Loss(self.Metadata[ac_task.Task_Name]["Loss"][aux_key_1][key_1][key_2][key_3][0],keep,loss)
                    
                    loss=self.Train_Ac_Probing_Model(
                        ac_Models[ac_task.Task_Name][aux_key_1][key_3][1],
                        ac_Optimizers[ac_task.Task_Name][aux_key_1][key_3][1],
                        bottom_input,
                        Expected_Output[aux_key_1]    
                    )
                    self.Add_Loss(self.Metadata[ac_task.Task_Name]["Loss"][aux_key_1][key_1][key_2][key_3][1],keep,loss)


    
    def Train_Probing_Models(
        self,
        Given_Input,
        Expected_Output,
        ac_task,
        ac_Models,
        ac_Optimizers):

        for ac_keep in self.Splittance:
            self.Train_Probing_Models_aux(
                ac_Models[ac_keep], 
                ac_Optimizers[ac_keep],
                Given_Input,
                Expected_Output,
                ac_keep,
                ac_task)
    


    def Get_Probing(self,Number_of_samples):
        
        self.Load_Metadata()
        
        if self.Metadata["Number_of_samples"]!=-1 and self.Metadata["Number_of_samples"]!=Number_of_samples:
            raise RuntimeError("Number of samples does not fit metadata")
        self.Metadata["Number_of_samples"]=Number_of_samples
        
        layer_idx_combos=[]
        for ac_layer in self.relevance_map:
            for ac_idx  in self.relevance_map[ac_layer]:
                layer_idx_combos.append([ac_layer,ac_idx])
        self.TimeMeasurer.start()
        while self.Metadata["progress"]<len(layer_idx_combos):
            Tasks_done=[0,0]
            torch.cuda.empty_cache()
            
            self.ac_combi=layer_idx_combos[self.Metadata["progress"]]
            processed_percentage=self.Metadata["progress"]/len(layer_idx_combos)
            print('[INFO] Working on Layer:',self.ac_combi[0],'; Index:',self.ac_combi[1],";Tasks done:",str(Tasks_done),"Remaining:",self.TimeMeasurer.stop(processed_percentage),flush=True)

            ac_Models,ac_Optimizers=self.init_Models()
            self.Testing=False
            while Tasks_done[0]<self.Metadata["Number_of_samples"] or Tasks_done[1]<self.Metadata["Number_of_samples"]:
                #print(Tasks_done)
                actual_task=None
                if Tasks_done[0]<self.Metadata["Number_of_samples"]:
                    actual_task=self.task_1
                    actual_task_key=self.task_1.Task_Name
                    computed=Tasks_done[0]
                    Tasks_done[0]+=1
                else: 
                    actual_task=self.task_2
                    actual_task_key=self.task_2.Task_Name
                    computed=Tasks_done[1]
                    Tasks_done[1]+=1
                
                #Ensure that all tasks get the same samples at the same step
                actual_task.Set_Random_Seed(self.Random_Offset+Tasks_done[0])
                actual_task.New_Task()
                Task_Text,Task_Result,Intermediate_Variables=actual_task.Generate_Task()
                
                Hidden_Features,_=self.Get_Results(Task_Text,Task_Result)
                
                self.Train_Probing_Models(
                    Hidden_Features,
                    Intermediate_Variables,
                    actual_task,
                    ac_Models,
                    ac_Optimizers)

            
            self.Testing=True
            for test_idx in range(self.testing_Samples):
                for actual_task in [self.task_1,self.task_2]:

                    actual_task_key=actual_task.Task_Name
                    Task_Text,Task_Result,Intermediate_Variables=actual_task.get_Train_Sample(test_idx)
                    Hidden_Features,_=self.Get_Results(Task_Text,Task_Result)
                    self.Train_Probing_Models(
                        Hidden_Features,
                        Intermediate_Variables,
                        actual_task,
                        ac_Models,
                        ac_Optimizers
                    )
                Tasks_done[0]+=1
                Tasks_done[1]+=1
            self.Metadata["progress"]+=1
            self.Save_Metadata()
        return self.Metadata[self.task_1.Task_Name]["Loss"],self.Metadata[self.task_2.Task_Name]["Loss"]

    def Get_Probing_Results(self,Number_of_samples):
        if self.Probing_Map_Method=="Probing":
            self.Get_Probing(Number_of_samples)
        ray.shutdown()
        
        
