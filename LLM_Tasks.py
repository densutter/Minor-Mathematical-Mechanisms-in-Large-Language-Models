import random
from collections import Counter
import numpy as np
import copy
from tqdm import tqdm


class Regression_Task_Int:
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,number_inputs=2,number_range_inputs=list(range(23)),number_range_weights=list(range(23)),Testing_samples_num=0,Task_Name=None):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.number_inputs=number_inputs
        self.number_range_inputs=number_range_inputs
        self.number_range_weights=number_range_weights
        self.last_input=None
        #self.output_range=range(+1)
        self.Task_Name=Task_Name
        if self.Task_Name is None:
            self.Task_Name='Regression_Task_Int_'+str(self.number_inputs)+'_Numbers'        
        self.Intermediate_Results_Names =['Weights',"X^T_y"]
        if self.Context is None:
            self.Context='The output represents the result of a linear regression given '+str(number_inputs)+' dimensions with output range [0-1000]: \n\n'

        
        self.Test_Samples=[]
        random_offset=92847
        self.Testing_samples_num=Testing_samples_num
        for aS in range(Testing_samples_num):
            self.Set_Random_Seed(random_offset+aS)
            self.New_Task()
            self.Test_Samples.append(self.task_weigths)
        
        self.New_Task()
            
    
    def New_Task(self):
        is_new= False
        while not is_new:
            self.task_weigths=[]
            for _ in range(self.number_inputs):
                self.task_weigths.append(random.choice(self.number_range_weights))
            if not self.Is_Test_Sample():
                is_new=True
                
    def Get_Inputs(self):
        new_inputs=[]
        for _ in range(self.number_inputs):
            new_inputs.append(random.choice(self.number_range_inputs))
        return new_inputs
        
    def Compute_Result(self,ac_input):
        result=0
        for ac_pos in range(self.number_inputs):
            result+=self.task_weigths[ac_pos]*ac_input[ac_pos]
        return result
        
    def New_Regression_Sample(self):
        new_inputs=self.Get_Inputs()
        res=self.Compute_Result(new_inputs)
        return new_inputs,res
        
    def Is_Test_Sample(self):
        return any(np.array_equal(self.task_weigths,arr) for arr in self.Test_Samples)

            
        
    def Generate_Task(self,uses_digits=False):
        td=''
        ac_X=[]
        ac_y=[]
        td_new=""
        while True:
            inN,res=self.New_Regression_Sample()
            for ac_nnp,ac_nn in enumerate(inN):
                td_new=td_new+'Feature '+str(ac_nnp)+": "+str(ac_nn)+"\n"
            td_new=td_new+'Output: '+str(res)+'\n\n'
            if self.tokenizer(td_new, return_tensors="pt").input_ids.shape[1]>self.max_examples_token_length:
                break
            else:
                td=td_new
                ac_X.append(inN)
                ac_y.append(res)


        self.last_input=inN
        for ac_nnp,ac_nn in enumerate(inN):
            td=td+'Feature '+str(ac_nnp)+": "+str(ac_nn)+"\n"
        td=td+'Output: '
        
        if self.Display_Context:
            td=self.Context+"\n\n\n"+td

        ac_X=np.array(ac_X)
        ac_y=np.array(ac_y)
        Intermediate_Results={}
        Intermediate_Results['Weights']=self.task_weigths
        Intermediate_Results["X^T_y"]=np.dot(ac_X.T, ac_y)

        return td,str(res),Intermediate_Results
    
    def Set_Random_Seed(self,seed):
        random.seed(seed) 

    def get_Train_Sample(self,idx,iteration):
        rand_off=4234453
        self.Set_Random_Seed(rand_off+iteration*self.Testing_samples_num+idx)
        self.task_weigths=self.Test_Samples[idx]
        return self.Generate_Task()

    def get_Sample_DAS(self,which_weight):

        intervention_type=0 #can be changed if there are more than one intervention
        self.New_Task()
        Source_Task_Text,Source_Task_Result,_=self.Generate_Task()
        source_weights=copy.deepcopy(self.task_weigths)

        while True:
            self.New_Task()
            Base_Task_Text,Base_Task_Result,_=self.Generate_Task()
            base_weights=copy.deepcopy(self.task_weigths)
            self.task_weigths[which_weight]=source_weights[which_weight]
            if not self.Is_Test_Sample():
                break

        
        Source_Task_Text_Tokens=self.tokenizer(Source_Task_Text, return_tensors="pt").input_ids[0].tolist()
        Base_Task_Text_Tokens=self.tokenizer(Base_Task_Text, return_tensors="pt").input_ids[0].tolist()
        
        Labels=[-100]*len(Base_Task_Text_Tokens)
        Labels[-1]=self.tokenizer(str(self.Compute_Result(self.last_input)), return_tensors="pt").input_ids[0].tolist()[1]
        return Base_Task_Text_Tokens, Source_Task_Text_Tokens, Labels, intervention_type

    def get_Dataset_DAS(self,num_samples,which_weight):
        new_Dataset=[[],[],[],[]]
        for _ in tqdm(range(num_samples)):
            Base_Task_Text_Tokens, Source_Task_Text_Tokens, Labels, intervention_type=self.get_Sample_DAS(which_weight)
            new_Dataset[0].append(Base_Task_Text_Tokens)
            new_Dataset[1].append(Source_Task_Text_Tokens)
            new_Dataset[2].append(Labels)
            new_Dataset[3].append(intervention_type)
        return new_Dataset

    def get_Eval_Sample(self):
        self.New_Task()
        ac_Task_Text,ac_Task_Result,_=self.Generate_Task()
        ac_Task_Text_Tokens=self.tokenizer(ac_Task_Text, return_tensors="pt").input_ids[0].tolist()
        ac_Task_Result_Tokens=[-100]*len(ac_Task_Text_Tokens)
        ac_Task_Result_Tokens[-1]=self.tokenizer(ac_Task_Result, return_tensors="pt").input_ids[0].tolist()[1]
        return ac_Task_Text_Tokens,ac_Task_Result_Tokens

    def get_Eval_Dataset(self,num_samples):
        new_Dataset=[[],[]]
        for _ in tqdm(range(num_samples)):
            ac_Task_Text_Tokens,ac_Task_Result_Tokens=self.get_Eval_Sample()
            new_Dataset[0].append(ac_Task_Text_Tokens)
            new_Dataset[1].append(ac_Task_Result_Tokens)
        return new_Dataset
        
        
        
        


# Dataset from:
# https://www.kaggle.com/datasets/prajwalkanade/sentiment-analysis-word-lists-dataset
# This Task is obsolete. It was made for a previous version and is therefore no longer compatible with the actual pipeline
class Word_Sentiment_Task: 
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,Dataset_Folder='./Datasets/'):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.Dataset_Folder=Dataset_Folder
        self.Task_Name='Word_Sentiment_Task'        
        self.Intermediate_Results_Names = None
        
        f=open(self.Dataset_Folder+'positive-words.txt',"r",encoding="ISO-8859-1")
        self.pos_words=f.read().split('\n')[:-1]
        f.close()
        f=open(self.Dataset_Folder+'negative-words.txt',"r",encoding="ISO-8859-1")
        self.neg_words=f.read().split('\n')[:-1]
        f.close()
        
        
        if self.Context is None:
            self.Context='The following words are classified by the sentiment they imply: \n\n'

    def Generate_Task(self):
        td=''
        
        while True:
            if random.randint(0, 1)==1:
                td_new=td+'input = '+self.pos_words[random.randint(0, len(self.pos_words)-1)]+' ; output = positiv \n'
            else:
                td_new=td+'input = '+self.neg_words[random.randint(0, len(self.neg_words)-1)]+' ; output = negativ \n'
            if self.tokenizer(td, return_tensors="pt").input_ids.shape[1]>self.max_examples_token_length:
                break
            else:
                td=td_new

        res=None
        if random.randint(0, 1)==1:
            td=td+'input = '+self.pos_words[random.randint(0, len(self.pos_words)-1)]+' ; output = '
            res='positiv'
        else:
            td=td+'input = '+self.neg_words[random.randint(0, len(self.neg_words)-1)]+' ; output = '
            res='negativ'
        if self.Display_Context:
            td=self.Context+td
        return td,res,None
        
    def New_Task(self):
        pass 
        
    def Set_Random_Seed(self,seed):
        random.seed(seed) 




#The calculations for this task are inspired by:
#https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372
#This task is also not used in the actual version of the experiments. It however is compatible with the pipeline 
class Multiclass_Logistic_Regression_Task(Regression_Task_Int):
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,dimension_input=2,classes=3,number_range_inputs=list(range(23)),number_range_weights=list(range(23)),Testing_samples_num=0,Task_Name=None):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.dimension_input=dimension_input
        self.classes=classes
        self.number_range_inputs=number_range_inputs
        self.number_range_weights=number_range_weights
        self.Task_Name=Task_Name
        if self.Task_Name is None:
            self.Task_Name='Linear_Classification_Task_Int_'+str(self.dimension_input)+'_Dimension'
        self.Intermediate_Results_Names =["Weights","Weights_X^T"]
       
    
        self.allowed_Tries_to_split=100
        self.min_split_example=10
        
        self.task_weigths=None
        self.last_input=None
    
        if self.Context is None:
            self.Context='The output represents the result of a linear classification given '+str(2)+' dimensions  with output range [0-3]: \n\n'

        self.Test_Samples=[]
        
        random_offset=9391
        self.Testing_samples_num=Testing_samples_num
        for aS in range(Testing_samples_num):
            self.Set_Random_Seed(random_offset+aS)
            self.New_Task()
            self.Test_Samples.append(self.task_weigths)

        
        self.New_Task()
        
    def Generate_Task(self):
        td=''
        ac_X=[]
        ac_y=[]
        td_new=""
        while True:
            inN=np.random.choice(self.number_range_inputs,size=(self.dimension_input,), replace=True)
            res=self.Compute_Result(inN)
            for ac_nnp,ac_nn in enumerate(inN):
                td_new=td_new+'Feature '+str(ac_nnp)+": "+str(ac_nn)+"\n"
            td_new=td_new+'Output: '+str(res)+'\n\n'
            if self.tokenizer(td_new, return_tensors="pt").input_ids.shape[1]>self.max_examples_token_length:
                break
            else:
                td=td_new
                ac_X.append(inN)

        
        self.last_input=inN
        for ac_nnp,ac_nn in enumerate(inN):
            td=td+'Feature '+str(ac_nnp)+": "+str(ac_nn)+"\n"
        td=td+'Output: '
        
        if self.Display_Context:
            td=self.Context+"\n\n\n"+td
         
        ac_X=np.array(ac_X)

        Intermediate_Results={}
        Intermediate_Results["Weights"]=self.task_weigths
        Intermediate_Results["Weights_X^T"]=ac_X @ self.task_weigths.T
        return td,str(res),Intermediate_Results

    def Compute_Result(self,ac_input):
        logits = ac_input @ self.task_weigths.T  
        return np.argmax(logits)  

    
    def Test_if_splittable(self):
        Labels=[]
        for _ in range(self.allowed_Tries_to_split):
            X_sample = np.random.choice(self.number_range_inputs, size=(1, self.dimension_input),replace=True)  
            Labels.append(self.Compute_Result(X_sample))
        
        label_counts = Counter(Labels)
    
        # Check if all elements in range(num_classes) have at least min_count occurrences
        for i in range(self.classes):
            if label_counts.get(i, 0) < self.min_split_example:
                return False
        return True
        
            
    def New_Task(self):

        self.task_weigths=np.random.choice(self.number_range_weights, size=(self.classes, self.dimension_input), replace=True)
        while any(np.array_equal(self.task_weigths,arr) for arr in self.Test_Samples) or (not self.Test_if_splittable()):
            self.task_weigths=np.random.choice(self.number_range_weights, size=(self.classes, self.dimension_input), replace=True)
            
        
        
    def Set_Random_Seed(self,seed):
        random.seed(seed) 


    def get_Train_Sample(self,idx,iteration):
        rand_off=2098452
        self.Set_Random_Seed(rand_off+iteration*self.Testing_samples_num+idx)
        self.task_weigths=self.Test_Samples[idx]
        rgt=self.Generate_Task()
        return rgt

    def get_Sample_DAS(self,which_weight):

        intervention_type=0 
        self.New_Task()
        Source_Task_Text,Source_Task_Result,_=self.Generate_Task()
        source_weights=copy.deepcopy(self.task_weigths)

        while True:
            self.New_Task()
            Base_Task_Text,Base_Task_Result,_=self.Generate_Task()
            base_weights=copy.deepcopy(self.task_weigths)
            self.task_weigths[which_weight[0]][which_weight[1]]=source_weights[which_weight[0]][which_weight[1]]
            if not self.Is_Test_Sample():
                break

        
        Source_Task_Text_Tokens=self.tokenizer(Source_Task_Text, return_tensors="pt").input_ids[0].tolist()
        Base_Task_Text_Tokens=self.tokenizer(Base_Task_Text, return_tensors="pt").input_ids[0].tolist()
        
        Labels=[-100]*len(Base_Task_Text_Tokens)
        Labels[-1]=self.tokenizer(str(self.Compute_Result(self.last_input)), return_tensors="pt").input_ids[0].tolist()[1]
        return Base_Task_Text_Tokens, Source_Task_Text_Tokens, Labels, intervention_type




class Manhattan_Distance_Problem_Int(Regression_Task_Int):
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,number_inputs=2,number_range_inputs=list(range(23)),number_range_weights=list(range(23)),Testing_samples_num=0,Task_Name=None):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.number_inputs=number_inputs
        self.number_range_inputs=number_range_inputs
        self.number_range_weights=number_range_weights
        self.last_input=None
        self.Task_Name=Task_Name
        if self.Task_Name is None:
            self.Task_Name='Manhattan_Distance_Int_'+str(self.number_inputs)+'_Numbers'        
        self.Intermediate_Results_Names =['Weights']
        if self.Context is None:
            self.Context='The output represents the Manhattan Distance to a point given '+str(number_inputs)+' dimensions with output range [0-1000]: \n\n'

        
        self.Test_Samples=[]
        random_offset=239847
        self.Testing_samples_num=Testing_samples_num
        for aS in range(Testing_samples_num):
            self.Set_Random_Seed(random_offset+aS)
            self.New_Task()
            self.Test_Samples.append(self.task_weigths)
        
        self.New_Task()
            


        
    def Compute_Result(self,ac_input):
        result=0
        for ac_pos in range(self.number_inputs):
            result+=abs(self.task_weigths[ac_pos]-ac_input[ac_pos])
        return result
    
    def Generate_Task(self,uses_digits=False):
        td=''
        ac_X=[]
        ac_y=[]
        td_new=""
        while True:
            inN,res=self.New_Regression_Sample()
            for ac_nnp,ac_nn in enumerate(inN):
                td_new=td_new+'Feature '+str(ac_nnp)+": "+str(ac_nn)+"\n"
            td_new=td_new+'Output: '+str(res)+'\n\n'
            if self.tokenizer(td_new, return_tensors="pt").input_ids.shape[1]>self.max_examples_token_length:
                break
            else:
                td=td_new
                ac_X.append(inN)
                ac_y.append(res)

        self.last_input=inN
        for ac_nnp,ac_nn in enumerate(inN):
            td=td+'Feature '+str(ac_nnp)+": "+str(ac_nn)+"\n"
        td=td+'Output: '
        
        if self.Display_Context:
            td=self.Context+"\n\n\n"+td

        ac_X=np.array(ac_X)
        ac_y=np.array(ac_y)
        Intermediate_Results={}
        Intermediate_Results['Weights']=self.task_weigths

        return td,str(res),Intermediate_Results
    
    def Set_Random_Seed(self,seed):
        random.seed(seed) 

    def get_Train_Sample(self,idx,iteration):
        rand_off=297524
        self.Set_Random_Seed(rand_off+iteration*self.Testing_samples_num+idx)
        self.task_weigths=self.Test_Samples[idx]
        return self.Generate_Task()

