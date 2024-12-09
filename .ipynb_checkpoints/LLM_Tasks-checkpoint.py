import random
from collections import Counter
import numpy as np

#Hidden variables are X^Ty and w
class Regression_Task_Int:
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,number_inputs=2,number_range=[0,22],Testing_samples_num=0):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.number_inputs=number_inputs
        self.number_range=number_range
        self.Task_Name='Regression_Task_Int_'+str(self.number_inputs)+'_Numbers'
        self.Intermediate_Results_Names =['Weights',"X^T_y"]

        if self.Context is None:
            self.Context='The output represents the result of this linear equation given the input as the '+str(self.number_inputs)+' input numbers: \n\n'


        self.Test_Samples=[]
        random_offset=92847
        for aS in range(Testing_samples_num):
            self.Set_Random_Seed(random_offset+aS)
            self.New_Task()
            self.Test_Samples.append(self.regression_weigths)

        self.New_Task()


    def New_Task(self):
        is_new= False
        while not is_new:
            self.regression_weigths=[]
            for _ in range(self.number_inputs):
                self.regression_weigths.append(random.randint(self.number_range[0], self.number_range[1]))
            if not any(np.array_equal(self.regression_weigths,arr) for arr in self.Test_Samples):
                is_new=True

    def Get_Inputs(self):
        new_inputs=[]
        for _ in range(self.number_inputs):
            new_inputs.append(random.randint(self.number_range[0], self.number_range[1]))
        return new_inputs

    def Compute_Result(self,ac_input):
        result=0
        for ac_pos in range(self.number_inputs):
            result+=self.regression_weigths[ac_pos]*ac_input[ac_pos]
        return result

    def New_Regression_Sample(self):
        new_inputs=self.Get_Inputs()
        res=self.Compute_Result(new_inputs)
        return new_inputs,res

    def Generate_Task(self):
        td=''
        ac_X=[]
        ac_y=[]
        while True:
            inN,res=self.New_Regression_Sample()
            if self.number_inputs==1:
                td_new=td+'input = '+str(inN[0])+' ; output = '+str(res)+' \n'
            else:
                td_new=td+'input = ( '+str(inN[0])
                for ac_nn in inN[1:]:
                    td_new=td_new+' , '+str(ac_nn)
                td_new=td_new+' ) ; output = '+str(res)+' \n'
            if self.tokenizer(td, return_tensors="pt").input_ids.shape[1]>self.max_examples_token_length:
                break
            else:
                td=td_new
                ac_X.append(inN)
                ac_y.append(res)

        #inN,res=self.New_Regression_Sample()
        if self.number_inputs==1:
            td=td+'input = '+str(inN[0])+' ; output = '
        else:
            td=td+'input = ( '+str(inN[0])
            for ac_nn in inN[1:]:
                td=td+' , '+str(ac_nn)
            td=td+' ) ; output = '

        if self.Display_Context:
            td=self.Context+td

        ac_X=np.array(ac_X)
        ac_y=np.array(ac_y)
        Intermediate_Results={}
        Intermediate_Results['Weights']=self.regression_weigths
        Intermediate_Results["X^T_y"]=np.dot(ac_X.T, ac_y)
        return td,str(res),Intermediate_Results

    def Set_Random_Seed(self,seed):
        random.seed(seed)

    def get_Train_Sample(self,idx):
        rand_off=4234453
        self.Set_Random_Seed(rand_off+idx)
        self.regression_weigths=self.Test_Samples[idx]
        return self.Generate_Task()



# Dataset from:
# https://www.kaggle.com/datasets/prajwalkanade/sentiment-analysis-word-lists-dataset
# No hidden variable for now
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




#https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372
#Hidden variables are XW and W
class Multiclass_Logistic_Regression_Task:
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,dimension_input=2,classes=3,number_range=[0,22],Testing_samples_num=0):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.dimension_input=dimension_input
        self.classes=classes
        self.number_range=number_range
        self.Task_Name='Linear_Classification_Task_Int_'+str(self.dimension_input)+'_Dimension'
        self.Intermediate_Results_Names =["Weights","Weights_X^T"]


        self.allowed_Tries_to_split=100
        self.min_split_example=10

        self.Weights=None


        if self.Context is None:
            self.Context='The output represents the result of this linear classification given '+str(self.dimension_input)+' dimensions and '+str(self.classes)+' classes: \n\n'

        self.Test_Samples=[]

        random_offset=9391
        for aS in range(Testing_samples_num):
            self.Set_Random_Seed(random_offset+aS)
            self.New_Task()
            self.Test_Samples.append(self.Weights)


        self.New_Task()

    def Generate_Task(self):
        td=''
        ac_X=[]
        ac_y=[]
        while True:
            inN=np.random.randint(self.number_range[0],self.number_range[1], size=(self.dimension_input,))
            res=self.Get_Label(inN)
            if self.dimension_input==1:
                td_new=td+'input = '+str(inN[0])+' ; output = '+str(res)+' \n'
            else:
                td_new=td+'input = ( '+str(inN[0])
                for ac_nn in inN[1:]:
                    td_new=td_new+' , '+str(ac_nn)
                td_new=td_new+' ) ; output = '+str(res)+' \n'
            if self.tokenizer(td, return_tensors="pt").input_ids.shape[1]>self.max_examples_token_length:
                break
            else:
                td=td_new
                ac_X.append(inN)


        #inN=np.random.randint(self.number_range[0],self.number_range[1], size=(1, self.dimension_input))
        #res=self.Get_Label(inN)
        if self.dimension_input==1:
            td=td+'input = '+str(inN[0])+' ; output = '
        else:
            td=td+'input = ( '+str(inN[0])
            for ac_nn in inN[1:]:
                td=td+' , '+str(ac_nn)
            td=td+' ) ; output = '

        if self.Display_Context:
            td=self.Context+td

        ac_X=np.array(ac_X)

        Intermediate_Results={}
        Intermediate_Results["Weights"]=self.Weights
        Intermediate_Results["Weights_X^T"]=ac_X @ self.Weights.T
        return td,str(res),Intermediate_Results

    def Get_Label(self,ac_input):
        logits = ac_input @ self.Weights.T
        return np.argmax(logits)


    def Test_if_splittable(self):
        Labels=[]
        for _ in range(self.allowed_Tries_to_split):
            X_sample = np.random.randint(1, 20, size=(1, self.dimension_input))
            Labels.append(self.Get_Label(X_sample))

        label_counts = Counter(Labels)

        # Check if all elements in range(num_classes) have at least min_count occurrences
        for i in range(self.classes):
            if label_counts.get(i, 0) < self.min_split_example:
                return False
        return True


    def New_Task(self):

        self.Weights=np.random.randint(self.number_range[0], self.number_range[1], size=(self.classes, self.dimension_input))
        while any(np.array_equal(self.Weights,arr) for arr in self.Test_Samples) or (not self.Test_if_splittable()):
            self.Weights=np.random.randint(self.number_range[0], self.number_range[1], size=(self.classes, self.dimension_input))



    def Set_Random_Seed(self,seed):
        random.seed(seed)


    def get_Train_Sample(self,idx):
        rand_off=2098452
        self.Set_Random_Seed(rand_off+idx)
        self.regression_weigths=self.Test_Samples[idx]
        return self.Generate_Task()