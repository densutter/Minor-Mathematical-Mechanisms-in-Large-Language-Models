import random

class Regression_Task_Int:
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,number_inputs=1,number_range=[1,100]):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.number_inputs=number_inputs
        self.number_range=number_range
        self.Task_Name='Regression_Task_Int_'+str(self.number_inputs)+'_Numbers'
        
        if self.Context is None:
            self.Context='The output represents the result of this linear equation given the input as the '+str(self.number_inputs)+' input numbers: \n\n'
        
        self.New_Task()
    
    def New_Task(self):
        self.regression_weigths=[]
        for _ in range(self.number_inputs):
            self.regression_weigths.append(random.randint(self.number_range[0], self.number_range[1]))
            
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

        inN,res=self.New_Regression_Sample()
        if self.number_inputs==1:
            td=td+'input = '+str(inN[0])+' ; output = '
        else:
            td=td+'input = ( '+str(inN[0])
            for ac_nn in inN[1:]:
                td=td+' , '+str(ac_nn)
            td=td+' ) ; output = '
        
        if self.Display_Context:
            td=self.Context+td
        return td,str(res)
    
    def Set_Random_Seed(self,seed):
        random.seed(seed) 
        


# Dataset from:
# https://www.kaggle.com/datasets/prajwalkanade/sentiment-analysis-word-lists-dataset
class Word_Sentiment_Task:
    def __init__(self, tokenizer,Display_Context=False,Context=None,max_examples_token_length=200,Dataset_Folder='./Datasets/'):
        self.tokenizer=tokenizer
        self.Context=Context
        self.Display_Context=Display_Context
        self.max_examples_token_length=max_examples_token_length
        self.Dataset_Folder=Dataset_Folder
        self.Task_Name='Word_Sentiment_Task'
        
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
        return td,res
        
    def New_Task(self):
        pass 
        
    def Set_Random_Seed(self,seed):
        random.seed(seed) 
  
