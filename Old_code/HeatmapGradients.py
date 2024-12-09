#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
import zipfile
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns


# In[2]:


def Get_Mean_and_Variance(URL,Task):
    Files=listdir(URL+Task) #[:2]
    print('Calculate Mean')
    mean={}
    for ac_file in tqdm(Files):
        with open(URL+Task+'/'+ac_file, 'rb') as f:  
            loaded_data = pickle.load(f)
        for ak1 in loaded_data:
            if ak1 not in mean:
                mean[ak1]={}
            for ak2 in loaded_data[ak1]:
                if ak2 not in mean[ak1]:
                    mean[ak1][ak2]= [None]*len(loaded_data[ak1][ak2])
                for p_ak3,ak3 in enumerate(loaded_data[ak1][ak2]):
                    processing_data=np.array(ak3)
                    if len(processing_data.shape)==3:
                        if processing_data.shape[0]!=1:
                            print('error 1')
                            exit()
                        else:
                            processing_data=processing_data[0]
                    elif len(processing_data.shape)>3:
                            print('error 2')
                            exit()
                                                
                    #print(ak1)
                    #print(processing_data.shape)
                    #print(np.mean(processing_data, axis=0).shape)
                    processing_data=np.abs(processing_data)
                    processing_data=np.mean(processing_data, axis=0)

                    if mean[ak1][ak2][p_ak3] is None:
                        mean[ak1][ak2][p_ak3]=np.zeros((processing_data.shape[0],))
                    mean[ak1][ak2][p_ak3]=mean[ak1][ak2][p_ak3]+processing_data
    for ak1 in mean:
        for ak2 in mean[ak1]:
            for p_ak3,ak3 in enumerate(mean[ak1][ak2]):
                mean[ak1][ak2][p_ak3]=mean[ak1][ak2][p_ak3]/len(Files)

    print()
    print('Calculate Variance')
    variance={}
    for ac_file in tqdm(Files):
        with open(URL+Task+'/'+ac_file, 'rb') as f: 
            loaded_data = pickle.load(f)
        for ak1 in loaded_data:
            if ak1 not in variance:
                variance[ak1]={}
            for ak2 in loaded_data[ak1]:
                if ak2 not in variance[ak1]:
                    variance[ak1][ak2]= [None]*len(loaded_data[ak1][ak2])
                for p_ak3,ak3 in enumerate(loaded_data[ak1][ak2]):
                    processing_data=np.array(ak3)

                    if len(processing_data.shape)==3:
                        processing_data=processing_data[0]

                    processing_data=np.abs(processing_data)
                    processing_data=np.mean(processing_data, axis=0)
                    
                    if variance[ak1][ak2][p_ak3] is None:
                        variance[ak1][ak2][p_ak3]=np.zeros((processing_data.shape[0],))
                    variance[ak1][ak2][p_ak3]=variance[ak1][ak2][p_ak3]+((mean[ak1][ak2][p_ak3]-processing_data)**2)
    for ak1 in variance:
        for ak2 in variance[ak1]:
            for p_ak3,ak3 in enumerate(variance[ak1][ak2]):
                variance[ak1][ak2][p_ak3]=variance[ak1][ak2][p_ak3]/len(Files)

    
    return mean,variance


# In[3]:


meanReg,varianceReg=Get_Mean_and_Variance('./Raw_Gradients/','0')


# In[4]:


meanClass,varianceClass=Get_Mean_and_Variance('./Raw_Gradients/','1')


# In[5]:


#make heat maps

for ak1 in meanReg:
        Diff_heatmapval=[None]*len(meanReg[ak1][0])
        meanReg_heatmapval=[None]*len(meanReg[ak1][0])
        meanClass_heatmapval=[None]*len(meanReg[ak1][0])
        varianceReg_heatmapval=[None]*len(meanReg[ak1][0])
        varianceClass_heatmapval=[None]*len(meanReg[ak1][0])

        norm_meanReg_heatmapval=[None]*len(meanReg[ak1][0])
        norm_meanClass_heatmapval=[None]*len(meanReg[ak1][0])
        norm_varianceReg_heatmapval=[None]*len(meanReg[ak1][0])
        norm_varianceClass_heatmapval=[None]*len(meanReg[ak1][0])



        for ak2 in meanReg[ak1]:
            for p_ak3,ak3 in enumerate(meanReg[ak1][ak2]):
                if Diff_heatmapval[p_ak3] is None:
                    Diff_heatmapval[p_ak3]=[]

                    meanReg_heatmapval[p_ak3]=[]
                    meanClass_heatmapval[p_ak3]=[]
                    varianceReg_heatmapval[p_ak3]=[]
                    varianceClass_heatmapval[p_ak3]=[]

                    norm_meanReg_heatmapval[p_ak3]=[]
                    norm_meanClass_heatmapval[p_ak3]=[]
                    norm_varianceReg_heatmapval[p_ak3]=[]
                    norm_varianceClass_heatmapval[p_ak3]=[]
                dev_from_0_Reg=meanReg[ak1][ak2][p_ak3]/((varianceReg[ak1][ak2][p_ak3]+varianceClass[ak1][ak2][p_ak3])/2)
                dev_from_0_Class=meanClass[ak1][ak2][p_ak3]/((varianceReg[ak1][ak2][p_ak3]+varianceClass[ak1][ak2][p_ak3])/2)
                Diff_heatmapval[p_ak3].append(dev_from_0_Reg-dev_from_0_Class)
                meanReg_heatmapval[p_ak3].append(meanReg[ak1][ak2][p_ak3])
                meanClass_heatmapval[p_ak3].append(meanClass[ak1][ak2][p_ak3])
                varianceReg_heatmapval[p_ak3].append(varianceReg[ak1][ak2][p_ak3])
                varianceClass_heatmapval[p_ak3].append(varianceClass[ak1][ak2][p_ak3])

                norm_meanReg_heatmapval[p_ak3].append(meanReg[ak1][ak2][p_ak3]/np.mean(meanReg[ak1][ak2][p_ak3],axis=None))
                norm_meanClass_heatmapval[p_ak3].append(meanClass[ak1][ak2][p_ak3]/np.mean(meanClass[ak1][ak2][p_ak3],axis=None))
                norm_varianceReg_heatmapval[p_ak3].append(varianceReg[ak1][ak2][p_ak3]/np.mean(varianceReg[ak1][ak2][p_ak3],axis=None))
                norm_varianceClass_heatmapval[p_ak3].append(varianceClass[ak1][ak2][p_ak3]/np.mean(varianceClass[ak1][ak2][p_ak3],axis=None))
                


        for p_ak2,ak2 in enumerate(Diff_heatmapval):
            ak2=np.array(ak2)
            #print()
            #print(ak1,p_ak2)
            #print('*'*100)
            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(ak2,cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Diff_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(meanReg_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Mean_Reg_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(meanClass_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Mean_Class_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(varianceReg_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Variance_Reg_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(varianceClass_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Variance_Class_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()


            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(norm_meanReg_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Norm_Mean_Reg_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(norm_meanClass_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Norm_Mean_Class_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(norm_varianceReg_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Norm_Variance_Reg_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(norm_varianceClass_heatmapval[p_ak2],cmap='icefire', center=0)
            plt.savefig('./Heatmaps/Norm_Variance_Class_Heatmap_'+ak1+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()


# In[ ]:




