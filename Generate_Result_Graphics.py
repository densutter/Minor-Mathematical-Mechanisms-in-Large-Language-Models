#Libraries
import LLM_Tasks
import Relevance_Maps
import Probing
from transformers import AutoTokenizer
import torch
torch.set_num_threads(6)
import numpy as np
import os
import json
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


#Hyperparameters:
#model_id = "Local-Meta-Llama-3.2-1B"
#Relevance_Map_Method='vanilla_gradient' #'smoothGrad'
#Probing_Method='Probing'
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#Use_Context=True

Testing_Samples=100
Task_1=LLM_Tasks.Regression_Task_Int(None,Display_Context=False)
Task_2=LLM_Tasks.Multiclass_Logistic_Regression_Task(None,Display_Context=False)
Task_1_Name=Task_1.Task_Name
Task_2_Name=Task_2.Task_Name
Interim_Results_URL='./interim_results/'
Results_URL="./results/"
keep_percentages=["1", "0.5", "0.25"]
statistical_significance_value=0.05

# Helper Functions

def ensure_folder_exists(folder_path):
    """
    Check if a folder exists at the specified path. If it doesn't, create the folder.

    Parameters:
    - folder_path (str): The path of the folder to check or create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def Make_Heatmap_Relevance_Map(HM_Values,Relevance_Heatmap_URL):
    ensure_folder_exists(Relevance_Heatmap_URL)
    for ac_Layer in HM_Values:
        Heatmap_Vals=[]
        Heatmap_Vals_Norm=[]
        for ac_Depth in HM_Values[ac_Layer]:
            if len(np.array(HM_Values[ac_Layer][ac_Depth]).shape)<2:
                HM_Values[ac_Layer][ac_Depth]=[HM_Values[ac_Layer][ac_Depth]]
            for ac_i_p,ac_i in enumerate(HM_Values[ac_Layer][ac_Depth]):
                while len(Heatmap_Vals)<=ac_i_p:
                    Heatmap_Vals.append([None]*len(HM_Values[ac_Layer]))
                    Heatmap_Vals_Norm.append([None]*len(HM_Values[ac_Layer]))
                Heatmap_Vals[ac_i_p][ac_Depth]=ac_i
                np_ac_i=np.array(ac_i)
                Heatmap_Vals_Norm[ac_i_p][ac_Depth]=(np_ac_i/np.abs(np_ac_i).max()).tolist()
        for p_ak2,ak2 in enumerate(Heatmap_Vals):
            ak2=np.array(ak2)
            #print()
            #print(ak1,p_ak2)
            #print('*'*100)
            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(ak2,cmap='icefire', center=0)
            plt.savefig(Relevance_Heatmap_URL+'Diff_Heatmap_'+ac_Layer+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()

            fig = plt.gcf()  # Get the current figure
            fig.set_size_inches(18, 8)  # Set the size in inches
            sns.heatmap(Heatmap_Vals_Norm[p_ak2],cmap='icefire', center=0)
            plt.savefig(Relevance_Heatmap_URL+'Diff_Heatmap_Normalized_'+ac_Layer+'_'+str(p_ak2)+'.png',bbox_inches='tight')
            plt.close()


def analyze_data(input_data_structure, test_loss_num):
    results = {}

    for intermed_var, layers in input_data_structure.items():
        results[intermed_var] = {}

        for layer_name, layer_depths in layers.items():
            results[intermed_var][layer_name] = {}

            for layer_depth, versions in layer_depths.items():
                if not isinstance(versions, dict):
                    versions_n={}
                    versions_n["0"]=versions
                    versions=versions_n
                for output_version_idx,output_version in versions.items():
                    if output_version_idx not in results[intermed_var][layer_name]:
                        results[intermed_var][layer_name][output_version_idx]=None
                    # Prepare to save metrics for each layer_depth and keep_percentage
                    metrics = {}

                    # Filter out the relevant loss values for keep_percentage == 1, 0.5, and 0.25
                    for keep_percentage in keep_percentages:
                        #print(output_version[0])
                        top_loss_values = output_version[0][keep_percentage][-test_loss_num:]
                        bottom_loss_values = output_version[1][keep_percentage][-test_loss_num:]


                        # Compute means, variances, and differences
                        metrics["mean_top_keep_"+keep_percentage] = np.mean(top_loss_values)
                        metrics["mean_bottom_keep_"+keep_percentage] = np.mean(bottom_loss_values)
                        metrics["variance_top_keep_"+keep_percentage] = np.var(top_loss_values)
                        metrics["variance_bottom_keep_"+keep_percentage] = np.var(bottom_loss_values)
                        if keep_percentage != "1":
                            # Mean difference
                            top_mean = np.mean(top_loss_values)
                            bottom_mean = np.mean(bottom_loss_values)
                            metrics[f"mean_diff_keep_{keep_percentage}"] = top_mean - bottom_mean

                            t_stat, p_val = stats.ttest_ind(top_loss_values, bottom_loss_values, equal_var=False)
                            metrics[f"p_value_diff_keep_{keep_percentage}"] = p_val

                    if results[intermed_var][layer_name][output_version_idx] is None:
                        results[intermed_var][layer_name][output_version_idx]=[None]*len(layer_depths.keys())
                    #print(int(layer_depth))
                    results[intermed_var][layer_name][output_version_idx][int(layer_depth)] = metrics
    return results


def Make_Heatmap_Probing_Helper(probing_results,probing_results_url):
    for intermed_var in probing_results:
        for layer_name in probing_results[intermed_var]:
            for output_version_idx in probing_results[intermed_var][layer_name]:
                mean={}
                mean["top"]={}
                mean["bottom"]={}
                vari={}
                vari["top"]={}
                vari["bottom"]={}
                ac_depth=len(probing_results[intermed_var][layer_name][output_version_idx])
                for keep_percentage in keep_percentages:
                    mean["top"][keep_percentage]=[None]*ac_depth
                    mean["bottom"][keep_percentage]=[None]*ac_depth
                    vari["top"][keep_percentage]=[None]*ac_depth
                    vari["bottom"][keep_percentage]=[None]*ac_depth

                differe={}
                p_value={}
                for keep_percentage in keep_percentages:
                    if keep_percentage!="1":
                        differe[keep_percentage]=[None]*ac_depth
                        p_value[keep_percentage]=[None]*ac_depth

                for depth in range(len(probing_results[intermed_var][layer_name][output_version_idx])):
                    for tb in mean:
                        for per in mean[tb]:
                            mean[tb][per][depth]=probing_results[intermed_var][layer_name][output_version_idx][depth]["mean_"+tb+"_keep_"+per]
                            vari[tb][per][depth]=probing_results[intermed_var][layer_name][output_version_idx][depth]["variance_"+tb+"_keep_"+per]

                    for per in differe:
                        differe[per][depth]=probing_results[intermed_var][layer_name][output_version_idx][depth]["mean_diff_keep_"+per]
                        p_value[per][depth]=probing_results[intermed_var][layer_name][output_version_idx][depth]["p_value_diff_keep_"+per]



                #Make mean and variance heatmaps
                mean_plot_data=[]
                plot_x_axis=[]
                plot_y_axis=[]

                vari_plot_data=[]

                plot_x_axis=list(range(ac_depth))
                for per in keep_percentages:
                    for tb in ["top","bottom"]:

                        plot_y_axis.append(per+"-"+tb)

                        mean_plot_data.append(mean[tb][per])
                        vari_plot_data.append(vari[tb][per])


                fig = plt.gcf()  # Get the current figure
                fig.set_size_inches(18, 8)  # Set the size in inches
                sns.heatmap(mean_plot_data,cmap='icefire', center=0,xticklabels=plot_x_axis, yticklabels=plot_y_axis)
                plt.savefig(probing_results_url+'/Mean_Heatmap_'+intermed_var+"-"+layer_name+'_'+str(output_version_idx)+'.png',bbox_inches='tight')
                plt.close()

                fig = plt.gcf()  # Get the current figure
                fig.set_size_inches(18, 8)  # Set the size in inches
                sns.heatmap(vari_plot_data,cmap='icefire', center=0,xticklabels=plot_x_axis, yticklabels=plot_y_axis)
                plt.savefig(probing_results_url+'/Variance_Heatmap_'+intermed_var+"-"+layer_name+'_'+str(output_version_idx)+'.png',bbox_inches='tight')
                plt.close()



                #Make difference p-value plot
                differe_plot_data=[]
                plot_x_axis=[]
                plot_y_axis=[]

                p_value_plot_data=[]

                plot_x_axis=list(range(ac_depth))
                for per in keep_percentages:
                    if per!="1":
                        plot_y_axis.append(per)

                        differe_plot_data.append(differe[per])
                        p_value_plot_data.append(p_value[per])

                fig = plt.gcf()  # Get the current figure
                fig.set_size_inches(18, 8)  # Set the figure size in inches

                # Create the heatmap
                sns.heatmap(differe_plot_data, cmap='icefire', center=0, xticklabels=plot_x_axis, yticklabels=plot_y_axis)

                # Get the current axes
                ax = plt.gca()

                # Loop over each cell in the p_vals array to check the p-value
                for i in range(len(p_value_plot_data)):  # Iterate over rows
                    for j in range(len(p_value_plot_data[i])):  # Iterate over columns
                        if p_value_plot_data[i][j] < statistical_significance_value:  # Condition to highlight
                            # Create a rectangle with a specific edge color and width
                            #print("nice")
                            rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor='#9ACD32', linewidth=2)
                            ax.add_patch(rect)

                # Save the figure
                plt.savefig(probing_results_url+'/Diff_Heatmap_'+intermed_var+"-"+layer_name+'_'+str(output_version_idx)+'.png', bbox_inches='tight')
                plt.close()



def Make_Heatmap_Probing(probing_interim_results_url,probing_results_url,Task_1_Name,Task_2_Name,Testing_Samples):
    for ac_Task_Name in [Task_1_Name,Task_2_Name]:

        probing_interim_results_url_h = probing_interim_results_url+ac_Task_Name+"/Metadata.json"
        probing_results_url_h         = probing_results_url+ac_Task_Name
        ensure_folder_exists(probing_results_url_h)

        with open(probing_interim_results_url_h, 'r') as file:
            probing_interim_results = json.load(file)
        probing_results=analyze_data(probing_interim_results["Loss"], Testing_Samples)
        with open(probing_results_url_h+"/Results.json", 'w') as file:
            json.dump(probing_results, file, indent=4)

        Make_Heatmap_Probing_Helper(probing_results,probing_results_url_h)


print("[INFO] Step 1: Generate Heatmaps for Relevance Maps")

RelMap=Relevance_Maps.Relevance_Map(
    None,
    "vanilla_gradient",
    Task_1,
    Task_2,
    None,
    Allowed_Model_Usage_Before_Refresh=10,
    max_memory='cpu',
    num_gpus=0,
    num_cpus=1,
    interim_results_folder=Interim_Results_URL
)

Rel_Map_Result=RelMap.Get_Relevance_Map(Number_of_samples=-1)
Make_Heatmap_Relevance_Map(Rel_Map_Result,Results_URL+"Relevance_Heatmaps/"+Task_1_Name+"_vs_"+Task_2_Name+"/")

print("[INFO] Step 1: Finished")



Interim_Results_URL='./interim_results/'
Results_URL="./results/"

print("[INFO] Step 2: Analyse probing loss")

probing_interim_results_url = Interim_Results_URL+'Probing_Probing_Layer_1/'
probing_results_url         = Results_URL+"Probing_Results/"

Make_Heatmap_Probing(probing_interim_results_url,probing_results_url,Task_1_Name,Task_2_Name,Testing_Samples)

print("[INFO] Step 2: Finished")