Note: Part of this code is copied from other code as well as using ChatGPT. I noted when I took some code from somewhere else directly in the code. However I do not specifically mark the sections made with the help from ChatGPT. To avoid any copyright issues, assume that the whole code was made with the help from ChatGPT.


To get the the results from my experiment pipeline in my report run: 

main.py


A description about the different files:

main.py: main programm to run experiments

LLM_Tasks.py: includes the code for generate the tasks for the LLMs

Relevance_Maps.py: Includes the computation of the relevance maps

Prediction_Model.py: Includes the code necessary to run and evaluate the relevance of features in a model

Probing.py: Code to run probing experiments

Interventions.py: Code to get the intervention analysis

Intervention_Model.py: Includes the code to run the model for intervention analysis

Generate\_Result\_Graphics.py: Generates Graphs using the interim results

captum_helper.py: Helper for captum (overwriting original code, as original code did not work)

Prediction\_Helpers.py: Helper code for Relevance_Maps.py and Probing.py

TimeMeasurer.py: Code to get some needed time predictions


