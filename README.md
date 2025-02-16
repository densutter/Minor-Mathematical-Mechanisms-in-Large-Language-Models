## **Notes on Code Usage**  
Some parts of this code were copied from other sources and generated using ChatGPT. Any directly copied code is explicitly noted within the code. However, sections written with ChatGPT's assistance are not specifically marked. **To avoid any copyright issues, assume the entire code was developed with the help of ChatGPT.**  

## **Running the Experiment Pipeline**  
To reproduce the results from my experiment pipeline in the report, run:  
```bash
python main.py
```

## **Project Structure**  

### **Main Script**  
- **`main.py`** – Main program to run the experiments.  

### **Task & Feature Relevance Analysis**  
- **`LLM_Tasks.py`** – Generates tasks for the LLMs.  
- **`Relevance_Maps.py`** – Computes relevance maps.  
- **`Prediction_Model.py`** – Runs and evaluates feature relevance in a model.  

### **Probing & Interventions**  
- **`Probing.py`** – Runs probing experiments.  
- **`Interventions.py`** – Performs intervention analysis.  
- **`Intervention_Model.py`** – Runs the model for intervention analysis.  

### **Results & Visualization**  
- **`Generate_Result_Graphics.py`** – Generates graphs from interim results.  

### **Helper Modules**  
- **`captum_helper.py`** – Overwrites Captum’s original code (fixing functionality issues).  
- **`Prediction_Helpers.py`** – Helper functions for `Relevance_Maps.py` and `Probing.py`.  
- **`TimeMeasurer.py`** – Provides time predictions for experiments.  
