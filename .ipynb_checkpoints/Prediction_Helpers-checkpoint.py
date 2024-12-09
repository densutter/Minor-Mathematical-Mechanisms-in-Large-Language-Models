import os
import Prediction_Model
import ray

class Prediction_Helper:

    def Get_Results(self,Task_Text,Task_Result):
        if self.Allowed_Model_Usage_Before_Refresh<=self.Actual_Model_Usage:
            if self.verbose:
                print("[RAY ] Killed Container")
            ray.kill(self.model_handler)
            self.model_handler=None
            self.Actual_Model_Usage=0
        if self.model_handler is None:
            if self.verbose:
                print("[RAY ] Generated Container")
            self.model_handler=Prediction_Model.create_llm_remote(self.model_id, self.GPU_Savings, self.max_memory , self.tokenizer, self.Relevance_Map_Method,num_gpus=self.num_gpus,num_cpus=self.num_cpus)
        self.Actual_Model_Usage+=1

        results=ray.get(self.model_handler.get_results.remote(Task_Text,Task_Result))

        return results


    def safe_listdir(self,path):
        try:
            return os.listdir(path)
        except FileNotFoundError:
            return None

    def ensure_folder_exists(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)