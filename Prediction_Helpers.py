import os
import Prediction_Model

class Prediction_Helper:

    

    def Get_Results(self,Task_Text,Task_Result):
        if self.model_handler is None:
            if self.verbose:
                print("[INFO] Generated Model")
            self.model_handler=Prediction_Model.create_llm_remote(self.model_id, self.max_memory , self.tokenizer, self.Relevance_Map_Method,self.Baselines)

        results=self.model_handler.get_results(Task_Text,Task_Result)

        return results


    def safe_listdir(self,path):
        try:
            return os.listdir(path)
        except FileNotFoundError:
            return None

    def ensure_folder_exists(self,folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
