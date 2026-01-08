# metrics_logger.py
import csv
import os
from typing import Optional
import pandas as pd

class MetricsLogger:
    def __init__(self, experiment_id, experiment_run, outdir="results"):
        self.experiment_id = experiment_id
        self.experiment_run = experiment_run
        self.outdir = outdir
        
          # 👇 ROUND GLOBAL ATUAL
        self.current_round: int | None = None



        self.metrics_per_client_mmd = pd.DataFrame(
            columns= [
                "experiment_id",
                "experiment_run",
                "round",
                "client_id",
                "mmd"
            ]
        )

        self.metrics_per_client_kl = pd.DataFrame(
            columns= [
                "experiment_id",
                "experiment_run",
                "round",
                "client_id",
                "kl_divergence"
            ]
        )

        self.metrics_per_client_js = pd.DataFrame(
            columns= [
                "experiment_id",
                "experiment_run",
                "round",
                "client_id",
                "js"
            ]
        )

        self.metrics_per_client = pd.DataFrame(
            columns=[
                "experiment_id",
                "experiment_run",
                "round",
                "client_id",
                "accuracy",

            ]
        )
        
       

        
        self.metrics_global = pd.DataFrame(
            columns= [
                "experiment_id",
                "experiment_run",
                "round",
                "global_accuracy"
            ]
        )


   
    def log_client_accuracy(self, client_id: int, round_number: int, accuracy: float):
        """Log accuracy of a single client for a given round."""

        new_row = {
            "experiment_id": self.experiment_id,
            "experiment_run": self.experiment_run,
            "round": round_number,
            "client_id": client_id,
            "accuracy": accuracy,
        }

        # Adiciona a nova linha ao DataFrame
        self.metrics_per_client.loc[len(self.metrics_per_client)] = new_row


    def log_global_accuracy(self,round_number:int,accuracy:float):
        new_row = {
            "experiment_id":self.experiment_id,
            "experiment_run":self.experiment_run,
            "round":round_number,
            "global_accuracy":accuracy,
        }
        self.metrics_global.loc[len(self.metrics_global)] = new_row


    

    def log_client_mmd(self,client_id,round_number,mmd):
        new_row = {
            "experiment_id":self.experiment_id,
            "experiment_run":self.experiment_run,
            "round":round_number,
            "client_id":client_id,
            "mmd":mmd,
        }
        self.metrics_per_client_mmd.loc[len(self.metrics_per_client_mmd)] = new_row
         



    def log_client_kl(self,client_id,round_number,kl):
        new_row = {
            "experiment_id":self.experiment_id,
            "experiment_run":self.experiment_run,
            "round":round_number,
            "client_id":client_id,
            "kl_divergence":kl,
        }
        self.metrics_per_client_kl.loc[len(self.metrics_per_client_kl)] = new_row
         
    def log_client_js(self,client_id,round_number,js):
        new_row = {
            "experiment_id":self.experiment_id,
            "experiment_run":self.experiment_run,
            "round":round_number,
            "client_id":client_id,
            "js":js,
        }
        self.metrics_per_client_js.loc[len(self.metrics_per_client_js)] = new_row 

    def save(self):
        # results/experiment_X/run_Y
        base_dir = os.path.join(
            self.outdir,f"experiment_{self.experiment_id}",
            f"run_{self.experiment_run}",
        )

        os.makedirs(base_dir,exist_ok=True)

        #Paths
        client_path = os.path.join(base_dir,"metrics_per_client.csv")
        global_path = os.path.join(base_dir,"metrics_global.csv")
        kl_path = os.path.join(base_dir,"metrics_per_client_kl.csv")
        js_path = os.path.join(base_dir,"metrics_per_client_js.csv")
        mmd_path = os.path.join(base_dir,"metrics_per_client_mmd.csv")
        #Save
        self.metrics_per_client.to_csv(client_path,index=False)
        self.metrics_per_client_kl.to_csv(kl_path,index=False)
        self.metrics_per_client_js.to_csv(js_path,index=False)
        self.metrics_per_client_mmd.to_csv(mmd_path,index=False)

        self.metrics_global.to_csv(global_path,index=False)
