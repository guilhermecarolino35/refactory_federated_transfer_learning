

class TransferabilityManager:
    def __init__(self,metrics):
        self.metrics = metrics

    def on_round_start(self,global_model,local_model,dataloader,round_number):
        for metric in self.metrics:
            metric.on_round_start(
                global_model,local_model,dataloader,round_number)
    
    def on_round_end(self,global_model,local_model,dataloader,round_number):
        results= {}
        for metric in self.metrics:
            metric.on_round_end(
                global_model,local_model,dataloader,round_number)
            
            results.update(metric.compute())

        return results