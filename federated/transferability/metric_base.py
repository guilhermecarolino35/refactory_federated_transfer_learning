from abc import ABC, abstractmethod

class TransferMetric(ABC):
   # API 

    @abstractmethod
    def on_round_start(self, global_model, local_model, dataloader, round_number):
        pass

    @abstractmethod
    def on_round_end(self, global_model, local_model, dataloader, round_number):
        pass

    @abstractmethod
    def compute(self) -> dict:
        pass