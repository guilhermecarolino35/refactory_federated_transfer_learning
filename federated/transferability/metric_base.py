from abc import ABC, abstractmethod

class TransferMetric(ABC):
   # API 


    @abstractmethod
    def compute(self,repr_cache: dict, round_number:int) -> dict:
        pass