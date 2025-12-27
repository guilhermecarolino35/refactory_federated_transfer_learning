class TransferabilityManager:
    def __init__(self,metrics):
        self.metrics = metrics

    def compute_all(self, repr_cache,round_number):
        results = {}
        for metric in self.metrics:
            results.update(metric.compute(repr_cache,round_number))
        return results