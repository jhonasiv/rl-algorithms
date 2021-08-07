from abc import ABC, abstractmethod
from dataclasses import dataclass


class BaseFunction(ABC):
    def __init__(self, val_init: float):
        self.val = val_init
    
    def value(self) -> float:
        return self.val
    
    @abstractmethod
    def step(self, episode: int):
        """
        Run a step on this function
        
        :return: updated value
        """


@dataclass
class Constant(BaseFunction):
    def __init__(self, val_init: float):
        super().__init__(val_init)
    
    def step(self, _=None):
        return self.val


class LinearAnnealing(BaseFunction):
    def __init__(self, val_init: float, val_thresh: float, max_itt: int):
        super(LinearAnnealing, self).__init__(val_init)
        self.val_thresh = val_thresh
        self.step_size = (val_init - val_thresh) / max_itt
        self.op = min if val_thresh > val_init else max
    
    def step(self, episode: int) -> float:
        self.val = self.op(self.val_thresh, self.val - self.step_size)
        return self.val
