from dataclasses import dataclass, field

from abc import ABC, abstractmethod

from rlalgs.utils.functions import exponential_function


@dataclass
class BaseBetaFunction(ABC):
    beta_initial: float
    value: float = field(init=False, default=0)
    
    def __post_init__(self):
        self.value = self.beta_initial
    
    @abstractmethod
    def step(self, episode: int):
        """
        Updates beta.
        
        :return: new beta value
        """


@dataclass
class ConstantBeta(BaseBetaFunction):
    def step(self, _=None):
        return self.value


@dataclass
class ExponentialAnnealingBeta(BaseBetaFunction):
    rate: float
    beta_max: float = 1.0
    
    def step(self, episode: int):
        self.value = 1 - exponential_function(a=1, x=episode, k=self.rate, b=15, exp=2)
        self.value = min(self.value, self.beta_max)
        return self.value
