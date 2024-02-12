from torch.utils.data import IterableDataset
from util.pool import Pool
import typing

class RLDataset(IterableDataset):  
 """ 
    Iterable Dataset containing the ReplayBuffer 
    which will be updated with new experiences during training 
    Args: 
        buffer: replay buffer 
        sample_size: number of experiences to sample at a time 
    """ 
 
 def __init__(self, dataPool: Pool, device, sample_size: int = 200) -> None:  
        self.dataPool = dataPool  
        self.sample_size = sample_size 
        self.device = device
 
 def __iter__(self) -> typing.Tuple:  
    states, actions, rewards, dones, new_states = self.dataPool.sample(self.sample_size)  
    for i in range(len(dones)):  
        yield states[i].to(self.device), actions[i].to(self.device), \
            rewards[i].to(self.device), dones[i].to(self.device), new_states[i].to(self.device)
