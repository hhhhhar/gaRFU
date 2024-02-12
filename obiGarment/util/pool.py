import random
import torch
import torch.utils.data as data

class Pool(data.Dataset):

    def __init__(self):
        self.pool = []

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    #获取一批数据样本
    def sample(self, sample_size):
        data = random.sample(self.pool, sample_size)[0]

        state = data[0].unsqueeze_(0).to(torch.float32)
        action = data[1].reshape([1, -1]).to(torch.float32)
        reward = torch.as_tensor([data[2]]).unsqueeze_(0).to(torch.float32)
        next_state = torch.from_numpy(data[3]).unsqueeze_(0).to(torch.float32)
        over = torch.as_tensor([data[4]]).unsqueeze_(0).to(torch.float32)

        return state, action, reward, next_state, over
