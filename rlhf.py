from dataclasses import dataclass
import tiktoken
import torch
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import time
import os
from model import GPT, GPTConfig
from util import strip_state_prefix


class DataLoaderPreference:

    def __init__(self, batch_size):

        self.data_list = [(1, -1)] * 10
        self.current_position = 0
        self.batch_size = batch_size
    
    def get_batch(self):

        data = self.data_list[self.current_position: self.current_position + self.batch_size]
        self.current_position += self.batch_size

        if self.current_position > len(self.data_list):
            self.current_position = self.current_position % len(self.data_list)
            data.extend(self.data_list[:self.current_position])

        return data

class RewardModel(torch.nn.Module):

    def __init__(self, gpt_model: GPT):

        self.gpt_model = gpt_model
        self.linear_head = torch.nn.Linear(gpt_model.config.n_embd, 1)
    
    def forward(self, prefered, unprefered, mask, last_token=True):
        B, T = idx.size()
        assert mask.shape == idx.shape

        # this is a batch
        output, _ = self.gpt_model(idx, skip_lm_head=True)
        masked_output = output * mask

        if last_token:
            last_indexes = mask.sum(dim=1) - 1
            tokens = masked_output[torch.arange(B),last_indexes,:]
            return self.linear_head(tokens)
        else:
            average = masked_output.sum(dim=1) / mask.sum(dim=1)
            return self.linear_head(average)



def reward_model_training():
    model_file = "./model/pretrain_0616.pth"
    model_dir = "model"
    step = 10
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 30

    config = GPTConfig(vocab_size=50304)
    gpt_model = GPT(config)

    # Load the state dict from the saved file
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    gpt_model.load_state_dict(strip_state_prefix(state_dict))

    # construct the reward model
    model = RewardModel(gpt_model)

    dataloader = DataLoaderPreference(30)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for i in range(step):
        optimizer.zero_grad()

        # [(prefered, not_prefered) ...]
        preference = dataloader.get_batch()
        
        prefered = torch.tensor([item[0] for item in preference])
        not_prefered = torch.tensor([item[1] for item in preference])

        # need to pad the data into the same length

        # mixed precision training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            prefered_target = model(prefered, skip_lm_head=True)
            not_prefered_target = model(not_prefered, skip_lm_head=True)

            loss = -torch.log(F.sigmoid(prefered_target) - F.sigmoid(not_prefered_target))
            loss = loss.mean()
        
        loss.backward()
        optimizer.step()
    


