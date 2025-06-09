import os
import torch
import numpy as np
import random

DATA_ROOT = "fineweb_edu_10B"

def load_token(filename):
    token_bytes = np.load(filename)
    toekn_tensor = torch.tensor(token_bytes, dtype=torch.long)
    return toekn_tensor

class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        # list the file
        data_root = DATA_ROOT
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for {split} under {data_root}"
        print(f"Found {len(shards)} shards for split {split}")

        self.reset()


    def reset(self):
        # set the starting shard randomly to create some randomness in data loading
        self.current_shard = random.randint(0, len(self.shards)) % len(self.shards)
        self.tokens = load_token(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):

        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # each dataloader only picks certain parts from each shard
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_token(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y