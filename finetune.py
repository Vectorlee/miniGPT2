from dataclasses import dataclass
import tiktoken
import torch
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import time
from model import GPT, GPTConfig

DATA_FILE = "./finetune_data/alpaca.parquet"

# set the random seed to ensure reproducibility
random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


def load_tokens_numpy_list():
    df = pd.read_parquet(DATA_FILE)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

    # Columns: instruction, input, output, text
    # filter all the conversation longer than the threshold
    df = df[df['text'].str.len() < 1024]
    # filter all the invalid output 
    df = df[~ df['output'].str.contains('<nooutput>')]

    texts = df['text'].tolist()
    outputs = df['output'].tolist()

    text_tokens = list(map(lambda s: np.array(enc.encode_ordinary(s)), texts))
    outputs_tokens = list(map(lambda s: np.array(enc.encode_ordinary(s)), outputs))
    label_tokens = []
    for t, o in zip(text_tokens, outputs_tokens):
        label = t.copy()
        # set to -100 so the cross entropy function will ignore these tokens
        label[0: len(t) - len(o)] = -100
        label_tokens.append(label)

    return list(zip(text_tokens, label_tokens))


def pad_finetune_batch(data_list):
    max_length = max(map(lambda s: len(s[0]), data_list))
    #print(f"max length in batch: {max_length}")

    padded_list = []
    for input, label in data_list:
        assert(len(input) == len(label))
        padding_size = max_length - len(input)

        padded_input = np.concatenate((input, [0] * padding_size), axis=0)
        padded_label = np.concatenate((label, [-100] * padding_size), axis=0)
        padded_list.append((padded_input, padded_label))
    
    return padded_list


class DataLoadeFinetune:

    def __init__(self, batch_size):
        # load the file
        token_list = load_tokens_numpy_list()

        # reserve the same constant part for validation test
        self.val_data = token_list[:1000]

        _tmp_input = token_list[1000:]
        random.shuffle(_tmp_input)

        # shuffle the remain data and get the train test split
        self.train_data = _tmp_input[len(_tmp_input) // 10:]
        self.test_data = _tmp_input[:len(_tmp_input) // 10]

        #print(f"data length, train_data: {len(self.train_data)}, test_data: {len(self.test_data)}")
        self.batch_size = batch_size
        self.train_position = 0
        self.test_position = 0
        self.val_position = 0
    

    def reset(self):
        self.train_position = 0
        self.test_position = 0
        self.val_position = 0


    def training_data_size(self):
        return len(self.train_data)


    def __fetch_data__(self, data_buf, position, batch_size):
        current_position = position

        buf = data_buf[current_position: current_position + batch_size]
        current_position += batch_size

        if current_position > len(data_buf):
            current_position = current_position % len(data_buf)
            buf.extend(data_buf[:current_position])

        padded_batch = pad_finetune_batch(buf)
        # train_data: list of tuples, [(text, label)]
        x = np.array(list(map(lambda s: s[0], padded_batch)))
        y = np.array(list(map(lambda s: s[1], padded_batch)))

        return x, y, current_position


    def get_train_batch(self):
        x, y, position = self.__fetch_data__(
            self.train_data, self.train_position, self.batch_size
        )
        self.train_position = position

        #print(f"train_position: {self.train_position}")
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)


    def get_test_batch(self, size=8):
        x, y, position = self.__fetch_data__(
            self.test_data, self.test_position, size
        )
        self.test_position = position
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)
    
    def get_val_batch(self, size=8):
        x, y, position = self.__fetch_data__(
            self.val_data, self.val_position, size
        )
        self.val_position = position
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)



def calc_finetune_loss(model, input, label):
    # input/label shape: B, T
    B, T = input.shape

    # logits shape: B, T, vocab_size
    logits, _ = model(input)

    # left shift label !!!
    target = label[:, 1:]

    pad = torch.zeros(B, 1, dtype=torch.int64)
    pad = torch.fill(pad, -100)
    pad = pad.to(target.device)
    target = torch.concat((target, pad), dim=1)

    # cross entropy by default will ignore index with -100
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    return loss


def configure_optimizers(model, weight_decay, learning_rate):
    # start with all of the parameters that require grad
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # do not weight decay bias, layernorm, and other less than 2 dimension weights
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed tensors: {len(decay_params)}, with {num_decay_params} parameters")
    print(f"num non-decayed tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

    fused = True if torch.cuda.is_available() else False
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=fused)
    return optimizer


def finetune_loop(model):
    batch_size = 2048
    learning_rate = 5e-5
    epoch = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = DataLoadeFinetune(batch_size)
    finetune_steps = dataloader.training_data_size() / batch_size * epoch
    print(f"Finetuning steps: {finetune_steps}")

    model = model.to(device)
    # compile the model, for kernel fuse
    model = torch.compile(model)

    # smaller weight decay as we are doing finetuning
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=learning_rate)

    model.train()
    for step in range(finetune_steps):
        t0 = time.time()

        optimizer.zero_grad()

        x, y = dataloader.get_train_batch()
        x, y = x.to(device), y.to(device)

        # mixed precision training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):            
            # without the no_sync context manager here
            loss = calc_finetune_loss(model, x, y)
            loss.backward()

        # Gradient Clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        t1 = time.time()

        dt = (t1 - t0)
        token_processed = batch_size * len(x[0])
        token_per_sec = token_processed / dt
        # the item() function ship the tensor back from gpu to cpu
        print(f"step {step}, loss: {loss.item():.6f}, dt: {dt * 1000:.2f}ms, \
              tok/sec: {token_per_sec}, norm: {norm:.4f}")




if __name__ == "__main__":
    dataloader = DataLoadeFinetune(20000)

    for i in range(3):
        x, y = dataloader.get_train_batch()

        print(x.shape, y.shape)
        print(len(x[3]))
        print(len(y[3]))