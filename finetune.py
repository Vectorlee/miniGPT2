from dataclasses import dataclass
import tiktoken
import torch
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import time
from model import GPT, GPTConfig

SFT_DATA_FILE = "./finetune_data/alpaca.parquet"
MODEL_FILE = "./model/pretrain_0616.pth"

# set the random seed to ensure reproducibility
random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


def load_tokens():
    df = pd.read_parquet(SFT_DATA_FILE)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

    # Alpaca Parquet Columns: "instruction", "input", "output", "text"
    # filter all the conversation longer than the model context window
    df = df[df['text'].str.len() < 1024]
    # filter all the invalid output 
    df = df[~ df['output'].str.contains('<nooutput>')]

    texts = df['text'].tolist()
    outputs = df['output'].tolist()

    # add end of text token to input and output
    text_tokens = list(map(lambda s: np.array(enc.encode_ordinary(s) + [eot]), texts))
    outputs_tokens = list(map(lambda s: np.array(enc.encode_ordinary(s) + [eot]), outputs))
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

    def __init__(self, validation_size=1000):
        # load the file
        token_list = load_tokens()

        # reserve the same constant part for validation test
        self.val_data = token_list[:validation_size]

        _tmp_input = token_list[validation_size:]
        random.shuffle(_tmp_input)

        # shuffle the remain data and get the train test split
        self.train_data = _tmp_input[len(_tmp_input) // 10:]
        self.test_data = _tmp_input[:len(_tmp_input) // 10]

        #print(f"data length, train_data: {len(self.train_data)}, test_data: {len(self.test_data)}")
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


    def get_train_batch(self, batch_size):
        x, y, position = self.__fetch_data__(
            self.train_data, self.train_position, batch_size
        )
        self.train_position = position

        #print(f"train_position: {self.train_position}")
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)


    def get_test_batch(self, batch_size=32):
        x, y, position = self.__fetch_data__(
            self.test_data, self.test_position, batch_size
        )
        self.test_position = position
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)
    
    def get_val_batch(self, batch_size=32):
        x, y, position = self.__fetch_data__(
            self.val_data, self.val_position, batch_size
        )
        self.val_position = position
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)



def get_loss(model, inputs, labels):
    # inputs/labels shape: B, T
    B, T = inputs.shape

    # logits shape: B, T, vocab_size
    logits, _ = model(inputs)

    # left shift label
    # as mentioned in https://huggingface.co/docs/transformers/v4.53.0/en/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.labels
    # "Note that the labels are shifted inside the model, i.e. you can set labels = input_ids"
    # "Indices are selected in [-100, 0, ..., config.vocab_size]"
    # "All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]"
    target = labels[:, 1:]

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


def instruction_finetune(model, dataloader, batch_size, learning_rate, epoch):
    device = model.device
    finetune_steps = dataloader.training_data_size() / batch_size * epoch
    print(f"Finetuning steps: {finetune_steps}")

    # smaller weight decay as we are doing finetuning
    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=learning_rate)

    for step in range(finetune_steps):

        # validation loop
        if step % 50 == 0 or step == finetune_steps - 1:
            model.eval()
            with torch.no_grad():
                test_steps = 20
                test_loss_accum = 0
                for _ in range(test_steps):
                    x, y = dataloader.get_test_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        loss = get_loss(model, x, y)
                    loss = loss / test_steps
                    test_loss_accum += loss.detach()
                print(f"validation loss: {test_loss_accum.item():.6f}")

        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        x, y = dataloader.get_train_batch(batch_size)
        x, y = x.to(device), y.to(device)

        # mixed precision training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):            
            # without the no_sync context manager here
            loss = get_loss(model, x, y)
            loss.backward()

        # Gradient Clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        t1 = time.time()

        dt = (t1 - t0)
        token_processed = x.shape[0] * x.shape[1]
        token_per_sec = token_processed / dt
        # the item() function ship the tensor back from gpu to cpu
        print(f"step {step}, loss: {loss.item():.6f}, dt: {dt * 1000:.2f}ms, \
              tok/sec: {token_per_sec}, norm: {norm:.4f}")
    
    return model


def strip_state_prefix(state_dict, prefix="_orig_mod.module."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict



def training_loop():
    batch_size = 512
    learning_rate = 5e-5
    epoch = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = DataLoadeFinetune()
    print(f"total training data: {dataloader.training_data_size()}")

    config = GPTConfig(vocab_size=50304)
    model = GPT(config)

    # Load the state dict from the saved file
    state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
    model.load_state_dict(strip_state_prefix(state_dict))
    
    model = model.to(device)
    # compile the model, for kernel fuse
    model = torch.compile(model)

    # instrunction finetuning
    instruction_finetune(
        model, 
        dataloader, 
        batch_size, 
        learning_rate, 
        epoch
    )


if __name__ == "__main__":
    training_loop()
