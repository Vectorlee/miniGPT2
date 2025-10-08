from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from model import GPT, GPTConfig
from util import strip_state_prefix
import tiktoken
import torch
import random

random.seed(1337)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#initilize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

MODEL_FILE = "./model/pretrain_0616.pth"

def generate(model, input_ids, attention_masks, temperature, max_steps):
    B, T = input_ids.shape
    finish_mask = torch.zeros(B, dtype=torch.int64, device=input_ids.device)
    
    for _ in range(max_steps):
        logits, _ = model(input_ids)   # [B, T, vocab_size]
        last_logit_indexes = attention_masks.sum(dim=1) - 1  # [B, ]
        last_logit = logits[torch.arange(B, device=input_ids.device), last_logit_indexes]

        # expend the length of input tensors
        pad_tensor = torch.zeros(B, 1, dtype=torch.int64, device=input_ids.device)
        input_ids = torch.cat((input_ids, pad_tensor), dim = 1)
        attention_masks = torch.cat((attention_masks, pad_tensor), dim = 1)

        next_tokens = torch.tensor([])
        if temperature == 0:
            # just pick the most likely token
            next_tokens = torch.argmax(last_logit, dim=1)
        else:
            # apply the temperature then do softmax
            probs = F.softmax(last_logit / temperature, dim=1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze()
        
        # assign the next token and attention mask
        input_ids[torch.arange(B, device=input_ids.device), last_logit_indexes + 1] = next_tokens
        attention_masks[torch.arange(B, device=input_ids.device), last_logit_indexes + 1] = 1

        # if we hit the endoftext token, mark it in the finish_mask 
        idx = torch.nonzero(next_tokens == eot, as_tuple=False).squeeze()
        finish_mask[idx] = 1
        if finish_mask.sum() == B:
            break

    return input_ids, attention_masks


def decode_generation(input_ids):
    B, T = input_ids.shape
    answer_list = []

    for i in range(B):
        sequence = input_ids[i].tolist()
        index = sequence.index(eot) if eot in sequence else len(sequence)
        sequence = sequence[:index]
        answer_list.append(enc.decode(sequence))

    return answer_list


def get_padding_batch_input(token_batch):
    input_list = []
    mask_list = []

    for tokens in token_batch:
        input_list.append(torch.tensor(tokens, dtype=torch.int64))
        mask_list.append(torch.ones(len(tokens), dtype=torch.int64))
    
    input_ids = pad_sequence(input_list, batch_first=True)
    attention_masks = pad_sequence(mask_list, batch_first=True)
    
    return input_ids, attention_masks



if __name__ == "__main__":
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)

    # Load the state dict from the saved file
    state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
    model.load_state_dict(strip_state_prefix(state_dict))
    model.eval()

    input_str1 = "I plan to visit Seattle next month, what are the places I can go?"
    input_str2 = "Help me write a short story about a young girl trying to establish herself in a new company."
    enc = tiktoken.get_encoding("gpt2")
    token_seq1 = enc.encode_ordinary(input_str1)
    token_seq2 = enc.encode_ordinary(input_str2)
    
    input_ids, attention_masks = get_padding_batch_input([token_seq1, token_seq2])

    with torch.no_grad():
        input_ids, attention_masks = generate(model, input_ids, attention_masks, 0.8, 100)
        answer_list = decode_generation(input_ids)
        for answer in answer_list:
            print(answer)



