import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
import tiktoken


MODEL_FILE = "./model/pretrain_0616.pth"

def generate(model, input_ids, attention_masks, temperature, steps):

    for s in range(steps):
        # input_ids: B, T
        B, T = input_ids.shape
        # logits: [B, T, vocab_size]
        logits, _ = model(input_ids)
        #print(logits.shape)

        # attention_masks: B, T, sum the value to get the length of each sequence
        input_lengths = torch.sum(attention_masks, dim=1, dtype=torch.int64)
        #print(input_lengths)

        # expend the length of input tensors
        input_ids = torch.concat((input_ids, torch.zeros(B, 1, dtype=torch.int64)), dim=1)
        attention_masks = torch.concat((attention_masks, torch.zeros(B, 1, dtype=torch.int64)), dim=1)

        for i in range(B):
            seq_size = input_lengths[i].item()
            last_logit = logits[i][seq_size - 1]

            if temperature == 0:
                # in this case we just pick the most likely token
                ans = torch.max(last_logit, dim=0)
                token = ans.indices
                # append the last token to the input sequence
                input_ids[i][seq_size] = token
            else:
                # apply the temperature
                last_logit = last_logit / temperature
                probs = F.softmax(last_logit, dim=0)
                token = torch.multinomial(probs, num_samples=1).item()
                input_ids[i][seq_size] = token

            # append the attention mask to the last position
            attention_masks[i][seq_size] = 1 
    
    #print(input_ids)
    #print(attention_masks)
    return input_ids, attention_masks


def strip_state_prefix(state_dict, prefix="_orig_mod.module."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def get_padding_batch_tensor(token_batch):

    max_length = max(map(lambda s: len(s), token_batch))
    input_ids = []
    attention_masks = []

    for tokens in token_batch:
        padded_tokens = tokens + [0] * (max_length - len(tokens))
        padded_masks = [1] * len(tokens) + [0] * (max_length - len(tokens))

        input_ids.append(padded_tokens)
        attention_masks.append(padded_masks)
    
    return \
        torch.tensor(input_ids, dtype=torch.int64), \
        torch.tensor(attention_masks, dtype=torch.int64)



if __name__ == "__main__":
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)

    # Load the state dict from the saved file
    state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
    model.load_state_dict(strip_state_prefix(state_dict))
    model.eval()

    input_str1 = "What is your favorite"
    input_str2 = "It is a sunny day here in Seattle"
    enc = tiktoken.get_encoding("gpt2")
    token_seq1 = enc.encode_ordinary(input_str1)
    token_seq2 = enc.encode_ordinary(input_str2)
    
    input_ids, attention_masks = get_padding_batch_tensor([token_seq1, token_seq2])
    print(input_ids)
    print(attention_masks)

    with torch.no_grad():
        input_ids, attention_masks = generate(model, input_ids, attention_masks, 0.8, 10)
        for i in range(input_ids.shape[0]):
            print(enc.decode(input_ids[i].tolist()))




