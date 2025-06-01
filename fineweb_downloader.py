import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "finweb_edu_10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M token per shard, total of 100 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split='train')

#initilize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# tokenize an individual document
def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # write a numpy array of uint16 tokens to a binary file
    with open(filename, "wb") as pfile:
        pfile.write(tokens_np.tobytes())


# tokenize all documents and write output shards
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0

    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # the "tokens" here are the the tokens of an individual fineweb doc
    for tokens in pool.imap(tokenize, fw, chunksize=32):

        # if it is enough space in the current shard for the new tokens
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)

            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_edu_{split}_{shard_index:05d}.npy")

            remainder = shard_size - token_count
            progress_bar.update(remainder)
            
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)

            shard_index += 1
            progress_bar = None

            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
    
    if token_count != 0:
        split = 'val' if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_edu_{split}_{shard_index:05d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])
