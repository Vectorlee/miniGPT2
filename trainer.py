import os
import torch
import math
import time
from dataloader import DataLoaderLite
from model import GPT, GPTConfig
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Set up DDP (Distributed Data Parallel)
# if we set the RANK environment variable, then it is a distributed training job
# torchrun creates these environment variables
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    # now we need cuda
    assert torch.cuda.is_available(), "Need CUDA for DDP"
    init_process_group(backend="nccl")

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # Vanilla
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # if iter > max_steps, use the costant min learing rate
    if it > max_steps:
        return min_lr

    # cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


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
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
    return optimizer

# gradient accumulate
# following the GPT-3 paper 0.5M batch size setting
total_batch_size = 524288 # 2**19, in number of tokens, nice number
B = 16 # micro batch size
T = 1024 # sequence length
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# divisible
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')


# Change the original 50257 token count into a nice number
# Nice numbers are the numbers that can be divided by large power of 2 numbers
config = GPTConfig(vocab_size=50304)
model = GPT(config)

#optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4)
if ddp:
    model = DDP(model, device_id=[ddp_local_rank])

# use the micro batch size in the data loader
dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

for step in range(max_steps):
    t0 = time.time()

    # zero grad
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = dataloader.next_batch()
        x = x.to(device)
        y = y.to(device)

        # mixed precision training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            not_last_step = micro_step < grad_accum_steps - 1

            if ddp and not_last_step:
                with model.no_sync():
                    # no_sync context requires the forward pass also resides in the context
                    logits, loss = model(x, y)

                    # the micro batch lost the normalizer
                    # so we divide the loss by the number of micro step count
                    loss = loss / grad_accum_steps
                    loss_accum += loss.detach()
                    # because we didn't zero the grad, the gradient will accumulate
                    loss.backward()
            else:
                # without the no_sync context manager here
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                # For the ddp case, the gradients will be synchronized across devices
                loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient Clipping
    # Before the optimizer.step, but after the loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # set the cosine decay learing rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    t1 = time.time()

    dt = (t1 - t0)
    token_processed = B * T * grad_accum_steps * ddp_world_size
    token_per_sec = token_processed / dt
    # the item() function ship the tensor back from gpu to cpu
    if master_process:
        print(f"step {step}, loss: {loss_accum.item()}, dt: {dt * 1000:.2f}ms, tok/sec: {token_per_sec}, norm: {norm:.4f}, lr: {lr}")

if ddp:
    destroy_process_group()