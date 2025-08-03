import os
import subprocess
from huggingface_hub import snapshot_download

directory_list = [
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
    "red-team-attempts"
]

snapshot_download(repo_id="Anthropic/hh-rlhf", repo_type="dataset", local_dir="./", allow_patterns="*.jsonl.gz")

for dir in directory_list:
    file_list = list(filter(lambda s: s.endswith(".jsonl.gz"), os.listdir(dir)))
    file_list = list(map(lambda s: os.path.join(dir, s), file_list))
    for filename in file_list:
        subprocess.run(["gzip", "-d", filename])