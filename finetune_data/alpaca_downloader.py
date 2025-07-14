import os
import subprocess
from huggingface_hub import snapshot_download

data_directory = "data/"

snapshot_download(repo_id="tatsu-lab/alpaca", repo_type="dataset", local_dir="./", allow_patterns="*.parquet")

file_list = os.listdir(data_directory)
assert(len(file_list) == 1)
filename = f"{data_directory}{file_list[0]}"

print(f"Download alpaca data: {filename}")

subprocess.run(["mv", filename, "./alpaca.parquet"])