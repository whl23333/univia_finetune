from safetensors.torch import load_file
import os
data = load_file("/home/yjh/UniVLA/trajectory_embeddings/1bba95d6209c1eb03d87b2bdb9b71fe2420c5413_batch_0.safetensors")
for k, v in data.items():
    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    if len(v.shape) == 1:
        print(k, v)