import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run KVHash Cache Compression")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="The Hugging Face model name to load.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cuda or cpu).")
    parser.add_argument("--cache_dir", type=str, default='./model',
                        help="Cache directory for model related configs, weights etc")
    parser.add_argument("--result_dir", type=str, default='./result',
                        help="Result directory that holds experiment output")
    parser.add_argument("--enable_kvhash", type=bool, default=True,
                        help="kvhash ratio")
    parser.add_argument("--hash_budget", type=float, default=0.5, help="kv hash budget")
    parser.add_argument("--num_planes", type=int, default=4, help="number of division plane")
    return parser.parse_args()