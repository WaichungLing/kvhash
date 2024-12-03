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
    parser.add_argument("--cache_budget", type=float, default=0.6, help="kv cache budget")
    parser.add_argument("--sink_protect_tokens", type=int, default=128, help="number of tokens to be protect at the head") # put back 256
    parser.add_argument("--recent_protect_budget", type=int, default=0.01, help="ration of tokens to be protect at the end") # put back 0.01
    parser.add_argument("--min_eviction_seqlen", type=int, default=5, help="sequence length that starts eviction") # put back 2048
    parser.add_argument("--num_planes", type=int, default=4, help="number of division plane")
    return parser.parse_args()