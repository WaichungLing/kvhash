import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run KVHash Cache Compression")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The Hugging Face model name to load.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu).")
    parser.add_argument("--cache_dir", type=str, default="model", help="Cache directory for model related configs, weights etc")
    parser.add_argument("--data_dir", type=str, default="data", help="Cache directory for dataset etc")
    parser.add_argument("--pred_dir", type=str, default="pred", help="Result directory that holds experiment output")
    # unicache
    parser.add_argument("--cache_budget", type=float, default=0.2, help="global kv cache budget")
    parser.add_argument("--recent_protect_budget", type=int, default=128, help="number of tokens to be protect at the end")  
    parser.add_argument("--proxy_total", type=int, default=64, help="number of query proxy tokens")
    parser.add_argument("--proxy_latest", type=int, default=16, help="number of latest window of proxy")
    parser.add_argument("--top_rank", type=int, default=20, help="top rank for PCA reduction")
    parser.add_argument("--n_recursive", type=int, default=1, help="number of recursion for elbow point allocation")
    # tasks:
    parser.add_argument("--task", type=str, default="narrativeqa", help="evaluation task")
    return parser.parse_args()
