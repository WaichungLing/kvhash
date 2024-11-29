import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache

from src.args import parse_args
from src.replace_llama import convert_llama_with_kv_hash
from src.kvhash import KVHashCache
from config import tokens
from datasets import load_dataset

LONGBENCH_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main():
    seed_everything(42)

    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir = args.cache_dir,
        token=tokens.HF_TOKEN
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading config...")
    config = AutoConfig.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        token=tokens.HF_TOKEN
    )
    if args.enable_kvhash:
        config.enable_kvhash = args.enable_kvhash
        config.min_eviction_seqlen = args.min_eviction_seqlen
    print(config)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        config=config,
        cache_dir = args.cache_dir,
        attn_implementation="eager",
        token=tokens.HF_TOKEN
    )

    if args.enable_kvhash:
        convert_llama_with_kv_hash(model)
        print("[KVHash] -- replacing llama attention wit KVHash")
    model.eval().to(args.device)

    print("Loading everything done")

    past_key_value = DynamicCache().to(args.device)
    if args.enable_kvhash:
        past_key_value = KVHashCache(
            config,
            cache_budget = args.cache_budget,
            sink_protect_tokens = args.sink_protect_tokens,
            recent_protect_budget = args.recent_protect_budget,
            num_planes=args.num_planes
        ).to(args.device)

    # Prepare dataset
    if args.task == "all":
        datasets = LONGBENCH_TASKS
    else:
        if args.task in LONGBENCH_TASKS:
            datasets = [args.task]
    
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    if not os.path.exists("pred"):
        os.makedirs("pred")

    for dataset in datasets:
        print("Prepareing dataset {dataset}")
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        if not os.path.exists(f"pred/{args.model_name}"):
            os.makedirs(f"pred/{args.model_name}")
        out_path = f"pred/{args.model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]


    input_text = "Compare the Llama model and GPT model"
    max_length = 50
    inputs = tokenizer(
        input_text, 
        return_tensors="pt",
    ).to(args.device)

    # print(f"\nInput Text: {input_text} -- token: {inputs.input_ids.shape[1]}")
    
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_length,
        attention_mask=inputs.attention_mask,
        use_cache=True,
        past_key_values = past_key_value
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the result
    print(f"\nInput Text: {input_text} -- token: {inputs.input_ids.shape[1]}")
    print("Generated Text:", generated_text)


if __name__ == "__main__":
    main()
