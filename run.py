import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.args import parse_args
from src.replace_llama import convert_llama_with_kv_hash
from src.kvhash import KVHashCache
from config import tokens

def main():
    # os.environ["HUGGINGFACE_TOKEN"] = tokens.HF_TOKEN

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
        config.hash_budget = args.hash_budget
    print(config)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        config=config,
        cache_dir = args.cache_dir,
        attn_implementation="eager",
        token=tokens.HF_TOKEN
    )

    use_cache = True
    if args.enable_kvhash:
        convert_llama_with_kv_hash(model)
        use_cache = False
        print("[KVHash] -- replacing llama attention wit KVHash")
    model.to(args.device)

    print("Loading everything done")

    # 在Generate时传入max_length,和cache，加入cache的reset函数

    # Encode input and generate output
    input_text = "Compare GPT with LLama briefly"
    max_length = 50
    inputs = tokenizer(
        input_text, 
        return_tensors="pt",
    ).to(args.device)
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_length,
        attention_mask=inputs.attention_mask,
        use_cache=use_cache
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the result
    print("\nInput Text:", input_text)
    print("Generated Text:", generated_text)


if __name__ == "__main__":
    main()
