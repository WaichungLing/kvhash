import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache

from src.args import parse_args
from src.replace_llama import convert_llama_with_kv_hash
from src.kvhash import KVHashCache
from config import tokens

def main():
    torch.manual_seed(42)

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

    if args.enable_kvhash:
        convert_llama_with_kv_hash(model)
        print("[KVHash] -- replacing llama attention wit KVHash")
    model.to(args.device)

    print("Loading everything done")

    past_key_value = DynamicCache().to(args.device)
    if args.enable_kvhash:
        past_key_value = KVHashCache(
            config,
            num_planes=args.num_planes    
        ).to(args.device)

    input_text = "Introduce the llama 3.1 model"
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
