import torch
import os
import json
import warnings
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache

from src.args import parse_args
from src.replace_llama import convert_llama_with_kv_hash
from src.kvhash import KVHashCache
from config import tokens
from datasets import load_dataset

LONGBENCH_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

MAX_CONTEXT = 4*1024

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import get_conversation_template
    #     conv = get_conversation_template("vicuna")
    #     conv.append_message(conv.roles[0], prompt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, past_key_value, data, max_gen, prompt_format, dataset, device, model_name, out_path):
    file_name = f'pred/sparsity_h2o/{dataset}_sparsity.jsonl'
    idx = 0
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > MAX_CONTEXT:
            half = int(MAX_CONTEXT/2)
            # prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            prompt = tokenizer.decode(tokenized_prompt[-MAX_CONTEXT:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                min_length=context_length+1,
                use_cache=True,
                past_key_values = past_key_value,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            if past_key_value == None:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    use_cache=True
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    use_cache=True,
                    past_key_values = past_key_value
                )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        if past_key_value is not None:
            print(f"===== done. KV {past_key_value.key_cache[0].shape[-2]}/{past_key_value._seen_tokens} ====")

            # ======= unicache expr ========
            with open(file_name, "a", encoding="utf-8") as f:
                json.dump({'h2o': past_key_value.attn_sparsity 
                           # 'tail': past_key_value.attn_sparstiy_hash
                        #    'pca_qk': past_key_value.attn_sparsity_pca_qk,
                        #    'pca_qq': past_key_value.attn_sparsity_pca_qq
                        }, f, ensure_ascii=False)
                f.write('\n')
            # ==============================

            # ======== get qka npy =========
            # qqq = past_key_value.query_cache
            # qqq_cpu = [layer.to(dtype=torch.float32).cpu().numpy() for layer in qqq]
            # np.save("query_state.npy", np.array(qqq_cpu, dtype=object))

            # aaa = past_key_value.attn_score
            # aaa_cpu = [layer.to(dtype=torch.float32).cpu().numpy() for layer in aaa]
            # np.save("attn_score.npy", np.array(aaa_cpu, dtype=object))

            # prefill_len = qqq_cpu[0].shape[2]

            # kkk = past_key_value.key_cache
            # kkk_cpu = [layer.to(dtype=torch.float32).cpu().numpy()[:,:,:prefill_len,:] for layer in kkk]
            # np.save("key_state.npy", np.array(kkk_cpu, dtype=object))

            # print(f"successfully saved. query {qqq_cpu[0].shape}, key {kkk_cpu[0].shape}")
            # ==============================

            past_key_value.clear()

        # if idx == 0:
        #     break
        idx += 1
        # with open(out_path, "a", encoding="utf-8") as f:
        #     json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
        #     f.write('\n')

    # dist.destroy_process_group()

def main():
    seed_everything(42)

    args = parse_args()

    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

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
    config.pad_token_id = config.eos_token_id[0]
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

    past_key_value = None
    if args.enable_kvhash:
        past_key_value = KVHashCache(
            config,
            cache_budget = args.cache_budget,
            sink_protect_tokens = args.sink_protect_tokens,
            recent_protect_budget = args.recent_protect_budget,
        ).to(args.device)

    # Prepare dataset
    if args.task == "all":
        datasets = LONGBENCH_TASKS
    else:
        if args.task in LONGBENCH_TASKS:
            datasets = [args.task]
    
    dataset2prompt = json.load(open("longbench/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench/dataset2maxlen.json", "r"))

    for dataset in datasets:
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        if not os.path.exists(f"{args.pred_dir}/{args.model_name}"):
            os.makedirs(f"{args.pred_dir}/{args.model_name}")
        out_path = f"{args.pred_dir}/{args.model_name}/{dataset}.jsonl"

        print(f"Prepareing dataset {dataset}, max_gen = {max_gen}, out_path = {out_path}")
        data = load_dataset('THUDM/LongBench', dataset, split='test',cache_dir=args.data_dir)
        print("Load dataset done")

        get_pred(model, tokenizer, past_key_value, data, max_gen, prompt_format, dataset, args.device, args.model_name, out_path)


if __name__ == "__main__":
    main()
