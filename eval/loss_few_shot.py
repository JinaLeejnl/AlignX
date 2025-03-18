from transformers import AutoConfig, AutoTokenizer
import torch
import numpy as np
from typing import List
import multiprocessing
import datasets
import queue
import time
import os
import pickle
import json
import logging
import re
from tqdm import tqdm
import sys
import os
import random
from tqdm import tqdm
import copy
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed


torch.set_printoptions(profile="full")
torch.multiprocessing.set_start_method('spawn',force=True)
num_workers = 8
gpu_num = 8
max_length = 8192
max_new_tokens = 512
IGNORE_INDEX = -100

def read_data(input_file, to_tokenize_queue):
    ds = []
    with open(input_file, "r") as f:
        ds = json.load(f)

    print(len(ds))

    for idx, data in tqdm(enumerate(ds), total=len(ds)):
        Like_Dislike = ""
        comments = ""
        persona = ""

        if "Pair-wise Comparative Feedback" in data:
            Like = data["Pair-wise Comparative Feedback"]
            if len(Like) > 0:
                Like_Dislike = f"**This person has chosen or rejected comments on some posts:**\n\n"
                for i, it in enumerate(Like):
                    Like_Dislike = (
                        f"{Like_Dislike}"
                        f"{i+1}. *Post:*\n{it['prompt']}\n\n"
                        f"*Chosen:*\n{it['chosen']}\n\n"
                        f"*Rejected:*\n{it['rejected']}\n\n"
                    )

        if "User-Generated Content" in data:
            comment = data["User-Generated Content"]
            if len(comment) > 0:
                comments = f"**This person has commented on some posts:**\n\n"
                for i, it in enumerate(comment):
                    comments = (
                        f"{comments}"
                        f"{i+1}. *Post:*\n{it['prompt']}\n\n"
                        f"*Comment:*\n{it['comment']}\n\n"
                    )

        if "Demographic Information" in data:
            persona = data["Demographic Information"]
        
        task = f"**Post:**\n{data['prompt']}\n\n"

        sft_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\nGenerate a task-specific response based on user historical behavior and preferences.\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"***Task***\n\n{task}"
            f"***Historical Behavior and User Preferences***\n\n{comments}{Like_Dislike}{persona}\n\n***Response:***\n\n<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
        
        data["format"] = sft_prompt
        data["idx"] = idx

        to_tokenize_queue.put(data)
        
    for i in range(num_workers):
        to_tokenize_queue.put(None)


def tokenize_data(to_tokenize_queue, to_output_queue, rank):
    model_name = 'model/Llama-3.1-8B-Instruct'
    config = AutoConfig.from_pretrained(model_name)
    config.remove_unused_columns = False
    config._attn_implementation = "flash_attention_2"
    config.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16).to(f"cuda:{rank%gpu_num}")
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    
    while True:
        data = to_tokenize_queue.get()
        if data is None:
            break

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = 0  
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left" 

        prompt = data["format"]
        chosen = data["chosen"]
        rejected = data["rejected"]

        examples_inputs = tokenizer(prompt, padding=False, truncation=True, add_special_tokens=False, max_length=max_length-max_new_tokens-1)["input_ids"]
        chosen_inputs = tokenizer(chosen, padding=False, truncation=True, add_special_tokens=False, max_length=max_length-max_new_tokens-1)["input_ids"]
        rejected_inputs = tokenizer(rejected, padding=False, truncation=True, add_special_tokens=False, max_length=max_length-max_new_tokens-1)["input_ids"]
            
        examples_inputs = [tokenizer.bos_token_id] + examples_inputs
        prompt_length = len(examples_inputs)
        en_chosen = examples_inputs + chosen_inputs + [tokenizer.eos_token_id]
        en_rejected = examples_inputs + rejected_inputs + [tokenizer.eos_token_id]

        label_chosen = copy.deepcopy(en_chosen)
        label_chosen[:prompt_length] = [IGNORE_INDEX] * prompt_length

        label_rejected = copy.deepcopy(en_rejected)
        label_rejected[:prompt_length] = [IGNORE_INDEX] * prompt_length

        with torch.no_grad():
            en_chosen = torch.tensor(en_chosen).reshape(1,-1)
            en_rejected = torch.tensor(en_rejected).reshape(1,-1)
            outputs_chosen = model(input_ids = en_chosen.to(f"cuda:{rank%gpu_num}"))
            outputs_rejected = model(input_ids = en_rejected.to(f"cuda:{rank%gpu_num}"))


            ##chosen
            logits = outputs_chosen["logits"] if isinstance(outputs_chosen, dict) else outputs_chosen[0]
            logits = logits[..., :-1, :].contiguous()
            label = torch.tensor(label_chosen).reshape(1,-1)
            label = label[..., 1:].contiguous()
            label = label.to(f"cuda:{rank%gpu_num}")

            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            if label.dim() == log_probs.dim() - 1:
                label = label.unsqueeze(-1)
            
            padding_mask_all = label.eq(IGNORE_INDEX)
            label = torch.clamp(label, min=0)
            nll_loss_all = log_probs.gather(dim=-1, index=label)
            nll_loss_all.masked_fill_(padding_mask_all, 0.0)
            num_active_elements_all = padding_mask_all.numel() - padding_mask_all.long().sum()
            nll_loss_all_chosen = nll_loss_all.sum()


            ##rejected
            logits = outputs_rejected["logits"] if isinstance(outputs_rejected, dict) else outputs_rejected[0]
            logits = logits[..., :-1, :].contiguous()
            label = torch.tensor(label_rejected).reshape(1,-1)
            label = label[..., 1:].contiguous()
            label = label.to(f"cuda:{rank%gpu_num}")

            log_probs = -nn.functional.log_softmax(logits, dim=-1)
            if label.dim() == log_probs.dim() - 1:
                label = label.unsqueeze(-1)
            
            padding_mask_all = label.eq(IGNORE_INDEX)
            label = torch.clamp(label, min=0)
            nll_loss_all = log_probs.gather(dim=-1, index=label)
            nll_loss_all.masked_fill_(padding_mask_all, 0.0)
            num_active_elements_all = padding_mask_all.numel() - padding_mask_all.long().sum()
            nll_loss_all_rejected = nll_loss_all.sum()

            result = { **data,
                    'nll_loss_all_chosen': nll_loss_all_chosen.item(),
                    'nll_loss_all_rejected': nll_loss_all_rejected.item()}
       

        to_output_queue.put(result)

    to_output_queue.put(None)

def output_data(to_output_queue):
    count = 0
    start_time = None
    finish_tag = 0
    
    while True:
        data = to_output_queue.get()
        if start_time is None:
            start_time = time.time()
        if data is None:
            finish_tag += 1
            if finish_tag == num_workers:
                print("End")
                break
            else:
                continue
        else:
            with open('/eval_data/Llama-3.1-8B-Instruct-PRISM.json', 'a') as f:
                try:
                    json.dump(data, f)
                    f.write('\n')
                except:
                    continue
            
            count += 1
            if count % 100 == 0:
                end_time = time.time()
                print(count)
                print(f"Spend:{(end_time-start_time)} s")
        

if __name__ == "__main__":
    import sys

    to_tokenize_queue = multiprocessing.Queue(maxsize=100000)
    to_output_queue = multiprocessing.Queue(maxsize=100000)
    
    reader_process = multiprocessing.Process(target=read_data, args=("/benchmark/PRISM.json", to_tokenize_queue))
    tokenizer_processes = [multiprocessing.Process(target=tokenize_data, args=(to_tokenize_queue, to_output_queue, rank)) for rank in range(num_workers)]
    output_process = multiprocessing.Process(target=output_data, args=(to_output_queue,))
    
    reader_process.start()
    for p in tokenizer_processes:
        p.start()
    output_process.start()

    start_time =  time.time()
    reader_process.join()
    for p in tokenizer_processes:
        p.join()
    output_process.join()
    end_time = time.time()
    print(end_time-start_time)