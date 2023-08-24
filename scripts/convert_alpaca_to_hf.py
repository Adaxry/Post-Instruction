##############################
# Function: convert alpaca data format to training data format: {'text': "", 'prefix': ""}
# Author: Wenxiang Jiao
# Last modified: 2023/04/15
##############################

import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
import csv, json



# Instrauct language
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"},
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese"},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語"},
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文"},
}

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
emphasize_start_tag = "<p>"
emphasize_end_tag = "</p>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_input_above": (
        "Below is an input sentence as context, paired with an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "double_prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{pre_instruction}\n\n### Input:\n{input}\n\n### Instruction:\n{post_instruction}\n\n### Response:"
    ), 
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_prompt(path, above_prompt, double_prompt, replace_fake_breaker, emphasize_source, no_alpaca):
    list_data_dict = read_json(path)
    prompt_input, prompt_input_above, double_prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"],  PROMPT_DICT["prompt_input_above"], PROMPT_DICT["double_prompt_input"],  PROMPT_DICT["prompt_no_input"]

    #sources = [
    #    prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    #    for example in list_data_dict
    #]
    sources = []
    for example in list_data_dict:
        if example.get("input", "") != "":
            if emphasize_source:
                example["input"] = emphasize_start_tag + example["input"] + emphasize_end_tag
            if no_alpaca:
                prompt_input = "{instruction}\n{input}\n"
                prompt_input_above = "{input}\n{instruction}\n"

            if double_prompt:
                return_prompt = double_prompt_input.format_map(example) 
            elif above_prompt:
                return_prompt = prompt_input_above.format_map(example)
            else:
                return_prompt = prompt_input.format_map(example)
        else:
             return_prompt = prompt_no_input.format_map(example)
        if replace_fake_breaker:
            return_prompt = return_prompt.replace("\\n", "\n")
        sources.append(return_prompt)         

    # add by yijin
    #outputs = example['output'] if not replace_fake_breaker else example['output'].replace("\\n", "\n")
    #targets = [f"{outputs}{DEFAULT_EOS_TOKEN}" if not outputs.endswith(DEFAULT_EOS_TOKEN) else outputs for example in list_data_dict]
    targets = []
    for example in list_data_dict:
        outputs = example['output'] if not replace_fake_breaker else example['output'].replace("\\n", "\n")
        targets.append(f"{outputs}{DEFAULT_EOS_TOKEN}" if not outputs.endswith(DEFAULT_EOS_TOKEN) else outputs)

    return targets, sources


def write_json(in_file, out_file, above_prompt, double_prompt, replace_fake_breaker, emphasize_source, no_alpaca):
    prompts, sources = create_prompt(in_file, above_prompt, double_prompt, replace_fake_breaker, emphasize_source, no_alpaca)
    with open(out_file, 'w', encoding='utf-8') as fo:
        for p,s in zip(prompts, sources):
            jsoned = json.dumps({'text': p, 'prefix': s}, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')


if __name__ == "__main__":
    """
    python3 ../create_prompt.py --in-file ./alpaca_data.json --out-file data_alp_hf.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--out-file','-o', type=str, required=True, help='output file')
    parser.add_argument('--above-prompt', action='store_true', default=False, help='output file')
    parser.add_argument('--double-prompt', action='store_true', default=False, help='output file')
    parser.add_argument('--emphasize-source', action='store_true', default=False, help='output file')
    parser.add_argument('--replace-fake-breaker', action='store_true', default=False, help='\\n -> \n')
    parser.add_argument('--no_alpaca', action='store_true', default=False, help='no ### instruct ### input format')
    args = parser.parse_args()
    in_file = args.in_file
    out_file = args.out_file
    double_prompt = args.double_prompt
    above_prompt = args.above_prompt
    replace_fake_breaker = args.replace_fake_breaker
    emphasize_source = args.emphasize_source
    no_alpaca = args.no_alpaca

    # Start
    write_json(in_file, out_file, above_prompt, double_prompt, replace_fake_breaker, emphasize_source, no_alpaca)



