##############################
# Function: inference
# Author: Wenxiang Jiao
# Last modified: 2023/04/06
##############################

import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import numpy as np


# Instruction language, default: 'en'
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"},
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese", 'fr': "French", "uk": "Ukrainian", "cs": "Czech", "hr": "Croatian", "ru": "Russian", "liv": "Livonian", "ja": "Japanese", "sah": "Yakut"},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語"},
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文"},
}

# Special tokens in llama
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
    "prompt_no_alpaca": (
        "{instruction}\n{input}\n"
    ),
    "prompt_no_alpaca_above": (
        "{input}\n{instruction}\n"
    ),
}


# Read task instruction, fill in languages
def read_instruct(path, src, tgt, lang_ins="en"):
    source, target = lang_instruction[lang_ins][src], lang_instruction[lang_ins][tgt]
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip().replace("[SRC]", source).replace("[TGT]", target)
            ins_list.append(line)
    return ins_list


# Read input data for inference
def read_input(path, emphasize_source):
    with open(path, 'r', encoding='utf-8') as f:
        input_data = f.readlines()
        if emphasize_source:
            input_data = [emphasize_start_tag + line.strip() + emphasize_end_tag  for line in input_data]
    return input_data


# Assembly instruction and input data, handle hints
def create_prompt(instruct, input_data, template="prompt_no_input", post_instruct=None, append_bos=False):
    if "###" in instruct:
        instruct, input_suffix = instruct.split("###")
        hint = "\n\n### Hint: {}".format(input_suffix)
    else:
        instruct =  instruct
        hint = ""
    prompt_input = PROMPT_DICT[template] if not append_bos else PROMPT_DICT[template] + tokenizer.bos_token 
    if template == "prompt_input":
        list_data_dict = [{"instruction": instruct, "input": p.strip() + hint} for p in input_data]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    elif template == "prompt_no_input":
        list_data_dict = [{"instruction": "\n\n".join([instruct, p.strip() + hint]).strip(), "input": ""} for p in input_data]
        #prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ]
    elif template == "prompt_input_above":
        #list_data_dict = [{"instruction": instruct, "input": p.strip() + hint} for p in input_data]
        list_data_dict = [{"instruction": instruct + hint, "input": p.strip()} for p in input_data]
        #prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ] 
    elif template == "double_prompt_input":
        assert post_instruct is not None
        list_data_dict = [{"pre_instruction": instruct, "post_instruction": post_instruct + hint, "input": p.strip()} for p in input_data]
        #prompt_input = PROMPT_DICT[template]
        sources = [ prompt_input.format_map(example) for example in list_data_dict ] 
    elif template == "prompt_no_alpaca" or template == "prompt_no_alpaca_above":
        list_data_dict = [{"instruction": instruct + hint, "input": p.strip()} for p in input_data]
        sources = [prompt_input.format_map(example) for example in list_data_dict ]  
    return sources


# Post-process the output, extract translations
def post_process(text):
    if "### Response:" in text:
        text = text.split("### Response:")[1].strip()
    else:
        text = text.split("\n")[-1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, required=True, help='model name in the hub or local path')
    parser.add_argument('--inst-file', '-ins', type=str, default=None, help='instruction file')
    parser.add_argument('--post-inst-file', type=str, default=None, help='post instruction file for testing double prompt mode')
    parser.add_argument('--input-file','-i', type=str, required=True, help='input file')
    parser.add_argument('--output-file','-o', type=str, required=True, help='output file')
    parser.add_argument('--lang-pair', '-lp', type=str, default='zh-en', help='language pair: zh-en, en-de')
    parser.add_argument('--search-algorithm', '-sa', type=str, default='beam', help='search algorithms: sample, beam')
    parser.add_argument('--batch', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--beam-size', type=int, default=4, help='beam size')
    parser.add_argument('--no-repeat-ngram-size', '-no', type=int, default=0, help='no_repeat_ngram_size')
    parser.add_argument('--template', '-tp', type=str, default="prompt_input", help='0: prompt_no_input, 1: prompt_input, 3: prompt_input_above')
    parser.add_argument('--temperature', '-t', type=float, default=0.1, help='temperature: 0.7 for text generation')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='temperature: 0.7 for text generation')
    parser.add_argument('--length-penalty', type=float, default=1.0, help='temperature: 0.7 for text generation')
    parser.add_argument('--emphasize-source', action='store_true', default=False, help='add <p> </p> for source')
    parser.add_argument('--append-bos', action='store_true', default=False, help='add bos before generation')
    parser.add_argument('--padding-side', type=str, default=None, help='defaultly use the same padding-side as pre-training, else use the specified padding side.')

    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    inst_file = args.inst_file
    post_inst_file = args.post_inst_file
    input_file = args.input_file
    output_file = args.output_file
    lang_pair = args.lang_pair
    search = args.search_algorithm
    batch = args.batch
    beam_size = args.beam_size
    temperature = args.temperature
    #temp = args.template
    template = args.template 
    no_repeat_ngram_size = args.no_repeat_ngram_size
    emphasize_source = args.emphasize_source
    append_bos = args.append_bos
    padding_side = args.padding_side
    max_new_tokens = args.max_new_tokens
    length_penalty = args.length_penalty
    #"prompt_input" if temp > 0 else "prompt_no_input"
    # Load checkpoints
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    print(model.hf_device_map)
    # bloom uses only fast tokenize
    #to_use_fast = False
    to_use_fast = True # TODO
    if "bloom" in model_name_or_path:
        to_use_fast = True
    if "llama" in model_name_or_path:
        to_use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    print("debug: tokenizer.padding_side=", tokenizer.padding_side)
    #tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = GenerationConfig(temperature=temperature,
                                  top_p=0.9,
                                  do_sample=True,
                                  num_beams=1,
                                  max_new_tokens=max_new_tokens,
                                  length_penalty=length_penalty,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token=tokenizer.pad_token_id,
                                  )

    if search == "beam":
        gen_config = GenerationConfig(temperature=temperature,
                                      top_p=0.9,
                                      num_beams=beam_size,
                                      max_new_tokens=max_new_tokens,
                                      length_penalty=length_penalty,
                                      no_repeat_ngram_size=no_repeat_ngram_size,
                                      eos_token_id=tokenizer.eos_token_id,
                                      pad_token=tokenizer.pad_token_id,
                                      )

    # Prepare input data
    srcl, tgtl = lang_pair.split('-')
    if inst_file is not None:
        instructs = read_instruct(inst_file, srcl, tgtl)
        instruct = instructs[0] if len(instructs) > 0 else ""
    else: # In case instruction file is missing, then use input as instruction
        instruct = ""
        template = "prompt_no_input"
    input_data = read_input(input_file, emphasize_source)
    post_instruct = None
    empty_detection = post_inst_file.split("/")[-1].strip(".txt") if post_inst_file is not None else None
    if empty_detection is not None and empty_detection != "":
    #if post_inst_file.split("/")[-1].strip(".txt") is not None:
        post_instructs = read_instruct(post_inst_file, srcl, tgtl)
        post_instruct = post_instructs[0] if len(post_instructs) > 0 else ""

    prompt = create_prompt(instruct, input_data, template, post_instruct, append_bos)
    #if append_bos:
    #    append_bos_prompt = []
    #    for p in prompt:
    #        #p = p + tokenizer.bos_token
    #        p = p + DEFAULT_BOS_TOKEN
    #        append_bos_prompt.append(p)
    #prompt = append_bos_prompt
    print ("debug: \n", prompt[0:2])

    # Generate
    torch.manual_seed(0)
    with open(output_file, 'w', encoding='utf-8') as fo,open(output_file+".hyp", 'w', encoding='utf-8') as fo2:
        for i in range(0, len(prompt), batch):
            p = prompt[i:i+batch]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            input_ids = input_ids[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else input_ids
            attn_mask = attn_mask[:, :-1] if input_ids[0, -1] == tokenizer.eos_token_id else attn_mask
            with torch.no_grad():
                #generated_ids = model.generate(inputs=input_ids,attention_mask=attn_mask, generation_config=gen_config)
                generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config, pad_token_id=tokenizer.eos_token_id)
            decoded_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for dec in decoded_tokens:
                print(dec, file=fo, flush=True)
                print(post_process(dec), file=fo2, flush=True)


