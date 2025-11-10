from datasets import Dataset
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from trl import KTOConfig, KTOTrainer
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import pandas as pd
import re
import json
import argparse
import time
from sklearn.model_selection import train_test_split
from models.utils import print_gpu_utilization

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    
    if answer_str is None:
        return ''
    answer_str=answer_str.strip()

    indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is']

    for indicator in indicators:
        if answer_str.find(indicator)>=0:
            answer_str = answer_str.split(indicator)[1].strip()
    
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')

    if answer_str.startswith(':'):
       return answer_str.replace(':', '').replace('.', '').strip()
    return ''

def tokenize_function(examples):
    results = tokenizer(examples['query'])
    return results


def load_prompt_templates(dataset):
    prompt_file = './models/prompts/{}.txt'.format(dataset)
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
        f.close()
    return prompt_template


def get_prompt_answer(x):
    problem = x['context']
    question = x['question'].strip()
    full_prompt = template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)

    if 'options' in x.keys():
        choices_str = '\n'.join([f'({choice.strip()}' for choice in x['options']]).strip()
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        
    answer = x['raw_logic_programs']
    text = full_prompt + answer
    return text

def get_prompt(x):
    problem = x['context']
    question = x['question'].strip()
    full_prompt = template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)

    if 'options' in x.keys():
        choices_str = '\n'.join([f'({choice.strip()}' for choice in x['options']]).strip()
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)

    return full_prompt

def get_program(x):
    if 'program' not in x.index:
        program = x['raw_logic_programs'][0]
    else:
        if x['program'] is None:
            program = x['raw_logic_programs'][0]
        else:
            program = x['program']

    program = program.rsplit("------", 1)[0]+"------"
    return program

def prepare_data(model_name, split, logic_inference, logic_programs, dataset_name):
    df_logic=[]
    for file in os.listdir(logic_inference):
        
        if model_name not in file:
            continue

        if split not in file:
            continue

        if 'self-refine' in file:
            refine, dataset, split, model, backup = file.split("_")
            refine = int(refine.split('-')[-1])
        else:
            dataset, split, model, backup = file.split("_")
            refine = 0
        
        result_file = os.path.join(logic_inference,file)
        with open(result_file, 'r') as f:
            all_samples = json.load(f)
            f.close()

        df=pd.DataFrame(all_samples)
        if len(df) == 0:
            continue
        df["clean_answer"] = df.predicted_answer.apply(get_choice)
        df = df.drop(columns=["question", "predicted_answer","context"])
        df["answer"] = df.answer.str.replace('(', '').replace(')', '')
        df["is_correct"] = df.answer == df.clean_answer
        df["is_empty"] = df.clean_answer == ''
        df["mode"] = 'Logic'
        df["dataset"] = dataset
        df["split"] = split
        df["model"] = model
        df["refiment"] = refine
        df_logic.append(df)

    df_programs=[]
    for file in os.listdir(logic_programs):
        
        if model_name not in file:
            continue

        if split not in file:
            continue

        if 'self-refine' in file:
            refine, dataset, split, model = file.split("_")
            refine = int(refine.split('-')[-1])
        else:
            dataset, split, model = file.split("_")
            refine = 0
        
        result_file = os.path.join(logic_programs,file)
        with open(result_file, 'r') as f:
            all_samples = json.load(f)
            f.close()

        df=pd.DataFrame(all_samples)
        if len(df) == 0:
            continue
        df["dataset"] = dataset
        df["split"] = split
        df["model"] = model[:-5]
        df["refiment"] = refine
        df_programs.append(df)

    df_programs2=pd.concat(df_programs)
    df_logic2=pd.concat(df_logic)
    # df_logic3 = df_logic2.loc[(df_logic2.flag == "success") & (df_logic2.is_correct) & (df_logic2.dataset==dataset_name)]
    df_logic3 = df_logic2.loc[(df_logic2.dataset==dataset_name)]
    df_merged = pd.merge(df_logic3, df_programs2, how="inner", on=["id","model","split","refiment"])
    # to handle legacy save setting
    df_merged["raw_logic_programs"] = df_merged.apply(lambda x: get_program(x), axis=1)

    df_merged["query"] = df_merged.apply(lambda x: get_prompt(x), axis=1)
    df_merged["label"] = (df_merged.flag == "success") & (df_merged.is_correct)
    return df_merged

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="ProntoQA")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--train_model_name', type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument('--benchmark_model_name', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', type=str, default="no")
    parser.add_argument('--logic_inference', type=str, default='./outputs/logic_inference')
    parser.add_argument('--logic_programs', type=str, default='./outputs/logic_programs')
    parser.add_argument('--demonstration_path', type=str, default='./models/prompts')
    parser.add_argument('--result_path', type=str, default="/mnt/evafs/groups/luckner-lab/models/")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--load_in_8bit', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=16)
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":

    overall_start = time.time()

    args = parse_args()
    model_name = args.train_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template = load_prompt_templates(args.dataset_name)
    df_merged = prepare_data(args.benchmark_model_name, args.split, args.logic_inference, args.logic_programs, args.dataset_name)    
    df_merged = df_merged.rename(columns={"query": "prompt", "raw_logic_programs": "completion"})
    df_merged = df_merged.drop(columns = df_merged.columns.drop(["prompt", "completion", "label"]))
    df_merged = df_merged.drop_duplicates()

    if args.resume_from_checkpoint == "resume":
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False

    training_args = KTOConfig(output_dir=os.path.join(args.result_path, model_name, args.dataset_name, "kto"),
                                    report_to="tensorboard",
                                    load_best_model_at_end = True,
                                    num_train_epochs = args.epochs,
                                    logging_steps = 10,
                                    save_total_limit = 2,
                                    eval_strategy="epoch", save_strategy = "epoch",
                                    save_safetensors=False,
                                    fp16=False,
                                    auto_find_batch_size=True,
                                    desirable_weight = 1/(df_merged.label.mean()+0.0001),
                                    )


    lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_r,
    lora_dropout=0.05,
    bias="none",
    target_modules = "all-linear",
    task_type="CAUSAL_LM",
    )

    quantization_config =None
    if bool(args.load_in_8bit):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = quantization_config,
    device_map="balanced",
    torch_dtype="auto",)

    model.config.use_cache = False
    model.generation_config.pad_token_id=tokenizer.pad_token_id

    data = df_merged.to_dict('records')
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)

    trainer = KTOTrainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        max_seq_length = 2048,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.result_path, model_name, args.dataset_name,"kto","best"))

    print(f"Total time: {time.time() - overall_start:.2f} secs")
    print_gpu_utilization()