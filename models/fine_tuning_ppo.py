import os

# clean Pyke temporary library
complied_krb_dir = './models/compiled_krb'
if os.path.exists(complied_krb_dir):
    os.system(f'rm -rf {complied_krb_dir}')

complied_krb_dir = './compiled_krb'
if os.path.exists(complied_krb_dir):
    os.system(f'rm -rf {complied_krb_dir}')

from datasets import Dataset
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from awq import AutoAWQForCausalLM
from transformers import TrainingArguments, Trainer, Adafactor
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
import re
import json
import argparse
import time
from tqdm import tqdm
import models.logic_inference as logic_inference
from models.utils import print_gpu_utilization

def tokenize_function(examples):
    results = tokenizer(examples['query'])
    return results


def load_prompt_templates(dataset):
    prompt_file = './models/prompts/{}.txt'.format(dataset)
    with open(prompt_file, 'r') as f:
        prompt_template = f.read()
        f.close()
    return prompt_template

def get_prompt(x):
    problem = x['context']
    question = x['question'].strip()
    full_prompt = template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)

    if 'options' in x.keys():
        choices_str = '\n'.join([f'({choice.strip()}' for choice in x['options']]).strip()
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)

    return full_prompt

def prepare_data(split, dataset_name):
        
    result_file = os.path.join('./data/',dataset_name, split+".json")
    with open(result_file, 'r') as f:
        all_samples = json.load(f)
        f.close()

    df=pd.DataFrame(all_samples)
    df["query"] = df.apply(lambda x: get_prompt(x), axis=1)
    return df

def tokenize_dataset(df, answer_map):
    df["label"] = df.apply(lambda x: answer_map[x["answer"]], axis=1)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(df.columns.drop("label"))
    return tokenized_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="ProntoQA")
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--train_model_name', type=str, default="trl-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument('--model_path', type=str, default='/mnt/evafs/groups/luckner-lab/models/')
    parser.add_argument('--use_fine_tuned', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default="no")
    parser.add_argument('--demonstration_path', type=str, default='./models/prompts')
    parser.add_argument('--result_path', type=str, default="/mnt/evafs/groups/luckner-lab/models/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mini_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--use_ada', type=int, default=0)
    parser.add_argument('--load_in_8bit', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    args, unknown = parser.parse_known_args()
    return args



if __name__ == "__main__":

    overall_start = time.time()
    args = parse_args()
    answer_map={"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7, None: -1}

    model_name = args.train_model_name
    if args.use_fine_tuned == 1:
        model_name = model_name.replace("/","-")

    if args.use_fine_tuned == 1:
        model_name_full = args.model_path + args.train_model_name
    else:
        model_name_full = args.train_model_name

    batch_size = args.batch_size

    # prepare logic engine
    args.model_name = args.train_model_name
    args.save_path = "./outputs/logic_inference"
    args.backup_strategy = 'Random'
    args.backup_LLM_result_path = ''
    args.refiment = -1
    args.mode = "CoT"
    logic_engine = logic_inference.LogicInferenceEngine(args)

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_full, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template = load_prompt_templates(args.dataset_name)
    df_merged = prepare_data(args.split, args.dataset_name)
    tokenized_dataset = tokenize_dataset(df_merged, answer_map)
    print(len(tokenized_dataset))
    # tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    # print(len(tokenized_dataset['train']))
    # print(len(tokenized_dataset['test']))

    if args.resume_from_checkpoint == "resume":
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False

    config = PPOConfig(
    model_name=args.train_model_name,
    project_kwargs={"logging_dir": os.path.join(args.result_path, model_name, args.dataset_name, "ppo", 'logs')},
    log_with='tensorboard',
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    ppo_epochs=args.ppo_epochs,
    kl_penalty = "abs",
    adap_kl_ctrl = False,
    init_kl_coef = 0.1,
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

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name_full,
    quantization_config = quantization_config,
    device_map="balanced",
    torch_dtype="auto",
    peft_config=lora_config)

    model.config.use_cache = False

    optimizer = None
    if args.use_ada:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )

    generation_kwargs = {
        "temperature": 1,
        "min_new_tokens": 10,
        "top_k": 0.0,
        "top_p": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "batch_size": batch_size,
        "max_new_tokens": 1024,
        "tokenizer":tokenizer,
        "stop_strings": ["------","-----","----"]
        }

    data_collator = DataCollatorWithPadding(tokenizer)

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator = data_collator,
        optimizer = optimizer,
    )

    def reward_model(results, labels, answer_map):
        return [(answer_map[res["predicted_answer"]] == lab)*1.0 + (answer_map[res["predicted_answer"]] != -1)*0.1  for res, lab in zip(results, labels)]

    print_gpu_utilization()

    epochs = args.epochs
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader): 
            query_tensors = list(batch["input_ids"])
        
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
            #### Compute reward score
            results = logic_engine.inference_on_text(batch["response"])
            rewards = reward_model(results, batch["labels"], answer_map)

            # rewards = [reward.to("cuda:0") for reward in rewards]
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            mean_reward = torch.mean(torch.stack(rewards))
            print(f"Mean reward in batch: {mean_reward}")

    print_gpu_utilization()
    ppo_trainer.save_pretrained(os.path.join(args.result_path, model_name, args.dataset_name,"ppo","last"))

    print(f"Total time: {time.time() - overall_start:.2f} secs")

    print_gpu_utilization()
