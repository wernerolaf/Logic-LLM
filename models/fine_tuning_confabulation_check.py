from datasets import Dataset
from transformers import pipeline
import torch
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model, prepare_model_for_kbit_training
import os
import pandas as pd
import numpy as np
import re
import json
import argparse
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from models.utils import print_gpu_utilization
from collections import Counter

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


def get_prompt_no_examples(x):
    problem = x['context']
    question = x['question'].strip()
    template_new = template.split("------")[-1]
    full_prompt = template_new.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)

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

    program = str(program).rsplit("------", 1)[0]+"------"
    return program

def tokenize_function(examples):
    return tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=2018)

def prepare_data(model_name, split, logic_inference, logic_programs, dataset_name):
    df_logic=[]
    for file in os.listdir(logic_inference):
        
        if model_name not in file:
            continue

        if split not in file:
            continue

        if dataset_name not in file:
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

        if dataset_name not in file:
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
    df_logic3 = df_logic2.loc[(df_logic2.flag == "success") & (df_logic2.dataset==dataset_name)]
    df_merged = pd.merge(df_logic3, df_programs2, how="inner", on=["id","model","split","refiment"])
    
    # to handle legacy save setting
    df_merged["raw_logic_programs"] = df_merged.apply(lambda x: get_program(x), axis=1)

    df_merged["query"] = df_merged.apply(lambda x: get_prompt_no_examples(x), axis=1)
    df_merged["label"] = ((df_merged.flag == "success") & (df_merged.is_correct)).astype(int)
    return df_merged

def reload_model_and_tokenizer(model_path):
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return loaded_model, loaded_tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="ProntoQA")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--train_model_name', type=str, default="google/bigbird-roberta-base")
    parser.add_argument('--benchmark_model_name', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', type=str, default="no")
    parser.add_argument('--logic_inference', type=str, default='./outputs/logic_inference')
    parser.add_argument('--logic_programs', type=str, default='./outputs/logic_programs')
    parser.add_argument('--demonstration_path', type=str, default='./models/prompts')
    parser.add_argument('--result_path', type=str, default="/mnt/evafs/groups/luckner-lab/models/")
    parser.add_argument('--save_dir', type=str, default="bert")
    parser.add_argument('--reload_model_for_test', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--load_in_8bit', type=int, default=1)
    parser.add_argument('--fine_tune_all_modules', type=int, default=0)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":

    overall_start = time.time()

    args = parse_args()
    model_name = args.train_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template = load_prompt_templates(args.dataset_name)
    df_merged = prepare_data(args.benchmark_model_name, args.split, args.logic_inference, args.logic_programs, args.dataset_name)    
    df_merged = df_merged.rename(columns={"query": "prompt"})
    df_merged = df_merged.drop(columns = df_merged.columns.drop(["prompt", "label"]))
    df_merged = df_merged.drop_duplicates()
    # Get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Save df_merged to a CSV file
    output_file = os.path.join('./outputs/prepared_data',f"{args.dataset_name}_{args.split}_merged_data_bigbird_{current_date}.csv")
    df_merged.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")

    if args.resume_from_checkpoint == "resume":
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False


    training_args = TrainingArguments(
    output_dir=os.path.join(args.result_path, model_name, args.dataset_name, "bert"),
    report_to="tensorboard",
    load_best_model_at_end = True,
    num_train_epochs = args.epochs,
    logging_steps = 10,
    save_total_limit = 2,
    eval_strategy = "epoch", 
    save_strategy = "epoch",
    save_safetensors = False,
    fp16 = False,
    auto_find_batch_size = True,
    # warmup_steps=500,
    weight_decay=0.01,
)

    lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    target_modules=None if args.fine_tune_all_modules==1 else "all-linear",
    task_type="SEQ_CLS",
    modules_to_save=["score"],
    )

    quantization_config =None
    if bool(args.load_in_8bit):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config = quantization_config,
    device_map="balanced",
    torch_dtype="auto",
    num_labels=2,
    )#.to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # model.config.use_cache = False

    data = df_merged.to_dict('records')
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)

    tokenized_train_data = train_data.map(tokenize_function, batched=True)
    tokenized_val_data = val_data.map(tokenize_function, batched=True)

    tokenized_train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Calculate label distribution for training set
    train_label_dist = Counter(tokenized_train_data['label'].tolist())
    print("Training Label Distribution:")
    for label, count in train_label_dist.items():
        print(f"Label {label}: {count} ({count/len(tokenized_train_data)*100:.2f}%)")

    # Calculate label distribution for validation set
    val_label_dist = Counter(tokenized_val_data['label'].tolist())
    print("\nValidation Label Distribution:")
    for label, count in val_label_dist.items():
        print(f"Label {label}: {count} ({count/len(tokenized_val_data)*100:.2f}%)")

    trainer = Trainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.result_path, model_name, args.dataset_name,args.save_dir,"best"))
    # # Save the entire model (base + PEFT)
    model.save_pretrained(os.path.join(args.result_path, model_name, args.dataset_name,args.save_dir,"best"))
    # # Save the tokenizer
    tokenizer.save_pretrained(os.path.join(args.result_path, model_name, args.dataset_name, args.save_dir,"best"))

    print(f"Total time: {time.time() - overall_start:.2f} secs")
    print_gpu_utilization()

    # Get predictions on the validation set
    predictions = trainer.predict(tokenized_val_data)

    # Get the predicted labels
    predicted_labels = predictions.predictions.argmax(-1)

    # Get the predicted probabilities
    predicted_probs = predictions.predictions[:, 1]  # Assuming binary classification, get probabilities for positive class

    # Get the true labels
    true_labels = tokenized_val_data['label']

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    auc_roc = roc_auc_score(true_labels, predicted_probs)

    print(f"\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    if args.reload_model_for_test:

        # After training and saving the model
        saved_model_path = os.path.join(args.result_path, model_name, args.dataset_name, args.save_dir, "best")

        # Reload the model and tokenizer
        loaded_model, loaded_tokenizer = reload_model_and_tokenizer(saved_model_path)

        loaded_model.config.pad_token_id = loaded_tokenizer.pad_token_id
        loaded_model.config.pad_token = loaded_tokenizer.pad_token

        # Create a new trainer with the loaded model
        loaded_trainer = Trainer(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_val_data,
        )

        # Get predictions using the loaded model
        loaded_predictions = loaded_trainer.predict(tokenized_val_data)

        # Compare predictions
        original_predicted_labels = predictions.predictions.argmax(-1)
        loaded_predicted_labels = loaded_predictions.predictions.argmax(-1)

        are_predictions_identical = np.array_equal(original_predicted_labels, loaded_predicted_labels)

        print("\nVerifying loaded model results:")
        print(f"Predictions are identical: {are_predictions_identical}")

        if are_predictions_identical:
            print("The loaded model produces identical results to the original model.")
        else:
            print("Warning: The loaded model produces different results from the original model.")
        
        # Calculate and print metrics for the loaded model
        loaded_accuracy = accuracy_score(true_labels, loaded_predicted_labels)
        loaded_auc_roc = roc_auc_score(true_labels, loaded_predictions.predictions[:, 1])
        
        print(f"\nLoaded Model Validation Metrics:")
        print(f"Accuracy: {loaded_accuracy:.4f}")
        print(f"AUC-ROC: {loaded_auc_roc:.4f}")
