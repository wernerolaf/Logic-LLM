"""
evaluate_sample computes exact match and F1 score for a single prediction-truth pair.

compute_exact_match and compute_f1 are helper functions to compute exact match and F1.

evaluate_QA computes accuracy over a list of QA prediction samples.

full_evaluation runs evaluation over a results JSON file and prints overall accuracy, executable rate, and accuracy over executable samples.

parse_args parses command line arguments.

Main calls full_evaluation on the parsed args result file path.
"""
import re
import json
import os
import argparse
import time
from models.utils import friendly_model_name

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))
    # return prediction == truth

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_sample(prediction, gold_answers):
    em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
    f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
    return em_score, f1_score

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')

    if answer_str.startswith(':'):
       return answer_str.replace(':', '').replace('.', '').strip()
    return None

def evaluate_QA(QA_results):
    total_first_em = 0.0
    total_any_em = 0.0
    sample_count = 0
    current_id = None
    current_answers = []

    for sample in QA_results:
        if sample['id'] != current_id:
            if current_id is not None:
                first_correct = current_answers[0] == gold_answer
                any_correct = any([answer ==  gold_answer for answer in current_answers])
                total_first_em += 1.0 if first_correct else 0.0
                total_any_em += 1.0 if any_correct else 0.0
                sample_count += 1

            current_id = sample['id']
            current_answers = []

        gold_answer = sample['answer'].replace('(', '').replace(')', '').strip()
        answer_str = sample['predicted_answer'].strip() if sample['predicted_answer'] is not None else ''
        prediction = get_choice(answer_str)

        indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is']
        if prediction is None:
            for indicator in indicators:
                if answer_str.find(indicator)>=0:
                    answer_str = answer_str.split(indicator)[1].strip()
                    prediction = get_choice(answer_str)
                    break
        
        current_answers.append(prediction)

    if current_id is not None:
        first_correct = current_answers[0] == gold_answer
        any_correct = any([answer ==  gold_answer for answer in current_answers])
        total_first_em += 1.0 if first_correct else 0.0
        total_any_em += 1.0 if any_correct else 0.0
        sample_count += 1

    if sample_count != 0:
        first_answer_accuracy = total_first_em / sample_count
        any_answer_accuracy = total_any_em / sample_count
    else:
        first_answer_accuracy = 0
        any_answer_accuracy = 0

    return first_answer_accuracy, any_answer_accuracy

def full_evaluation(result_file):
    with open(result_file, 'r') as f:
        all_samples = json.load(f)

    if len(all_samples) == 0:
        print(f"No samples loaded from {result_file}. Executable rate: 0")
        return

    executable_samples = [sample for sample in all_samples if sample['flag'] == 'success']
    first_acc, any_acc = evaluate_QA(all_samples)
    exe_first_acc, exe_any_acc = evaluate_QA(executable_samples)
    
    print(f"Overall first answer accuracy: {first_acc}")
    print(f"Overall any answer accuracy: {any_acc}")
    print(f'Executable rate (Exe_Rate): {len(executable_samples)/len(all_samples)}')
    print(f"Executable first answer accuracy: {exe_first_acc}")
    print(f"Executable any answer accuracy: {exe_any_acc}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str, default='text-davinci-003')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--backup", type=str, default='random')
    parser.add_argument('--result_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--refiment', type=int, default=0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_beam_groups', type=int, default=1)
    parser.add_argument('--zero_shot', type=int, default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model_name = friendly_model_name(args.model_name)
    num_beams = args.num_beams

    if num_beams > 1:
        model_name = f"{model_name}-beam{num_beams}-group{args.num_beam_groups}"

    if args.zero_shot >= 0:
        model_name=model_name+f"-zero-{args.zero_shot}"

    if args.mode == '':
        if args.refiment == 0:
            result_file = os.path.join(args.result_path, f'{args.dataset_name}_{args.split}_{model_name}_backup-{args.backup}.json')
        else:
            result_file = os.path.join(args.result_path, f'self-refine-{args.refiment}_{args.dataset_name}_{args.split}_{model_name}_backup-{args.backup}.json')
    else:
        result_file = os.path.join(args.result_path, f'{args.mode}_{args.dataset_name}_{args.split}_{model_name}.json')
    
    print(f'Evaluating {result_file}')
    full_evaluation(result_file)
