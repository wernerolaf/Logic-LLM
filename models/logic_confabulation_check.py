import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.logic_program import LogicProgramGenerator
from models.logic_inference import LogicInferenceEngine
from models.utils import print_gpu_utilization
import time

class LogicConfabulationChecker:
    def __init__(self, args):
        self.args = args
        self.program_generator = LogicProgramGenerator(args)
        self.inference_engine = LogicInferenceEngine(args)
        self.model, self.tokenizer = self.load_trained_model()

        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.model_name=self.model_name.replace("/","-")
        self.save_path = args.save_path
        self.refiment = args.refiment
        self.num_beams = args.num_beams
        self.num_beam_groups = args.num_beam_groups
        self.num_return_sequences = args.num_return_sequences
        
        if args.use_existing_programs:
            self.dataset = self.load_logic_programs()

        if self.num_beams > 1:
            self.model_name = f"{self.model_name}-beam{self.num_beams}-group{self.num_beam_groups}"

    def load_trained_model(self):
        model_path = f"{self.args.result_path}/{self.args.train_model_name}/{self.args.dataset_name}/bert/best"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.args.train_model_name)
        return model, tokenizer

    def load_logic_programs(self):
        dataset = {}
        if self.refiment == -1:
            return dataset
        
        if self.refiment == 0:
            file_path = os.path.join('./outputs/logic_programs', f'{self.dataset_name}_{self.split}_{self.model_name}.json')
        else:
            file_path = os.path.join('./outputs/logic_programs', f'self-refine-{self.refiment}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            print(f"Loaded {len(dataset)} examples from {self.split} split.")
        else:
            print(f"Dataset file not found: {file_path}")
        
        return dataset

    def generate_and_evaluate(self, example):
        if args.use_existing_programs:
            program = next((item for item in self.dataset if item['id'] == example['id']), None)
            if program:
                logic_programs = program['raw_logic_programs']
            else:
                logic_programs = self.program_generator.llm_model.generate(self.program_generator.prompt_creator[self.args.dataset_name](example))
        else:
            logic_programs = self.program_generator.llm_model.generate(self.program_generator.prompt_creator[self.args.dataset_name](example))

        results = []
        for program in logic_programs:
            # Execute the logic program
            answer, flag, error_message = self.inference_engine.safe_execute_program(example['id'], program)
            
            # Prepare input for the trained model
            input_text = f"{example['context']} {example['question']} {program} {answer}"
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
            
            # Get prediction from the trained model
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=1)
                prediction = probabilities[0][1].item()  # Probability of the positive class

            
            results.append({
                'id': example['id'],
                'program': program,
                'answer': answer,
                'flag': flag,
                'error_message': error_message,
                'confabulation_score': prediction
            })

        return results

    def check_dataset(self):
        dataset = self.program_generator.load_raw_dataset(self.args.split)
        outputs = []

        for example in dataset:
            results = self.generate_and_evaluate(example)
            outputs.append({
                'id': example['id'],
                'context': example['context'],
                'question': example['question'],
                'answer': example['answer'],
                'results': results
            })

        return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str ,default='AR-LSAT')
    parser.add_argument('--split', type=str, default='minitest')
    parser.add_argument('--save_path', type=str, default='./outputs/confabulation_check')
    parser.add_argument('--model_name', type=str, default='mistralai-Mistral-7B-v0.1-AR-LSAT-sft-best/AR-LSAT/ppo/last')
    parser.add_argument('--model_path', type=str, default='/mnt/evafs/groups/luckner-lab/models/')
    parser.add_argument('--use_fine_tuned', type=int, default=1)
    parser.add_argument('--train_model_name', type=str, default="google/bigbird-roberta-base")
    parser.add_argument('--result_path', type=str, default="/mnt/evafs/groups/luckner-lab/models/")
    parser.add_argument('--framework_to_use', type=str, default='HuggingFace')
    parser.add_argument('--num_beams', type=int, default=2)
    parser.add_argument('--num_return_sequences', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_existing_programs', type=int, default=1)
    parser.add_argument('--logic_programs_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--refiment', type=int, default=0)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--early_stopping', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--is_AWQ', type=str, default="auto")
    parser.add_argument('--timeout_time', type=int, default=600)
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--mode', type=str, default='CoT')
    parser.add_argument('--backup_LLM_result_path', type=str, default='./outputs/results')
    args, unknown = parser.parse_known_args()
    return args

if __name__ == '__main__':
    overall_start = time.time()
    args = parse_args()
    confabulation_checker = LogicConfabulationChecker(args)
    results = confabulation_checker.check_dataset()
    print(f"Total time: {time.time() - overall_start:.2f} secs")
    print_gpu_utilization()
    # Add code here to save or process the results