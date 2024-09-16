"""
Generates logic programs from a dataset of problems and questions by prompting an LLM.

Loads a dataset, creates prompts for each example, generates programs with an LLM, 
and saves the outputs. Includes batch generation for efficiency.
"""
# generate facts and rules based on the problem description

import os
import json
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from models.utils import OpenAIModel, HuggingFaceModel, LLMClass
import argparse
from models.utils import print_gpu_utilization
import time

class LogicProgramGenerator:
    def __init__(self, args, llm_model=None):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        if args.use_fine_tuned == 1:
            self.model_name = args.model_path + args.model_name
        self.save_path = args.save_path
        self.framework_to_use = args.framework_to_use
        self.num_beams = args.num_beams
        self.num_beam_groups = args.num_beam_groups
        self.diversity_penalty = args.diversity_penalty
        self.num_return_sequences = args.num_return_sequences
        if llm_model is None:
            if self.framework_to_use == "OpenAI":
                self.llm_model = OpenAIModel(args.api_key, 'gpt-4', args.stop_words, args.max_new_tokens)
            elif self.framework_to_use == "HuggingFace":
                self.llm_model = HuggingFaceModel(model_id=self.model_name, stop_words = args.stop_words, max_new_tokens=args.max_new_tokens,
                 is_AWQ=args.is_AWQ, timeout_time=args.timeout_time, batch_size=args.batch_size,
                 num_beams=args.num_beams, num_beam_groups=args.num_beam_groups, diversity_penalty=args.diversity_penalty, num_return_sequences=args.num_return_sequences, early_stopping = bool(args.early_stopping))
            else:
                self.llm_model = LLMClass()
        else:
            self.llm_model = llm_model

        if args.use_fine_tuned == 1:
            self.model_name = args.model_name
        
        self.model_name=self.model_name.replace("/","-")

        self.prompt_creator = {'FOLIO': self.prompt_folio,
                           'ProntoQA': self.prompt_prontoqa,
                           'ProofWriter': self.prompt_proofwriter,
                           'LogicalDeduction': self.prompt_logicaldeduction, 
                           'AR-LSAT': self.prompt_arlsat}
        self.zero_shot = args.zero_shot
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = './models/prompts/{}.txt'.format(self.dataset_name)
        if self.dataset_name == 'AR-LSAT' and self.model_name == 'gpt-4':
            prompt_file = './models/prompts/{}-long.txt'.format(self.dataset_name)
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
        if self.zero_shot:
            self.prompt_template = self.prompt_template.split("------", 1)[0] + "------" + self.prompt_template.rsplit("------", 1)[-1]

    def prompt_folio(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_arlsat(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join(['{}'.format(choice.strip()) for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt
    
    def prompt_prontoqa(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_proofwriter(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_logicaldeduction(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join(['{}'.format(choice.strip()) for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, '{}.json'.format(split))) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def logic_program_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print("Loaded {} examples from {} split.".format(len(raw_dataset),self.split))

        outputs = []
        for example in tqdm(raw_dataset):
            # create prompt
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                programs = self.llm_model.generate(full_prompt)
                # create output
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'options': example['options'],
                        'raw_logic_programs': programs}
                outputs.append(output)
            except Exception as e:
                print(e)
                print('Error in generating logic programs for example: ', example['id'])

        # save outputs        
        with open(os.path.join(self.save_path, '{}_{}_{}.json'.format(self.dataset_name,self.split,self.model_name,)), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''
    def batch_logic_program_generation(self, batch_size = 10, auto_batch_size=True):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print("Loaded {} examples from {} split.".format(len(raw_dataset),self.split))
        if auto_batch_size:
            chunk = raw_dataset[:batch_size]
            # create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            batch_outputs = self.llm_model.batch_generate(full_prompts)
            relative_increase = print_gpu_utilization()
            batch_size = max(batch_size, int(0.9 * (batch_size * (1+relative_increase) - 1)))
            print('New batch size: ', batch_size)

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            try:
                batch_outputs = self.llm_model.batch_generate(full_prompts)
                # create output
                for sample, programs in zip(chunk, batch_outputs):
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            'options': sample['options'],
                            'raw_logic_programs': programs}
                    outputs.append(output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.llm_model.generate(full_prompt)
                        programs = output
                        output = {'id': sample['id'], 
                                'context': sample['context'],
                                'question': sample['question'], 
                                'answer': sample['answer'],
                                'options': sample['options'],
                                'raw_logic_programs': programs}
                        outputs.append(output)
                    except Exception as e:
                        print(e)
                        print('Error in generating logic programs for example: ', sample['id'])

        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print("Generated {} examples.".format(len(outputs)))
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.num_beams > 1:
            filename = f'{self.dataset_name}_{self.split}_{self.model_name}-beam{self.num_beams}-group{self.num_beam_groups}-zero-{self.zero_shot}.json'
            with open(os.path.join(self.save_path, filename), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
        else:
            with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}-zero-{self.zero_shot}.json'), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str, default='KEY')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--model_path', type=str, default='/mnt/evafs/groups/luckner-lab/models/')
    parser.add_argument('--use_fine_tuned', type=int, default=0)
    parser.add_argument('--framework_to_use', type=str, default='HuggingFace')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--is_AWQ', type=str, default="auto")
    parser.add_argument('--zero_shot', type=int, default=0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_beam_groups', type=int, default=1)
    parser.add_argument('--diversity_penalty', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--timeout_time', type=int, default=1200)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    overall_start = time.time()
    args = parse_args()
    logic_program_generator = LogicProgramGenerator(args)
    logic_program_generator.batch_logic_program_generation(batch_size = args.batch_size)
    print(f"Total time: {time.time() - overall_start:.2f} secs")
    print_gpu_utilization()