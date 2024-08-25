"""Logic inference engine to execute logic programs.

Loads logic programs from a dataset, executes them using different logic solvers, 
and generates outputs with execution results. Handles errors during parsing and 
execution using backup strategies. Saves outputs to file after inference.
"""
import json
import os
from tqdm import tqdm
from models.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from models.symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from models.symbolic_solvers.csp_solver.csp_solver import CSP_Program
from models.symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random
from models.backup_answer_generation import Backup_Answer_Generator
from models.utils import print_gpu_utilization
import time

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.model_name=self.model_name.replace("/","-")
        self.save_path = args.save_path
        self.backup_strategy = args.backup_strategy
        self.refiment = args.refiment
        self.num_beams = args.num_beams
        self.num_return_sequences = args.num_return_sequences

        if self.num_beams>1:
            self.model_name = self.model_name + "-beam" + str(self.num_beams)

        self.dataset = self.load_logic_programs()
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'ProntoQA': Pyke_Program, 
                                'ProofWriter': Pyke_Program,
                                'LogicalDeduction': CSP_Program,
                                'AR-LSAT': LSAT_Z3_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(args.mode, self.dataset_name, args.split, args.model_name, self.backup_strategy, self.args.backup_LLM_result_path)

    def load_logic_programs(self):
        if self.refiment == -1:
            dataset = {}
        elif self.refiment == 0:
            with open(os.path.join('./outputs/logic_programs', f'{self.dataset_name}_{self.split}_{self.model_name}.json')) as f:
                dataset = json.load(f)
        else:
            with open(os.path.join('./outputs/logic_programs', f'self-refine-{self.refiment}_{self.dataset_name}_{self.split}_{self.model_name}.json')) as f:
                dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset

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
    
    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        if self.refiment == 0:
            with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_backup-{self.backup_strategy}.json'), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
        else:
            with open(os.path.join(self.save_path, f'self-refine-{self.refiment}_{self.dataset_name}_{self.split}_{self.model_name}_backup-{self.backup_strategy}.json'), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):

        # preprocess string as ------ means end of problem
        logic_program = logic_program.replace("------","").strip()
        program = self.program_executor(logic_program, self.dataset_name)
        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'parsing error', ''
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', ''

    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            answers = []
            flags = []
            error_messages = []
            
            for program in example['raw_logic_programs']:
                # execute each logic program
                answer, flag, error_message = self.safe_execute_program(example['id'], program.strip())
                answers.append(answer)
                flags.append(flag)
                error_messages.append(error_message)
                if flag != 'success':
                    error_count += 1

                # create output
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'program': program,
                        'flag': flag,
                        'predicted_answer': answer,
                        'error_message': error_message}
                outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            # print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

    def cleanup_partial(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            # print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}/*')
            
    def inference_on_text(self, texts):
        outputs = []
        
        for text in texts:
            # execute the logic program
            answer, flag, error_message = self.safe_execute_program('text', text)
            
            # create output
            output = {'text': text,
                    'flag': flag,
                    'predicted_answer': answer}
            outputs.append(output)

        self.cleanup_partial()
            
        return outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='./outputs/results')
    parser.add_argument('--use_fine_tuned', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='/mnt/evafs/groups/luckner-lab/models/')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--mode', type=str, default='CoT')
    parser.add_argument('--refiment', type=int, default=0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    overall_start = time.time()
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()
    print(f"Total time: {time.time() - overall_start:.2f} secs")
    print_gpu_utilization()
