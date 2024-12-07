"""
Utility functions for working with large language models.

Includes functions for generating text from models like GPT-3, handling OpenAI API requests, and timing out long-running requests.

The main classes are:

- HuggingFaceModel: Generates text using a HuggingFace model.
- OpenAIModel: Generates text using an OpenAI model. 
- LLMClass: Abstract base class for LLM models.
- Timeout: Context manager for timing out functions.

The OpenAIModel and HuggingFaceModel classes handle generating text with different underlying models.
The Timeout context manager allows limiting execution time.
"""
import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

from signal import signal, alarm, SIGALRM
import time

class TimeoutError(Exception):
    ...

class Timeout:
    def __init__(self, seconds=1, message="Timed out"):
        self._seconds = seconds
        self._message = message

    @property
    def seconds(self):
        return self._seconds

    @property
    def message(self):
        return self._message
    
    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def handle_timeout(self, *_):
        raise TimeoutError(self.message)

    def __enter__(self):
        self.handler = signal(SIGALRM, self.handle_timeout)
        alarm(self.seconds)
        return self

    def __exit__(self, *_):
        alarm(0)
        signal(SIGALRM, self.handler)


from pynvml import *

def print_gpu_utilization():
    try:
        nvmlInit()
        gpus = nvmlDeviceGetCount()
        min_increase = 100
        for g in range(gpus):
            handle = nvmlDeviceGetHandleByIndex(g)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {g} memory occupied: {info.used//1024**2} MB.")
            print(f"GPU {g} memory occupied: {info.used/info.total*100:.2f} %.")
            if info.used > 0:
                possible_increase = (info.free / info.used)
                min_increase = min(min_increase, possible_increase)
        return min_increase
    except Exception as e:
        print("problems with reading GPU")
        return 0

from typing import List

class LLMClass:
    def generate(self, input_string: str, temperature: float = 0.0) -> str:
        return input_string

    def batch_generate(self, messages_list: List[str], temperature: float = 0.0) -> List[str]:
        return messages_list


from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessorList, LogitsProcessor

from awq import AutoAWQForCausalLM

class StoppingCriteriaSeq(StoppingCriteria):

    def __init__(self, tokenizer, stops = []):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        
        generated_text = self.tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True)

        for stop in self.stops:
            if stop in generated_text:
                return True
                
        return False

class IgnoreEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        scores[:, self.eos_token_id] = -float('inf')
        return scores

class HuggingFaceModel(LLMClass):
    def __init__(self, model_id, stop_words, force_words="", max_new_tokens=1024, is_AWQ = "auto", timeout_time=300, batch_size=10, num_beams=1, num_beam_groups=1, diversity_penalty=1.0, num_return_sequences=1, early_stopping = True) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.timeout_time = timeout_time
        self.num_beams = int(num_beams)
        self.num_beam_groups = int(num_beam_groups)
        self.diversity_penalty = diversity_penalty
        if num_beam_groups <= 1:
            self.diversity_penalty = None

        self.num_return_sequences = int(num_return_sequences)
        self.early_stopping = early_stopping
    
        if is_AWQ == "auto":
            if "AWQ" in model_id:
                is_AWQ = True
            else:
                is_AWQ = False
        else:
            is_AWQ = bool(is_AWQ)

        if is_AWQ:
            model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True, device_map = 'balanced',
                                          trust_remote_code=False, safetensors=True)
            self.model = model.model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="balanced", torch_dtype="auto")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.model.config.eos_token_id

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSeq(stops=stop_words.split(" "), tokenizer=self.tokenizer)])

        self.force_words = force_words
        if len(self.force_words) > 0:
            self.force_words_ids = [self.tokenizer.encode(word, add_special_tokens=False) for word in force_words.split(" ")]
            # Flatten the list of lists using list comprehension
            self.force_words_ids = [item for sublist in self.force_words_ids for item in sublist]
            print(self.force_words_ids)
        else:
            self.force_words_ids = None


        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "device_map": "balanced",
            "do_sample": False,
            "top_p": 1.0,
            "return_full_text": False,
            # "stopping_criteria": stopping_criteria,
            "stop_strings": stop_words.split(" "),
            "num_return_sequences": self.num_return_sequences,
            "early_stopping": self.early_stopping,
        }

        if self.num_beams > 1:
            pipeline_kwargs.update({
                "num_beams": self.num_beams,
                "num_beam_groups": self.num_beam_groups,
                "diversity_penalty": self.diversity_penalty,
                "force_words_ids": self.force_words_ids,
            })

        self.ignore_eos_processor = None
        if len(self.force_words) > 0:
            eos_token_id = self.tokenizer.eos_token_id
            ignore_eos_processor = LogitsProcessorList([IgnoreEOSLogitsProcessor(eos_token_id)])
            self.ignore_eos_processor = ignore_eos_processor
            pipeline_kwargs.update({
                "logits_processor": ignore_eos_processor,
            })

        self.pipe = pipeline("text-generation", **pipeline_kwargs)


    def generate(self, input_string, temperature = 0.0):
        with Timeout(self.timeout_time): # time out after 5 minutes
            try:
                response = self.pipe(input_string, temperature=temperature, tokenizer=self.tokenizer, logits_processor=self.ignore_eos_processor)
                generated_text = [response[i]["generated_text"].strip() for i in range(len(response))]
                return generated_text
            except TimeoutError as e:
                print(e)
                # print(input_string)
                return ['Time out!']

    def batch_generate(self, messages_list, temperature = 0.0):
        with Timeout(self.timeout_time): # time out after 5 minutes
            try:
                responses = self.pipe(messages_list, temperature=temperature, tokenizer=self.tokenizer, logits_processor=self.ignore_eos_processor)
                generated_text = [[response[i]["generated_text"].strip() for i in range(len(response))] for response in responses]
                return generated_text
            except TimeoutError as e:
                print(e)
                # print(messages_list)
                return ['Time out!' for m in messages_list]


class OpenAIModel(LLMClass):
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature = 0.0):
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['message']['content'].strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text