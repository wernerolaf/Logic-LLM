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
import random
import json
# OpenAI SDK (new style)
try:
    from openai import (
        OpenAI,
        RateLimitError,
        APIError,
        APIConnectionError,
        APITimeoutError,
    )
except Exception:
    OpenAI = None  # Fallback handled in OpenAIModel
    RateLimitError = APIError = APIConnectionError = APITimeoutError = Exception
import os
import asyncio
from typing import Any
from signal import signal, alarm, SIGALRM
import time
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False



def strip_comments(block: str) -> str:
    return "\n".join(line.split(":::")[0].strip() for line in block.splitlines() if line.strip())


def friendly_model_name(name: str) -> str:
    """
    Shorten a potentially absolute model path for use in filenames.
    Keeps the last few path segments and replaces slashes with dashes so
    logic_program and logic_inference stay in sync.
    """
    parts = [p for p in name.strip(os.sep).split(os.sep) if p]
    if os.path.isabs(name) and len(parts) > 4:
        parts = parts[-4:]
    safe = "-".join(parts) if parts else name
    return safe.replace("/", "-")


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
try:
    from datasets import Dataset
    from transformers.pipelines.pt_utils import KeyDataset
except Exception:
    Dataset = None
    KeyDataset = None

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
    def __init__(self, model_id, stop_words, force_words="", max_new_tokens=1024, is_AWQ = "auto", timeout_time=300, batch_size=10, num_beams=1, num_beam_groups=1, diversity_penalty=1.0, num_return_sequences=1, early_stopping = True, backend="vllm",
    tensor_parallel_size=1, max_model_len=None) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.timeout_time = timeout_time
        self.batch_size = int(batch_size)
        self.num_beams = int(num_beams)
        self.num_beam_groups = int(num_beam_groups)
        self.diversity_penalty = diversity_penalty
        if num_beam_groups <= 1:
            self.diversity_penalty = None

        self.num_return_sequences = int(num_return_sequences)
        self.early_stopping = early_stopping
        self.backend = backend.lower()

        # Try to set a reasonable pad token before any heavy model init
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if self.tokenizer.pad_token_id is None and eos_token_id is not None:
            self.tokenizer.pad_token_id = eos_token_id

        # Many causal models require left padding
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass

        # Detect instruction-tuned chat models and enable chat templating if available
        self.use_chat_template = False
        try:
            if isinstance(self.model_id, str) and ("instruct" in self.model_id.lower()):
                has_apply = hasattr(self.tokenizer, "apply_chat_template")
                has_template = getattr(self.tokenizer, "chat_template", None) is not None
                self.use_chat_template = bool(has_apply and has_template)
        except Exception:
            self.use_chat_template = False

        quantization_config =None
        if False:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        if self.backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed but backend='vllm' was requested")

            llm_kwargs = dict(
                model=model_id,
                tensor_parallel_size=tensor_parallel_size,
                dtype="auto",
                trust_remote_code=True,
            )
            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len

            self.llm = LLM(**llm_kwargs)
            self.stop_words = stop_words.split(" ")
            self.max_new_tokens = max_new_tokens
        else:

            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quantization_config, device_map="auto", torch_dtype="auto")

            # If padding is still unset, borrow it from the loaded model config
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = getattr(self.model.config, "eos_token_id", self.tokenizer.pad_token_id)

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
                "batch_size": self.batch_size,
                "device_map": "auto",
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


    def _maybe_apply_chat_template(self, inp):
        """If model is instruction-tuned and tokenizer defines a chat template,
        wrap the plain string as a single user message and render it.
        """
        if not self.use_chat_template:
            return inp
        try:
            style = os.getenv("LLM_CHAT_STYLE", "logic")  # or use args
            if style == "CoT":
                system_text = "Explain your reasoning step-by-step, then give the final answer."
            elif style == "Direct":
                system_text = "Return the answer immediately."
            else:
                system_text = "Return only the logic program in the correct format."

            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": inp},
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered
        except Exception:
            # Fallback to the raw input if templating fails for any reason
            return inp

    def _vllm_generate(self, prompts, temperature):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            stop=self.stop_words,
            n=self.num_return_sequences,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for out in outputs:
            texts = [o.text.strip() for o in out.outputs]
            results.append(texts)

        return results


    def generate(self, input_string, temperature = 0.0):
        with Timeout(self.timeout_time): # time out after 5 minutes
            try:
                to_generate = self._maybe_apply_chat_template(input_string)

                if self.backend == "vllm":
                    return self._vllm_generate([to_generate], temperature)[0]

                response = self.pipe(to_generate, temperature=temperature, tokenizer=self.tokenizer, logits_processor=self.ignore_eos_processor)
                generated_text = [response[i]["generated_text"].strip() for i in range(len(response))]
                return generated_text
            except TimeoutError as e:
                print(e)
                # print(input_string)
                return ['Time out!']

    def batch_generate(self, messages_list, temperature = 0.0):
        # with Timeout(self.timeout_time): # time out after 5 minutes
        # try:
        if self.use_chat_template:
            prepped = []
            for m in messages_list:
                prepped.append(self._maybe_apply_chat_template(m))
        else:
            prepped = messages_list

        if self.backend == "vllm":
            return self._vllm_generate(prepped, temperature)

        # Prefer dataset streaming to allow pipeline to batch on-GPU efficiently
        if Dataset is not None and KeyDataset is not None:
            ds = Dataset.from_dict({"text": prepped})
            generated_text = []
            for response in tqdm(
                self.pipe(
                    KeyDataset(ds, "text"),
                    temperature=temperature,
                    tokenizer=self.tokenizer,
                    logits_processor=self.ignore_eos_processor,
                    batch_size=self.batch_size,
                ),
                total=len(prepped),
                desc="Generating"
            ):
                generated_text.append(
                    [response[i]["generated_text"].strip() for i in range(len(response))]
                )
            return generated_text

        # Fallback if datasets is unavailable
        responses = self.pipe(
            prepped,
            temperature=temperature,
            tokenizer=self.tokenizer,
            logits_processor=self.ignore_eos_processor,
            batch_size=self.batch_size,
        )
        generated_text = [
            [response[i]["generated_text"].strip() for i in range(len(response))]
            for response in responses
        ]
        return generated_text
            # except TimeoutError as e:
            #     print(e)
            #     # print(messages_list)
            #     return [['Time out!'] for m in messages_list]


class OpenAIModel(LLMClass):
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        # Prefer the new OpenAI SDK client. If unavailable, raise a helpful error.
        if OpenAI is None:
            raise ImportError(
                "OpenAI SDK v1+ is required. Install/update with `pip install --upgrade openai`."
            )

        # Initialize client using explicit key or env vars.
        # Preference: provided API_KEY -> env OPENAI_KEY -> env OPENAI_API_KEY
        # Treat placeholder values like "KEY" as unset and prefer env vars
        provided_key = (str(API_KEY).strip() if API_KEY is not None else "")
        use_provided = bool(provided_key) and provided_key.upper() != "KEY"
        api_key = (provided_key if use_provided else None) \
            or os.environ.get("OPENAI_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Pass API_KEY or set OPENAI_KEY/OPENAI_API_KEY."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = str(model_name)
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens else None
        self.stop_words = stop_words
        # Retry/backoff settings for rate limits and transient errors
        self.batch_max_retries = 6
        self.retry_base_seconds = 1.0
        self.retry_max_sleep = 60.0

    def _sleep_backoff(self, attempt: int):
        base = self.retry_base_seconds
        delay = min(self.retry_max_sleep, base * (2 ** attempt) + random.uniform(0, 0.25 * base))
        time.sleep(delay)

    def _with_retries(self, func, *args, **kwargs):
        last_err = None
        for attempt in range(self.batch_max_retries):
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_err = e
            except APIError as e:
                # Retry only for 5xx API errors
                status = getattr(e, "status_code", None)
                if status is not None and int(status) < 500:
                    raise
                last_err = e
            if attempt == self.batch_max_retries - 1:
                break
            self._sleep_backoff(attempt)
        raise last_err

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature = 0.0):
        response = self._with_retries(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[{"role": "user", "content": input_string}],
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.stop_words,
        )
        return (response.choices[0].message.content or "").strip()
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        response = self._with_retries(
            self.client.completions.create,
            model=self.model_name,
            prompt=input_string,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=self.stop_words,
        )
        return (response.choices[0].text or "").strip()

    def generate(self, input_string, temperature = 0.0):
        name = self.model_name.lower()
        # Use chat for modern models (e.g., gpt-4, gpt-4o, gpt-5, o-series)
        if name.startswith("gpt-") or name.startswith("o"):
            return self.chat_generate(input_string, temperature)
        # Fallback to legacy completions for non-chat base models
        return self.prompt_generate(input_string, temperature)
    
    def batch_chat_generate(self, messages_list, temperature = 0.0):
        outputs = []
        for idx, message in enumerate(messages_list):
            try:
                outputs.append(self.chat_generate(message, temperature))
            except Exception as e:
                outputs.append(f"<error at {idx}: {e}>")
        return outputs
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        outputs = []
        for idx, prompt in enumerate(prompt_list):
            try:
                outputs.append(self.prompt_generate(prompt, temperature))
            except Exception as e:
                outputs.append(f"<error at {idx}: {e}>")
        return outputs

    def batch_generate(self, messages_list, temperature = 0.0):
        name = self.model_name.lower()
        if name.startswith("gpt-") or name.startswith("o"):
            return self.batch_chat_generate(messages_list, temperature)
        return self.batch_prompt_generate(messages_list, temperature)

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = self._with_retries(
            self.client.completions.create,
            model=self.model_name,
            prompt=input_string,
            suffix=suffix,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return (response.choices[0].text or "").strip()

    # =====================
    # Batch API (JSONL flow)
    # =====================
    def _normalize_stop(self):
        return self.stop_words

    def write_batch_jsonl(
        self,
        items,
        jsonl_path,
        *,
        endpoint = "/v1/chat/completions",
        temperature: float = 0.0,
        system_prompt = None,
        metadata_per_item = None,
    ):
        """Writes a batch input JSONL file for the Batch API.

        - items: list of prompts (strings). For chat endpoint, treated as user messages.
        - jsonl_path: file to write.
        - endpoint: one of "/v1/chat/completions" or "/v1/completions".
        - returns: list of custom_ids (req-0, req-1, ...)
        """
        if endpoint not in {"/v1/chat/completions", "/v1/completions"}:
            raise ValueError("Unsupported endpoint for batch input: " + endpoint)

        stop = self._normalize_stop()
        custom_ids = []
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(items):
                cid = f"request-{i}"
                custom_ids.append(cid)
                if endpoint == "/v1/chat/completions":
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": item})
                    body = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": self.max_new_tokens,
                        "temperature": temperature,
                        "top_p": 1.0,
                    }
                    if stop is not None:
                        body["stop"] = stop
                else:  # /v1/completions
                    body = {
                        "model": self.model_name,
                        "prompt": item,
                        "max_tokens": self.max_new_tokens,
                        "temperature": temperature,
                        "top_p": 1.0,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                    }
                    if stop is not None:
                        body["stop"] = stop

                line = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": endpoint,
                    "body": body,
                }
                if metadata_per_item and i < len(metadata_per_item) and metadata_per_item[i]:
                    line["metadata"] = metadata_per_item[i]
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return custom_ids

    def create_batch(
        self,
        input_file_path,
        *,
        endpoint = "/v1/chat/completions",
        completion_window = "24h",
        metadata = None,
    ):
        """Uploads a JSONL file and creates a Batch job. Returns the Batch object."""
        batch_input_file = self.client.files.create(
            file=open(input_file_path, "rb"),
            purpose="batch",
        )
        return self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )

    def wait_for_batch(self, batch_id, poll_seconds: int = 10):
        """Polls the Batch until a terminal state. Returns final Batch object."""
        running = {"validating", "in_progress", "finalizing", "cancelling"}
        batch_obj = self.client.batches.retrieve(batch_id)
        while batch_obj.status in running:
            print(f"Current status: {batch_obj.status}")
            time.sleep(poll_seconds)
            batch_obj = self.client.batches.retrieve(batch_id)
        print(f"Final status: {batch_obj.status}")
        return batch_obj

    def download_batch_output_text(self, output_file_id: str) -> str:
        """Downloads the batch output file content as text."""
        file_response = self.client.files.content(output_file_id)
        # New SDK returns a response-like object with .text
        return getattr(file_response, "text", str(file_response))

    def parse_batch_output(self, output_text: str):
        """Parses the output JSONL into a mapping by custom_id.

        Returns a dict where each value is a dict with keys: response, error, and a
        convenience 'content' field containing text content when available.
        """
        results = {}
        for line in output_text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            response = obj.get("response")
            error = obj.get("error")
            content_val = None
            try:
                if response and isinstance(response, dict):
                    body = response.get("body") or {}
                    # Chat completions
                    choices = body.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                        if msg and isinstance(msg, dict):
                            content_val = msg.get("content")
                        # Completions legacy fallback
                        if content_val is None:
                            content_val = choices[0].get("text")
            except Exception:
                pass
            results[cid] = {"response": response, "error": error, "content": content_val}
        return results

    def batch_chat_generate_via_batch_api(
        self,
        messages_list,
        *,
        jsonl_path,
        temperature: float = 0.0,
        system_prompt = None,
        completion_window: str = "24h",
        metadata = None,
        poll_seconds: int = 10,
    ):
        """High-level convenience method to run a chat batch and return outputs in input order."""
        # 1) Build JSONL
        custom_ids = self.write_batch_jsonl(
            messages_list,
            jsonl_path,
            endpoint="/v1/chat/completions",
            temperature=temperature,
            system_prompt=system_prompt,
        )
        # 2) Create batch
        batch_obj = self.create_batch(
            jsonl_path,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )
        # 3) Wait for completion
        batch_obj = self.wait_for_batch(batch_obj.id, poll_seconds=poll_seconds)
        if getattr(batch_obj, "status", None) != "completed":
            raise RuntimeError(f"Batch did not complete successfully: {getattr(batch_obj, 'status', None)}")
        # 4) Download and parse output
        out_text = self.download_batch_output_text(batch_obj.output_file_id)
        mapping = self.parse_batch_output(out_text)
        # 5) Return aligned outputs in input order
        outputs = []
        for cid in custom_ids:
            entry = mapping.get(cid) or {}
            content = entry.get("content")
            if content is None:
                err = entry.get("error")
                outputs.append(f"<error: {err}>")
            else:
                outputs.append(str(content))
        return outputs
