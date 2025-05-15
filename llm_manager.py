import os
import gc
import json
import torch
import pandas as pd
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig, 
    logging as hf_logging
)
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

tqdm.pandas()
load_dotenv()
hf_logging.set_verbosity_error() 

os.environ["OPENAI_API_KEY"] = os.getenv('First_Key')
client_openai = OpenAI(api_key= os.getenv('First_Key') )

llama_pipeline = None
qwen_pipeline = None
qwen_4b_pipeline = None
mistral_pipeline = None
deepseek_qwen_pipeline = None

# Run prompt on LLM, for every text
# Odd function that uses get_funct from llm_manager.py
# I started with the function and did not have time to clean this up
def run_prompt_on_llm(get_funct, directory, prompt, tweets):
    os.makedirs(f"data/{directory}", exist_ok=True)
    responses = []
    for i, row in tqdm(tweets.iterrows(), total=tweets.shape[0]):                          
        text = row["text"]
        i_prompt = prompt.replace("{{tweet_text}}", text)
        response = get_funct(i_prompt)
        # save each to file in data/directory
        with open(f"data/{directory}/{i}.json", "w") as f:
            f.write(response)
        responses.append(response)
    return responses

###############################################################################################
# PAID SERVICES
###############################################################################################
def get_openai_gpt4omini_response(prompt):
    response = client_openai.responses.create(
        model="gpt-4o-mini", #"gpt-4o-mini""o3-mini" "o3-mini-2025-01-31" "gpt-4.1-nano" "gpt-4.1-mini"
        input=prompt,
    )
    return response.output_text

def get_openai_o4mini_response(prompt):
    response = client_openai.responses.create(
        model="o4-mini", #"gpt-4o-mini""o3-mini" "o3-mini-2025-01-31" "gpt-4.1-nano" "gpt-4.1-mini"
        input=prompt,
    )
    return response.output_text

###############################################################################################
# LOCAL MODELS
###############################################################################################
def get_llama_response(prompt):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    global llama_pipeline
    if llama_pipeline is None:
        llama_pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            #device_map="auto",
            device=0,
        )
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = llama_pipeline(
        messages,
        max_new_tokens=2048,
    )
    return response[0]["generated_text"][-1]["content"]

def get_deepseek_qwen_response(prompt):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    global deepseek_qwen_pipeline
    if deepseek_qwen_pipeline is None:
        deepseek_qwen_pipeline = transformers.pipeline(
            "text-generation", 
            device=0,
            model=model_id
        )
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = deepseek_qwen_pipeline(
        messages,
        max_new_tokens=2048,
    )
    return response[0]["generated_text"][-1]["content"]


def get_qwen_response(prompt):
    model_id = "Qwen/Qwen3-8B"  #"qwen/qwen2.5-7b-instruct" # Qwen/Qwen2.5-14B-Instruct
    global qwen_pipeline
    if qwen_pipeline is None:
        qwen_pipeline = transformers.pipeline(
            "text-generation", 
            device=0,
            model=model_id
        )
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = qwen_pipeline(
        messages,
        max_new_tokens=2048,
    )
    return response[0]["generated_text"][-1]["content"]

def get_qwen_4b_response(prompt):
    model_id = "Qwen/Qwen3-4B"
    global qwen_4b_pipeline
    if qwen_4b_pipeline is None:
        qwen_4b_pipeline = transformers.pipeline(
            "text-generation", 
            device=0,
            model=model_id
        )
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = qwen_4b_pipeline(
        messages,
        max_new_tokens=2048,
    )
    return response[0]["generated_text"][-1]["content"]


def get_mistral_response(prompt):
    model_id = "mistralai/Mistral-7B-Instruct-v0.3" #"mistralai/Mistral-7B-v0.3" mistralai/Ministral-8B-Instruct-2410
    global mistral_pipeline
    if mistral_pipeline is None:
        mistral_pipeline = transformers.pipeline(
            "text-generation", 
            device=0,
            model=model_id
        )
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = mistral_pipeline(
        messages,
        #max_new_tokens=2048,
    )
    return response[0]["generated_text"][-1]["content"]

###############################################################################################
# CLEANUP
###############################################################################################

def cleanup_llama():
    global llama_pipeline
    if llama_pipeline is not None:
        # ── unload everything the pipeline pinned to GPU ──
        llama_pipeline.model.to("cpu")
        llama_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_qwen():
    global qwen_pipeline
    if qwen_pipeline is not None:
        # ── unload everything the pipeline pinned to GPU ──
        qwen_pipeline.model.to("cpu")
        qwen_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_qwen_4b():
    global qwen_4b_pipeline
    if qwen_4b_pipeline is not None:
        # ── unload everything the pipeline pinned to GPU ──
        qwen_4b_pipeline.model.to("cpu")
        qwen_4b_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_mistral():
    global mistral_pipeline
    if mistral_pipeline is not None:
        # ── unload everything the pipeline pinned to GPU ──
        mistral_pipeline.model.to("cpu")
        mistral_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()

def cleanup_deepseek_qwen():
    global deepseek_qwen_pipeline
    if deepseek_qwen_pipeline is not None:
        # ── unload everything the pipeline pinned to GPU ──
        deepseek_qwen_pipeline.model.to("cpu")
        deepseek_qwen_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()


###############################################################################################
# other paid services
###############################################################################################
# from anthropic import Anthropic
# client_claude = Anthropic(api_key=os.getenv('anthropic-first-key'))
# client_perplexity = OpenAI(api_key=os.getenv('perplexity-api-key'), base_url="https://api.perplexity.ai")
#!huggingface-cli login --token os.getenv('my-token')


# def get_claude_response(prompt):
#     response = client_claude.messages.create(
#         model="claude-3-7-sonnet-20250219",
#         max_tokens=32000,
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.content[0].text

# def get_perplexity_response(prompt):
#     response = client_perplexity.chat.completions.create(
#         model="sonar-pro", # sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro, sonar-deep-research
#         messages = [
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content
