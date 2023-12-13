import argparse, os, sys
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.distributed as td

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKEND = "nccl" if torch.cuda.is_available() else "gloo"

def training_pipeline():
    """Container for the entire finetuning script for the Verte model. Any external call to Verte training must call this method. 

    Args :
        Port(str): Port String for the distributed framework.  
    Returns :
        None
    
    """
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    prompt = "So how can I best use the Minstral ai for my project?"
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=1024)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

if __name__ == "__main__":
    training_pipeline()