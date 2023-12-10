import argparse, os, sys
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.distributed as td

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKEND = "nccl" if torch.cuda.is_available() else "gloo"

def training_pipeline(**args):
    """Container for the entire finetuning script for the Verte model. Any external call to Verte training must call this method. 

    Args :
        Port(str): Port String for the distributed framework.  
    Returns :
        None
    
    """
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser
    parser.add_argument('-p', '--port', help="Designated port for selected distributed framework of training.", default="12055")
    args = parser.parse_args()
    training_pipeline(args)