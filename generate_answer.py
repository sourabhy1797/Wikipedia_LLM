import os
import torch
import pandas as pd
import time
from datasets import Dataset, load_from_disk
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)

tokenizers = {
    "flan_t5": T5Tokenizer.from_pretrained("google/flan-t5-base"),
    "bart": BartTokenizer.from_pretrained("facebook/bart-base")
}

models = {
    "flan_t5": T5ForConditionalGeneration.from_pretrained("./flan_t5_wikipedia_model"),
    "bart": BartForConditionalGeneration.from_pretrained("./bart_wikipedia_model")
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models.values():
    model.to(device)

def generate_model_answers(query):
    """
    Generates an answer separately from FLAN-T5, BART, and the Ensemble model.
    """
    model_outputs = {}
    
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()} 
        
        output_tokens = model.generate(**inputs, max_length=256)
        decoded_output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        model_outputs[model_name] = decoded_output  
    
    ensemble_output = " ".join(model_outputs.values())
    model_outputs["ensemble"] = ensemble_output
    
    return model_outputs  


test_query = "Answer the following question: What is artificial intelligence?"
model_outputs = generate_model_answers(test_query)

for model_name, answer in model_outputs.items():
    print(f"Generated Answer ({model_name}):", answer)





