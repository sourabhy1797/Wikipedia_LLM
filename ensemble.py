import os
import torch
import pandas as pd
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

def generate_ensemble_answer(query):
    """
    Generates an answer using ensemble of FLAN-T5 and BART models.
    Uses per-word probability voting to determine the best output.
    """
    model_outputs = {}
    token_probs = {}
    max_length = 256  
    
    
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  
        
        output_tokens = model.generate(**inputs, max_length=max_length, return_dict_in_generate=True, output_scores=True)
        token_ids = output_tokens.sequences[0]
        probabilities = torch.nn.functional.softmax(output_tokens.scores[0], dim=-1)  
        
        decoded_output = tokenizer.decode(token_ids, skip_special_tokens=True)
        model_outputs[model_name] = token_ids.tolist()  
        token_probs[model_name] = probabilities if probabilities.shape[0] > 0 else torch.zeros((1, probabilities.shape[-1]), device=device)  # Ensure no empty array
    
    final_output = []
    min_length = min(len(model_outputs["bart"]), len(model_outputs["flan_t5"]))  
    for word_pos in range(min_length):
        best_model = max(models.keys(), key=lambda model: token_probs[model][word_pos].max().item() if word_pos < token_probs[model].shape[0] else 0)
        best_token_id = model_outputs[best_model][word_pos] if word_pos < len(model_outputs[best_model]) else None
        
        if best_token_id is not None:
            best_token = tokenizers[best_model].decode([best_token_id])  
            final_output.append(best_token)
    
    return " ".join(final_output) 

test_query = "Answer the following question: What is artificial intelligence?"
print("Generated Answer (Ensemble):", generate_ensemble_answer(test_query))