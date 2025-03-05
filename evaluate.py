import os
import torch
import pandas as pd
import time
from datasets import Dataset, load_from_disk
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)
from evaluate import load

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
        model_outputs[model_name] = decoded_output  
    
    return model_outputs  

def evaluate_model_efficiency(model_name, model, tokenizer, query):
    """Evaluates inference time and number of parameters for a given model."""
    inputs = tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  
    
    start_time = time.time()
    output_tokens = model.generate(**inputs, max_length=256)
    end_time = time.time()

    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    inference_time = end_time - start_time
    num_params = sum(p.numel() for p in model.parameters())

    return {
        "Model": model_name,
        "Inference Time (s)": inference_time,
        "Num Parameters": num_params,
        "Generated Output": output_text
    }

results = []
test_query = "Answer the following question: What is artificial intelligence?"
for model_name, model in models.items():
    results.append(evaluate_model_efficiency(model_name, model, tokenizers[model_name], test_query))

ensemble_outputs = generate_ensemble_answer(test_query)
ensemble_output = " ".join(ensemble_outputs.values())
results.append({
    "Model": "Ensemble",
    "Inference Time (s)": "N/A",
    "Num Parameters": "N/A",
    "Generated Output": ensemble_output
})

reference = ["Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn."] * len(results)
predictions = [r["Generated Output"] for r in results]

rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")

rouge_scores = [rouge.compute(predictions=[pred], references=[ref]) for pred, ref in zip(predictions, reference)]
bleu_scores = [bleu.compute(predictions=[pred], references=[ref]) for pred, ref in zip(predictions, reference)]
meteor_scores = [meteor.compute(predictions=[pred], references=[ref]) for pred, ref in zip(predictions, reference)]

# import ace_tools as tools
df_results = pd.DataFrame(results)
metrics_df = pd.DataFrame([{**r, **b, **m} for r, b, m in zip(rouge_scores, bleu_scores, meteor_scores)])

metrics_df.to_csv('metric.csv')
# tools.display_dataframe_to_user(name="Model Efficiency Comparison", dataframe=df_results)
# tools.display_dataframe_to_user(name="Evaluation Metrics", dataframe=metrics_df)

print("Generated Answer (Ensemble):", ensemble_output)
