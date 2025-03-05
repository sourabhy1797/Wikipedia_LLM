import os
import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq

df = pd.read_csv("clean_wikipedia_data.csv").head(100000)

df["query"] = "Answer the following question: " + df["title"]
df["answer"] = df["clean_text"]

dataset = Dataset.from_pandas(df[["query", "answer"]])

print("File reading complete")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def preprocess_function(examples):
    """Tokenizes inputs and outputs efficiently using batch processing."""
    inputs = [str(text) for text in examples["query"]]
    targets = [str(text) for text in examples["answer"]]

    model_inputs = tokenizer(
        inputs,
        max_length=128,  
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=128,  
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,  
    batch_size=256,  
    num_proc=1,  
    remove_columns=["query", "answer"]  
)

tokenized_datasets.save_to_disk("tokenized_wikipedia_flan_t5")
print("Tokenization complete")

dataset = load_from_disk("tokenized_wikipedia_flan_t5")
dataset = dataset.train_test_split(test_size=0.1)  # 90% train, 10% test
print("Tokenized Data Onboarded")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

training_args = TrainingArguments(
    output_dir="./flan_t5_wikipedia_model",
    evaluation_strategy="steps",  
    eval_steps=500,  
    save_strategy="steps",  
    save_steps=1000,  
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4, 
    num_train_epochs=3, 
    learning_rate=3e-4,  
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,  
    bf16=False,  
    save_total_limit=2,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    warmup_steps=1000, 
    lr_scheduler_type="linear", 
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Training Initiated")
trainer.train()

model.save_pretrained("./flan_t5_wikipedia_model")
tokenizer.save_pretrained("./flan_t5_wikipedia_model")

print("Training completed")

# def generate_answer(query):
#     device = model.device  
#     inputs = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)
#     inputs = {key: value.to(device) for key, value in inputs.items()}  
    
#     outputs = model.generate(
#         **inputs, max_length=150, min_length=20, do_sample=True, temperature=0.7
#     )
    
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test_query = "Answer the following question: What is artificial intelligence?"
# print("Generated Answer:", generate_answer(test_query))