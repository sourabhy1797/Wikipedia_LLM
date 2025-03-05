import os
import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq


df = pd.read_csv("clean_wikipedia_data.csv").head(100000)

df["query"] = "Answer the following question: " + df["title"]
df["answer"] = df["clean_text"]

dataset = Dataset.from_pandas(df[["query", "answer"]])

print("File reading complete")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


def preprocess_function(examples):
    """Tokenizes inputs and outputs efficiently using batch processing."""
    
    inputs = [str(text) for text in examples["query"]]
    targets = [str(text) for text in examples["answer"]]

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,  
    batch_size=512,  
    num_proc=1,  
    remove_columns=["query", "answer"]  
)

tokenized_datasets.save_to_disk("tokenized_wikipedia_bart")
print("Tokenization complete")


dataset = load_from_disk("tokenized_wikipedia_bart")
dataset = dataset.train_test_split(test_size=0.1)  
print("Tokenized Data Onboarded")


model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# training_args = TrainingArguments(
#     output_dir="./bart_wikipedia_model",
#     evaluation_strategy="epoch",  
#     save_strategy="epoch",
#     per_device_train_batch_size=32,  
#     per_device_eval_batch_size=32,
#     gradient_accumulation_steps=2,  
#     num_train_epochs=3,
#     learning_rate=3e-4,  
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=250,  
#     bf16=True,  
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     remove_unused_columns=False  
# )

training_args = TrainingArguments(
    output_dir="./bart_wikipedia_model",
    evaluation_strategy="steps",  
    save_strategy="steps",  
    save_steps=1000, 
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  
    num_train_epochs=5,  
    learning_rate=5e-5, 
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,  
    bf16=False, 
    save_total_limit=3,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    warmup_ratio=0.1,  
    lr_scheduler_type="cosine",  
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

model.save_pretrained("./bart_wikipedia_model")
tokenizer.save_pretrained("./bart_wikipedia_model")

print("Training completed")


# def generate_answer(query):
#     inputs = tokenizer(query, return_tensors="pt", max_length=256, truncation=True)
#     outputs = model.generate(**inputs)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# query = "Answer the following question: What is artificial intelligence?"
# print("Generated Answer:", generate_answer(query))