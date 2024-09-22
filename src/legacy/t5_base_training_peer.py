import json
import os
import pandas as pd
import torch.optim as optim
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, TrainerCallback, get_linear_schedule_with_warmup
from termcolor import colored

# Load the training and validation datasets
def load_datasets(train_path, test_path):
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding='utf-8') as f:
        test_data = json.load(f)
    
    return train_data, test_data

# Convert the dataset to Datasets library format
def convert_to_dataset(data):
    flattened_data = []
    for item in data:
        paper = (item["paper"])
        review = (item["review"])
        flattened_data.append({
            "paper": paper,
            "review": review
        })
    return Dataset.from_pandas(pd.DataFrame(flattened_data))

# Tokenize the dataset
def preprocess_function(examples):
    inputs = [ex for ex in examples["paper"]]
    targets = [ex for ex in examples["review"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Compute the metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    # Convert predictions to the correct format if necessary
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert predictions to numpy array and take the argmax
    preds = np.argmax(preds, axis=-1)

    # Decode the predicted and actual labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute metrics
    bleu_result = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    # Return metrics in a readable format
    metrics = {
        'eval_bleu': bleu_result['bleu'],
        'eval_meteor': meteor_result['meteor'],
        'eval_bertscore_precision': np.mean(bertscore_result['precision']),
        'eval_bertscore_recall': np.mean(bertscore_result['recall']),
        'eval_bertscore_f1': np.mean(bertscore_result['f1']),
    }

    return metrics

# Custom callback for printing metrics
class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            epoch = state.epoch
            metrics = logs

            if 'eval_loss' in metrics:
                print(colored(f"\n\nEpoch: {epoch}", "blue"))
                print(colored(f"- Loss: {metrics.get('eval_loss', 'N/A')}", "green"))
                print(colored(f"- BLEU: {metrics.get('eval_bleu', 'N/A')}", "yellow"))
                print(colored(f"- METEOR: {metrics.get('eval_meteor', 'N/A')}", "yellow"))
                print(colored(f"- BERTScore Precision: {metrics.get('eval_bertscore_precision', 'N/A')}", "yellow"))
                print(colored(f"- BERTScore Recall: {metrics.get('eval_bertscore_recall', 'N/A')}", "yellow"))
                print(colored(f"- BERTScore F1: {metrics.get('eval_bertscore_f1', 'N/A')}\n", "yellow"))

if __name__ == "__main__":
    train_path = os.path.abspath("../data/processed/peer_train_dataset.json")
    test_path = os.path.abspath("../data/processed/peer_test_dataset.json")

    # Load the training and test datasets
    train_data, test_data = load_datasets(train_path, test_path)

    train_dataset = convert_to_dataset(train_data)
    test_dataset = convert_to_dataset(test_data)

    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Print dataset features and sizes
    print(colored(f"Train Dataset Features: {datasets['train'].features}", "cyan"))
    print(colored(f"Test Dataset Features: {datasets['test'].features}", "cyan"))
    print(colored(f"Train Dataset Size: {len(datasets['train'])}", "cyan"))
    print(colored(f"Test Dataset Size: {len(datasets['test'])}\n", "cyan"))

    # Load the T5 model and tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="../results", # The output directory
        eval_strategy="epoch", # Evaluation is done at the end of each epoch
        num_train_epochs=5, # Number of training epochs
        learning_rate=3e-4, # Learning rate
        per_device_train_batch_size=4, # Batch size for training
        per_device_eval_batch_size=4, # Batch size for evaluation
        weight_decay=0.01, # Weight decay to avoid overfitting
        save_total_limit=5, # Limit the total number of checkpoints
        save_strategy="epoch", # Save a checkpoint at the end of each epoch
        logging_steps=10, # Log metrics every 10 steps
        load_best_model_at_end=True, # Load the best model when training ends
        metric_for_best_model="eval_loss", # Use loss to determine the best model
        greater_is_better=False, # A lower loss is better
        dataloader_drop_last=False, # Consider last batch even if it's smaller
    )

    # Define the optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    total_steps = len(tokenized_datasets["train"]) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Define the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    # Load metric calculators
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].shuffle(seed=42),
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[early_stopping_callback, PrintMetricsCallback()],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()

    print(colored(f"\nTraining complete!", "green"))

    trainer.log_metrics("eval", metrics)