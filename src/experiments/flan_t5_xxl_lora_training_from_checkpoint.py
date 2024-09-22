import json
import os
import pandas as pd
import torch.optim as optim
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, TrainerCallback, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from termcolor import colored
from torch import torch

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
        flattened_data.append({
            "paper": item["paper"],
            "review": item["review"]
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
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    # Return metrics in a readable format
    metrics = {
        'eval_bleu': bleu_result['bleu'],
        'eval_meteor': meteor_result['meteor'],
        'eval_bertscore_precision': np.mean(bertscore_result['precision']),
        'eval_bertscore_recall': np.mean(bertscore_result['recall']),
        'eval_bertscore_f1': np.mean(bertscore_result['f1']),
        'eval_rouge1': rouge_result['rouge1'],
        'eval_rouge2': rouge_result['rouge2'],
        'eval_rougeL': rouge_result['rougeL'],
        'eval_rougeLsum': rouge_result['rougeLsum'],
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
                print(colored(f"- BERTScore F1: {metrics.get('eval_bertscore_f1', 'N/A')}", "yellow"))
                print(colored(f"- ROUGE-1: {metrics.get('eval_rouge1', 'N/A')}", "yellow"))
                print(colored(f"- ROUGE-2: {metrics.get('eval_rouge2', 'N/A')}", "yellow"))
                print(colored(f"- ROUGE-L: {metrics.get('eval_rougeL', 'N/A')}", "yellow"))
                print(colored(f"- ROUGE-Lsum: {metrics.get('eval_rougeLsum', 'N/A')}\n", "yellow"))

def load_checkpoint_info(checkpoint_path):
    trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(trainer_state_file):
        with open(trainer_state_file, "r") as f:
            trainer_state = json.load(f)
            epoch = trainer_state['epoch']
            global_step = trainer_state['global_step']
            best_metric = trainer_state['best_metric'] if 'best_metric' in trainer_state else None
            return epoch, global_step, best_metric
    else:
        print(colored(f"Warning: trainer_state.json not found in {checkpoint_path}", "red"))
        return None, None, None

if __name__ == "__main__":
    train_path = os.path.abspath("../../data/processed/train_dataset.json")
    test_path = os.path.abspath("../../data/processed/test_dataset.json")

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

    # Load the T5 model and tokenizer from checkpoint
    checkpoint_path = "../../results/checkpoint"
    model_name = "philschmid/flan-t5-xxl-sharded-fp16"

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    device_map = {'' : torch.cuda.current_device()}

    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint_path,
        quantization_config=quantization_config,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Apply Lora
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none"
    )
    # Prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)

    # Add Lora adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="../../results", # The output directory
        eval_strategy="epoch", # Evaluation is done at the end of each epoch
        num_train_epochs=8, # Continue training for a total of 8 epochs
        learning_rate=1e-4, # Learning rate
        per_device_train_batch_size=1, # Batch size for training
        per_device_eval_batch_size=1, # Batch size for evaluation
        weight_decay=0.01, # Weight decay to avoid overfitting
        save_total_limit=5, # Limit the total number of checkpoints
        save_strategy="epoch", # Save a checkpoint at the end of each epoch
        logging_steps=10, # Log metrics every 10 steps
        load_best_model_at_end=True, # Load the best model when training ends
        metric_for_best_model="eval_loss", # Use loss to determine the best model
        greater_is_better=False, # A lower loss is better
        dataloader_drop_last=True, # Consider last batch even if it's smaller
        resume_from_checkpoint=checkpoint_path, # Resume training from the checkpoint
    )

    # Print information about the checkpoint and training
    epoch, global_step, best_metric = load_checkpoint_info(checkpoint_path)
    print(colored(f"\nResuming training from checkpoint: {checkpoint_path}", "cyan"))
    print(colored(f"Checkpoint info - Epoch: {epoch}, Step: {global_step}, Best Metric: {best_metric}", "cyan"))
    print(colored(f"Total epochs: {training_args.num_train_epochs}", "cyan"))
    print(colored(f"Learning rate: {training_args.learning_rate}", "cyan"))
    print(colored(f"Train batch size: {training_args.per_device_train_batch_size}", "cyan"))
    print(colored(f"Eval batch size: {training_args.per_device_eval_batch_size}\n", "cyan"))

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
    rouge = evaluate.load("rouge")

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

    print(colored(f"\nOptimizer: {trainer.optimizer}\n", "cyan"))

    # Train the model
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # Evaluate the model
    metrics = trainer.evaluate()

    print(colored(f"\nTraining complete!", "green"))

    trainer.log_metrics("eval", metrics)
