import json
import os
import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from termcolor import colored

# Load the validation dataset
def load_dataset(test_path):
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data

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

    # Load metric calculators
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

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

if __name__ == "__main__":
    model_path = "../../models/t5_base_el=2.94"

    test_path = os.path.abspath("../../data/processed/test_dataset.json")
    test_data = load_dataset(test_path)

    test_dataset = convert_to_dataset(test_data)

    # Load the T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Define the evaluation arguments
    training_args = TrainingArguments(
        output_dir="../../results",
        per_device_eval_batch_size=8,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    metrics = trainer.evaluate()

    print(colored(f"\nValidation complete!", "green"))

    trainer.log_metrics("eval", metrics)