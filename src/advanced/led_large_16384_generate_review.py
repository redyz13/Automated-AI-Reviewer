import json
import os
import torch
from transformers import LEDForConditionalGeneration, LEDTokenizer

# Load the model and tokenizer
def load_model(checkpoint):
    model = LEDForConditionalGeneration.from_pretrained(
        checkpoint,
    ).to("cuda").half()
    tokenizer = LEDTokenizer.from_pretrained(checkpoint)
    return model, tokenizer

# Generate a review for a given paper
def generate_review(model, tokenizer, paper_text):
    inputs = tokenizer(paper_text, return_tensors="pt", max_length=16384, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1
    
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        global_attention_mask=global_attention_mask,
        max_length=1024, 
        num_beams=5,
        repetition_penalty=1.3, 
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    
    review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return review

# Load the test dataset
def load_test_set(test_path):
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data

# Write the review to a file
def write_review_to_file(review, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(review)

if __name__ == "__main__":
    checkpoint = "../../models/led_large_16384_el=2.60"
    test_path = os.path.abspath("../../data/processed/test_dataset.json")
    output_review_path = os.path.abspath("../../results/generated_review.txt")

    model, tokenizer = load_model(checkpoint)

    test_data = load_test_set(test_path)

    paper_text = test_data[0]["paper"]

    print("Generating review...")

    review = generate_review(model, tokenizer, paper_text)
    write_review_to_file(review, output_review_path)
    print("Review generated and written to:", output_review_path)
