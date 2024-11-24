import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
def load_model(checkpoint):
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    return model, tokenizer

# Generate a review for a given paper
def generate_review(model, tokenizer, paper_text):
    inputs = tokenizer(paper_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        max_length=512, 
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
    checkpoint = "../../models/t5_base_el=2.94"
    test_path = os.path.abspath("../../data/processed/test_dataset.json")
    output_review_path = os.path.abspath("../../results/generated_review.txt")

    model, tokenizer = load_model(checkpoint)

    test_data = load_test_set(test_path)

    paper_text = test_data[0]["paper"]

    review = generate_review(model, tokenizer, paper_text)
    write_review_to_file(review, output_review_path)
    print("Review generated and written to:", output_review_path)