import json
import os
from peft import PeftModel
from torch import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from termcolor import colored

# Load the model and tokenizer
def load_model(checkpoint):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False,
    )
    device_map = {"": torch.cuda.current_device()}
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        quantization_config=quantization_config,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer

# Chunk the paper text into smaller sections
def chunk_paper(paper_text, chunk_size=512, overlap=50):
    words = paper_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Generate a review section progressively for each given promp using chunked paper text
def generate_review_section_progressive(model, tokenizer, paper_text, section_prompt, chunk_size=512, overlap=50, num_tokens_per_chunk=25):
    paper_chunks = chunk_paper(paper_text, chunk_size=chunk_size, overlap=overlap)
    review_section = ""

    # Process each chunk and generate progressively
    for chunk_idx, chunk in enumerate(paper_chunks):
        # Prepare the input by concatenating the section prompt and the current chunk of the paper
        inputs_text = f"Instruction: {section_prompt}\n\nContext: {chunk}"

        print(f"Input to model for chunk {chunk_idx + 1}:\n{inputs_text}\n")
        
        # Tokenize the chunk
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True, padding="max_length", max_length=chunk_size)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generate the progressive text for the chunk
        outputs = model.base_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=num_tokens_per_chunk,
            num_beams=5, 
            repetition_penalty=1.3, 
            no_repeat_ngram_size=3, 
            early_stopping=True
        )

        # Decode the generated text and append it to the review section
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        review_section += generated_text + " "

        print(colored(f"Processed chunk {chunk_idx + 1}/{len(paper_chunks)}", "green"))
        
        print(colored('Chunk generated:', 'blue'), f"{generated_text}\n")

    return review_section.strip()

# Write the review in a structured format to a file
def generate_structured_review_progressive(model, tokenizer, paper_text, output_path, num_tokens_per_chunk=50):
    sections = {
        "summary": "Provide a concise summary of the research paper. Include the main objectives, methodology, results, and conclusions.",
        "methodological_soundness": "Evaluate the methodological soundness of the study. Are the research methods appropriate and well-documented? If there are deviations from empirical standards, are they reasonably justified?",
        "scientific_contribution": "Assess the scientific contribution of the paper. Does the study provide novel insights or improvements in the field of software engineering?",
        "evidence_robustness": "Evaluate the robustness of the evidence provided in the paper. Are the claims appropriately validated, and is there transparency in the presentation of results?",
        "limitations": "Identify the limitations of the study and discuss how they are addressed by the authors. Are there significant threats to validity?",
        "reporting_quality": "Evaluate the quality of reporting in the paper. Is the research well-structured and clear? Does the paper provide sufficient supplementary materials for replication?",
        "strengths": "Highlight the strengths of the study. What aspects of the methodology or results stand out?",
        "weaknesses": "Identify the weaknesses of the study. What aspects could be improved, and where does the paper fall short of empirical standards?",
        "conclusion": "Summarize the overall assessment of the paper. Provide recommendations for the authors to improve the paper and suggest if the paper should be accepted, revised, or rejected."
    }

    review = {}

    for section, prompt in sections.items():
        print(f"Generating section: {section}")
        
        section_text = ""
        section_text += generate_review_section_progressive(model, tokenizer, paper_text, prompt, num_tokens_per_chunk=num_tokens_per_chunk)

        if section not in review:
            review[section] = f"\n\n[{section.upper()}]\n{section_text}"

            with open(output_path, "a", encoding="utf-8") as output_file:
                output_file.write(review[section])
        
        print(f"Section {section.upper()} generated.")

    return review

# Load the test dataset
def load_test_set(test_path):
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return test_data

if __name__ == "__main__":
    checkpoint = "../../models/checkpoint-6505"
    test_path = os.path.abspath("../../data/processed/test_dataset.json")
    output_review_path = os.path.abspath("../../results/generated_review.txt")

    model, tokenizer = load_model(checkpoint)

    model = PeftModel.from_pretrained(model, checkpoint)

    test_data = load_test_set(test_path)

    paper_text = test_data[0]["paper"]

    print("Generating review...")

    review = generate_structured_review_progressive(model, tokenizer, paper_text, output_review_path, num_tokens_per_chunk=100)
    
    print("Review generated and written to:", output_review_path)
