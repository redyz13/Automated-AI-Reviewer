import threading
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import torch
import fitz
from peft import PeftModel

class ModelManager:
    def __init__(self, callback):
        self.callback = callback
        self.model = None
        self.tokenizer = None
        self.model_thread = threading.Thread(target=self.load_model)
        self.model_thread.start()

    def load_model(self):
        checkpoint = "../../models/flan_t5_xxl_el=2.09"
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {'': torch.cuda.current_device()}
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            device_map=device_map
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.model = PeftModel.from_pretrained(model, checkpoint)
        # Call the callback function to notify that the model has been loaded
        self.callback()

    def model_inference(self, paper_text):
        inputs = self.tokenizer(paper_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model.base_model.generate(
            input_ids, attention_mask=attention_mask, max_length=512, num_beams=5,
            repetition_penalty=1.3, top_p=0.9, top_k=50, no_repeat_ngram_size=3,
            early_stopping=False, do_sample=True
        )
        review = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return review

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with fitz.open(pdf_path) as pdf:
                for page in pdf:
                    text += page.get_text()
                text = ' '.join(text.split())
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text

    def write_review_to_file(self, review, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(review)

    def load_test_set(self, test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        return test_data
