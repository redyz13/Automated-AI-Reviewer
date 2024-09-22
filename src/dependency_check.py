import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def check_transformers():
    try:
        model_name = "philschmid/flan-t5-xxl-sharded-fp16"
        print("Transformers imported successfully.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Transformers is working correctly with the T5 model.")
    except Exception as e:
        print(f"Error with Transformers: {e}")

def check_torch():
    try:
        print("Torch imported successfully.")
        if torch.cuda.is_available():
            print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
    except Exception as e:
        print(f"Error with Torch: {e}")

if __name__ == "__main__":
    print("Dependency check for T5 model fine-tuning task:")
    print("------------------------------------------------------")
    check_transformers()
    print("------------------------------------------------------")
    check_torch()
    print("------------------------------------------------------")
