import zipfile
import os
import fitz
import json
from sklearn.model_selection import train_test_split

def extract_zip(zip_path, extract_to):
    # Extracts the contents of a zip file to a target directory
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def extract_text_from_pdf(pdf_path):
    # Extracts text from a PDF file and removes unnecessary line breaks
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
            # Remove extra whitespaces
            text = ' '.join(text.split())
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def find_review_file(paper_path):
    # Find the review file corresponding to the paper
    review_path_txt = paper_path.replace("Paper", "Review").replace(".pdf", ".txt")
    if os.path.exists(review_path_txt):
        return review_path_txt
    
    # Try to find a file without extension
    review_path_no_ext = paper_path.replace("Paper", "Review").replace(".pdf", "")
    if os.path.exists(review_path_no_ext):
        return review_path_no_ext
    
    return None

def create_dataset(zip_files, output_train_json, output_val_json):
    # Create a dataset from zip files containing papers and their reviews
    dataset = []
    extract_dir = os.path.abspath("../../data/extracted_files")
    os.makedirs(extract_dir, exist_ok=True)

    total_papers = 0
    total_reviews = 0

    for zip_file in zip_files:
        print(f"Extracting {zip_file}...")
        extract_zip(zip_file, extract_dir)

    for root, dirs, _ in os.walk(extract_dir):
        for dir_name in dirs:
            # Ignore __MACOSX and other hidden directories
            if dir_name.startswith('.'):
                continue
            researcher_name = dir_name
            researcher_dir = os.path.join(root, dir_name)
            
            review_dirs = [d for d in os.listdir(researcher_dir) if os.path.isdir(os.path.join(researcher_dir, d))]
            for review_dir in review_dirs:
                review_path = os.path.join(researcher_dir, review_dir)
                paper_files = [f for f in os.listdir(review_path) if f.endswith(".pdf") and not f.startswith('._')]
                for paper_file in paper_files:
                    paper_path = os.path.join(review_path, paper_file)
                    review_text_path = find_review_file(paper_path)
                    print(f"Looking for review file: {review_text_path}")

                    if review_text_path is None or not os.path.exists(review_text_path):
                        print(f"Review file not found for {paper_path}")
                        continue

                    paper_text = extract_text_from_pdf(paper_path)
                    try:
                        with open(review_text_path, 'r', encoding='utf-8') as f:
                            review_text = f.read()
                    except Exception as e:
                        print(f"Error reading review {review_text_path}: {e}")
                        continue

                    dataset.append({
                        "researcher": researcher_name,
                        "review_id": review_dir,
                        "paper": paper_text,
                        "review": review_text
                    })
                    total_papers += 1
                    total_reviews += 1
                    print(f"Added paper and review for {paper_path}")

    # Split the dataset into train and test sets
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    output_dir = os.path.dirname(output_train_json)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    print(f"\nTraining dataset written to {output_train_json}")

    with open(output_val_json, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=4)
    print(f"Validation dataset written to {output_val_json}")

    if total_papers == total_reviews:
        print("All operations completed successfully! All papers have corresponding reviews.")
    else:
        print(f"Warning: Mismatch in total papers and reviews. Papers: {total_papers}, Reviews: {total_reviews}")

if __name__ == "__main__":
    # Zip files path
    zip_dir = os.path.abspath("../../data/Reviews_Dataset")
    zip_files = [
        os.path.join(zip_dir, file) for file in os.listdir(zip_dir) if file.endswith(".zip")
    ]
    print(f"Found {len(zip_files)} zip files")

    # Output JSON file path
    output_train_json = os.path.abspath("../../data/processed/train_dataset.json")
    output_test_json = os.path.abspath("../../data/processed/test_dataset.json")

    create_dataset(zip_files, output_train_json, output_test_json)
