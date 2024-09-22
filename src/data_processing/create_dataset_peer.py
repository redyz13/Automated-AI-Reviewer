import os
import json
from sklearn.model_selection import train_test_split

def load_paper_review_pairs(parsed_pdfs_dir, reviews_dir):
    dataset = []
    
    # Loop over all files in the parsed_pdfs directory
    for pdf_file in os.listdir(parsed_pdfs_dir):
        if pdf_file.endswith(".json"):
            # Full paths to the parsed pdf and corresponding review
            pdf_path = os.path.join(parsed_pdfs_dir, pdf_file)
            review_file = pdf_file.replace(".pdf.json", ".json")
            review_path = os.path.join(reviews_dir, review_file)
            
            # Check if corresponding review exists
            if os.path.exists(review_path):
                with open(pdf_path, 'r', encoding='utf-8') as pdf_f:
                    paper_data = json.load(pdf_f)
                
                with open(review_path, 'r', encoding='utf-8') as review_f:
                    review_data = json.load(review_f)
                
                # Safely extract sections, if available
                sections = paper_data.get('metadata', {}).get('sections', [])
                if sections:  # Only process if sections exist
                    paper_text = " ".join([section.get('text', '') for section in sections])
                    review_text = " ".join([rev.get('comments', '') for rev in review_data.get('reviews', [])])
                    
                    dataset.append({
                        "paper": paper_text,
                        "review": review_text
                    })
                else:
                    print(f"No sections found in paper {pdf_file}")
            else:
                print(f"Review file not found for {pdf_file}")
    
    return dataset

def create_unified_dataset(root_dir, output_train_json, output_val_json):
    dataset = []

    # Iterate over all datasets
    for dataset_name in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset_name)
        if os.path.isdir(dataset_path):
            print(f"Processing dataset: {dataset_name}")
            for subset in ['train', 'dev', 'test']:
                subset_dir = os.path.join(dataset_path, subset)
                parsed_pdfs_dir = os.path.join(subset_dir, "parsed_pdfs")
                reviews_dir = os.path.join(subset_dir, "reviews")
                
                subset_data = load_paper_review_pairs(parsed_pdfs_dir, reviews_dir)
                dataset.extend(subset_data)
    
    if not dataset:
        print("No data collected. Exiting.")
        return

    # Split the combined dataset into train and validation
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Save to output JSON files
    output_dir = os.path.dirname(output_train_json)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_train_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    print(f"\nTraining dataset written to {output_train_json}")
    
    with open(output_val_json, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=4)
    print(f"Validation dataset written to {output_val_json}")

if __name__ == "__main__":
    # Root directory containing the datasets
    root_dir = "../../data/PeerRead/data"
    
    # Output JSON file paths
    output_train_json = "../../data/processed/peer_train_dataset.json"
    output_val_json = "../../data/processed/peer_test_dataset.json"
    
    create_unified_dataset(root_dir, output_train_json, output_val_json)
