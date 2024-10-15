import os
import json
import numpy as np
from datasets import load_dataset
import argparse

# Metadata for datasets including dataset name and the column name to extract
dataset_metas = [
    ("Cohere/wikipedia-22-12-simple-embeddings", "emb"),
    ("MongoDB/subset_arxiv_papers_with_embeddings", "embedding"),
    ("MongoDB/airbnb_embeddings", "text_embeddings"),
]

# Function to calculate the precision of floating-point numbers and dump to JSON
def calculate_precision_and_dump_to_json(dataset_metas, output_dir):
    precision_data = {}
    
    for dataset_name, column_name in dataset_metas:
        try:
            # Load the dataset
            dataset = load_dataset(dataset_name)
            column_data = dataset['train'][column_name]
            
            # Calculate the precision of floating-point numbers
            def calculate_precision(value):
                str_value = f"{value:.16f}".rstrip('0')
                if '.' in str_value:
                    return len(str_value.split('.')[1])
                return 0
            
            precisions = [calculate_precision(x) for array in column_data for x in array]
            n = np.percentile(precisions, 90)
            dataset_key = f"{dataset_name.replace('/', '_')}_{column_name}"
            precision_data[dataset_key] = n
            
            # Write the embedding data to a binary file
            output_path = os.path.join(output_dir, f"{dataset_key}.bin")
            with open(output_path, "wb") as bin_file:
                np.array(column_data, dtype=np.float64).tofile(bin_file)
            
            print(f"Processed dataset '{dataset_name}' with column '{column_name}'")
        
        except KeyError:
            print(f"Column '{column_name}' not found in dataset '{dataset_name}'")
        except Exception as e:
            print(f"Failed to process dataset '{dataset_name}': {e}")
    
    # Dump precision data to JSON
    with open(os.path.join(output_dir, "precision_data.json"), "w") as json_file:
        json.dump(precision_data, json_file, indent=4)

# Main function to parse command-line arguments and execute the processing
def main():
    parser = argparse.ArgumentParser(description="Process datasets and calculate precision.")
    parser.add_argument("output_dir", type=str, help="Directory to output the .bin files and precision_data.json")
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute the function with the provided output directory
    calculate_precision_and_dump_to_json(dataset_metas, args.output_dir)

if __name__ == "__main__":
    main()