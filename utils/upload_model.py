from huggingface_hub import HfApi
import argparse
import os

def upload_model_folder(model_path, repo_id):
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        path_in_repo=".",   
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model folder to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    args = parser.parse_args()

    upload_model_folder(args.model_path, args.repo_id)