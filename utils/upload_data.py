from huggingface_hub import HfApi
import os

def upload_data(path):
    api = HfApi()
    api.upload_folder(
        folder_path=f"{path}",
        path_in_repo=".",
        repo_id="MartinJYHuang/jailbreak-agent",
        repo_type="dataset",
    )
    
def upload_single_data(folder_path, file_name):
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=os.path.join(folder_path, file_name),
        path_in_repo=os.path.relpath(folder_path, "data"),
        repo_id="MartinJYHuang/jailbreak-agent",
        repo_type="dataset",
    )
    
if __name__ == "__main__":
    # upload_single_data("./data/raw_data/test/rollout5","tree.json")
    upload_data("data")