from huggingface_hub import snapshot_download



if __name__ == "__main__":
    

    local_path = snapshot_download(
        repo_id="MartinJYHuang/jailbreak-seed", 
        repo_type="dataset",
        local_dir="./data/prompt_seed",
        local_dir_use_symlinks=False  
    )

