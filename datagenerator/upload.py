from huggingface_hub import upload_folder
from huggingface_hub import HfApi
# upload_folder(
#     repo_id="MartinJYHuang/MultiturnJailbreak",  # 替换成你自己的用户名和数据集名
#     repo_type="dataset",
#     folder_path="mcts/mcts_data",    # 本地数据路径
#     commit_message="Initial upload"
# )

api = HfApi()

