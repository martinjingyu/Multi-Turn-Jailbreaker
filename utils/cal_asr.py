
import os
import json

def read_json_files_from_dir(dir_path: str, sort_files: bool = True) -> list[dict]:

    json_data_list = []

    filenames = [f for f in os.listdir(dir_path) if f.endswith('.json')]

    if sort_files:
        filenames.sort()

    for filename in filenames:
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                json_data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error in file: {filename}, error: {e}")

    return json_data_list

if __name__ == "__main__":
    list_json = read_json_files_from_dir("data/result/MartinJYHuang/JA-v1-Llama-3.1-8B-Instruct")
    total_4 = 0
    total_5 = 0
    for item in list_json:
        has_4 = False
        has_5 = False
        for node in item:
            if node["reward"] == "4.0":
                has_4 = True
            elif node["reward"] == "5.0":
                has_5 = True
        if has_4:
            total_4 += 1
        if has_5: 
            total_5 += 1
    print(total_4)
    print(total_5)