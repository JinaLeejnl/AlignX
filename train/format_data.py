import os
import pandas as pd
from profile_utils import get_profile

folder_path = "/AlignX"

parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

data_list = []
print(len(parquet_files))

for idx, file in enumerate(parquet_files):
    file_path = os.path.join(folder_path, file)
    data = pd.read_parquet(file_path)
    data_list.extend(data.to_dict(orient='records'))
    print(idx)
    
print(len(data_list))

import json

res = []

for item in data_list:
    newitem = {}

    task = f'{item["prompt"]}\n\n'

    num_ugc_to_keep = min(random.randint(0, 4), len(item["User-Generated Content"]))
    num_pair_to_keep = min(random.randint(0, 4), len(item["Pair-wise Comparative Feedback"]))

    if not (num_ugc_to_keep or num_pair_to_keep):
        num_ugc_to_keep = min(4, len(item["User-Generated Content"]))
        num_pair_to_keep = min(4, len(item["Pair-wise Comparative Feedback"]))

    if num_ugc_to_keep > 0:
        random_ugc = random.sample(item["User-Generated Content"], num_ugc_to_keep)
  
    if num_pair_to_keep > 0:
        random_pair = random.sample(item["Pair-wise Comparative Feedback"], num_pair_to_keep)

    history = random_ugc + random_pair
    profile = get_profile(history)


    pba_prompt = (
        "<|start_header_id|>system<|end_header_id|>\n\nGenerate a task-specific response based on user preferences.\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"***Task***\n\n{task}"
        f"***User Preferences***\n\n{profile}\n\n***Response:***\n\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )

    ugcs = ""
    pairs = ""

    if num_ugc_to_keep > 0:
        ugcs = f"**This person has commented on some posts:**\n\n"
        for i, it in enumerate(random_ugc):
            ugcs = (
                f"{ugcs}"
                f"{i+1}. *Post:*\n{it['prompt']}\n\n"
                f"*Comment:*\n{it['comment']}\n\n"
            )

    if num_pair_to_keep > 0:
        pairs = f"**This person has chosen or rejected comments on some posts:**\n\n"
        for i, it in enumerate(random_pair):
            pairs = (
                f"{pairs}"
                f"{i+1}. *Post:*\n{it['prompt']}\n\n"
                f"*Chosen:*\n{it['chosen']}\n\n"
                f"*Rejected:*\n{it['rejected']}\n\n"
            )

    chosen = item['chosen']
    rejected = item['rejected']

    ica_prompt = (
        "<|start_header_id|>system<|end_header_id|>\n\nGenerate a task-specific response based on user historical behavior.\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"***Task***\n\n{task}"
        f"***Historical Behavior***\n\n{ugcs}{pairs}\n\n***Response:***\n\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )

    newitem["ica_prompt"] = ica_prompt
    newitem["pba_prompt"] = pba_prompt
    newitem["chosen"] = chosen
    newitem["rejected"] = rejected

    res.append(newitem)

with open("/train.json", 'w') as f:
    json.dump(res, f, indent=4)