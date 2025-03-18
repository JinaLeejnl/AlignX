import json

patha = "/eval_data/Llama-3.1-8B-Instruct-PRISM.json"
pathb = "/eval_data/Llama-3.1-8B-Instruct-ICA-PRISM.json"

dataA = []
dataB = []

with open(patha, 'r') as f:
    for line in f:
        dataA.append(json.loads(line))

with open(pathb, 'r') as f:
    for line in f:
        dataB.append(json.loads(line))

dataA.sort(key=lambda x: x['idx'])
dataB.sort(key=lambda x: x['idx'])

a_idx_set = set(item['idx'] for item in dataB)

dataA = [item for item in dataA if item['idx'] in a_idx_set]

count = 0
num = 0

for i, aa in enumerate(dataA):
    if aa["idx"] != dataB[i]["idx"]:
        continue

    chosen_rewards = 0.1 * ((-dataB[i]["nll_loss_all_chosen"]) - (-aa["nll_loss_all_chosen"]))
    rejected_rewards = 0.1 * ((-dataB[i]["nll_loss_all_rejected"]) - (-aa["nll_loss_all_rejected"]))

    if chosen_rewards>rejected_rewards:
        count += 1
        dataB[i]["predict"] = True
    else:
        dataB[i]["predict"] = False
    
    num += 1

print(num)
print(count)
print("acc:", count/num)