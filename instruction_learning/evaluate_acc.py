import json
import os
import re

for file in os.listdir('./results/'):
    with open(os.path.join('./results/', file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    correct, invalid = 0, 0
    for item in data:

        response = item['response'] if 'response' in item.keys() else item['predict']
        try:
            correct += int(response) == item['label']
        except ValueError:
            pattern = re.compile(r'\d+')
            predict = pattern.findall(response)
            if not predict:
                invalid += 1
            # the qwen3-8B need predict[-1]
            correct += (int(predict[0] if predict else -1)) == item['label']

            
    print(f'{".".join(file.split(".")[:-1])} Acc : {correct / len(data)} ivalid: {invalid}')


            