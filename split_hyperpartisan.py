import random
import json
import random
import os
max_seq_length = 1024
random.seed(42)
data_dir = 'hyperpartisan'


label_0_list = []
label_1_list = []
L = []
for line in open(os.path.join(data_dir, "original_data.txt")):
    line = line.strip()
    a = line.find(' ')
    b = line.find(' ', a+1)

    label = line[a+1: b]
    text = line[b+1: ]
    
    label = int(label)
    if label==0:
        label_0_list.append({'text': text, 'label': label})
    elif label==1:
        label_1_list.append({'text': text, 'label': label})
    else: 
        assert(False)

l_0 = len(label_0_list)
l_1 = len(label_1_list)


random.shuffle(label_0_list)
random.shuffle(label_1_list)

num_0_dev = int(l_0/10 + 0.5)
num_1_dev = int(l_0/10 + 0.5)

train_set = label_0_list[ : -2*num_0_dev] + label_1_list[ : -2*num_1_dev]
dev_set = label_0_list[-2*num_0_dev : -num_0_dev] + label_1_list[-2*num_1_dev : -num_1_dev]
test_set = label_0_list[-num_0_dev : ] + label_1_list[-num_1_dev : ]


if not os.path.exists(data_dir):
    os.mkdir(data_dir)

for name, dataset in zip(['train.jsonl', 'dev.jsonl', 'test.jsonl'], [train_set, dev_set, test_set]):
    w = open(os.path.join(data_dir, name), "w")
    for item in dataset:
        w.write(json.dumps(item)+'\n')


