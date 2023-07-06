# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from itertools import count
import jsonlines
import numpy as np
from tqdm import tqdm
import jsonlines

def preprocess_predict(file, dataset):
    # load dictionary
    with open(file, 'r') as F:
        data = F.readlines()
        data = [item.strip('\n').strip() for item in data]
        removed_tagend_data = []
        if dataset == 'Vine':
            for item in data:
                if item[-1] == '#':
                    removed_tagend_data.append(item[:-1].strip())
                else:
                    removed_tagend_data.append(item)
            output = []
            for line in tqdm(removed_tagend_data):
                hashtags = line.split(' ')
                output.append(hashtags)
        elif dataset == 'YFCC':
            output = []
            for line in tqdm(data):
                hashtags = line.split('[sep]')
                output.append(hashtags)
        return output

def preprocess_gold(file):
    # load dictionary
    with jsonlines.open(file, 'r') as F:
        data = [obj for obj in F]        
        data = [item['filtered_hashtag'] for item in data]
        output = []
        for hashtags in tqdm(data):
            output.append(hashtags)
        return output

def cal_f1_k1(predicts, reference):
    p = []
    r = []
    for i in range(len(predicts)):
        counter_p = 0
        for item in predicts[i]:
            if item in reference[i]:
                counter_p += 1
        predicts_length = len(predicts[i])
        if predicts_length > 0:
            this_p = counter_p/len(predicts[i])
        else:
            this_p = 0
        p.append(this_p)

        counter_r = 0
        for item in reference[i]:
            if item in predicts[i]:
                counter_r += 1
        this_r = counter_r/len(reference[i])
        r.append(this_r)
    p = np.mean(p)
    r = np.mean(r)
    f1 = (2*p*r)/(p+r)
    return p, r, f1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict_path', default='', type=str)
    parser.add_argument('-golden_path', default='', type=str)
    parser.add_argument('-k', default=1, type=int)
    parser.add_argument('-dataset', default='Vine', type=str)

    args = parser.parse_args()

    predicts = preprocess_predict(args.predict_path, args.dataset)
    reference = preprocess_gold(args.golden_path)
    
    print(f'first 10 predicts = {predicts[:10]}')
    print(f'first 10 references = {reference[:10]}')
    p, r, f1 = cal_f1_k1(predicts, reference)
    print(f'F1@1 = {f1*100}')
    print(f'P1@1 = {p*100}')
    print(f'R1@1 = {r*100}')