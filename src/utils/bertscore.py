# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import jsonlines
from tqdm import tqdm
from datasets import load_metric
import numpy as np

def preprocess_predict(file, dataset):
    # load dictionary
    with open(file, 'r') as F:
        data = F.readlines()
        data = [item.strip('\n') for item in data]
        output = []
        if dataset == 'Vine':
            for line in tqdm(data):
                if line[-1] == '#':
                    sents = line[:-1]
                else:
                    sents = line
                output.append(sents)
        elif dataset == 'YFCC':
            for line in tqdm(data):
                hashtags = line.split('[sep]')
                sents = ' '.join(hashtags)
                sents = sents.strip()
                output.append(sents)
        return output

def preprocess_gold(file, dataset):
    # load dictionary
    with jsonlines.open(file, 'r') as F:
        data = [obj for obj in F]
        data = [item['filtered_hashtag'] for item in data]
        
        output = []
        if dataset == 'Vine':
            for hashtags in tqdm(data):
                hashtags = [item for item in hashtags]
                sents = ' '.join(hashtags)
                output.append(sents)
        elif dataset == 'YFCC':
            for hashtags in tqdm(data):
                sents = ' '.join(hashtags)
                output.append(sents)
        return output

if __name__ == '__main__':
    # python src/utils/bertscore.py -predict_path=./datasets/Vine/test_results/vision_audio_run13_seen_1.txt -golden_path=./datasets/Vine/final_split/test_seen.jsonl -dataset=Vine
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict_path', default='', type=str)
    parser.add_argument('-golden_path', default='', type=str)
    parser.add_argument('-dataset', default='Vine', type=str)

    args = parser.parse_args()

    bertscore = load_metric("bertscore")
    predicts = preprocess_predict(args.predict_path, args.dataset)
    reference = preprocess_gold(args.golden_path, args.dataset)
    
    results = bertscore.compute(predictions=predicts, references=reference, model_type="microsoft/deberta-xlarge-mnli")
    bert_f1score = np.mean(results['f1'])
    print(f'bert score = {bert_f1score*100}')

