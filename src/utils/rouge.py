# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import jsonlines
import wordninja
from tqdm import tqdm
from datasets import load_metric

def preprocess_predict(file, dataset):
    # load dictionary
    with open(file, 'r') as F:
        data = F.readlines()
        data = [item.strip('\n') for item in data]
        output = []
        if dataset == 'Vine':
            for line in tqdm(data):
                hashtags = line.split(' ')
                hashtags = [item.strip('#') for item in hashtags]
                sents = []
                for item in hashtags:
                    sents += wordninja.split(item)
                sents = ' '.join(sents)
                sents = sents.strip()
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
                hashtags = [item.strip('#') for item in hashtags]
                sents = []
                for item in hashtags:
                    sents += wordninja.split(item)
                sents = ' '.join(sents)
                output.append(sents)
        elif dataset == 'YFCC':
            for hashtags in tqdm(data):
                sents = ' '.join(hashtags)
                output.append(sents)
        return output

def calrouge(summary, reference, rouge):
    rouge.add_batch(predictions=summary, references=reference)
    final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
    R1_F1 = final_results["rouge1"].mid.fmeasure * 100
    R2_F1 = final_results["rouge2"].mid.fmeasure * 100
    RL_F1 = final_results["rougeL"].mid.fmeasure * 100
    R1_R = final_results["rouge1"].mid.recall * 100
    R2_R = final_results["rouge2"].mid.recall * 100
    RL_R = final_results["rougeL"].mid.recall * 100
    R1_P = final_results["rouge1"].mid.precision * 100
    R2_P = final_results["rouge2"].mid.precision * 100
    RL_P = final_results["rougeL"].mid.precision * 100
    return R1_F1, R2_F1, RL_F1, R1_R, R2_R, RL_R, R1_P, R2_P, RL_P

if __name__ == '__main__':
    # python src/utils/cal_rouge_bert_score.py -predict_path=./datasets/Vine/test_results/vision_audio_run13_seen_1.txt -golden_path=./datasets/Vine/final_split/test_seen.jsonl -dataset=Vine
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict_path', default='', type=str)
    parser.add_argument('-golden_path', default='', type=str)
    parser.add_argument('-dataset', default='Vine', type=str)

    args = parser.parse_args()
    
    rouge = load_metric('rouge', experiment_id='testset evaluation')

    predicts = preprocess_predict(args.predict_path, args.dataset)
    reference = preprocess_gold(args.golden_path, args.dataset)
    
    R1_F1, R2_F1, RL_F1, R1_R, R2_R, RL_R, R1_P, R2_P, RL_P = calrouge(predicts, reference, rouge)
    print(f'ROUGE-1 = {R1_F1}')
    print(f'ROUGE-2 = {R2_F1}')
    print(f'ROUGE-L = {RL_F1}')