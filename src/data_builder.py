# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, ViTFeatureExtractor, CLIPProcessor, BartTokenizer
import soundfile as sf
from PIL import Image
from tqdm import tqdm
import torch
import jsonlines
import numpy as np
import wav2clip
import random
import wordninja
from transformers.models.bart.modeling_bart import shift_tokens_right

class OurDataset(Dataset):
    """Summarization dataset"""
    def __init__(self, args, mode):
        self.args = args
        self.processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        if self.args.knowledge_distillation:
            self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
        if self.args.hubert:
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        elif self.args.wavlm:
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        elif self.args.wav2clip:
            self.wav2clip_model = wav2clip.get_model()
        if mode == 'train':
            data_path = self.args.train_path
        elif mode == 'val':
            data_path = self.args.val_path
        elif mode == 'test':
            data_path = self.args.test_path
        self.ids, self.image_paths, self.wav_path, self.video_feature_path, self.audio_feature_path, self.tgt, self.constrains, self.descriptions = self.file_reader(data_path)
        print(f'Reading and processing {mode} done. Number of data sample: {len(self.image_paths)}')
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.ids[idx], self.image_paths[idx], self.wav_path[idx], self.video_feature_path[idx], self.audio_feature_path[idx], self.tgt[idx], self.constrains[idx], self.descriptions[idx]

    def file_reader(self, file_path):
        with jsonlines.open(file_path, 'r') as reader:
            id = []
            image_paths = []
            wav_path = []
            video_feature_path = []
            audio_feature_path = []
            tgt = []
            constrains = []
            descriptions = []
            for obj in reader:
                id.append(obj['id'])
                image_paths.append(obj['image_path'])
                wav_path.append(obj['wave_path'])
                video_feature_path.append(obj['video_feature_path'])
                audio_feature_path.append(obj['audio_feature_path'])
                tgt.append(obj['filtered_hashtag'])
                constrains.append(obj['50_constrains'])
                description = obj['text']
                description = " ".join([item for item in description.strip().split(' ') if len(item) != 0])
                description = " ".join([item for item in description.strip().split(' ') if item[0] != '#'])
                descriptions.append(description)
        return id, image_paths, wav_path, video_feature_path, audio_feature_path, tgt, constrains, descriptions

    def collate_fn(self, data):
        # rebuild the raw text and truncate to max length
        max_output_len = self.args.max_output_len
        batch_size = len(data)
        ids = [pair[0] for pair in data] # [1, 2, 3]
        image_paths = [pair[1] for pair in data] # [[], [], []]
        wav_path = [pair[2] for pair in data] # ['', '', '']
        fram_feature_paths = [pair[3] for pair in data] # [tensor1, tensor2, tensor3]
        audio_feature_path = [pair[4] for pair in data] # [tensor1, tensor2, tensor3]
        raw_tgt = [pair[5] for pair in data] # ['', '', '']
        descriptions = [pair[7] for pair in data] # ['', '', '']
        # shuffle order
        if self.args.label_shuffle:
            new_raw_tgt = []
            for item in raw_tgt:
                random.shuffle(item)
                new_raw_tgt.append(item)
            if self.args.one2one:
                raw_tgt = ['<s>'.join(item) for item in new_raw_tgt]
                tgt = [item[0] for item in new_raw_tgt]
            else:
                raw_tgt = ['<s>'.join(item) for item in new_raw_tgt]
        else:
            raw_tgt = ['<s>'.join(item) for item in raw_tgt]
        constrains = [pair[6] for pair in data] # ['', '', '']
        constrains = [item.replace('[sep]', '</s>') for item in constrains]

        # make pixel_values and encoder_attention_mask
        list_pixel_values = []
        list_encoder_attention_mask = []
        clip_list_pixel_values = []
        for i in range(len(fram_feature_paths)):
            if self.args.use_frame_features:
                one_id_features_path = fram_feature_paths[i]
                one_batch_pixel_values = torch.load(one_id_features_path) # [N, 3, 244, 244]
            else:
                images = [Image.open(path) for path in image_paths[i]]
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                one_batch_pixel_values = inputs.pixel_values
                if self.args.knowledge_distillation:
                    clip_inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
                    one_batch_clip_pixel_values = clip_inputs.pixel_values
            if len(one_batch_pixel_values) >= self.args.frame_number:
                one_batch_pixel_values = one_batch_pixel_values[:self.args.frame_number]
                list_pixel_values.append(one_batch_pixel_values)
                list_encoder_attention_mask.append(torch.ones(self.args.frame_number*197))
                if self.args.knowledge_distillation:
                    one_batch_clip_pixel_values = one_batch_clip_pixel_values[:self.args.frame_number]
                    clip_list_pixel_values.append(one_batch_clip_pixel_values)
            else:
                original_frames = len(one_batch_pixel_values)
                left_frames = self.args.frame_number - len(one_batch_pixel_values)
                left_frams_pixel_values = torch.zeros([left_frames, 3, 224, 224])
                one_batch_pixel_values = torch.cat((one_batch_pixel_values, left_frams_pixel_values))
                list_pixel_values.append(one_batch_pixel_values)
                list_encoder_attention_mask.append(torch.cat((torch.ones(original_frames*197), torch.zeros(left_frames*197))))
                if self.args.knowledge_distillation:
                    one_batch_clip_pixel_values = torch.cat((one_batch_clip_pixel_values, left_frams_pixel_values))
                    clip_list_pixel_values.append(one_batch_clip_pixel_values)
        pixel_values = torch.stack(list_pixel_values)
        encoder_attention_mask = torch.stack(list_encoder_attention_mask) # this is for images mask
        if self.args.knowledge_distillation:
            clip_pixel_values = torch.stack(clip_list_pixel_values)
        else:
            clip_pixel_values = None

        # make audio input and audio length
        if self.args.wav2clip:
            audio_input = []
            for path in wav_path:
                speech, sampling_rate = sf.read(path)
                # speech = torch.tensor(speech, dtype=torch.float32)
                speech = np.array(speech, dtype=np.float32)
                embeddings = wav2clip.embed_audio(speech, self.wav2clip_model)
                audio_input.append(torch.tensor(embeddings))
            audio_input = torch.stack(audio_input,0)
            audio_len = []
        elif self.args.use_audio:
            audio_features = [torch.load(item) for item in audio_feature_path] # [B, N, 2015]
            max_audio_feature_len = min([max([len(item) for item in audio_features]), self.args.max_audio_feature_len])
            audio_input = torch.zeros([batch_size, max_audio_feature_len, self.args.d_audio_input])
            audio_len = []
            for i in range(batch_size):
                audio_input[i][:audio_features[i].shape[0]] = audio_features[i][:max_audio_feature_len]
                audio_len.append(min([audio_features[i].shape[0], max_audio_feature_len]))
        else:
            audio_input = None
            audio_len = None
        
        # make constrains
        if self.mode == 'train':
            new_constrains = []
            for i in range(len(constrains)):
                this_constrain = constrains[i]
                this_constrain_list = this_constrain.split('</s>')
                this_label = raw_tgt[i]
                this_label_list = this_label.split('<s>')
                this_new_constrain_list = []
                for hashtag in this_constrain_list:
                    if hashtag not in this_label_list:
                        this_new_constrain_list.append(hashtag)
                    else:
                        if np.random.choice([0,1,2]) in [1,2]:
                            this_new_constrain_list.append(hashtag)
                if len(this_new_constrain_list) >= 45:
                    new_constrains.append('</s>'.join(this_new_constrain_list[:45]))
                else:
                    new_constrains.append('</s>'.join(this_new_constrain_list))
            constrains = new_constrains
        else:
            constrains = ["</s>".join(item.split('</s>')[:45]) for item in constrains]
        final_text_input = []
        for i in range(len(descriptions)):
            final_text_input.append('Descriptions: ' + descriptions[i] + " Related Hashtags: " + constrains[i].replace('</s>',' ')) #for vine cggm
        constrains_input = self.tokenizer(final_text_input, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors='pt')
        if self.args.one2one:
            raw_tgt_ids = self.tokenizer(tgt, add_special_tokens=True, padding=True, truncation=True, max_length=self.args.max_output_len, return_tensors='pt')['input_ids']
        else:
            raw_tgt_ids = self.tokenizer(raw_tgt, add_special_tokens=True, padding=True, truncation=True, max_length=self.args.max_output_len, return_tensors='pt')['input_ids']
        tgt_ids = shift_tokens_right(raw_tgt_ids, 1, 2)
        raw_tgt_ids[raw_tgt_ids[:, :] == 1] = -100
        raw_tgt = [item.replace('<s>', ' ') for item in raw_tgt]
        return {'ids': ids,
                'pixel_values': pixel_values,
                'encoder_attention_mask': encoder_attention_mask,
                'audio_input': audio_input,
                'audio_len': audio_len, # list of length
                'label_ids':raw_tgt_ids,
                'labels': raw_tgt,
                'constrains_input': constrains_input,
                'clip_pixel_values': clip_pixel_values,
                'decoder_ids': tgt_ids,}