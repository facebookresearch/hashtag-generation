# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from attr import s
import wordninja
import pytorch_lightning as pl
from transformers import GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel, BartTokenizer, BartModel, BartForConditionalGeneration, ViTModel
from models.fid import BartForMultiConditionalGeneration
from datasets import load_metric
import torch
from models.model_clip import BaseModelCLIP
import numpy as np

class BaseModel(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = BartForMultiConditionalGeneration.from_pretrained("facebook/bart-base", args=self.args)
        self.model.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # for param in self.model.vit.parameters(): # for vg_bart, comment when use cggm
        #         param.requires_grad = False
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.rouge = load_metric('rouge', experiment_id=self.args.log_name[-3:])
        
        if self.args.knowledge_distillation:
            class Args:
                model = 'openai/clip-vit-base-patch14'
                log_name= '/checkpoints/tiezheng/HashtagGeneration/VideoCLIP/vision_only/test'
                checkpoint_name = '/checkpoints/tiezheng/HashtagGeneration/VideoCLIP/vision_only/run5/lightning_logs/version_1/checkpoints/epoch=2-step=4068.ckpt'
            args_clip=Args()
            clip_model = BaseModelCLIP.load_from_checkpoint(args_clip.checkpoint_name, args=args_clip)
            print('?'*100)
            self.model.clip_vision_model = clip_model.model.vision_model
            self.model.visual_projection = clip_model.model.visual_projection
            for param in self.model.clip_vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, encoder_attention_mask, audio_input, audio_len, label_ids, constrains_input, clip_pixel_values,decoder_input_ids):
        # loss = self.model(input_ids=constrains_input.input_ids, attention_mask=constrains_input.attention_mask, decoder_input_ids=decoder_input_ids, labels=label_ids)[0]
        loss = self.model(pixel_values=pixel_values, 
                          video_attention_mask=encoder_attention_mask, 
                          audio_input=audio_input, 
                          audio_len=audio_len,
                          input_ids=constrains_input.input_ids,
                          attention_mask=constrains_input.attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          labels=label_ids,
                          clip_pixel_values=clip_pixel_values,).loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(pixel_values=batch['pixel_values'], 
                    encoder_attention_mask=batch['encoder_attention_mask'], 
                    audio_input=batch['audio_input'], 
                    audio_len=batch['audio_len'], 
                    label_ids=batch['label_ids'], 
                    constrains_input=batch['constrains_input'],
                    clip_pixel_values=batch['clip_pixel_values'],
                    decoder_input_ids=batch['decoder_ids'])
        # logs
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.one2one:
            summary_ids = self.model.generate(
                                                pixel_values=batch['pixel_values'],
                                                video_attention_mask=batch['encoder_attention_mask'],
                                                audio_input=batch['audio_input'],
                                                audio_len=batch['audio_len'],
                                                input_ids=batch['constrains_input'].input_ids,
                                                attention_mask=batch['constrains_input'].attention_mask,
                                                num_beams=self.args.n_beams,
                                                max_length=self.args.max_output_len,
                                                min_length=self.args.min_output_len,
                                                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                                num_return_sequences=5)
        else:
            summary_ids = self.model.generate(
                                                pixel_values=batch['pixel_values'],
                                                video_attention_mask=batch['encoder_attention_mask'],
                                                audio_input=batch['audio_input'],
                                                audio_len=batch['audio_len'],
                                                input_ids=batch['constrains_input'].input_ids,
                                                attention_mask=batch['constrains_input'].attention_mask,
                                                num_beams=self.args.n_beams,
                                                max_length=self.args.max_output_len,
                                                min_length=self.args.min_output_len,
                                                no_repeat_ngram_size=self.args.no_repeat_ngram_size,)
                                                
        return [summary_ids, batch['labels']]

    def validation_epoch_end(self, outputs):
        if self.args.one2one:
            summary = []
            reference = []
            for item in outputs:
                summary_id = item[0]
                one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_id]
                batch_final_summary = []
                counter = 0
                one_item_summary = ''
                for this_summary in one_summary:
                    this_summary = this_summary.replace('<s>','')
                    one_item_summary = one_item_summary + this_summary + '<s>'
                    counter += 1
                    if counter == 5:
                        batch_final_summary.append(one_item_summary)
                        counter = 0
                        one_item_summary = ''
                summary += batch_final_summary
                reference += item[1]
        else:
            summary = []
            reference = []
            for item in outputs:
                summary_id = item[0]
                one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in summary_id] # </s><s>nyc</s><pad>
                summary += one_summary
                reference += item[1]
        p, r, f1 = self.cal_f1_k1(summary,reference)
        R1_F1, R2_F1, RL_F1, R1_R, R2_R, RL_R, R1_P, R2_P, RL_P = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge/f1', f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge1_F1', R1_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge2_F1', R2_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rougeL_F1', RL_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge1_R', R1_R, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge2_R', R2_R, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rougeL_R', RL_R, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge1_P', R1_P, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge2_P', R2_P, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rougeL_P', RL_P, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.val_save_file, summary)

    def test_step(self, batch, batch_idx):
        loss = self(pixel_values=batch['pixel_values'], 
                    encoder_attention_mask=batch['encoder_attention_mask'], 
                    audio_input=batch['audio_input'], 
                    audio_len=batch['audio_len'], 
                    label_ids=batch['label_ids'], 
                    constrains_input=batch['constrains_input'],
                    clip_pixel_values=batch['clip_pixel_values'],
                    decoder_input_ids=batch['decoder_ids'])
        if self.args.one2one:
            summary_ids = self.model.generate(
                                                pixel_values=batch['pixel_values'],
                                                video_attention_mask=batch['encoder_attention_mask'],
                                                audio_input=batch['audio_input'],
                                                audio_len=batch['audio_len'],
                                                input_ids=batch['constrains_input'].input_ids,
                                                attention_mask=batch['constrains_input'].attention_mask,
                                                num_beams=self.args.n_beams,
                                                max_length=self.args.max_output_len,
                                                min_length=self.args.min_output_len,
                                                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                                num_return_sequences=5)
        else:
            summary_ids = self.model.generate(
                                                pixel_values=batch['pixel_values'],
                                                video_attention_mask=batch['encoder_attention_mask'],
                                                audio_input=batch['audio_input'],
                                                audio_len=batch['audio_len'],
                                                input_ids=batch['constrains_input'].input_ids,
                                                attention_mask=batch['constrains_input'].attention_mask,
                                                num_beams=self.args.n_beams,
                                                max_length=self.args.max_output_len,
                                                min_length=self.args.min_output_len,
                                                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                                length_penalty=self.args.length_penalty,)
        return [summary_ids, batch['labels'],loss]

    def test_epoch_end(self, outputs):
        if self.args.one2one:
            summary = []
            reference = []
            for item in outputs:
                summary_id = item[0]
                one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_id]
                batch_final_summary = []
                counter = 0
                one_item_summary = ''
                for this_summary in one_summary:
                    this_summary = this_summary.replace('<s>','')
                    one_item_summary = one_item_summary + this_summary + '<s>'
                    counter += 1
                    if counter == 5:
                        batch_final_summary.append(one_item_summary)
                        counter = 0
                        one_item_summary = ''
                summary += batch_final_summary
                reference += item[1]
        else:
            summary = []
            reference = []
            loss = []
            for item in outputs:
                summary_id = item[0]
                one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in summary_id] # </s><s>nyc</s><pad>
                summary += one_summary
                reference += item[1]
                loss.append(item[2].detach().cpu().numpy().tolist())
        p, r, f1 = self.cal_f1_k1(summary,reference)
        R1_F1, R2_F1, RL_F1, R1_R, R2_R, RL_R, R1_P, R2_P, RL_P = self.calrouge(summary, reference, self.rouge)
        self.log('test_Rouge/loss', np.mean(loss), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/f1', f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge1_F1', R1_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge2_F1', R2_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rougeL_F1', RL_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge1_R', R1_R, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge2_R', R2_R, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rougeL_R', RL_R, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge1_P', R1_P, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge2_P', R2_P, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rougeL_P', RL_P, on_epoch=True, prog_bar=True, sync_dist=True)
        if 'Vine' in self.args.train_path:
            summary = [item.replace('<s>',' ').replace('</s>', '').replace('<pad>', '').strip() for item in summary]
            summary = [item.strip(' ') for item in summary]
        elif 'YFCC100M' in self.args.train_path:
            summary = [item.replace('<pad>', ' ').strip() for item in summary]
        self.save_txt(self.args.test_save_file, summary)
# ============================================================== Useful Functions ==============================================================
    def hashtag2str(self, input):
        output = []
        for line in input:
            hashtags = line.split(' ')
            hashtags = [item.strip('#') for item in hashtags]
            sents = []
            for item in hashtags:
                sents += wordninja.split(item)
            sents = ' '.join(sents)
            sents = sents.strip()
            output.append(sents)
        return output

    def calrouge(self, summary, reference, rouge):
        summary = [item.replace('<s>',' ').replace('</s>',' ').replace('<pad>', ' ').strip() for item in summary]
        reference = [item.replace('<s>',' ').replace('</s>',' ').replace('<pad>', ' ').strip() for item in reference]
        summary = self.hashtag2str(summary)
        reference = self.hashtag2str(reference)
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

    def preprocess_f1(self, data):
        output = []
        for line in data:
            line = line.strip('<s>')
            hashtags = line.split('<s>')
            output.append(hashtags)
        return output
        
    def cal_f1_k1(self, predicts, reference):
        predicts = [item.replace('<pad>', ' ').replace('</s>', ' ').strip(' ').strip('<s>') for item in predicts]
        reference = [item.replace(' ','<s>').strip() for item in reference]
        predicts = self.preprocess_f1(predicts)
        reference = self.preprocess_f1(reference)
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
        f1 = (2.0*p*r)/(p+r)
        return p*100, r*100, f1*100

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return [optimizer]

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # warm up lr
        if self.trainer.global_step < self.args.warmup:
            lr_scale = min(1.0, max(0.2,float(self.trainer.global_step + 1) / self.args.warmup))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.args.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
