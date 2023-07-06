# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from attr import s
import wordninja
import pytorch_lightning as pl
from transformers import CLIPTokenizer
from models.modeling_clip import CLIPModel
from datasets import load_metric
import torch

class BaseModelCLIP(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = CLIPModel.from_pretrained(self.args.model)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.args.model)
        self.rouge = load_metric('rouge', experiment_id=self.args.log_name[-3:])
        self.sigm = torch.nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, pixel_values, encoder_attention_mask, audio_input, audio_len):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values, 
                            encoder_attention_mask=encoder_attention_mask, 
                            audio_input=audio_input, 
                            audio_len=audio_len,
                            return_loss=True,
                            )
        return output

    def training_step(self, batch, batch_idx):
        # get loss
        # print('============================================================ Testing case ============================================================')
        # print(batch)
        # print(f'ids = {batch["ids"]}')
        # print(f'pixel_values = {batch["pixel_values"].shape}')
        # print(f'encoder_attention_mask = {batch["encoder_attention_mask"].shape}')
        # print(f'audio_input = {batch["audio_input"].shape}')
        # print(f'audio_len = {batch["audio_len"]}')
        # print(f'label_ids = {batch["label_ids"].shape}')
        # print('======================================================================================================================================')
        # exit()
        output = self(input_ids=batch['label_ids'],
                    attention_mask=batch['decoder_attention_mask'],
                    pixel_values=batch['pixel_values'], 
                    encoder_attention_mask=batch['encoder_attention_mask'], 
                    audio_input=batch['audio_input'], 
                    audio_len=batch['audio_len'], 
                    )
        loss = output.loss
        # logs
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(input_ids=batch['label_ids'],
                    attention_mask=batch['decoder_attention_mask'],
                    pixel_values=batch['pixel_values'], 
                    encoder_attention_mask=batch['encoder_attention_mask'], 
                    audio_input=batch['audio_input'], 
                    audio_len=batch['audio_len'], 
                    )
        loss = output.loss

        return [loss]

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([item[0] for item in outputs]))
        self.log('validation/val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        output = self(input_ids=batch['label_ids'],
                    attention_mask=batch['decoder_attention_mask'],
                    pixel_values=batch['pixel_values'], 
                    encoder_attention_mask=batch['encoder_attention_mask'], 
                    audio_input=batch['audio_input'], 
                    audio_len=batch['audio_len'], 
                    )
        print(output)
        exit()

    def test_epoch_end(self, outputs):
        predicts = []
        for item in outputs:
            predicts += item[0]
        self.save_txt(self.args.test_save_file, predicts)

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
