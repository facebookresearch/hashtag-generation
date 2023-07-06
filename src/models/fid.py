# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from models.modeling_wavlm import WavLMModel
from transformers.modeling_outputs import BaseModelOutput
from models.modeling_hubert import HubertModel
from models.audio_transformer import AudioTransformerEncoder
from transformers import BartForConditionalGeneration

class BartForMultiConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.vision_projector = nn.Linear(self.args.d_model, self.args.d_model)
        if self.args.use_audio:
            if self.args.hubert:
                self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
                self.audio_projecter = nn.Linear(self.args.d_model, self.args.d_model)
                self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_model, self.args.d_model)
                self.fusion_weight = nn.Parameter(torch.Tensor([self.args.reweight]))
            elif self.args.wavlm:
                self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
                self.audio_projecter = nn.Linear(self.args.d_model, self.args.d_model)
                self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_model, self.args.d_model)
                self.fusion_weight = nn.Parameter(torch.Tensor([self.args.reweight]))
            elif self.args.wav2clip:
                if self.args.fusion_approach == 'dot_product':
                    self.audio_projecter = nn.Linear(512, self.args.d_model)
                    self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_model, self.args.d_model)
                elif self.args.fusion_approach == 'cross_attention':
                    # self._vtlinear_1_audio_ = nn.Linear(512, self.args.d_model) # K
                    # self._vtlinear_2_audio_ = nn.Linear(512, self.args.d_model) # V
                    # self._vtlinear_3_audio_ = nn.Linear(self.args.d_model, self.args.d_model) # Q
                    # self.multi_head_attn_audio_ = torch.nn.MultiheadAttention(embed_dim=self.args.d_model, num_heads=self.args.num_fusion_heads, batch_first=True)
                    self.audio_projecter = nn.Linear(512, self.args.d_model)
                    self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_model, self.args.d_model)
            else:
                self.audio_transformer = AudioTransformerEncoder(d_model=self.args.d_audio_input, num_layers=self.args.num_layers, num_heads=self.args.num_heads, dim_feedforward=self.args.dim_feedforward)
                self.audio_projecter = nn.Linear(self.args.d_audio_input, self.args.d_model)
                self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_model, self.args.d_model)
                self.fusion_weight = nn.Parameter(torch.Tensor([self.args.reweight]))
                self.layer_norm = nn.LayerNorm(self.args.d_audio_input)
                if self.args.fusion_approach == 'multi_head':
                    self._linear_1 = nn.Linear(self.args.d_model, self.args.d_common) # K
                    self._linear_2 = nn.Linear(self.args.d_model, self.args.d_common) # V
                    self._linear_3 = nn.Linear(self.args.d_model, self.args.d_common) # Q
                    self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim=self.args.d_common, num_heads=self.args.num_fusion_heads, batch_first=True)
                    self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_common, self.args.d_model)
        if self.args.constrained_generation and 'cross_attention' in self.args.fusion_approach:
            self._vtlinear_1 = nn.Linear(self.args.d_model, self.args.d_model) # K
            self._vtlinear_2 = nn.Linear(self.args.d_model, self.args.d_model) # V
            self._vtlinear_3 = nn.Linear(self.args.d_model, self.args.d_model) # Q
            self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim=self.args.d_model, num_heads=self.args.num_fusion_heads, batch_first=True)
            self._vt_attn_layer_norm = nn.LayerNorm(self.args.d_model)
        elif self.args.constrained_generation and 'dot_product' in self.args.fusion_approach:
            self.fusion_layer = nn.Linear(self.args.d_model + self.args.d_model, self.args.d_model)

    def multi_encode(
        self,
        pixel_values=None,
        video_attention_mask=None,
        input_ids=None,
        attention_mask=None,
        return_dict=None,
        audio_input=None,
        audio_len=None,
        constrains_input=None,
        clip_pixel_values=None,
    ):
        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D)
        # (B, N, L) -> (B*N, L) -> (B, N*L)
        distillation_loss = None
        B = pixel_values.size(0)  # batch-size
        N = pixel_values.size(1)  # num-image
        C = pixel_values.size(2)  # num-channel
        H = pixel_values.size(3)  # H
        W = pixel_values.size(4)  # W
        pixel_values = pixel_values.contiguous().view(B * N, C, H, W)
        encoder_outputs = self.vit(
            pixel_values=pixel_values,
            return_dict=return_dict,
            output_hidden_states=True,
        )
        if return_dict:
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]
        # hidden_states: (B * N, 197, 768)
        L = hidden_states.size(1)
        D = hidden_states.size(2)
        stacked_source_reps = hidden_states.contiguous().view(B, N * L, D)
        stacked_source_reps = self.vision_projector(stacked_source_reps)
        '''
        # ================== knowledge distillation ===========================
        if self.args.knowledge_distillation and clip_pixel_values != None:
            # generetive_hidden_states = encoder_outputs.hidden_states[-2].contiguous().view(B, N * L, D)
            clip_pixel_values = clip_pixel_values.contiguous().view(B * N, C, H, W)
            vision_outputs = self.clip_vision_model(
                                                    pixel_values=clip_pixel_values,
                                                    output_attentions=False,
                                                    output_hidden_states=True,
                                                    return_dict=return_dict,
                                                    )
            generetive_features = encoder_outputs.last_hidden_state[:, 0, :]  # [120,768]
            clip_pooler_output = vision_outputs.pooler_output
            clip_features = self.visual_projection(clip_pooler_output) # [120,768]
            clip_attention_mask = video_attention_mask.contiguous().view(B,N,L)
            clip_attention_mask = clip_attention_mask[:,:,0]
            clip_attention_mask = clip_attention_mask.contiguous().view(-1)
            clip_attention_mask = torch.unsqueeze(clip_attention_mask, -1)
            clip_attention_mask = clip_attention_mask.repeat(1,D)
            generetive_features = generetive_features * clip_attention_mask
            clip_features = clip_features * clip_attention_mask
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_score = cos(generetive_features, clip_features)
            distillation_loss = 1 - torch.mean(cos_score)
            # clip__hidden_states = vision_outputs.hidden_states[-2].contiguous().view(B, N * L, D)
            # clip_attention_mask = torch.unsqueeze(video_attention_mask, -1)
            # clip_attention_mask = clip_attention_mask.repeat(1,1,D)
            # generetive_hidden_states = generetive_hidden_states * clip_attention_mask
            # clip__hidden_states = clip__hidden_states * clip_attention_mask
            # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            # cos_score = cos(generetive_hidden_states, clip__hidden_states)
            # distillation_loss = 1 - torch.mean(cos_score)
        else:
            distillation_loss = None
        '''
        # ================== using audio ===========================
        if self.args.use_audio:
            if self.args.hubert:
                if self.args.fusion_approach == 'dot_product':
                    hubert_hidden_states = self.hubert(input_values=audio_input.input_values).last_hidden_state
                    attn = torch.bmm(stacked_source_reps, hubert_hidden_states.transpose(1, 2))
                    attn = F.softmax(attn, dim=1)
                    hubert_hidden_states = torch.bmm(attn, hubert_hidden_states) # (S_t, D_v)
                    fusion_output = self.fusion_layer(torch.cat((stacked_source_reps, hubert_hidden_states), 2))
                    stacked_source_reps = stacked_source_reps + fusion_output
                elif self.args.fusion_approach == 'concanten':
                    hubert_hidden_states = self.hubert(input_values=audio_input.input_values).last_hidden_state
                    hubert_hidden_states = self.audio_projecter(hubert_hidden_states)
                    audio_mask = torch.ones([hubert_hidden_states.shape[0],hubert_hidden_states.shape[1]]).cuda()
                    stacked_source_reps = torch.cat((stacked_source_reps, hubert_hidden_states), 1)
                    video_attention_mask = torch.cat((video_attention_mask, audio_mask), 1)
            elif self.args.wavlm:
                if self.args.fusion_approach == 'dot_product':
                    hubert_hidden_states = self.wavlm(input_values=audio_input.input_values).last_hidden_state
                    attn = torch.bmm(stacked_source_reps, hubert_hidden_states.transpose(1, 2))
                    attn = F.softmax(attn, dim=1)
                    hubert_hidden_states = torch.bmm(attn, hubert_hidden_states) # (S_t, D_v)
                    fusion_output = self.fusion_layer(torch.cat((stacked_source_reps, hubert_hidden_states), 2))
                    stacked_source_reps = stacked_source_reps + fusion_output
                elif self.args.fusion_approach == 'concanten':
                    hubert_hidden_states = self.wavlm(input_values=audio_input.input_values, attention_mask=audio_input.attention_mask).last_hidden_state
                    hubert_hidden_states = self.audio_projecter(hubert_hidden_states)
                    audio_mask = torch.ones([hubert_hidden_states.shape[0],hubert_hidden_states.shape[1]]).cuda()
                    stacked_source_reps = torch.cat((stacked_source_reps, hubert_hidden_states), 1)
                    video_attention_mask = torch.cat((video_attention_mask, audio_mask), 1)
            elif self.args.wav2clip:
                if self.args.fusion_approach == 'dot_product':
                    audio_input = self.audio_projecter(audio_input)
                    attn = torch.bmm(stacked_source_reps, audio_input.transpose(1, 2))
                    attn = F.softmax(attn, dim=1)
                    audio_input = torch.bmm(attn, audio_input) # (S_t, D_v)
                    fusion_output = self.fusion_layer(torch.cat((stacked_source_reps, audio_input), 2))
                    stacked_source_reps = stacked_source_reps + fusion_output
                elif self.args.fusion_approach == 'cross_attention':
                    # K = self._vtlinear_1_audio_(audio_input)
                    # V = self._vtlinear_2_audio_(audio_input)
                    # Q = self._vtlinear_3_audio_(stacked_source_reps)
                    # attn_output, _ = self.multi_head_attn_audio_(query=Q, key=K, value=V)
                    # stacked_source_reps = stacked_source_reps + attn_output
                    audio_input = self.audio_projecter(audio_input)
                    attn = torch.bmm(stacked_source_reps, audio_input.transpose(1, 2))
                    attn = F.softmax(attn, dim=1)
                    audio_input = torch.bmm(attn, audio_input) # (S_t, D_v)
                    fusion_output = self.fusion_layer(torch.cat((stacked_source_reps, audio_input), 2))
                    stacked_source_reps = stacked_source_reps + fusion_output
            else:
                if self.args.fusion_approach == 'dot_product':
                    audio_input = self.layer_norm(audio_input)
                    audio_features = self.audio_transformer(audio_input, audio_len)[-1] # [n*768]
                    audio_features = self.audio_projecter(audio_features)
                    attn = torch.bmm(stacked_source_reps, audio_features.transpose(1, 2))
                    attn = F.softmax(attn, dim=1)
                    audio_features = torch.bmm(attn, audio_features) # (S_t, D_v)
                    output = self.fusion_layer(torch.cat((stacked_source_reps, audio_features), 2))
                    stacked_source_reps = stacked_source_reps + output
                elif self.args.fusion_approach == 'multi_head':
                    raw_audio_features = self.audio_projecter(audio_input)
                    raw_audio_features = self.layer_norm(raw_audio_features)
                    audio_features = self.audio_transformer(raw_audio_features, audio_len)[-1] # [n*768]
                    K = self._linear_1(audio_features)
                    V = self._linear_2(audio_features)
                    Q = self._linear_3(stacked_source_reps)
                    attn_output, _ = self.multi_head_attn(Q, K, V)
                    output = self.fusion_layer(torch.cat((stacked_source_reps, attn_output), 2))
                    stacked_source_reps = stacked_source_reps + output
                elif self.args.fusion_approach == 'concanten':
                    raw_audio_features = self.audio_projecter(audio_input)
                    raw_audio_features = self.layer_norm(raw_audio_features)
                    audio_features = self.audio_transformer(raw_audio_features, audio_len)[-1] # [n*768]
                    audio_mask = torch.ones([audio_features.shape[0],audio_features.shape[1]]).cuda()
                    stacked_source_reps = torch.cat((stacked_source_reps, audio_features), 1)
                    video_attention_mask = torch.cat((video_attention_mask, audio_mask), 1)
        
        # ================== constrained_generation ===========================
        if self.args.constrained_generation:
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask,return_dict=return_dict)
            last_hidden_states = outputs.last_hidden_state
            
            
            if self.args.fusion_approach == 'cross_attention':
                K = self._vtlinear_1(stacked_source_reps)
                V = self._vtlinear_2(stacked_source_reps)
                Q = self._vtlinear_3(last_hidden_states)
                key_padding_mask = 1 - video_attention_mask
                attn_output, _ = self.multi_head_attn(query=Q, key=K, value=V, key_padding_mask=key_padding_mask)
                last_hidden_states = last_hidden_states + attn_output

            elif self.args.fusion_approach == 'dot_product':
                attn = torch.bmm(last_hidden_states, stacked_source_reps.transpose(1, 2))
                attn = F.softmax(attn, dim=2)
                stacked_source_reps = torch.bmm(attn, stacked_source_reps) # (S_t, D_v)
                fusion_output = self.fusion_layer(torch.cat((last_hidden_states, stacked_source_reps), 2))
                last_hidden_states = last_hidden_states + fusion_output
            elif self.args.fusion_approach == 'concat':
                last_hidden_states = torch.cat((last_hidden_states, stacked_source_reps), 1)
                attention_mask = torch.cat((attention_mask, video_attention_mask), 1)

        else:
            last_hidden_states = stacked_source_reps
            attention_mask = video_attention_mask
            
        if return_dict:
            encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_states)
        else:
            encoder_outputs = (last_hidden_states,)
        return encoder_outputs, attention_mask, distillation_loss

    def generate(
        self,
        pixel_values=None,
        video_attention_mask=None,
        input_ids=None,
        attention_mask=None,
        audio_input=None,
        audio_len=None,
        constrains_input=None,
        **kwargs,
    ):
        encoder_outputs, attention_mask, distillation_loss = self.multi_encode(
                pixel_values=pixel_values,
                video_attention_mask=video_attention_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                audio_input=audio_input,
                audio_len=audio_len,
            )
        
        return super().generate(
            input_ids=None,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,   
            **kwargs
        )
    
    def forward(
        self,
        pixel_values=None,
        video_attention_mask=None,
        audio_input=None,
        audio_len=None,
        clip_pixel_values=None,
        distillation_loss=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        
        if pixel_values is None:
            if encoder_outputs is None:
                raise ValueError("Encoder outputs is required when no pixel_values passed")
        else:
            encoder_outputs, attention_mask, distillation_loss = self.multi_encode(
                pixel_values=pixel_values,
                video_attention_mask=video_attention_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                audio_input=audio_input,
                audio_len=audio_len,
                clip_pixel_values=clip_pixel_values,
            )
        output = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=None,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if distillation_loss != None:
            output.loss = output.loss + distillation_loss
        return output