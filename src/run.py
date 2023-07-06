# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from torch.utils.data import DataLoader
from data_builder import OurDataset
from models.model import BaseModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for data_path
    parser.add_argument('-train_path', default='datasets/Vine/split_data/train.jsonl', type=str)
    parser.add_argument('-val_path', default='datasets/Vine/split_data/val.jsonl', type=str)
    parser.add_argument('-test_path', default='datasets/Vine/split_data/test.jsonl', type=str)
    parser.add_argument('-val_save_file', default='datasets/Vine/split_data/val_save.txt', type=str)
    parser.add_argument('-test_save_file', default='datasets/Vine/split_data/test_results/test_results.txt', type=str)

    # for model settings
    parser.add_argument('-model', default='nlpconnect/vit-gpt2-image-captioning', type=str)
    parser.add_argument('-checkpoint', default='None', type=str)

    # for training
    parser.add_argument('-log_name', default='', type=str)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-learning_rate', default=5e-5, type=float)
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-warmup', type=int, default=20)
    parser.add_argument('-grad_accum', type=int, default=10)
    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-do_train', action='store_true')
    parser.add_argument('-do_test', action='store_true')
    parser.add_argument('-limit_val_batches', type=float, default=1.0)
    parser.add_argument('-val_check_interval', type=float, default=1.0)
    parser.add_argument('-split_hashtag', action='store_true')

    # vision settings
    parser.add_argument('-frame_number', type=int, default=12)
    parser.add_argument('-use_frame_features', action='store_true')

    # Audio settings
    parser.add_argument('-use_audio', action='store_true')
    parser.add_argument('-fusion_approach', default='', type=str)
    parser.add_argument('-wav2clip', action='store_true')
    parser.add_argument('-d_audio_input', type=int, default=1025)
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-num_layers', type=int, default=4)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-dim_feedforward', type=int, default=768)
    parser.add_argument('-max_audio_feature_len', type=int, default=512)
    parser.add_argument('-reweight', type=float, default=0.0)

    # setting for fusion
    parser.add_argument('-d_common', type=int, default=256)
    parser.add_argument('-num_fusion_heads', type=int, default=4)
    parser.add_argument('-constrained_generation', action='store_true')
    parser.add_argument('-knowledge_distillation', action='store_true')

    # decoding settings
    parser.add_argument('-do_sample', action='store_true')
    parser.add_argument('-top_p', type=float, default=1.0)
    parser.add_argument('-top_k', type=int, default=50)
    parser.add_argument('-max_output_len', type=int, default=64)
    parser.add_argument('-min_output_len', type=int, default=0)
    parser.add_argument('-n_beams', type=int, default=5)
    parser.add_argument('-no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('-early_stopping', action='store_true')
    parser.add_argument('-length_penalty', type=float, default=1)

    # order settings
    parser.add_argument('-label_shuffle', action='store_true')
    parser.add_argument('-use_multisoftmax', action='store_true')
    parser.add_argument('-one2one', action='store_true')


    args = parser.parse_args()
    print('================================================== augments ==================================================')
    print(args)
    # random seed
    seed_everything(args.random_seed)

    # set logger
    logger = pl_loggers.TensorBoardLogger(args.log_name)

    # save checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='validation_Rouge/rouge1_F1',
                                          save_last=False,
                                          save_top_k=3,
                                          mode='max')

    # make trainer
    if args.checkpoint == 'None':
        resume_checkpoint = None
    else:
        resume_checkpoint = args.checkpoint
    trainer = Trainer(
                      deterministic=True,
                      num_sanity_val_steps=2,
                      resume_from_checkpoint=resume_checkpoint,
                      logger=logger,
                      gpus='-1',
                      accelerator='gpu',
                      strategy="ddp",
                      precision=16,
                      gradient_clip_val=1.0,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accum,
                      log_every_n_steps=1,
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback])

    # make dataloader & model
    train_set = OurDataset(args, 'train')
    val_set = OurDataset(args, 'val')
    test_set = OurDataset(args, 'test')
    train_loader = DataLoader(dataset=train_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=16, \
                                    shuffle=True, \
                                    collate_fn=train_set.collate_fn)
    val_loader = DataLoader(dataset=val_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=16, \
                                    shuffle=False, \
                                    collate_fn=val_set.collate_fn)
    test_loader = DataLoader(dataset=test_set, \
                                    batch_size=args.batch_size, \
                                    num_workers=16, \
                                    shuffle=False, \
                                    collate_fn=test_set.collate_fn)
    model = BaseModel(args)

    # Fit the instantiated model to the data
    if args.do_train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.do_test:
        trainer.test(model=model, ckpt_path=args.checkpoint, dataloaders=test_loader)

