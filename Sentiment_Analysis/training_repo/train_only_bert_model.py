#!/usr/bin/env python
# coding: utf-8
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from argparse import Namespace, ArgumentParser
from data import SentimentDataset, create_input, create_data_loader
from util import cleanup_text
from models import BERTModel
from models import train_bert_model, eval_bert_model

if __name__ == "__main__":
    args = Namespace(
        hidden_dim=256,
        learning_rate=0.001,
        batch_size=4,
        num_epochs=5,
        dropout=0.3,
        output_dim=1,
        max_input_length=128
    )

    parser = ArgumentParser()
    parser.add_argument('--train-csv',
                        help='Number of worker processes for background data loading',
                        default='../raw_data/train.zip',
                        type=str)

    cli_args = parser.parse_args()
    args.train_csv = cli_args.train_csv



    X_train, X_val, X_test, y_train, y_val, y_test = create_input(args.train_csv, cleanup_text)
    X_train, X_val, y_train, y_val = X_train[:1000], X_val[:1000], y_train[:1000], y_val[:1000]
    
    device = "cuda" if torch.cuda.is_available else "cpu"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_data_loader = create_data_loader(SentimentDataset, X_train.to_list(),
                                           y_train.to_list(), tokenizer,
                                           args.max_input_length, args.batch_size,
                                           shuffle=True, num_workers=8)

    val_data_loader = create_data_loader(SentimentDataset, X_val.to_list(),
                                         y_val.to_list(), tokenizer,
                                         args.max_input_length, args.batch_size,
                                         shuffle=True, num_workers=8)

    bert = BertModel.from_pretrained('bert-base-uncased')

    model = BERTModel(bert=bert,
                      output_dim=args.output_dim,
                      dropout=args.dropout)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * args.num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(args.num_epochs):
        train_bert_model(epoch, model, train_data_loader, device, loss_fn, optimizer)
        eval_bert_model(epoch, model, val_data_loader, device, loss_fn)
