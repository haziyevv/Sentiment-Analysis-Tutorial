#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import torch.nn as nn
from argparse import Namespace
from transformers import BertTokenizer, BertModel
from util import cleanup_text
from data import SentimentDataset, create_input, create_data_loader
from models import BERTLSTMModel, train_bert_model, eval_bert_model


if __name__ == "__main__":
    args = Namespace(
        hidden_dim=256,
        learning_rate=0.001,
        train_csv="../raw_data/train.zip",
        batch_size=4,
        num_epochs=5,
        num_layers=1,
        embedding_size=100,
        dropout=0.5,
        output_dim=1,
        max_input_length=128
    )

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

    model = BERTLSTMModel(bert=bert,
                          hidden_dim=args.hidden_dim,
                          output_dim=args.output_dim,
                          n_layers=1,
                          dropout=0.5)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    for epoch in range(args.num_epochs):
        train_bert_model(epoch, model, train_data_loader, device, loss_fn, optimizer)
        eval_bert_model(epoch, model, val_data_loader, device, loss_fn)
