#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vocab
from argparse import Namespace
from util import cleanup_text, create_vocabulary
from data import collate_batch, create_input, create_lstm_data_loader
from models import LSTMModel
from models import train_lstm_model, eval_lstm_model


if __name__ == "__main__":

    args = Namespace(
        hidden_dim=256,
        learning_rate=0.001,
        train_csv="../raw_data/train.zip",
        batch_size=128,
        num_epochs=5,
        num_layers=1,
        embedding_size=100,
        dropout=0.5,
        output_dim=1
    )

    X_train, X_val, X_test, y_train, y_val, y_test = create_input(args.train_csv, cleanup_text)


    TweetVocabulary = Vocab(create_vocabulary(X_train),
                            min_freq=1,
                            specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))

    model = LSTMModel(input_dim=len(TweetVocabulary), embedding_size=args.embedding_size,
                      hidden_dim=args.hidden_dim, output_dim=1, padding_id=3,
                      n_layers=args.num_layers, dropout=args.dropout)

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    train_loader = create_lstm_data_loader(TweetVocabulary, X_train, y_train,
                                           args.batch_size, shuffle=True, collate_func=collate_batch)
    val_loader = create_lstm_data_loader(TweetVocabulary, X_val, y_val,
                                         args.batch_size, shuffle=True, collate_func=collate_batch)

    for epoch in range(args.num_epochs):
        train_lstm_model(epoch, model, train_loader, device, loss_fn, optimizer)
        eval_lstm_model(epoch, model, val_loader, device, loss_fn)
