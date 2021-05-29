import torch.nn as nn
import torch
from util import calculate_accuracy


class BERTModel(nn.Module):
    def __init__(self, bert, output_dim, dropout=0.3):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_ids, attention_mask):
        pooled_out = self.bert(input_ids, attention_mask)
        out = self.dropout(pooled_out[1])
        out = self.fc(out)
        return out.squeeze(1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_size, hidden_dim,
                 output_dim, padding_id, n_layers, dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_id)

        self.rnn = nn.LSTM(embedding_size,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu())

        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = self.fc(hidden)
        return out.squeeze(1)
        
        
class BERTLSTMModel(nn.Module):
    def __init__(self, bert, hidden_dim,
                 output_dim, n_layers, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True,
                           dropout=0 if n_layers < 2 else dropout,
                           batch_first=True)

        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, texts, attention_mask):
        with torch.no_grad():
            embedded = self.bert(
                texts,
                attention_mask
            )[0]
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(embedded)
        output = self.dropout(torch.cat((hidden[-2, :, :],
                                         hidden[-1, :, :]),
                                        dim=1))
        out = self.fc(output)
        return out.squeeze(1)
        

def train_lstm_model(epoch, model, data_loader, device, loss_fn, optimizer):
    total_train_loss = 0.0
    total_train_accuracy = []
    model = model.train()
    for ys, xs, text_lengths in data_loader:
        xs = xs.to(device)
        ys = ys.to(device)
        predicted = model(xs, text_lengths.cpu())
        train_loss = loss_fn(predicted, ys)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
        accuracy = calculate_accuracy(predicted.cpu(), ys.cpu())
        total_train_accuracy.append(accuracy)
    print(f"Epoch {epoch}, Training Loss:  {total_train_loss}")
    print(f"Epoch {epoch}, Training accuracy: {sum(total_train_accuracy) / len(total_train_accuracy)}")


def eval_lstm_model(epoch, model, data_loader, device, loss_fn):
    total_eval_loss = 0.0
    total_eval_accuracy = []
    model = model.eval()
    with torch.no_grad():
        for ys, xs, text_lengths in data_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            predicted = model(xs, text_lengths.cpu())
            loss = loss_fn(predicted, ys)

            total_eval_loss += loss.item()
            accuracy = calculate_accuracy(predicted.cpu(), ys.cpu())
            total_eval_accuracy.append(accuracy)
    print(f"Epoch {epoch}, Eval Loss:  {total_eval_loss}")
    print(f"Epoch {epoch}, Eval accuracy: {sum(total_eval_accuracy) / len(total_eval_accuracy)}")


def train_bert_model(epoch, model, data_loader, device, loss_fn, optimizer):
    total_train_loss = 0.0
    total_train_accuracy = []
    model = model.train()

    for data in data_loader:
        xs = data["input_ids"].to(device)
        ys = data["labels"].to(device)
        attention_mask = data["attention_mask"].to(device)

        predicted = model(xs, attention_mask)
        train_loss = loss_fn(predicted, ys)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
        accuracy = calculate_accuracy(predicted.cpu(), ys.cpu())
        total_train_accuracy.append(accuracy)

    print(f"Epoch {epoch}, Training Loss:  {total_train_loss}")
    print(f"Epoch {epoch}, Training accuracy: {sum(total_train_accuracy) / len(total_train_accuracy)}")


def eval_bert_model(epoch, model, data_loader, device, loss_fn):
    total_val_loss = 0.0
    total_val_accuracy = []
    model = model.eval()

    with torch.no_grad():
        for data in data_loader:
            xs = data["input_ids"].to(device)
            ys = data["labels"].to(device)
            attention_mask = data["attention_mask"].to(device)
            predicted = model(xs, attention_mask)
            val_loss = loss_fn(predicted, ys)

            total_val_loss += val_loss.item()
            accuracy = calculate_accuracy(predicted.cpu(), ys.cpu())
            total_val_accuracy.append(accuracy)
    print(f"Epoch {epoch}, Validation loss: {total_val_loss}")
    print(f"Epoch {epoch}, Validation accuracy: {sum(total_val_accuracy) / len(total_val_accuracy)}")
