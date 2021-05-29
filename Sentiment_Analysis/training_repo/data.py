import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from util import initialize
from torch.utils.data import DataLoader


def create_data_loader(DatasetClass, x_data, label_data,
                       tokenizer, max_length, batch_size, shuffle, num_workers):
    dataset = DatasetClass(x_data, label_data, tokenizer, max_length)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def create_lstm_data_loader(Vocabulary, x_data, label_data,
                            batch_size, shuffle, collate_func):
    dataset = []
    for (text, label) in zip(x_data, label_data):
        dataset.append((text_transform(text, Vocabulary), label))

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_func)


def create_input(filename, cleaner):
    initialize()
    df = pd.read_csv(filename,
                     error_bad_lines=False)

    df["SentimentText"] = df["SentimentText"].apply(cleaner)

    X_train, X_val, y_train, y_val = train_test_split(df['SentimentText'], df['Sentiment'], test_size=0.1)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.05)
    return X_train, X_val, X_test, y_train, y_val, y_test


class SentimentDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


def text_transform(text, Vocabulary):
    return [Vocabulary['<BOS>']] + [Vocabulary[token]
                                    for token in text.split(" ")] + [Vocabulary['<EOS>']]


def collate_batch(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    label_list, text_list, text_lengths = [], [], []
    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text)
        text_list.append(processed_text)
        text_lengths.append(len(processed_text))
    return torch.tensor(label_list, dtype=torch.float32), pad_sequence(text_list, padding_value=3.0), torch.tensor(
        text_lengths, dtype=torch.int64, device="cpu")
