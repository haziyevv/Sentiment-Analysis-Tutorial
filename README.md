# Python Ds Tasks

In this repository two projects are applied, first one is a crawling project where BeautifulSoup is used to crawl wikipedia and prove that starting from any random wikipedia page, you will end up in the Philosophy page. Second project is about Sentiment Analysis, where I have applied three different approaches.

## Philosophy Crawling

```
python3 main.py wikipedia_url
```

## Sentiment Analysis

```
cd Sentiment_Analysis
ls

--> returns

ckpt             Data_Sentiment_analysis.ipynb  raw_data
contractions.py  lstm_model.zip                 training_repo
```

**Data\_Sentiment\_analysis.ipynb** -->  In this jupyter notebook, Sentiment Analysis training data is analysed and 3 different Sentiment Analysis models are evaluated and compared.

Pretrained models are loaded in this notebook. To use those download pretrained models and put in **ckpt** folder.

## Trained Sentiment Analysis Models

To train the models create **raw\_data** folder and put **train.csv** Sentiment Analydis Dataset to there.

**Bidirectional LSTM** model --> located in **training\_repo/train\_lstm\_model.py** 

to train

```
python train_lstm_model.py --train-csv training_file.csv
```

BERT + Bidirectional LSTM model --> located in **training\_repo/train_bert_with_lstm_model.py** 

```
python train_bert_with_lstm_model.py --train-csv training_file.csv
```

BERT model --> located in **training\_repo/train\_only\_bert\_model.py** 

```
python train_only_bert_model.py --train-csv training_file.csv
```