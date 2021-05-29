import re
import string
from contractions import CONTRACTION_MAP
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import torch
import random

def initialize():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)



def expand_contractions(text):
    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),
                                      flags=re.IGNORECASE)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match) \
            if CONTRACTION_MAP.get(match) \
            else CONTRACTION_MAP.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text



def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)



def cleanup_text(line: str) -> str:
    """
    Args:
        line: text line of the opinion
    Returns:
        text line where not alphabetical characters removed
    """
    corrections = {"&quot": "", "&amp": ""}
    line_lower = line.lower()
    
    for wrong, cor in corrections.items():
        line_lower = line_lower.replace(wrong, cor)
    
    # remove html
    line_lower = remove_html(line_lower)
    
    # remove url
    line_lower = remove_url(line_lower)
    
    # remove emoji
    line_lower = remove_emoji(line_lower)
    #words = re.findall(r"[\w']+", line_lower)

    return remove_punct(line_lower)

def create_vocabulary(texts):    
    counter = Counter()
    for line in texts:
        counter.update([x for x in line.split(" ") if x not in STOP_WORDS and len(x) > 1])
    return counter



def calculate_accuracy(y_pred, y):
    y = y.detach().numpy()
    y_pred = torch.sigmoid(y_pred).detach().numpy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    correct = len(y_pred[y_pred == y])
    return (correct/y_pred.shape[0]) * 100
