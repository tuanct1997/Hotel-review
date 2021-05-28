import numpy as np
import pandas as pd
import nltk
import torch
import gensim
from sklearn.manifold import TSNE
import random
# from torchtext.data import Field
import spacy
from matplotlib import pyplot as plt
import string
from keras.preprocessing.sequence import pad_sequences
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from gensim.models import KeyedVectors
from tensorflow import keras
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from wikipedia2vec import Wikipedia2Vec
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.current_device())

# IN NEED TO SPECIFY FREE GPU
# torch.cuda.set_device(0)

#LOAD DATASET
DATA = pd.read_csv('tripadvisor_hotel_reviews.csv')

print(DATA.info())
txt_sequence = []
rate = []

#convert phrase to list of words
for idx,val in DATA.iterrows():
    temp = []
    for w in val['Review'].split():
        w = w.translate(str.maketrans('', '', string.punctuation))
        if all(str.isdigit(c) for c in w):
            continue
        temp.append(w)
    txt_sequence.append(temp)
    rate.append(val['Rating'] - 1)

print(len(txt_sequence))
encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]
print('!!!!!!!!!!!!!!!!!!!!!')

def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

