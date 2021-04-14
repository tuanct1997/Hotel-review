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
## TORCHTEXT == 0.8 
## >>> import ssl
## >>> ssl._create_default_https_context = ssl._create_unverified_context

class LSTM(nn.Module):
	"""docstring for LSTM"""
	def __init__(self):
		super(LSTM, self).__init__()
		# self.word_embeddings = nn.Embedding.from_pretrained(w2v_weights)
		self.lstm = nn.LSTM(100, 300,2,bidirectional = False)
		self.dropout = nn.Dropout(0.5)
		self.dense = nn.Linear(300, 5)
		self.act = nn.ReLU()

	def forward(self,x):
		# text_emb = self.word_embeddings(x)
		# packed_input = pack_padded_sequence(text_emb, 1931, batch_first=True, enforce_sorted=False)
		lstm_out, lstm_hidden = self.lstm(x)
		lstm_out = lstm_out[:,-1,:]
		lstm_out = self.act(lstm_out)
		drop_out = self.dropout(lstm_out)
		output = self.dense(drop_out)
		return output
		

def check_acc(result,prediction):
    count = 0
    # check each prediction
    for index,val in enumerate(prediction):
        # if predict correct => plus 1 point
        if val == result[index]:
            count += 1
    # return the percentage accuracy
    return count/len(result)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.current_device())

#torch.cuda.set_device(0)
DATA = pd.read_csv('tripadvisor_hotel_reviews.csv')

print(DATA.info())
txt_sequence = []
rate = []
for idx,val in DATA.iterrows():
	temp = []
	for w in val['Review'].split():
		w = w.translate(str.maketrans('', '', string.punctuation))
		temp.append(w)
	txt_sequence.append(temp)
	rate.append(val['Rating'] - 1)
print(len(txt_sequence))
print('!!!!!!!!!!!!!!!!!!!!!')

#~~~~~~~~~~~~~~~~~~~~ TRAIN W2v ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# w2v_model = gensim.models.Word2Vec(txt_sequence, min_count = 1)
# w2v_model.save('./model/word2vec.model')
# word_vectors = w2v_model.wv
# word_vectors.save('./model/word2vec.wordvectors')

#~~~~~~~~~~~~~~~~~~~~ LOAD W2v ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
w2v_model = gensim.models.Word2Vec.load('./model/word2vec.model')
w2v_weights = torch.from_numpy(w2v_model.wv.vectors)
# CHECK WEIGHT
print(w2v_weights.shape)
word_vectors = KeyedVectors.load('./model/word2vec.wordvectors', mmap = 'r')
print('----------------------')

# EMBEDDING WORD BASE ON WEIGHT OF W2V
x = []
i = 0
for sentence in txt_sequence:
	temp = []
	for word in sentence :
		temp.append(w2v_model.wv[word])
	x.append(temp)
x = np.array(x)
print('!!!!!!!!')
# PAD SEQUENCE
x = keras.preprocessing.sequence.pad_sequences(x, dtype='float32')
rate = np.array(rate)
# print(a)
x_train, x_test, y_train, y_test = train_test_split(x, rate, test_size=0.2, shuffle = False)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test).to(device)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test).to(device)
trainset = TensorDataset(x_train,y_train)
loader = DataLoader(trainset, batch_size = 64)
i1,l1 = next(iter(loader))

model = LSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []
val_losses = []
paitent = 0
old = 0
# Make sure that the weights in the embedding layer are not updated
# ~~~~~~~~~~~~~~~~ ONLY NEED EMBEDDING LAYER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# model.word_embeddings.weight.data.copy_(torch.from_numpy(w2v_weights))
# model.word_embeddings.weight.requires_grad=False


model.load_state_dict(torch.load('./model/entire_model.pt'))
model.to(device)
trainset = TensorDataset(x_test,y_test)
testloader = DataLoader(trainset, batch_size = 64)

acc = []
for i, data in enumerate(testloader, 0):
    inputs,labels = data[0].to(device), data[1].to(device)
    outputs_val = model(inputs)
    _, predicted = torch.max(outputs_val,1)
    print(predicted)
    print('!!!!!!!!!!!!')
    val_acc = check_acc(labels, predicted)
    print(val_acc)
    acc.append(val_acc)
    del inputs
    del labels
    torch.cuda.empty_cache()

final_acc = sum(acc)/len(acc)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("VAL_ACC : {}".format(final_acc))
