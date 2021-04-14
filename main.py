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
## TORCHTEXT == 0.8 
## >>> import ssl
## >>> ssl._create_default_https_context = ssl._create_unverified_context

class LSTM(nn.Module):
	"""docstring for LSTM"""
	def __init__(self):
		super(LSTM, self).__init__()
		# self.word_embeddings = nn.Embedding.from_pretrained(w2v_weights)
		self.lstm = nn.LSTM(100, 300,2,bidirectional = True)
		self.dropout = nn.Dropout(0.5)
		self.dense = nn.Linear(600, 5)
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
# w2v_model = gensim.models.Word2Vec(txt_sequence, min_count = 1)
# w2v_model.save('./model/word2vec.model')
# word_vectors = w2v_model.wv
# word_vectors.save('./model/word2vec.wordvectors')
w2v_model = gensim.models.Word2Vec.load('./model/word2vec.model')
w2v_weights = torch.from_numpy(w2v_model.wv.vectors)
print(w2v_weights.shape)
word_vectors = KeyedVectors.load('./model/word2vec.wordvectors', mmap = 'r')
print('----------------------')
x = []
i = 0
for sentence in txt_sequence:
	temp = []
	for word in sentence :
		temp.append(w2v_model.wv[word])
	x.append(temp)
x = np.array(x)
print(x)
print('!!!!!!!!')
x = keras.preprocessing.sequence.pad_sequences(x, dtype='float32')
print(x)
rate = np.array(rate)
# print(a)
x_train, x_test, y_train, y_test = train_test_split(x, rate, test_size=0.2, shuffle = False)

x_train = torch.from_numpy(x_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_test = torch.from_numpy(y_test).to(device)
trainset = TensorDataset(x_train,y_train)
loader = DataLoader(trainset, batch_size = 64)
i1,l1 = next(iter(loader))

model = LSTM()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []
val_losses = []
paitent = 0
old = 0
# model.word_embeddings.weight.data.copy_(torch.from_numpy(w2v_weights))

# Make sure that the weights in the embedding layer are not updated
# model.word_embeddings.weight.requires_grad=False

for epoch in range(100):
    running_loss = 0.0
    # begin update with each mini-batch
    for i, data in enumerate(loader, 0):
        inputs,labels = data
        optimizer.zero_grad()
        # remember to replace model if we want another network 
        #model => MLP
        # model2 => LSTM
        # model3 => RNN
        outputs = model(inputs)
        # outputs = outputs.permute(0, 2, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #reach to the end of mini-batch
        if i == len(loader)-1 :
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 270))

outputs_val = model(x_test)
_, predicted = torch.max(outputs_val,1)
val_acc = check_acc(y_test, predicted)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("VAL_ACC : {}".format(val_acc))