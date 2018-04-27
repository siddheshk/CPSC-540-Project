from collections import Counter
from gensim.models import Word2Vec
from random import random
from nltk import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from torch.autograd import Variable
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score, precision_recall_curve

import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import math
import random
import re, string
import time

vocabulary = pickle.load(open('../Data/vocabulary_glove.p','rb'))
w2v_embeddings = pickle.load(open('../Data/w2v_embeddings_glove.p','rb'))
word2index = pickle.load(open('../Data/word2index_glove.p','rb'))
ind2word = pickle.load(open('../Data/ind2word_glove.p','rb'))

params = {}
params['embedding_size'] = 300
params['num_classes'] = 2
params['cnn_filter_size'] = [3,4,5]
params['word2index'] = word2index
params['ind2word'] = ind2word
params['cnn_output_channels'] = 100
params['dropout'] = 0.5
params['batch_size'] = 50
params['USE_CUDA'] = True
params['gpu'] = 0
params['epochs'] = 20
params['test_batch_size'] = 500
params['sentence_length'] = 60
params['number_of_sentences'] = 15
params['hidden_dim'] = 128
params['bidirectional'] = True

class HierarchicalLSTM(nn.Module):
    def __init__(self, params, embedding_weights=None):
        super(HierarchicalLSTM, self).__init__()
        self.params = params
        self.embedding_layer = nn.Embedding(len(self.params['word2index'].keys()), self.params['embedding_size'], padding_idx=0)
        if embedding_weights is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.word_lstm = nn.LSTM(self.params['embedding_size'], self.params['hidden_dim'], bidirectional=self.params['bidirectional'])
        self.dropout = nn.Dropout(self.params['dropout'])
        if self.params['bidirectional']:
            self.sentence_lstm = nn.LSTM(self.params['hidden_dim']*2, self.params['hidden_dim'], bidirectional=self.params['bidirectional'])
            self.linear = nn.Linear(self.params['hidden_dim']*2, self.params['num_classes'])
        else:
            self.sentence_lstm = nn.LSTM(self.params['hidden_dim'], self.params['hidden_dim'], bidirectional=self.params['bidirectional'])
            self.linear = nn.Linear(self.params['hidden_dim'], self.params['num_classes'])

    def forward(self, input_indices, sentence_length, number_of_sentences):
        batch_size = input_indices.size(0)
        sentence_length = sentence_length + (sentence_length == 0).long()
        input_embedding = self.embedding_layer(input_indices.view(-1, self.params['sentence_length']))
        input_len_sorted, perm_idx = sentence_length.view(-1,).sort(0, descending=True)
        _, perm_idx_resort = perm_idx.sort(0, descending=False)

        input_embedding_sorted = input_embedding.index_select(0, perm_idx)
        input_embedding_sorted = input_embedding_sorted.transpose(0,1)
        packed_input_embedding_tensor = nn.utils.rnn.pack_padded_sequence(input_embedding_sorted, input_len_sorted.data.cpu().numpy())
        lstm_out_packed, lstm_hidden = self.word_lstm(packed_input_embedding_tensor)
        #lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed)
        sentence_encoding = lstm_hidden[0].transpose(0,1).index_select(0, perm_idx_resort).view(batch_size*self.params['number_of_sentences'], -1).view(batch_size, self.params['number_of_sentences'], -1)

        input_num_sentence_sorted, perm_num_sentence_idx = number_of_sentences.sort(0, descending=True)
        _, perm_num_sentence_idx_resort = perm_num_sentence_idx.sort(0, descending=False)

        sentence_embedding_sorted = sentence_encoding.index_select(0, perm_num_sentence_idx)
        sentence_embedding_sorted = sentence_embedding_sorted.transpose(0,1)
        packed_input_sentence_tensor = nn.utils.rnn.pack_padded_sequence(sentence_embedding_sorted, input_num_sentence_sorted.data.cpu().numpy())
        sentence_lstm_out_packed, sentence_lstm_hidden = self.sentence_lstm(packed_input_sentence_tensor)
        document_encoding = sentence_lstm_hidden[0].transpose(0,1).index_select(0, perm_num_sentence_idx_resort).view(batch_size, -1)

        linear_input = F.tanh(self.dropout(document_encoding))
        output = self.linear(linear_input)
        return output

def clean(comment):
    comment = comment.lower()
    comment = re.sub("\\n"," ",comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    comment = re.sub("\[\[.*\]","",comment)
    comment = re.sub("(http://.*?\s)|(http://.*)",'',comment)
    comment = re.sub("\[\[User.*",'',comment)
    comment = re.sub('\.+', '. ', comment)
    return comment

train_toxic_text, val_toxic_text, test_toxic_text, train_non_toxic_text, val_non_toxic_text, test_non_toxic_text = pickle.load(open('../Data/train_test_whole_base_val.pkl','rb'))
X_train = train_toxic_text + train_non_toxic_text
X_val = val_toxic_text + val_non_toxic_text
X_test = test_toxic_text + test_non_toxic_text
y_train = [1]*len(train_toxic_text) + [0]*len(train_non_toxic_text)
y_val = [1]*len(val_toxic_text) + [0]*len(val_non_toxic_text)
y_test = [1]*len(test_toxic_text) + [0]*len(test_non_toxic_text)
X_train_sentences = [sent_tokenize(clean(comment)) for comment in X_train]
X_val_sentences = [sent_tokenize(clean(comment)) for comment in X_val]
X_test_sentences = [sent_tokenize(clean(comment)) for comment in X_test]
max_sentence_length = params['number_of_sentences']
X_train_sentences = [x[:min(len(x), max_sentence_length)] for x in X_train_sentences if len(x) > 0]
X_val_sentences = [x[:min(len(x), max_sentence_length)] for x in X_val_sentences if len(x) > 0]
X_test_sentences = [x[:min(len(x), max_sentence_length)] for x in X_test_sentences if len(x) > 0]
X_train_sentences_len = [len(x) for x in X_train_sentences]
X_val_sentences_len = [len(x) for x in X_val_sentences]
X_test_sentences_len = [len(x) for x in X_test_sentences]
tokenizer = TweetTokenizer()
X_train_words = [[tokenizer.tokenize(x) for x in sentences] for sentences in X_train_sentences]
X_val_words = [[tokenizer.tokenize(x) for x in sentences] for sentences in X_val_sentences]
X_test_words = [[tokenizer.tokenize(x) for x in sentences] for sentences in X_test_sentences]


def preprocess_numberize(sentence):
    numberized = [word2index.get(word, word2index["<UNK>"]) for word in sentence]
    return numberized

train_sentences = X_train_words
val_sentences = X_val_words
train_targets = y_train
val_targets = y_val
train_sentences_index = [[preprocess_numberize(sentence)[:params['sentence_length']] for sentence in sentences] for sentences in train_sentences]
val_sentences_index = [[preprocess_numberize(sentence)[:params['sentence_length']] for sentence in sentences] for sentences in val_sentences]
train_sentences_len = [[len(sentence) for sentence in sentences] for sentences in train_sentences_index]
val_sentences_len = [[len(sentence) for sentence in sentences] for sentences in val_sentences_index]
train_sentences_index = [[sentence + [0]*(params['sentence_length'] - len(sentence)) for sentence in sentences] for sentences in train_sentences_index]
val_sentences_index = [[sentence + [0]*(params['sentence_length'] - len(sentence)) for sentence in sentences] for sentences in val_sentences_index]
train_sentences_matrix = np.zeros((len(train_sentences_index), params['number_of_sentences'], params['sentence_length']), dtype=np.int64)
val_sentences_matrix = np.zeros((len(val_sentences_index), params['number_of_sentences'], params['sentence_length']), dtype=np.int64)
for index, sentences in enumerate(train_sentences_index):
    for sent_ind, sentence in enumerate(sentences):
        train_sentences_matrix[index, sent_ind, :] = sentence

for index, sentences in enumerate(val_sentences_index):
    for sent_ind, sentence in enumerate(sentences):
        val_sentences_matrix[index, sent_ind, :] = sentence

train_targets = np.array(train_targets)
val_targets = np.array(val_targets)
train_sentences_len = np.array([x + [0]*(params['number_of_sentences'] - len(x)) for x in train_sentences_len])
val_sentences_len = np.array([x + [0]*(params['number_of_sentences'] - len(x)) for x in val_sentences_len])
train_number_sentences = np.array(X_train_sentences_len + X_val_sentences_len)
val_number_sentences = np.array(X_val_sentences_len)



np.random.seed(1)
print (w2v_embeddings.shape)
model = HierarchicalLSTM(params, w2v_embeddings)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

if params['USE_CUDA']:
    model = model.cuda(params['gpu'])
    criterion = criterion.cuda(params['gpu'])
number_of_batches = math.ceil(len(train_sentences_matrix)/params['batch_size'])
number_of_val_batches = math.ceil(len(val_sentences_matrix)/params['test_batch_size'])
indexes = np.arange(len(train_sentences_matrix))
val_indexes = np.arange(len(val_sentences_matrix))
for epoch in range(params['epochs']):
    np.random.shuffle(indexes)
    avgLoss = 0.0
    start_time = time.time()
    model.train()
    for batch in range(number_of_batches):
        batch_indexes = indexes[batch*params['batch_size'] : (batch+1)*params['batch_size']]
        batch_sentences = Variable(torch.from_numpy(train_sentences_matrix[batch_indexes]), volatile=False)
        batch_targets = Variable(torch.from_numpy(train_targets[batch_indexes]), volatile=False)
        batch_sentence_length = Variable(torch.from_numpy(train_sentences_len[batch_indexes]), volatile=False)
        batch_number_of_sentences = Variable(torch.from_numpy(train_number_sentences[batch_indexes]), volatile=False)
        if params['USE_CUDA']:
            batch_sentences = batch_sentences.cuda(params['gpu'])
            batch_targets = batch_targets.cuda(params['gpu'])
            batch_sentence_length = batch_sentence_length.cuda(params['gpu'])
            batch_number_of_sentences = batch_number_of_sentences.cuda(params['gpu'])
        predictions = model(batch_sentences, batch_sentence_length, batch_number_of_sentences)
        loss = criterion(predictions, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avgLoss += loss.data.cpu().numpy()[0]
        if batch%500 == 0:
            print ("Time:",time.time() - start_time,"\tAverage Loss per Batch:", avgLoss/(batch+1))

    model.eval()
    avgValLoss = 0.0
    start_time = time.time()
    predictions_arr = np.zeros_like(val_targets)
    prediction_probs = np.zeros_like(val_targets, dtype=np.float64)
    for batch in range(number_of_val_batches):
        batch_indexes = val_indexes[batch*params['test_batch_size'] : (batch+1)*params['test_batch_size']]
        batch_sentences = Variable(torch.from_numpy(val_sentences_matrix[batch_indexes]), volatile=True)
        batch_targets = Variable(torch.from_numpy(val_targets[batch_indexes]), volatile=True)
        batch_sentence_length = Variable(torch.from_numpy(val_sentences_len[batch_indexes]), volatile=True)
        batch_number_of_sentences = Variable(torch.from_numpy(val_number_sentences[batch_indexes]), volatile=True)
        if params['USE_CUDA']:
            batch_sentences = batch_sentences.cuda(params['gpu'])
            batch_targets = batch_targets.cuda(params['gpu'])
            batch_sentence_length = batch_sentence_length.cuda(params['gpu'])
            batch_number_of_sentences = batch_number_of_sentences.cuda(params['gpu'])
        predictions = model(batch_sentences, batch_sentence_length, batch_number_of_sentences)
        predictions_arr[batch_indexes] = np.argmax(predictions.data.cpu().numpy(),1)
        prediction_probs[batch_indexes] = F.softmax(predictions,1).data.cpu().numpy()[:,1]
        loss = criterion(predictions, batch_targets)
        avgValLoss += loss.data.cpu().numpy()[0]
    print ("Time:",time.time() - start_time,"\tAverage Val Loss per Batch:", avgValLoss/number_of_val_batches,"\tAccuracy:", accuracy_score(val_targets, predictions_arr))
    scores = precision_recall_fscore_support(val_targets, predictions_arr)
    auc = roc_auc_score(val_targets, prediction_probs)
    precision, recall, _ = precision_recall_curve(val_targets, prediction_probs, pos_label=1)
    pickle.dump([precision, recall], open('results/hierarchical_lstm_'+str(epoch),'wb'),-1)
    print ("precision:",scores[0],"\trecall:",scores[1],"\tfbeta_score:",scores[2],"\tsupport:",scores[3], "\tauc:", auc)
    torch.save({'epoch': epoch ,'model': model.state_dict(), 'optimizer':optimizer.state_dict()}, "outputs/hierarchical_lstm_model_glove_"+str(epoch))
