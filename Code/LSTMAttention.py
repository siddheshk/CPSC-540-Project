
# coding: utf-8

# In[311]:


from collections import Counter
from gensim.models import Word2Vec
from random import random
from nltk import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from torch.autograd import Variable
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, roc_auc_score,precision_recall_curve

import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import math
import random
import re, string
import time


# In[22]:


train_toxic_text, val_toxic_text, test_toxic_text, train_non_toxic_text, val_non_toxic_text, test_non_toxic_text = pickle.load(open('../Data/train_test_whole_base_val.pkl','rb'))


# In[23]:


print (len(train_non_toxic_text), len(test_non_toxic_text), len(val_non_toxic_text), len(train_non_toxic_text)+len(test_non_toxic_text) + len(val_non_toxic_text))
print (len(train_toxic_text), len(test_toxic_text), len(val_toxic_text), len(train_toxic_text)+len(test_toxic_text) + len(val_toxic_text))


# In[44]:


def clean(comment):
    comment = comment.lower()
    comment = re.sub("\\n"," ",comment)
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    comment = re.sub("\[\[.*\]","",comment)
    comment = re.sub("(http://.*?\s)|(http://.*)",'',comment)
    comment = re.sub("\[\[User.*",'',comment)
    comment = re.sub('\.+', '. ', comment)
    return comment


# In[47]:


train_toxic_sentences = train_toxic_text
train_non_toxic_sentences = train_non_toxic_text

val_toxic_sentences = val_toxic_text
val_non_toxic_sentences = val_non_toxic_text

train_non_toxic_sentences = [clean(x) for x in train_non_toxic_sentences]
train_toxic_sentences = [clean(x) for x in train_toxic_text]
val_toxic_sentences = [clean(x) for x in val_toxic_sentences]
val_non_toxic_sentences = [clean(x) for x in val_non_toxic_text]


# In[48]:


tokenizer = TweetTokenizer()


# Lower-case the sentence, tokenize them
train_toxic_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in train_toxic_sentences]
train_non_toxic_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in train_non_toxic_sentences]
val_toxic_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in val_toxic_sentences]
val_non_toxic_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in val_non_toxic_sentences]


# In[ ]:


train_toxic_sentences = [sentence for sentence in train_toxic_text]
train_non_toxic_sentences = [sentence for sentence in train_non_toxic_text]
val_toxic_sentences = [sentence for sentence in val_toxic_text]
val_non_toxic_sentences = [sentence for sentence in val_non_toxic_text]


# In[49]:


vocabularySize = 50001


# In[50]:


vocabulary = pickle.load(open('../Data/vocabulary_glove.p','rb'))
w2v_embeddings = pickle.load(open('../Data/w2v_embeddings_glove.p','rb'))
word2index = pickle.load(open('../Data/word2index_glove.p','rb'))
ind2word = pickle.load(open('../Data/ind2word_glove.p','rb'))


# In[51]:


maxSequenceLength = 100

def preprocess_numberize(sentence):
    """
    Given a sentence, in the form of a string, this function will preprocess it
    into list of numbers (denoting the index into the vocabulary).
    """
    #tokenized = word_tokenize(sentence.lower())
        
    # Add the <SOS>/<EOS> tokens and numberize (all unknown words are represented as <UNK>).
    numberized = [word2index.get(word, 1) for word in sentence]
    
    return numberized



# In[189]:


params = {}
params['embedding_size'] = 300
params['num_classes'] = 2
params['cnn_filter_size'] = [3,4,5]
params['word2index'] = word2index
params['ind2word'] = {word2index[key]:key for key in word2index.keys()}
params['cnn_output_channels'] = 100
params['dropout'] = 0.5
params['batch_size'] = 50
params['USE_CUDA'] = True
params['gpu'] = 0
params['epochs'] = 10
params['test_batch_size'] = 500
params['hidden_dim'] = 128
params['num_layers'] = 1
params['vocabulary_size'] = 50001
params['bidirectional'] = True
params['sentence_length'] = 100


# In[53]:


train_sentences = train_toxic_sentences + train_non_toxic_sentences
val_sentences = val_toxic_sentences + val_non_toxic_sentences

train_targets = [1]*len(train_toxic_sentences) + [0]*len(train_non_toxic_sentences)
val_targets = [1]*len(val_toxic_sentences) + [0]*len(val_non_toxic_sentences)

train_sentences_index = [preprocess_numberize(sentence)[:maxSequenceLength] for sentence in train_sentences]
val_sentences_index = [preprocess_numberize(sentence)[:maxSequenceLength] for sentence in val_sentences]
#train_sentences_index = [sentence[:maxSequenceLength] for sentence in train_sentences]
#test_sentences_index = [sentence[:maxSequenceLength] for sentence in test_sentences]

train_sentences_len = [len(sentence) for sentence in train_sentences_index]
val_sentences_len = [len(sentence) for sentence in val_sentences_index]

train_sentences_index = [sentence + [0]*(maxSequenceLength - len(sentence)) for sentence in train_sentences_index]
val_sentences_index = [sentence + [0]*(maxSequenceLength - len(sentence)) for sentence in val_sentences_index]

train_sentences_index = np.array(train_sentences_index)
val_sentences_index = np.array(val_sentences_index)
train_targets = np.array(train_targets)
val_targets = np.array(val_targets)
train_sentences_len = np.array(train_sentences_len)
val_sentences_len = np.array(val_sentences_len)


# In[199]:


class AttentionLSTM(nn.Module):
    def __init__(self, params, embedding_weights=None):
        super(AttentionLSTM, self).__init__()
        self.params = params
        self.embedding_layer = nn.Embedding(len(self.params['word2index'].keys()), self.params['embedding_size'], padding_idx=0)
        if embedding_weights is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.word_lstm = nn.LSTM(self.params['embedding_size'], self.params['hidden_dim'], bidirectional=self.params['bidirectional'])
        self.dropout = nn.Dropout(self.params['dropout'])
        if self.params['bidirectional']:
            self.sentence_lstm = nn.LSTM(self.params['hidden_dim']*2, self.params['hidden_dim'], bidirectional=self.params['bidirectional'])
            self.linear = nn.Linear(self.params['hidden_dim']*2, self.params['num_classes'])
            self.word_matrix = nn.Linear(self.params['hidden_dim']*2, self.params['hidden_dim']*2)
            self.word_attention = nn.Linear(self.params['hidden_dim']*2, 1)
            
        else:
            self.sentence_lstm = nn.LSTM(self.params['hidden_dim'], self.params['hidden_dim'], bidirectional=self.params['bidirectional'])
            self.linear = nn.Linear(self.params['hidden_dim'], self.params['num_classes'])
            self.word_matrix = nn.Linear(self.params['hidden_dim'], self.params['hidden_dim'])
            self.word_attention = nn.Linear(self.params['hidden_dim'], 1)
            
    def forward(self, input_indices, sentence_length):
        batch_size = input_indices.size(0)
        sentence_length = sentence_length + (sentence_length == 0).long()
        input_embedding = self.embedding_layer(input_indices.view(-1, self.params['sentence_length']))
        input_len_sorted, perm_idx = sentence_length.view(-1,).sort(0, descending=True)
        _, perm_idx_resort = perm_idx.sort(0, descending=False)
        
        input_embedding_sorted = input_embedding.index_select(0, perm_idx)
        input_embedding_sorted = input_embedding_sorted.transpose(0,1)
        packed_input_embedding_tensor = nn.utils.rnn.pack_padded_sequence(input_embedding_sorted, input_len_sorted.data.cpu().numpy())
        lstm_out_packed, lstm_hidden = self.word_lstm(packed_input_embedding_tensor)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed)
        sentence_encoding = lstm_out.transpose(0,1).index_select(0, perm_idx_resort)
        word_mask = 1 - sentence_encoding.eq(0).float()
        word_att_mask = 1 - torch.sum(word_mask,2).eq(0).float()
        word_ui = F.tanh(torch.mul(self.word_matrix(sentence_encoding), word_mask))
        word_att = torch.mul(torch.exp(self.word_attention(word_ui)),word_att_mask.unsqueeze(2))
        word_att_sum = torch.sum(word_att,1,keepdim=True).expand_as(word_att)
        word_att_softmax = torch.div(word_att, word_att_sum)
        sentence_encoding_attended = torch.sum(torch.mul(sentence_encoding, word_att_softmax.expand_as(sentence_encoding)),1)
        #print(sentence_encoding_attended.size())
        #print(word_att_softmax.size())
        #aa
        linear_input = F.tanh(self.dropout(sentence_encoding_attended))
        output = self.linear(linear_input)
        #print(output.size())
        return output,word_att_softmax


# In[313]:


np.random.seed(1)
model = AttentionLSTM(params, w2v_embeddings)
optimizer = torch.optim.Adam(model.parameters(),lr = .0001)
criterion = nn.CrossEntropyLoss()
if params['USE_CUDA']:
    model = model.cuda(params['gpu'])
    criterion = criterion.cuda(params['gpu'])
number_of_batches = math.ceil(len(train_sentences_index)/params['batch_size'])
#number_of_batches = 10
number_of_val_batches = math.ceil(len(val_sentences_index)/params['test_batch_size'])
#number_of_test_batches = 1

indexes = np.arange(len(train_sentences_index))
val_indexes = np.arange(len(val_sentences_index))
for epoch in range(params['epochs']):
    np.random.shuffle(indexes)
    avgLoss = 0.0
    start_time = time.time()
    model.train()
    print(number_of_batches)
    for batch in range(number_of_batches):
        batch_indexes = indexes[batch*params['batch_size'] : (batch+1)*params['batch_size']]
        batch_sentences = Variable(torch.from_numpy(train_sentences_index[batch_indexes]), volatile=False)
        batch_targets = Variable(torch.from_numpy(train_targets[batch_indexes]), volatile=False)
        batch_len = Variable(torch.from_numpy(train_sentences_len[batch_indexes]), volatile=False)
        if params['USE_CUDA']:
            batch_sentences = batch_sentences.cuda(params['gpu'])
            batch_targets = batch_targets.cuda(params['gpu'])
            batch_len = batch_len.cuda(params['gpu'])
        predictions,_ = model(batch_sentences, batch_len)
        loss = criterion(predictions, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avgLoss += loss.data.cpu().numpy()
        if batch%500 == 0:
            print ("Time:",time.time() - start_time,"\tAverage Loss per Batch:", avgLoss/(batch+1))
    
    model.eval()
    avgValLoss = 0.0
    start_time = time.time()
    predictions_arr = np.zeros_like(val_targets)
    prediction_probs = np.zeros_like(val_targets, dtype=np.float64)
    for batch in range(number_of_val_batches):
        batch_indexes = val_indexes[batch*params['test_batch_size'] : (batch+1)*params['test_batch_size']]
        batch_sentences = Variable(torch.from_numpy(val_sentences_index[batch_indexes]), volatile=True)
        batch_targets = Variable(torch.from_numpy(val_targets[batch_indexes]), volatile=True)
        batch_len = Variable(torch.from_numpy(val_sentences_len[batch_indexes]), volatile=True)
        if params['USE_CUDA']:
            batch_sentences = batch_sentences.cuda(params['gpu'])
            batch_targets = batch_targets.cuda(params['gpu'])
            batch_len = batch_len.cuda(params['gpu'])
        predictions,_ = model(batch_sentences, batch_len)
        
        #val,idx = torch.max(predictions,1)
        #print(idx)
        predictions_arr[batch_indexes] = np.argmax(predictions.data.cpu().numpy(),1)
        prediction_probs[batch_indexes] = F.softmax(predictions,1).data.cpu().numpy()[:,1]
        loss = criterion(predictions, batch_targets)
        avgValLoss += loss.data.cpu().numpy()
    print ("Time:",time.time() - start_time,"\tAverage Test Loss per Batch:", avgValLoss/number_of_val_batches,"\tAccuracy:", accuracy_score(val_targets, predictions_arr))
    scores = precision_recall_fscore_support(val_targets, predictions_arr)
    auc = roc_auc_score(val_targets, prediction_probs)
    precision, recall, _ = precision_recall_curve(val_targets, prediction_probs, pos_label=1)
    pickle.dump([precision, recall], open('results/hierarchical_lstm_'+str(epoch),'wb'),-1)
    print ("precision:",scores[0],"\trecall:",scores[1],"\tfbeta_score:",scores[2],"\tsupport:",scores[3], "\tauc:", auc)
    torch.save({'epoch': epoch ,'model': model.state_dict(), 'optimizer':optimizer.state_dict()}, "outputs/lstm_model_glove_"+str(epoch))

