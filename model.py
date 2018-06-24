import re
from itertools import compress
import numpy as np
import pandas as pd
from nltk import word_tokenize, download
download('stopwords')
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

STOP_WORDS = set(stopwords.words('english'))

def create_test_filter(data, test_pct):
    '''
    splits up train and test by song - we don't want song-specific words to
    validate out of sample
    '''
    n_songs = max(data['song_id']) + 1
    test_songs = np.random.choice(np.arange(n_songs), int(n_songs * test_pct),
                                  replace=False)
    return data['song_id'].isin(test_songs)

def format_lyrics(lyrics):
    '''
    lowercases, tokenizes, and removes punctuation and stop words
    '''
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^\w\s]', '' , lyrics) # removes punctuation
    lyrics = word_tokenize(lyrics)
    return [word for word in lyrics if word not in STOP_WORDS]

def lyric_to_int(lyrics):
    '''
    takes a list of tokenized verse lyrics and returns a list of lists of ints
    along with the word to index dictionary
    '''
    vocab = set([])
    for lyric in lyrics:
        vocab = vocab.union(lyric)
    vocab_dict = {word:i for i, word in enumerate(vocab)}
    return [[vocab_dict[word] for word in lyric] for lyric in lyrics], vocab_dict

def prep_data(data, test_pct):
    '''
    preps lyrics dataset for modeling
    returns: train data tuple, test data tutple, word dictionary, artist dictionary
    '''
    lyrics = data.sample(frac=1) # shuffle verse order
    test_filter = create_test_filter(lyrics, test_pct)

    formatted_lyrics = [format_lyrics(lyric) for lyric in lyrics['verse_lyrics']]
    number_lyrics, word_dict = lyric_to_int(formatted_lyrics)

    artist_dict = {artist:i for i, artist in enumerate(set(lyrics['primary_artist']))}
    labels = [artist_dict[artist] for artist in lyrics['primary_artist']]

    useful_verses = [len(verse) > 0 for verse in number_lyrics]

    X_train = list(compress(number_lyrics, useful_verses & np.logical_not(test_filter)))
    y_train = list(compress(labels, useful_verses & np.logical_not(test_filter)))

    X_test = list(compress(number_lyrics, useful_verses & test_filter))
    y_test = list(compress(labels, useful_verses & test_filter))

    return (X_train, y_train), (X_test, y_test), word_dict, artist_dict

class WhoRappedIt(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, n_rappers):
        super(WhoRappedIt, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.linear2 = nn.Linear(128, n_rappers)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        '''
        resets the hidden state, do this before each pass of lstm
        '''
        return (torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(inputs), self.batch_size, -1), self.hidden)
        out = F.relu(self.linear1(lstm_out[-1]))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train_model(model, train_data, test_data, epochs):
    '''
    trains the neural network
    '''
    X_train, y_train = train_data
    X_test, y_test = train_data

    loss_function = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    losses = []

    for epoch in range(epochs):
        print("Epoch %d\n" % epoch)

        train_loss = 0
        train_correct = 0
        for i in range(len(X_train)):

            model.zero_grad()
            model.hidden = model.init_hidden()
            lyric_tensor = torch.autograd.Variable(torch.tensor(X_train[i], dtype=torch.long))
            target_tensor = torch.autograd.Variable(torch.tensor([y_train[i]], dtype=torch.long))

            log_probs = model(lyric_tensor)

            train_correct += (log_probs.view(-1).max(0)[1] == y_train[i])
            loss = loss_function(log_probs.view(1, -1), target_tensor)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print("Train Loss %.2f" % train_loss)
        print("Train Accuracy: %0.3f\n" % (int(train_correct)/len(X_train)))
        losses.append((train_loss, int(train_correct)/len(X_train)))

        test_loss = 0
        correct = 0
        for i in range(len(X_test)):
            model.hidden = model.init_hidden()
            lyric_tensor = torch.autograd.Variable(torch.tensor(X_test[i], dtype=torch.long))
            log_probs = model(lyric_tensor)

            correct += (log_probs.view(-1).max(0)[1] == y_test[i])
            loss = loss_function(log_probs.view(1, -1), target_tensor)
            test_loss += loss.item()

        print("Test Loss %.2f" % test_loss)
        print("Test Accuracy: %0.3f\n" % (int(correct)/len(X_test)))

    return model

if __name__ == '__main__':

    lyrics = pd.read_csv('data/prepped_rap_data.csv')

    train_data, test_data, word_dict, artist_dict = prep_data(lyrics, test_pct=0.25)

    model = WhoRappedIt(vocab_size=len(word_dict),
                        embedding_dim=20,
                        hidden_dim=15,
                        batch_size=1,
                        n_rappers=len(artist_dict))

    train_model(model, train_data, test_data, epochs=20)
