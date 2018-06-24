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

lyrics = pd.read_csv('data/prepped_rap_data.csv')

n_songs = max(lyrics['song_id']) + 1
# split up train and test by song - we don't want song-specific words to
# validate out of sample
test_songs = np.random.choice(np.arange(n_songs), n_songs//4, replace=False)
test_filter = lyrics['song_id'].isin(test_songs)

def formatted_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^\w\s]', '' , lyrics) # removes punctuation
    lyrics = word_tokenize(lyrics)
    return [word for word in lyrics if word not in STOP_WORDS]

formatted_lyrics = [formatted_lyrics(lyric) for lyric in lyrics['verse_lyrics']]

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

number_lyrics, word_dict = lyric_to_int(formatted_lyrics)

artist_dict = {artist:i for i, artist in enumerate(set(lyrics['primary_artist']))}
labels = [artist_dict[artist] for artist in lyrics['primary_artist']]

useful_verses = [len(verse) > 0 for verse in number_lyrics]

X_train = list(compress(number_lyrics, useful_verses & np.logical_not(test_filter)))
X_test = list(compress(number_lyrics, useful_verses &test_filter))

y_train = list(compress(labels, np.logical_not(test_filter)))
y_test =list(compress(labels, test_filter))
#
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


loss_function = nn.NLLLoss()
model = WhoRappedIt(vocab_size=len(word_dict),
                    embedding_dim=20,
                    hidden_dim=15,
                    batch_size=1,
                    n_rappers=len(artist_dict))

optimizer = optim.SGD(model.parameters(), lr=0.1)

losses = []

for epoch in range(20):
    total_loss = torch.Tensor([0])
    for i in range(len(X_train)):



        model.zero_grad()
        model.hidden = model.init_hidden()
        lyric_tensor = torch.autograd.Variable(torch.tensor(X_train[i], dtype=torch.long))
        target_tensor = torch.autograd.Variable(torch.tensor([y_train[i]], dtype=torch.long))
        log_probs = model(lyric_tensor)
        loss = loss_function(log_probs.view(1, -1), target_tensor)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    print(total_loss)
