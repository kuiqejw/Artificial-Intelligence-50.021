import csv
import torch
import torch.nn as nn
from torch import optim
from typing import List
import string
from numpy.random import choice, randint
import matplotlib.pyplot as plt
import time

ALL_LETTERS = string.printable
NUM_CLASSES = len(ALL_LETTERS)
BATCH_SIZE = 1
P1 = [0 for i in range(10)]
FILEPATH = "week9/startrek/star_trek_transcripts_all_episodes.csv"


def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st'] = []
    filterwords = ["NEXTEPISODE"]
    with open(FILEPATH, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and len(el) > 1:
                    # print(el)
                    # .replace(’=’,’’) #.replace(’/’,’ ’).replace(’+’,’ ’)
                    v = el.strip().replace(":", "").replace('\"', '')
                    category_lines['st'].append(v)

    n_categories = len(all_categories)
    # print(len(all_categories), len(category_lines['st']))
    # print(n_categories)
    # print("done")

    return category_lines, all_categories



def one_hot(word: List[str]) -> torch.Tensor:
    out = torch.zeros(len(word), 1, NUM_CLASSES)
    for i in range(len(word)):
        out[i][0][ALL_LETTERS.find(word[i])] = 1
    return out


def one_hot_single(letter) -> torch.Tensor:
    '''
    Returns a one-hot tensor encoding of the letter on ALL_LETTERS
    '''
    output = torch.zeros(NUM_CLASSES)
    output[ALL_LETTERS.find(letter)] = 1
    return output

# x = get_data()
# print(type(x[0]), type(x[1]))
# print(len(x[0]['st']))
# print(x[0]['st'][:100])


class LSTM(nn.Module):
    def __init__(self, input_size=NUM_CLASSES, lr=0.01):
        super(LSTM, self).__init__()
        # Input is a sequence of characters, output is the sentence we want to make.
        # That is how we can calculate loss.
        hidden_size = 200
        num_layers = 2
        dropout = 0.1
        num_classes = input_size
        temperature = 1
        self.LSTM = nn.LSTM(num_classes, hidden_size,
                            num_layers=num_layers, dropout=dropout)
        # hidden size is the same as output size?
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.temperature = temperature
        self.num_layers = num_layers
        self.learning_rate = lr
        self.softmax = nn.Softmax(dim=-1)

    # our lstm outputs another sequence

    def forward(self, inp, hidden):
        start = time.time()
        out, hidden = self.LSTM(inp, hidden)
        # Temperature sampling is done in testing
        output = self.fc(out)
        # # it should be a probability of selecting instead of an argmax?
        P1[9] += time.time()-start
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size), torch.zeros(self.num_layers, 1, self.hidden_size)

# train is per input word.
def train(model, input_word):
    start = time.time()
    model.zero_grad()
    hidden = model.initHidden()
    # We are using _ as a END and * as START

    truth = torch.empty(len(input_word)+1)
    for i, char in enumerate(input_word):
        truth[i] = ALL_LETTERS.find(char)
    truth[-1] = ALL_LETTERS.find("_")
    
    input_tensor = one_hot("*" + input_word)
    
    P1[0] += time.time()-start
    start = time.time()
    
    input_tensor = input_tensor.to(device)
    truth = truth.to(device, dtype=torch.long)
    hidden = hidden[0].to(device), hidden[1].to(device)
    P1[1] += time.time()-start

    # x is len(input_word)
    # truth.shape should be (x)
    # input_tensor.shape should be (x, 1, NUM_CLASSES)
    start = time.time()
    out, hid = model(input_tensor, hidden)
    P1[2] += time.time()-start
    start = time.time()

    out = out.squeeze(1).to(device)
    loss = nn.functional.cross_entropy(out, truth)
    loss.backward()
    optimizer.step()
    P1[3] += time.time()-start

    return loss.item()


def gen(model):
    model.eval()
    model.temperature = 0.5  # we might want to change the temperature while generating
    word = ""
    next_char = "*"
    hidden = model.initHidden()
    with torch.no_grad():
        while(next_char != "_"):  # _ is the end character
            char_tensor = one_hot_single(next_char)

            char_tensor = char_tensor.to(device)
            hidden = hidden[0].to(device), hidden[1].to(device)

            out, hidden = model(char_tensor.reshape(1, 1, -1), hidden)
            out = nn.functional.softmax(out/model.temperature, -1)
            out = out.flatten().cpu().numpy()
            out[out.argmax()] += 1-sum(out)
            # print(sum(out))
            next_char = choice(list(ALL_LETTERS), p=out)
            word += next_char
    return word[:-1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn = LSTM(NUM_CLASSES, lr=0.01)
rnn = rnn.to(device)

optimizer = optim.SGD(rnn.parameters(), rnn.learning_rate)

# --------------------------------

possible_sentences = get_data()[0]['st']

all_losses = []
for i in range(130000):
    ind = randint(len(possible_sentences))
    loss = train(rnn, possible_sentences[ind])
    if (i%100 == 0):
        all_losses.append(loss)
print(P1)

torch.save(rnn.state_dict(), "train130k")

plt.plot(all_losses)
plt.show()

print(gen(rnn))

# rnn.load_state_dict(torch.load("week9/train130k"))
# for i in range(10):
#     print(gen(rnn))

