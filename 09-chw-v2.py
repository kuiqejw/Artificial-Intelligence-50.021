from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import csv
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


#%%


all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
n_letters = len(all_letters) + 1 # Plus EOS marker

cutData = False


#%%

class chw09Dataset(Dataset):

    def __init__(self,lines,device):
        self.lines = lines
        self.device = device

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        data = inputTensor(self.lines[idx],self.device)
        labels = targetTensor(self.lines[idx],self.device)
        return (data,labels)
    
    
class chw09DataLoader(): 
    
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.count = 0
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count < len(self.dataset):
            batch = self.dataset[self.count]
            self.count += 1
            return batch
        else:
            self.count = 0
            raise StopIteration()


class RNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=dropout).to(self.device)
        self.linear = nn.Linear(hidden_size,output_size).to(self.device)

    def forward(self, x):
        output,hc = self.lstm(x,(self.h0,self.c0))
        return self.linear(output.squeeze(1)),hc

    
    def initHidden(self,hc=None):
        if hc == None:
            self.h0 = torch.zeros(self.num_layers,1,self.hidden_size).to(self.device)
            self.c0 = torch.zeros(self.num_layers,1,self.hidden_size).to(self.device)
        else:
            self.h0,self.c0 = hc


# Sample from a category and starting letter
def sample(args,device,model,start_letter='S'):
    with torch.no_grad():  # no need to track history in sampling
        letter_tensor = inputTensor(start_letter,device)
        model.initHidden()

        output_line = start_letter

        for i in range(140): # Max length of a tweet
            output, hc = model(letter_tensor)
            model.initHidden(hc)
            probs = np.array((F.softmax(output.squeeze(0)/0.5,dim=0)).cpu()) # Temperature = 0.5\
            probs /= probs.sum() # because it didn't sum to 1
            next_letter_idx = np.random.choice(range(len(probs)),p=probs)
            if next_letter_idx == n_letters - 1:
                break
            else:
                letter = all_letters[next_letter_idx]
                output_line += letter
            letter_tensor = inputTensor(letter,device)

        return output_line


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line,device):
    tensor = torch.zeros(len(line), 1, n_letters).to(device)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line,device):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes).to(device)


def get_data(cut=False):
    category_lines = {}
    category_lines['st'] = []
    filterwords = ['NEXTEPISODE']
    
    with open('star_trek_transcripts_all_episodes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',quotechar='"')
        i = 0
        for row in reader:
            if cut:
                i += 1
            for el in row:
                if (el not in filterwords) and (len(el)>1) and i <= 10:
#                    print(el)
                    v = el.strip().replace(';','').replace('\"','')#.replace('=','')#.replace('/','')
                    category_lines['st'].append(v)
    
    return category_lines


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
    

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    toSampleLog(args,'\nEpoch %d\n' % epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.initHidden()
        output,_ = model(data)
        loss = F.cross_entropy(output, target,reduction='mean')
        train_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % (args.log_interval) == 0:
            toLog(args,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if batch_idx % args.sampling_interval == 0 and batch_idx != 0:
            drawSamples(args,device,model)
            
    train_loss /= len(train_loader)
    
    return train_loss


def val(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    num_char = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            model.initHidden()
            output,_ = model(data)
            val_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.argmax(dim=1, keepdim=True) 
            num_char += target.shape[0]
            correct += pred.eq(target.view_as(pred)).sum().item()

#    val_loss /= len(val_loader.dataset)
    val_loss /= len(val_loader)
    val_acc = 100. * correct / num_char

    toLog(args,'\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, num_char, val_acc))
    
    drawSamples(args,device,model)
    
    return val_acc, val_loss


def adjust_learning_rate(args,optimizer, epoch):
    #Sets the learning rate to the initial LR decayed by 10 every 16 epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * (0.1 ** (epoch / 16))
        

def toLog(args,string):
    print(string)
    with open(args.log,'a') as f:
        f.write(string)
        f.write('\n')
        
        
def toSampleLog(args,string):
    with open(args.samples,'a') as f:
        f.write(string)
        f.write('\n')
        
        
def drawSamples(args,device,model,num=10):
    
    toLog(args,'\nSampling:')
    for i in range(num):
        line = sample(args,device,model,randomChoice(all_letters[26:52]))
        toSampleLog(args,line)
        
        if i < 3:
            if len(line) > 50:
                toLog(args,line[:47]+'...')
            else:
                toLog(args,line)
    toLog(args,'')
        
    
#%%

def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='chw9')
    parser.add_argument('--epochs', type=int, default=24, metavar='N',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='D',
                        help='Dropout (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='L',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log', type=str, default='09-chw-log.txt', metavar='l',
                        help='log file')
    parser.add_argument('--samples', type=str, default='09-chw-samples.txt', metavar='sam',
                        help='samples file')
    parser.add_argument('--sampling-interval', type=int, default=5000, metavar='SI',
                        help='how many batches to wait before sampling lines')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
        
    
    # Get data, create datasets and dataloaders
    data = get_data(cutData)['st']
    tr,te = train_test_split(data,test_size=0.25,random_state=1)
    train_data = chw09Dataset(tr,device)
    val_data = chw09Dataset(te,device)
    for_plotting = []
    train_loader = chw09DataLoader(args,train_data)
    val_loader = chw09DataLoader(args,val_data)
    
    # Create model and optimizer
    n_layers = 2
    n_hidden = 200
    model = RNN(device,n_letters,n_hidden,n_letters,n_layers,0.1)
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum)
    
    # Train and test
    train_losses = []
    val_losses = []
    val_accs = []
    best_loss = 0
    best_acc = 0
    best_state_dict = dict()
    for epoch in range(1, args.epochs + 1):
        epoch_train_loss = train(args, model, device, train_loader, optimizer,
                                 epoch)
        epoch_val_acc,epoch_val_loss = val(args, model, device, val_loader)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        if epoch == 1:
            best_loss = epoch_val_loss
            best_acc = epoch_val_acc
            best_state_dict = model.state_dict()
        else:
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_acc = epoch_val_acc
                best_state_dict = model.state_dict()
        adjust_learning_rate(args,optimizer,epoch)
    
    state_dict_file = "chw9-dict.pt"
    torch.save(best_state_dict,os.path.join(state_dict_file))
    for_plotting.append((train_losses,val_losses,val_accs))

    ###############################################################################
    # Results
    ###############################################################################
    
    for train_losses,val_losses,val_accs in for_plotting:
        fig1 = plt.figure(figsize=(20,5))
        fig1.add_subplot(131)
#        plt.ylabel("Training Loss")
        plt.xlabel("Epoch")
        plt.title("Training Loss")
        plt.plot(np.arange(1,args.epochs+1), train_losses)
        fig1.add_subplot(132)
#        plt.ylabel("Test Loss")
        plt.xlabel("Epoch")
        plt.title("Test Loss")
        plt.plot(np.arange(1,args.epochs+1), val_losses)
        fig1.add_subplot(133)
#        plt.ylabel("Test Accuracy")
        plt.xlabel("Epoch")
        plt.title("Test Accuracy")
        plt.plot(np.arange(1,args.epochs+1), val_accs)
#        plt.show()
        fig1.savefig("chw9-plots.png")
    
    toLog(args,'Best test accuracy: %.2f%%' % (best_acc))
            
if __name__ == '__main__':
    main()
    
    