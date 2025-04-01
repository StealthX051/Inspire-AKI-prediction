import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

file = '/home/server/Projects/data/AKI/time_series_cleaned.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

def set_seed(seed=42):
    random.seed(seed)                           # Python random module
    np.random.seed(seed)                        # NumPy
    torch.manual_seed(seed)                     # PyTorch CPU
    torch.cuda.manual_seed(seed)                # PyTorch GPU
    torch.cuda.manual_seed_all(seed)            # For multi-GPU setups
    torch.backends.cudnn.deterministic = True   # Makes cuDNN deterministic
    torch.backends.cudnn.benchmark = False      # Avoids dynamic optimizations


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state

        # model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = torch.tanh #lol

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Use Xavier initialization
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self,x, seq_len):
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        packed_input = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        output, (hn, cn) = self.lstm(packed_input)#, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # out = self.relu(hn)
        out = hn
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

import pickle
lstm_input_file_seq_len = '/home/server/Projects/data/AKI/andrew_temp_lstm_midinput_seq_len.pt'
with open(lstm_input_file_seq_len, "rb") as fp:   
    sequence_lengths = pickle.load(fp)

lstm_input_file = '/home/server/Projects/data/AKI/andrew_temp_lstm_midinput.pt'
tensors = torch.load(lstm_input_file)


num_samples_tot = len(sequence_lengths)
split_idx = num_samples_tot // 10

X_train = tensors[split_idx:, :, :-1].to(device)
y_train = tensors[split_idx:, 0, -1].unsqueeze(1).to(device)
seq_len_train = sequence_lengths[split_idx:]
X_test = tensors[:split_idx, :, :-1].to(device)
y_test = tensors[:split_idx, 0, -1].unsqueeze(1).to(device)
seq_len_test = sequence_lengths[:split_idx]

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

extra_string = 'tanh_hs32'
writer = SummaryWriter('/home/server/Projects/data/AKI/runs/lstm_' + datetime.now().strftime("%D:%H:%M:%S") + extra_string)

num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 24 #number of features
hidden_size = 2 * 16 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 

batch_size = 5000
num_train_samples = len(X_train)
num_batches = int(np.ceil(num_train_samples / batch_size))

for learning_rate in [0.001]:

  set_seed(42)
  lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers).to(device)

  criterion = torch.nn.MSELoss()    # mean-squared error for regression
  optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 
  for epoch in tqdm(range(num_epochs)):
    batch_loss = 0
    for i in range(num_batches):
      beginning_idx = i * batch_size
      ending_idx = min(num_train_samples, (i + 1) * batch_size)
      X = X_train[beginning_idx: ending_idx]
      y = y_train[beginning_idx: ending_idx]
      seq_len = seq_len_train[beginning_idx: ending_idx]
      outputs = lstm1.forward(X, seq_len) #forward pass
      
      optimizer.zero_grad() #calculate the gradient, manually setting to 0
      loss = criterion(outputs, y)
      loss.backward() #calculates the loss of the loss function
      optimizer.step() #improve from loss, i.e backprop
      batch_loss += loss.item()

    if epoch % 10 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, batch_loss / num_batches))
      writer.add_scalar('training loss',
                            batch_loss / num_batches,
                            epoch * num_batches + i)
  print("----------")

outputs = lstm1(X_test, seq_len_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
print(classification_report((y_test > 0.3).T.tolist()[0], (outputs > 0.3).T.tolist()[0]))