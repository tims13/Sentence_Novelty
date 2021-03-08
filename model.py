import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(self, vocab, dimension=128):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size = 300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # get the output of 2 directions and then concat them
        out_forward = output[range(len(output)), text_len-1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_combine = torch.cat((out_forward, out_reverse), 1)
        text_features = self.drop(out_combine)
        # get the deep features
        self.deep_features = text_features.detach()
        text_features = torch.squeeze(self.fc(text_features), 1)
        text_output = torch.sigmoid(text_features)
        return text_output

