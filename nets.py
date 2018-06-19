import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        # Load the pretrained ResNet and replace top fc layer.
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet152(pretrained=True)
        resnet = models.resnet10l(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        # Extract feature vectors from input ROI
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_class, num_layers, max_seq_length=20):
        # Set the hyper-parameters and build the layers.
        super(DecoderRNN, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, lengths):
        # Decode image feature vectors and generates captions.
        # features shape [batch, sequence length, feature_size]
        packed = pack_padded_sequence(features, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        # [0] for extract from a pack sequence
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        # Generate positions for given sequence 
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
