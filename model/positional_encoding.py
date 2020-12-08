import torch
import torch.nn as nn
import math

# Positional encoding (section 5.1)
class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L= L
        
    def forward(self, inputs):
        '''
        '''
        L = self.L
        encoded = []
        for l in range(L):
            encoded.append(torch.sin((2 ** l * math.pi) * inputs))
            encoded.append(torch.cos((2 ** l * math.pi) * inputs))
        return torch.cat(encoded, -1)
