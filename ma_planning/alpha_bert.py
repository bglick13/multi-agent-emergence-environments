import warnings
from enum import Enum
from typing import Union
from IPython import embed

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

def swish(x):
    return F.sigmoid(x) * x


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x) * x

class AlphaBert(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, ff_dim, n_head, n_encoder_layers, out_ff_dim, n_actions):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers

        # self.embedding_layer = torch.nn.Embedding(input_dim, embedding_dim)
        self.embedding_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, embedding_dim), torch.nn.LayerNorm(embedding_dim), Swish())

        self.encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, dim_feedforward=ff_dim, dropout=0.2)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        self.next_x_out = torch.nn.Linear(out_ff_dim, n_actions)
        self.next_x_hidden = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish())
        self.next_x_output = torch.nn.Sequential(self.next_x_hidden, self.next_x_out)

        self.next_y_out = torch.nn.Linear(out_ff_dim, n_actions)
        self.next_y_hidden = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish())
        self.next_y_output = torch.nn.Sequential(self.next_y_hidden, self.next_y_out)

        self.next_z_out = torch.nn.Linear(out_ff_dim, n_actions)
        self.next_z_hidden = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish())
        self.next_z_output = torch.nn.Sequential(self.next_z_hidden, self.next_z_out)

        self.value_output = torch.nn.Sequential(torch.nn.Linear(embedding_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish(), torch.nn.Linear(out_ff_dim, n_actions))

        self.sep = None
        self.cls = None
        self.le = None


    def embed_observations(self, observations):
        if isinstance(observations, (list, np.ndarray)):
            observations = torch.LongTensor(observations)
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        if cuda:
            observations = observations.cuda()
        return self.embedding_layer(observations)


    def get_attn_maps(self, src: torch.FloatTensor):
        src = self.embedding_layer(src)
        src = src + np.sqrt(self.embedding_dim)
        attn_maps = []
        for i in range(self.n_encoder_layers):
            output = self.encoder.layers[i].self_attn(src, src, src, need_weights=True)
            attn_maps.append(output)
        return attn_maps


    def forward(self, src: torch.FloatTensor):
        src = self.embedding_layer(src)
        src = src + np.sqrt(self.embedding_dim)
        out = self.encoder(src)
        return out


    def get_next_action_output(self, x):
        next_action = []
        for i in range(self.encoder.num_layers):
            output = [self.next_x_output(x[i]).detach().numpy(),
                      self.next_y_output(x[i]).detach().numpy(),
                      self.next_z_output(x[i]).detach().numpy()]
            next_action.append(output)
        return np.array(next_action)

    def get_value_output(self, x):
        return self.value_output(x)


def main():
    obs = torch.FloatTensor(np.random.randint(1, 10, (2, 10)))
    model = AlphaBert(10, 64, 2048, 2, 2, 128, 11)
    x = model.forward(obs)
    a = model.get_next_action_output(x)
    v = model.get_value_output(x)
    print(a)
    print(v)
    embed()

if __name__ == '__main__':
    main()