import warnings
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

class AlphaBert(torch.nn.module):
    def __init__(self, embedding_dim, ff_dim, n_head, n_encoder_layers, out_ff_dim, mask_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers

        self.embedding_layer = torch.nn.Sequential(torch.nn.Linear(embedding_dim, embedding_dim), torch.nn.LayerNorm(embedding_dim), Swish())

        self.n_encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, n_head, dim_feedforward=ff_dim, dropout=0.2)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)

        self.move_x_layer = torch.nn.Sequential(torch.nn.Linear(ff_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish())
        self.move_y_layer = torch.nn.Sequential(torch.nn.Linear(ff_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish())
        self.move_z_layer = torch.nn.Sequential(torch.nn.Linear(ff_dim, out_ff_dim), torch.nn.LayerNorm(out_ff_dim), Swish())

        self.pe = PositionalEncoding(embedding_dim, 0, max_len=n_encoder_layers)

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


    def get_attn_maps(self, src: torch.LongTensor):
        src = self.embedding_layer(src)
        src = src + np.sqrt(self.embedding_dim)

        src = src.permute(1, 0, 2)
        attn_maps = []
        for i in range(self.n_encoder_layers):
            output = self.encoder.layers[i].self_attn(src, src, src, need_weights=True)
            attn_maps.append(output)
        return attn_maps


    def forward(self, src: torch.LongTensor):
        src = self.embedding_layer(src)
        src = src + np.sqrt(self.embedding_dim)
        src = src.permute(1, 0, 2)


    def get_next_action(self, x):
        next_action = []
        next_action.append(np.argmax(np.softmax(self.move_x_layer(x))))
        next_action.append(np.argmax(np.softmax(self.move_x_layer(y))))
        next_action.append(np.argmax(np.softmax(self.move_x_layer(z))))
        return np.array(next_action)


    def predict(self, src: torch.LongTensor, mask: torch.BoolTensor, task: AlphaBertTasks, **predict_kwargs):

        if isinstance(src, (list, np.ndarry)):
            src = torch.LongTensor(src)
        if isinstance(mask, (list, np.ndarray)):
            mask = torch.BoolTensor(mask)
        self.eval()

        if task == AlphaBertTasks.