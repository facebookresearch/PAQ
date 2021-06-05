#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
import os
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModel


logger = logging.getLogger(__name__)


def _get_proj_keys_from_state_dict(state_dict):
    weight_key = [k for k in state_dict.keys() if 'encode_proj' in k and 'weight' in k]
    bias_key = [k for k in state_dict.keys() if 'encode_proj' in k and 'bias' in k]
    assert len(weight_key) == 1 == len(bias_key)
    weight_key, bias_key = weight_key[0], bias_key[0]
    return weight_key, bias_key


def _get_proj_dim_from_model_path(model_name_or_path):
    state = torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
    proj_dim = None
    if any('encode_proj' in k for k in state.keys()):
        _, bias_key = _get_proj_keys_from_state_dict(state)
        proj_dim = state[bias_key].shape[0]
    return proj_dim


def load_retriever(model_name_or_path):
    logger.info(f'Loading model from: {model_name_or_path}')
    model = RetrieverEncoder.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
    model.eval()
    return model, tokenizer


class RetrieverEncoder(nn.Module):
    """A wrapper for HF models, with an optional projection"""

    def __init__(self, config, proj_dim):
        super().__init__()
        # EncoderBase.__init__(self, config.hidden_size, project_dim)
        self.model = AutoModel.from_config(config)
        self.encode_proj = nn.Linear(config.hidden_size, proj_dim) if proj_dim is not None else None
        self.model.init_weights()

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        proj_dim = _get_proj_dim_from_model_path(model_name_or_path)

        retriever = cls(config, proj_dim)
        state = torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        retriever.model.load_state_dict({k.replace('albert.',''):v for k,v in state.items() if 'encode_proj' not in k}, strict=True)

        if proj_dim is not None:
            weight_key, bias_key = _get_proj_keys_from_state_dict(state)
            retriever.encode_proj.load_state_dict({'weight': state[weight_key], 'bias': state[bias_key]}, strict=True)

        return retriever

    def forward(self, *args, **kwargs):
        seq_outputs = self.model(*args, **kwargs)['last_hidden_state']
        return self.encode_proj(seq_outputs[:, 0]) if self.encode_proj is not None else seq_outputs[:, 0]
