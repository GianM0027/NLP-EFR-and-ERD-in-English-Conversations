import math
from typing import List, Dict

import torch
from torch import Tensor
import numpy as np

from DrTorch.modules import TrainableModule


class BertFreezed(TrainableModule):

    def __init__(self, bert_model, hidden_size=768, n_emotions=7):
        super().__init__()

        self.bert = bert_model

        # freezing params of bert layer
        for param in self.bert.parameters():
            param.requires_grad = False

        self.emotion_classifier = torch.nn.Linear(hidden_size, n_emotions)
        self.trigger_classifier = torch.nn.Linear(hidden_size, 2)
        self.softmax = torch.nn.functional.softmax
        self.sigmoid = torch.nn.functional.sigmoid

    def __get_n_chunk(self, input_shape):
        batch_n, n_sentence, n_token = input_shape
        x = math.ceil((n_sentence * n_token) / self.bert.config.max_position_embeddings)
        while n_sentence % x != 0:
            x += 1

        return x

    @staticmethod
    def __chunk_input(n_chunk, inputs):
        batch_n = inputs['input_ids'].shape[0]

        for key, data in inputs.items():
            inputs[key] = data.view((batch_n * n_chunk, -1))

        return inputs

    def __reshape(self, input, shape):
        batch_n, n_sentence, n_token = shape
        return input.view((batch_n, n_sentence, -1))


    def forward(self, inputs: dict) -> dict[str, Tensor]:
        n_chunk = self.__get_n_chunk(inputs['input_ids'].shape)
        chunked_input = self.__chunk_input(n_chunk, inputs.copy())
        features = self.bert(**chunked_input).last_hidden_state #todo indagare last hidden_state
        reshaped_features = self.__reshape(features, inputs['input_ids'].shape)
        porcodio = outputs.last_hidden_state.view((inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], -1))
        print(porcodio.shape)
        # sequence_features = outputs.last_hidden_state

        # emotion_logits = self.emotion_classifier(sequence_features)
        # trigger_logits = self.trigger_classifier(splitted_features)

        return 0  # {'emotions': self.softmax(emotion_logits, dim=1), 'triggers': self.sigmoid(trigger_logits)}
