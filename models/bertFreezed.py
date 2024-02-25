from typing import List, Dict

import torch
from torch import Tensor

from DrTorch.modules import TrainableModule


class BertFreezed(TrainableModule):

    def __split_utterances_features(self, features, mask):
        result = []
        for sentence, sentence_mask in zip(features, mask):
            sep_indices = torch.nonzero(sentence_mask, as_tuple=False).squeeze()
            splits = []
            for i in range(len(sep_indices) - 1):
                start, end = sep_indices[i] + 1, sep_indices[i + 1]
                if start < end:
                    splits.append(sentence[start:end])
            result.append(splits)
        return result

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

    def forward(self, inputs: dict) -> dict[str, Tensor]:
        separator_mask = inputs.pop('t_sep_index')  # n_token
        outputs = self.bert(**inputs)  # torch.Size([32, n_max_frasi, x, 768])     se paddi i dialoghi allo stesso numero di frasi x è uguale per tutte le frasi altrimenti e variabile
        sequence_features = outputs.last_hidden_state  # torch.Size([32, 3, 31, 768])
        print(sequence_features.shape)

        splitted_features = self.__split_utterances_features(features=sequence_features, mask=separator_mask)

        print(len(splitted_features))  # lista di list di tensori

        for l in splitted_features:
            print(len(l))
            print((l[0].shape))
            print((l[1].shape))
            print((l[2].shape))

        emotion_logits = self.emotion_classifier(splitted_features)  # B, N_frasi, C

        trigger_logits = self.trigger_classifier(splitted_features)  # B, N_frasi, 1

        return {'emotions': self.softmax(emotion_logits), 'triggers': self.sigmoid(trigger_logits)}
