from typing import List

import torch
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

    def forward(self, inputs) -> List[torch.Tensor]:
        input_ids, sep_positions = inputs
        outputs = self.bert(input_ids=input_ids)
        sequence_output = outputs.last_hidden_state #[B, INPUT_SIZE, HIDDEN_SIZE]     INPUT_SIZE:numero di token presenti nelle utterance

        # Extracting [CLS] vectors for every sentence by using their index
        cls_inputs = torch.stack([sequence_output[:, pos, :] for pos in sep_positions]) # B, n_frasi, hiddensize



        emotion_logits = self.emotion_classifier(cls_inputs)
        trigger_logits = self.trigger_classifier(cls_inputs)

        return [self.softmax(emotion_logits), self.sigmoid(trigger_logits)]
