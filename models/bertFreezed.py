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

    def forward(self, input_ids, cls_positions) -> List[torch.Tensor]:

        outputs = self.bert(input_ids=input_ids)
        sequence_output = outputs.last_hidden_state

        # Extracting [CLS] vectors for every sentence by using their index
        cls_outputs = torch.stack([sequence_output[0, pos, :] for pos in cls_positions])

        emotion_logits = self.emotion_classifier(cls_outputs)

        trigger_logits = self.trigger_classifier(cls_outputs)

        return [emotion_logits, trigger_logits]