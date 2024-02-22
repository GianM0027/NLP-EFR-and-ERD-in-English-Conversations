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

    def forward(self, input_ids, sep_positions) -> List[torch.Tensor]:
        outputs = self.bert(input_ids=input_ids)
        sequence_output = outputs.last_hidden_state

        # Initialize a list to hold SEP output vectors for each example in the batch
        sep_outputs = []

        # Iterate over the batch and extract SEP vectors for each example
        for i, pos in enumerate(sep_positions):
            sep_output = sequence_output[i, pos, :]
            sep_outputs.append(sep_output)

        # Stack the SEP vectors along a new dimension
        sep_outputs = torch.stack(sep_outputs)

        emotion_logits = self.emotion_classifier(sep_outputs)
        trigger_logits = self.trigger_classifier(sep_outputs)

        return [emotion_logits, trigger_logits]
