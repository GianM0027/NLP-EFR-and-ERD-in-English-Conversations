import torch
from DrTorch.modules import TrainableModule


class BertFreezed(TrainableModule):

    def __init__(self, bert_model, hidden_size=768, n_emotions=7):
        super().__init__()

        self.bert = bert_model

        self.emotion_classifier = torch.nn.Linear(hidden_size, n_emotions)
        self.trigger_classifier = torch.nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, cls_positions):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Estrai i vettori [CLS] per ogni battuta utilizzando gli indici di cls_positions
        cls_outputs = torch.stack([sequence_output[i, pos, :] for i, pos in enumerate(cls_positions)])

        # Classificazione delle emozioni per ogni battuta
        emotion_logits = self.emotion_classifier(cls_outputs)

        # Identificazione dei trigger per ogni battuta
        trigger_logits = self.trigger_classifier(cls_outputs)

        return emotion_logits, torch.sigmoid(trigger_logits)