import torch
from DrTorch.modules import TrainableModule


class BertFreezed(TrainableModule):
    def __init__(self, bert_model, bert_tokenizer, hidden_size=768, n_emotions=7):
        super().__init__()

        self.bert = bert_model
        self.tokenizer = bert_tokenizer

        self.emotions_classification_head = torch.nn.Linear(hidden_size, n_emotions)
        self.triggers_classification_head = torch.nn.Linear(hidden_size, 2)

    def forward(self, kwards: dict):
        pass