from typing import Dict, Tuple

import torch
from torch import Tensor
from transformers import LongformerModel

from DrTorch.modules import TrainableModule


class BigBertOne(TrainableModule):
    def __init__(self, bert_model: LongformerModel,
                 cls_input_size: int,
                 hidden_dim: int,
                 n_emotions: int,
                 n_triggers: int,
                 freeze_bert_weights: bool = False,
                 name: str = 'BigdBertOne'):

        super().__init__()

        self.name = name
        self.bert = bert_model
        self.config = self.bert.config

        if freeze_bert_weights:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.emotion_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=cls_input_size, out_features=hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=hidden_dim, out_features=n_emotions)
        )

        self.trigger_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=cls_input_size, out_features=hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=hidden_dim, out_features=n_triggers)
        )

        self.__init_classifier_weights()

    @staticmethod
    def __reshape_input(inputs: Dict[str, torch.Tensor], shape: Tuple[int, int, int]) -> dict[str, Tensor]:
        """
        Reshapes input tensors to their original shape in order to aggregate the chunks.

        :params inputs: Input tensor to reshape.
        :params shape: Target shape.

        :returns:  Reshaped input tensor.

        """

        batch_n, n_sentence, n_token = shape

        for key, data in inputs.items():
            inputs[key] = data.view((batch_n, n_sentence * n_token))

        return inputs

    @staticmethod
    def __reshape_features(features: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Reshapes input tensors to their original shape in order to aggregate the chunks.

        :params inputs: Input tensor to reshape.
        :params shape: Target shape.

        :returns:  Reshaped input tensor.

        """

        batch_n, n_sentence, n_token = shape
        return features.view((batch_n, n_sentence, -1))

    def __init_classifier_weights(self):
        """
        Initializes the weights of the linear layers in the model using Xavier initialization.

        Xavier's initialization (also known as Glorot initialization) sets the initial weights of the neural network
        layers in a way that ensures the activations neither vanish to zero nor explode to very large values during
        forward propagation. This helps in stabilizing the training process and improving convergence.

        For each module in the model, if it is an instance of a linear layer (`torch.nn.Linear`),
        the weights of that layer are initialized using Xavier normal initialization.

        """

        for name, module in self.named_children():
            if name == 'emotion_classifier' or name == 'trigger_classifier':
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, torch.nn.Linear):
                        torch.nn.init.kaiming_uniform_(sub_module.weight, a=0, mode='fan_in', nonlinearity='relu')
                        if sub_module.bias is not None:
                            torch.nn.init.constant_(sub_module.bias, 0.0)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the BertFreezed module.

        :params inputs: Dictionary containing input tensors. It should contain the following keys:
                        - 'input_ids': Tensor of shape (batch_size, sequence_length) containing input IDs.
                        - 'attention_mask': Tensor of shape (batch_size, sequence_length) containing attention mask.
                        - 'token_type_ids': Tensor of shape (batch_size, sequence_length) containing token type IDs.

        :returns: Dictionary containing emotion and trigger logits.

        """

        reshaped_input = self.__reshape_input(inputs.copy(), inputs['input_ids'].shape)
        features = self.bert(**reshaped_input).last_hidden_state
        reshaped_features = self.__reshape_features(features, inputs['input_ids'].shape)

        emotion_logits = self.emotion_classifier(reshaped_features)
        trigger_logits = self.trigger_classifier(reshaped_features)

        return {'emotions': emotion_logits, 'triggers': trigger_logits}
