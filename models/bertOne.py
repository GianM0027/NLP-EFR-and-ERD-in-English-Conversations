from typing import Tuple, Dict

import numpy as np

from DrTorch.modules import TrainableModule

import torch
import math


class BertOne(TrainableModule):
    """
       A custom module implementing a fine-tuning mechanism for BERT-based models.

       This module allows for fine-tuning a pre-trained BERT model for emotion and trigger classification tasks. It includes
       methods for initializing the module, performing forward passes, and handling input data.

       Args:
           bert_model (BertModel): Pre-trained BERT model.
           cls_input_size (int): Size of the input for the classifier.
           n_emotions (int): Number of classes for emotion classification.
           n_triggers (int): Number of classes for trigger classification.
           freeze_bert_weights (bool): If True, freezes the weights of the BERT model.


       Example:
            ```
              inputs = {
              'input_ids': torch.tensor([[101, 2054, 2003, 1996, 2017, 102],
                                         [101, 2074, 2017, 2024, 2035, 102]]),
              'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1],
                                              [1, 1, 1, 1, 1, 1]]),
              'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0]])
              }
              logits = model.forward(inputs)
             ```

       """

    def __init__(self, bert_model, cls_input_size, n_emotions, n_triggers, freeze_bert_weights=False):
        """
        Initializes the BertFreezed module.

        :params bert_model: Pre-trained BERT model.
        :params cls_input_size: Size of the input for the classifier.
        :params n_emotions: Number of classes for emotion classification.
        :params n_triggers: Number of classes for trigger classification.
        :params freeze_bert_weights: If it is True the bert weights are freezed.

        :returns:None

        """

        super().__init__()

        self.bert = bert_model
        self.config = self.bert.config

        if freeze_bert_weights:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Emotion and Trigger classifiers
        self.emotion_classifier = torch.nn.Linear(in_features=cls_input_size, out_features=n_emotions)
        self.trigger_classifier = torch.nn.Linear(in_features=cls_input_size, out_features=n_triggers)

    def __get_n_chunk(self, input_shape: torch.Size) -> int:
        """
        Calculates the number of chunks needed based on the input shape.

        :params input_shape: Shape of the input tensor.

        :returns:  Number of chunks.

        """
        batch_n, n_sentence, n_token = input_shape

        for chunks in range(1, n_sentence):
            if ((n_sentence % chunks) == 0) and ((n_sentence * n_token) / chunks <= self.bert.config.max_position_embeddings):
                return chunks

        return n_sentence

    @staticmethod
    def __chunk_input(n_chunk, inputs) -> Dict[str, torch.Tensor]:
        """
        Chunks input data into multiple parts for processing.

        :params n_chunk: Number of chunks.
        :params inputs: Dictionary containing input tensors.

        :returns: Chunked input tensors.

        """

        batch_n = inputs['input_ids'].shape[0]

        for key, data in inputs.items():
            inputs[key] = data.view((batch_n * n_chunk, -1))

        return inputs

    @staticmethod
    def __reshape(inputs: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Reshapes input tensors to their original shape in order to aggregate the chunks.

        :params inputs: Input tensor to reshape.
        :params shape: Target shape.

        :returns:  Reshaped input tensor.

        """

        batch_n, n_sentence, n_token = shape
        return inputs.view((batch_n, n_sentence, -1))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the BertFreezed module.

        :params inputs: Dictionary containing input tensors. It should contain the following keys:
                        - 'input_ids': Tensor of shape (batch_size, sequence_length) containing input IDs.
                        - 'attention_mask': Tensor of shape (batch_size, sequence_length) containing attention mask.
                        - 'token_type_ids': Tensor of shape (batch_size, sequence_length) containing token type IDs.

        :returns: Dictionary containing emotion and trigger logits.

        """

        n_chunk = self.__get_n_chunk(inputs['input_ids'].shape)
        chunked_input = self.__chunk_input(n_chunk, inputs.copy())

        features = self.bert(**chunked_input).last_hidden_state
        reshaped_features = self.__reshape(features, inputs['input_ids'].shape)

        emotion_logits = self.emotion_classifier(reshaped_features)
        trigger_logits = self.trigger_classifier(reshaped_features)

        return {'emotions': emotion_logits, 'triggers': trigger_logits}
