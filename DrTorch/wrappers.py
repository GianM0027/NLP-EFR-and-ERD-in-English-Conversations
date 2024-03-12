"""



 /$$$$$$$         /$$$$$$$$                            /$$
| $$__  $$       |__  $$__/                           | $$
| $$  \ $$  /$$$$$$ | $$  /$$$$$$   /$$$$$$   /$$$$$$$| $$$$$$$
| $$  | $$ /$$__  $$| $$ /$$__  $$ /$$__  $$ /$$_____/| $$__  $$
| $$  | $$| $$  \__/| $$| $$  \ $$| $$  \__/| $$      | $$  \ $$
| $$  | $$| $$      | $$| $$  | $$| $$      | $$      | $$  | $$
| $$$$$$$/| $$      | $$|  $$$$$$/| $$      |  $$$$$$$| $$  | $$
|_______/ |__/      |__/ \______/ |__/       \_______/|__/  |__/



# ----------------------------------------------------------------------------#
# Copyright(C) 2023-2024 Davide Femia and Riccardo Murgia. Italy              #
#                                                                             #
#                                                                             #
# All rights reserved.                                                        #
#                                                                             #
# This software is the property of Davide Femia and Riccardo Murgia           #
# and may not be modified, distributed, or used without explicit permission.  #
#                                                                             #
#                                                                             #
#                                                                             #
# File:     wrappers.py                                                       #
# Authors:  Davide Femia     <femiadavide04@gmail.com>                        #
#           Riccardo Murgia  <murgiariccardo96@gmail.com>                     #
#                                                                             #
# Date:      11.03.2024                                                       #
#                                                                             #
# ----------------------------------------------------------------------------#



"""


from typing import Iterable, Optional, Callable, Any, Dict, List
from abc import ABC

import torch


class AbstractCriterion(ABC):
    def __init__(self,
                 name: str,
                 reduction_function: Optional[Callable] = None):
        """
        Initialize a custom loss criterion.

        :param name: A name for the criterion.
        :param reduction_function: Specifies the reduction function method that you want to use
        """

        self.name = name
        self.reduction_function = reduction_function

    def __str__(self):
        return self.name


class Criterion(AbstractCriterion):
    """

    A class representing a custom loss criterion for training neural networks.
    This Class is designed to give the possibility to customize your reduction technique simpling modifying the
    constructor method.

    Attributes:
        - name (str): A name for the criterion.
        - loss_function (Callable): A callable loss function instantiated without a reduction function logic.
                              In case of Pytorch loss function instantiated using reduction='none'.
        - reduction_function (str, optional): Specifies the reduction to apply to the output. .


    Methods:
        -__init__(self, name: str, loss_function: Callable, reduction_function: Callable): Construct the object.
        - __call__(predicted_labels, target_labels): Compute the loss between predicted and target labels.
        -_str__(self): Return the name.

    Example:
        w2 = torch.ones(10)  # Replace with actual class weights
        criterion = Criterion('loss', torch.nn.CrossEntropyLoss(reduction='none', weight=w2), reduction_function=torch.mean)

    Note
      - The reduction is set to 'none' to allow the flexibility of applying any desired operation,
        as specified by the `reduction_function` parameter.

    """

    def __init__(self,
                 name: str,
                 loss_function: Callable,
                 reduction_function: Optional[Callable] = None,
                 reshape_loss_input_f: Optional[Callable] = None):
        """
        Initialize a custom loss criterion.

        :param name: A name for the criterion.
        :param loss_function: Instantiated Pytorch loss function or custom loss function instantiated without a reduction function logic.
                              In case of Pytorch loss function instantiated using reduction='none'.
        :param reduction_function: Specifies the reduction function method that you want to use
        """

        super().__init__(name=name, reduction_function=reduction_function)
        self.loss_function = loss_function
        self.reshape_loss_input_f = reshape_loss_input_f

    def __call__(self,
                 predicted_labels: torch.Tensor | Any,
                 target_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between predicted and target labels as a torch.Tensor instantiated on the device
        where the input tensor are located.

        :param predicted_labels: Predicted labels.
        :param target_labels: Target labels.

        :return: The computed loss.

        """

        if self.reshape_loss_input_f is not None:
            predicted_labels = self.reshape_loss_input_f(predicted_labels)
            target_labels = self.reshape_loss_input_f(target_labels)

        output = self.loss_function(predicted_labels, target_labels)

        return output


class MultyHeadCriterion(AbstractCriterion):
    """
    A class representing a multi-head loss criterion for training neural networks.
    This class allows for applying different loss functions to different subsets of predictions and targets.

    Attributes:
        - name (str): A name for the multi-head criterion.
        - loss_functions (Dict[str, Callable]): A dictionary containing loss functions for each head.
        - loss_weights (List[int]): Optional list of weights for each loss when aggregating results.
        - reduction_function (Optional[Callable]): Optional function for aggregating individual head losses.

    Methods:
        - __init__(self, name: str, loss_functions: Dict[str, Callable], loss_weights: List[int],
                   reduction_function: Optional[Callable] = None): Constructor method to initialize the multi-head criterion.
        - __call__(self, predicted_labels: Dict[str, torch.Tensor], target_labels: Dict[str, torch.Tensor]) -> torch.Tensor:
                  Main method to compute the loss values based on the predicted and target labels.

    Example:
        # Replace with actual loss functions and weights
        loss_functions = {
            'head1': torch.nn.CrossEntropyLoss(),
            'head2': torch.nn.MSELoss(),
            ...
        }
        loss_weights = [1, 2, ...]  # Replace with actual weights

        # Instantiate the MultyHeadCriterion class
        criterion = MultyHeadCriterion(name='loss', loss_functions=loss_functions, loss_weights=loss_weights)

        # Example usage of the criterion
        predicted_labels = {'head1': torch.randn(10, 10), 'head2': torch.randn(10, 1), ...}  # Example predicted labels
        target_labels = {'head1': torch.randint(0, 10, (10,)), 'head2': torch.randn(10, 1), ...}  # Example target labels

        # Calculate the loss
        my_loss = criterion(predicted_labels, target_labels)

    """

    def __init__(self,
                 name: str,
                 loss_functions: Dict[str, Callable],
                 loss_weights: Dict[str, float],
                 reduction_function: Optional[Callable] = None,
                 aggregate_losses_f: Callable = torch.sum):
        """
       Initialize a multi-head loss criterion.

       :param name: A name for the multi-head criterion.
       :param loss_functions: A dictionary mapping head keys to loss functions.
       :param loss_weights: Dictionary of weights for each loss when aggregating results.
       :param reduction_function: Optional function for aggregating individual head losses.

       """

        super().__init__(name=name, reduction_function=reduction_function)
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.aggregate_losses_f = aggregate_losses_f

    def __call__(self,
                 predicted_labels: Dict[str, torch.Tensor],
                 target_labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the loss between predicted and target labels as a torch.Tensor instantiated on the device
        where the input tensor are located.

        :param predicted_labels: Predicted labels.
        :param target_labels: Target labels.

        :return: The computed loss.

        """

        losses = []
        for head_key, current_head_loss in self.loss_functions.items():

            losses.append(current_head_loss(predicted_labels=predicted_labels[head_key],
                                            target_labels=target_labels[head_key]))

        if self.loss_weights is not None:
            losses = [self.loss_weights[head_key] * current_loss for head_key, current_loss in zip(self.loss_functions.keys(), losses)]

        losses = torch.stack(losses, dim=0)
        loss = self.aggregate_losses_f(losses, dim=0)

        return loss


class OptimizerWrapper:
    """
       Wrapper class for creating and managing PyTorch optimizers.

       Attributes:
           name (str): A human-readable identifier for the optimizer.
           optimizer_constructor (Type[torch.optim.Optimizer]): The optimizer constructor class.
           optimizer_partial_params (Dict[str, Any]): Partial parameters for the optimizer.

       Methods:
           __init__(self, optimizer_constructor, identifier='', optimizer_partial_params=None):
               Initializes the OptimizerWrapper.

           __str__(self) -> str:
               Returns a human-readable representation of the optimizer.

           get_optimizer(self, net_params: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
               Constructs and returns an instance of the specified optimizer.

    """

    def __init__(self,
                 optimizer_constructor: type(torch.optim.Optimizer),
                 identifier: str = '',
                 optimizer_partial_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the OptimizerWrapper.

        :param optimizer_constructor: The optimizer constructor class.
        :param identifier: Additional identifier for the optimizer (optional).
        :param optimizer_partial_params: Partial parameters for the optimizer (optional).

        """

        name = repr(optimizer_constructor).split("'")[1].split('.')[-1]
        if identifier:
            name += " " + identifier
        self.name = name
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_partial_params = dict() if optimizer_partial_params is None else optimizer_partial_params

    def __str__(self) -> str:
        """
        Get a human-readable representation of the optimizer.

        :return: A string representation of the optimizer.

        """

        return self.name

    def get_optimizer(self, net_params: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
        """
        Construct and return an instance of the specified optimizer.

        :param net_params: Iterable of model parameters.

        :return: An instance of the PyTorch optimizer.

        """

        return self.optimizer_constructor(net_params, **self.optimizer_partial_params)
