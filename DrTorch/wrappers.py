"""



 /$$$$$$$         /$$$$$$$$                            /$$
| $$__  $$       |__  $$__/                           | $$
| $$  \ $$  /$$$$$$ | $$  /$$$$$$   /$$$$$$   /$$$$$$$| $$$$$$$
| $$  | $$ /$$__  $$| $$ /$$__  $$ /$$__  $$ /$$_____/| $$__  $$
| $$  | $$| $$  \__/| $$| $$  \ $$| $$  \__/| $$      | $$  \ $$
| $$  | $$| $$      | $$| $$  | $$| $$      | $$      | $$  | $$
| $$$$$$$/| $$      | $$|  $$$$$$/| $$      |  $$$$$$$| $$  | $$
|_______/ |__/      |__/ \______/ |__/       \_______/|__/  |__/



"""


from typing import Iterable, Optional, Callable, Any, Dict
import torch


class Criterion:
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
                 reduction_function: Callable):
        """
        Initialize a custom loss criterion.

        :param name: A name for the criterion.
        :param loss_function: Instantiated Pytorch loss function or custom loss function instantiated without a reduction function logic.
                              In case of Pytorch loss function instantiated using reduction='none'.
        :param reduction_function: Specifies the reduction function method that you want to use
        """

        self.name = name
        self.loss_function = loss_function
        self.reduction_function = reduction_function

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

        output = self.loss_function(predicted_labels, target_labels)

        return output

    def __str__(self):
        return self.name


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
