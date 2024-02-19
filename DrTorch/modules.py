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
from typing import Union, Any, Dict, List, Tuple, Callable, Optional

import wandb

from .wrappers import Criterion, OptimizerWrapper
from .metrics import Metric
from .callbacks import EarlyStopper

import os
import time
import sys

import torch
import numpy as np

from torchviz import make_dot
from torchsummary import summary
from thop import profile, clever_format
from IPython.display import display


class DrTorchModule(torch.nn.Module):
    """
     Base class for creating PyTorch modules with additional functionalities.

     Attributes:
         device (str): A string indicating the target device for the model (default is 'cpu').

     Methods:
         to(*args, **kwargs):
             Moves and/or casts the parameters and buffers of this module to the specified device.
             Overrides the to method of the parent class.

         visualize_network_graph(input_element, save_image=True, file_folder='.', file_name='my_model', file_format='png'):
             Visualizes the network graph of the model using torchviz.
             Displays the graph using the IPython display module.
             Optionally saves the graph image to a file.

         get_summary(input_size):
             Prints the summary of the model, including information about input/output shapes and parameters.
    """

    def __init__(self):
        """
        Initializes a new instance of the DrTorchModule class.
        """

        super(DrTorchModule, self).__init__()
        self.device = 'cpu'

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers of this module to the specified device.
        Overrides the to method of the parent class.

        :param args: Arguments that specify the target device (e.g., 'cpu' or 'cuda').
        :param kwargs: Additional arguments for the to method.

        :return: Module with parameters and buffers moved/cast to the specified device.

        """

        self.device = args[0]
        return super(DrTorchModule, self).to(*args, **kwargs)

    def visualize_network_graph(self,
                                input_element: torch.Tensor,
                                save_image: bool = True,
                                file_folder: str = './',
                                file_name: str = 'my_model',
                                file_format: str = 'png') -> None:
        """
        Visualizes the network graph of the model using torchviz.

        :param input_element: A sample input element (tensor) to generate the network graph.
        :param save_image: Whether to save the graph image.
        :param file_folder: Folder Path.
        :param file_name: The name of the saved image file (if save_image is True).
        :param file_format: The format of the saved image file (if save_image is True).

        :return: None

        """

        dot = make_dot(self(input_element), params=dict(self.named_parameters()))
        display(dot)
        if save_image:
            dot.render(filename=os.path.join(file_folder, file_name), format=file_format, cleanup=True)

    def get_summary(self, input_size: torch.Size, **kwargs) -> None:
        """
        Prints the summary of the model, including information about input/output shapes and parameters.

        :param input_size: The size of the input tensor (e.g., torch.Size([channels, height, width])).

        :return: None

        """

        input_data = torch.rand(input_size).unsqueeze(0).to(self.device)
        macs, _ = profile(self, inputs=(input_data,), verbose=False)
        macs = clever_format([macs], "%.3f")

        summary(self, input_size=input_size, **kwargs)
        print(f"Number of MAC operations: {macs}")


class TrainableModule(DrTorchModule):
    """
        A class for creating trainable PyTorch modules with convenient training and evaluation methods.
        This class extend DrTorchModule.

        Methods:
            - validate(self, data_loader, criterion, metrics, aggregate_on_dataset=True)
            - fit(self, train_loader, val_loader, criterion, metrics, optimizer, num_epochs, early_stopper=None,
                  aggregate_on_dataset=True, verbose=True)
            - predict(self, data, batch_size=32)

        Attributes:
            No new attributes are introduced in this class.
    """

    def __init__(self):
        super(TrainableModule, self).__init__()

    def __to_device(self,
                    data: Union[torch.tensor, Dict, List],
                    device: Union[torch.device | str]) -> Union[torch.tensor, Dict, List]:
        """
        Transfers the input data or nested data structures to the specified PyTorch device.

        :params: data : The input data or nested data structures to be transferred.
        :params: device The target PyTorch device.
        :returns: torch.Tensor or dict or list or other: The data or nested data structures transferred to the specified device.

        Notes:
        - For torch.Tensor objects, the method uses the `to` method to transfer the tensor to the specified device.
        - For dictionaries, the method recursively applies itself to each value in the dictionary.
        - For lists, the method recursively applies itself to each element in the list.
        - Other data types are returned unchanged.

        Example:
        ```python
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = {
            'image': torch.rand((3, 256, 256)),
            'mask': torch.tensor([0, 1, 2])
        }

        transformed_data = __to_device(data, device)
        ```

        In this example, the `image` tensor and the `mask` tensor inside the `data` dictionary are transferred to the specified device.

        """

        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self.__to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.__to_device(item, device) for item in data]
        else:
            return data

    def validate(self,
                 data_loader: torch.utils.data.DataLoader,
                 criterion: Criterion,
                 metrics: Optional[List[Metric]] = None,
                 aggregate_loss_on_dataset: bool = True) -> Dict[str, float] | Tuple[Dict[str, float], torch.Tensor]:

        """
        Validate the model on the given data loader.

        :param data_loader: DataLoader containing data.
        :param criterion: A dictionary containing the name and the loss function to use.
        :param metrics: A list of dictionaries containing the name and the functions of the metrics to calculate.
        :param aggregate_loss_on_dataset: If True, the reduce strategy is applied over all the loss of the samples of the dataset.
                                          Otherwise, the reduce strategy is applied over all the batches to get a partial loss for each batch,
                                          then the partial losses are reduced to get a unique loss for the epoch.
                                          In the first case, the result is more accurate, but more RAM is used.
                                          In the second case, the result is less accurate due to numerical approximation, and less RAM is used.

        :return: A dictionary containing the validation results, including the loss value and metrics.
                 If `return_predictions` is True, returns a tuple containing the results dictionary and a tensor with
                 model predictions.

        """

        if metrics is None:
            metrics = []

        results = {criterion.name: []}
        for metric in metrics:
            results[metric.name] = []

        aggregated_losses = torch.tensor([], device='cpu')

        self.eval()
        with torch.no_grad():
            for iteration, (inputs, labels) in enumerate(data_loader):
                inputs, labels = self.__to_device(inputs, self.device), labels.to(
                    self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                if aggregate_loss_on_dataset:
                    aggregated_losses = torch.cat((aggregated_losses, loss.to('cpu')))
                else:
                    reduced_batch_loss = criterion.reduction_function(loss)
                    aggregated_losses = torch.cat((aggregated_losses, reduced_batch_loss.unsqueeze(0).to('cpu')))

                for metric in metrics:
                    metric.update_state(outputs, labels)

        results[criterion.name] = criterion.reduction_function(aggregated_losses).item()

        for metric in metrics:
            results[metric.name] = metric.get_result()
            metric.reset_state()

        return results

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            criterion: Criterion,
            optimizer: OptimizerWrapper,
            num_epochs: int,
            metrics: Optional[List[Metric]] = None,
            early_stopper: EarlyStopper = None,
            aggregate_loss_on_dataset: bool = True,
            verbose: bool = True,
            interaction_with_wandb: bool = False,
            interaction_function_with_wandb: Callable = None) -> Dict[str, Dict[str, List[Any]]]:
        """
        Train the model.


        :param train_loader: Training data loader.
        :param val_loader: Validation data loader.
        :param criterion: Loss function.
        :param optimizer: Model optimizer.
        :param num_epochs: Number of training epochs.
        :param metrics: List of dictionaries containing two fields name and function.
        :param early_stopper: An object of type EarlyStopper, if None training goes on until num_epochs has been done
        :param aggregate_loss_on_dataset: If True, the reduce strategy is applied over all the loss of the samples of the dataset.
                                          Otherwise, the reduce strategy is applied over all the batches to get a partial loss for each batch,
                                          then the partial losses are reduced to get a unique loss for the epoch.
                                          In the first case, the result is more accurate, but more RAM is used.
                                          In the second case, the result is less accurate due to numerical approximation, and less RAM is used.
        :param verbose: If true print the training process results.
        :param interaction_with_wandb: If True, log metrics to WandB per epoch.
        :param interaction_function_with_wandb: A callable function to be executed per batch for interaction logging.
                                                Note: This function has to produce a dictionary containing the multimedia data
                                                that you want to send to the wandb servers. Furthermore, it is important that the
                                                callable function shifts the model into evaluation mode before performing any
                                                operations and then switches it back to train mode.
                                                :Example:
                                                    def prepare_multimedia_data(data):
                                                        # Switch model to evaluation mode
                                                        model.eval()

                                                        # Perform operations to prepare multimedia data
                                                        # For example, you might preprocess images, videos, or audio data here

                                                        # Create a dictionary to hold the multimedia data
                                                        multimedia_data = {
                                                            'images': data['images'],
                                                            'videos': data['videos'],
                                                            'audio': data['audio']
                                                        }

                                                        # Switch model back to training mode
                                                        model.train()

                                                        return multimedia_data


        :returns: A dictionary that contain the history for loss and each metric.

        """

        optimizer = optimizer.get_optimizer(self.parameters())

        train_history = {criterion.name: []}
        val_history = {criterion.name: []}

        if metrics is None:
            metrics = []

        for metric in metrics:
            train_history[metric.name] = []
            val_history[metric.name] = []

        iterations_per_epoch = len(train_loader)

        if early_stopper and early_stopper.restore_weights:
            early_stopper.create_directory()

        try:
            for epoch in range(num_epochs):

                log_params = {}
                start_time = time.time()
                self.train()

                for iteration, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = self.__to_device(inputs, self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss = criterion.reduction_function(loss)
                    loss.backward()
                    optimizer.step()
                    metrics_value = []

                    for metric in metrics:
                        metrics_value.append(metric(outputs, labels))

                    if verbose:
                        out_str = f"\r Epoch: {epoch + 1}/{num_epochs} Iterations: {iteration + 1}/{iterations_per_epoch} - {criterion.name}: {loss.item()}"
                        for idx, metric in enumerate(metrics):
                            out_str += f" - {metric.name}: {metrics_value[idx]}"
                        sys.stdout.write(out_str)
                        sys.stdout.flush()

                train_results = self.validate(train_loader,
                                              criterion,
                                              metrics,
                                              aggregate_loss_on_dataset=aggregate_loss_on_dataset)
                val_results = self.validate(val_loader,
                                            criterion,
                                            metrics,
                                            aggregate_loss_on_dataset=aggregate_loss_on_dataset)

                end_time = time.time()

                for key, value in train_results.items():
                    train_history[key].append(value)
                for key, value in val_results.items():
                    val_history[key].append(value)

                if interaction_with_wandb:
                    for (train_key, train_value), (val_key, val_value) in zip(train_results.items(),val_results.items()):
                        log_params['train_' + 'loss' if criterion.name is train_key else train_key] = train_value
                        log_params['val_' + 'loss' if criterion.name is train_key else train_key] = val_value

                    additional_interaction_data = interaction_function_with_wandb(self)
                    wandb.log({**log_params, **additional_interaction_data})

                if verbose:
                    out_str = f"\r Epoch: {epoch + 1}/{num_epochs} Iterations: {iterations_per_epoch}/{iterations_per_epoch} Time: {np.round(end_time - start_time, decimals=3)}s"
                    for key, value in train_results.items():
                        out_str += f" - {key}: {np.round(value, 15)}"
                    for key, value in val_results.items():
                        out_str += f" - {'val_' + key}: {np.round(value, 15)}"

                    sys.stdout.write("\r" + " " * len(out_str) + "\r")
                    sys.stdout.flush()
                    sys.stdout.write(out_str)
                    sys.stdout.flush()
                    print()

                if early_stopper and early_stopper(val_history[early_stopper.monitor], self):
                    if verbose:
                        print(early_stopper.get_message())
                    break
        finally:
            if early_stopper and early_stopper.restore_weights:
                early_stopper.delete_directory()
        if verbose:
            print("Train Completed")

        return {'train': train_history, 'val': val_history}

    def predict(self,
                data: Union[torch.Tensor, Dict],
                model_output_function_transformation: Callable = torch.round) -> torch.Tensor:
        """
        Generate predictions for the given input data using the trained model.

        :param data: Input data for which predictions are to be generated. It can be either a tensor or a dictionary
                    containing input data (e.g., features or images).
        :param model_output_function_transformation: A callable function applied to the model's output for post-processing
                    predictions. Default is torch.round, which is commonly used for binary classification tasks.

        :return: Tensor containing the predicted labels.

        """

        predicted_labels_list = []

        for batch_data, y in data:
            batch_data_device = self.__to_device(data=batch_data, device=self.device)
            batch_output = self(batch_data_device)
            batch_output = model_output_function_transformation(batch_output).detach().cpu()
            predicted_labels_list.append(batch_output)
        predicted_labels = torch.cat(predicted_labels_list, dim=0)

        return predicted_labels
