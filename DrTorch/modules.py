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
# File:     modules.py                                                        #
# Authors:  Davide Femia     <femiadavide04@gmail.com>                        #
#           Riccardo Murgia  <murgiariccardo96@gmail.com>                     #
#                                                                             #
# Date:      11.03.2024                                                       #
#                                                                             #
# ----------------------------------------------------------------------------#



"""

from typing import Union, Any, Dict, List, Tuple, Callable, Optional

import wandb
import netron

from .wrappers import Criterion, OptimizerWrapper, MultiHeadCriterion
from .metrics import Metric, MultiHeadMetric, SingleHeadMetric
from .callbacks import EarlyStopper, MultipleEarlyStoppers

import time
import sys

import torch
import numpy as np

from collections import OrderedDict
from thop import profile, clever_format
from IPython.lib.display import IFrame


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

    def _to_device(self,
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

        transformed_data = _to_device(data, device)
        ```

        In this example, the `image` tensor and the `mask` tensor inside the `data` dictionary are transferred to the specified device.

        """

        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, Dict):
            return {key: self._to_device(value, device) for key, value in data.items()}
        elif isinstance(data, List):
            return [self._to_device(item, device) for item in data]
        else:
            return data

    def visualize_network_graph(self,
                                input: torch.Tensor | Dict[str, torch.Tensor],
                                file_nme: str = "default_name.onnx",
                                input_names: Optional[List[str]] = None,
                                output_names: Optional[List[str]] = None,
                                export_params: bool = True,
                                visualize_in_browser: bool = False,
                                width: int = 1400,
                                height: int = 700,
                                verbosity: int = 0,
                                **kwargs: Dict[str, Any]):
        """
        Visualizes the network graph of the model using ONNX.

        :param input: Input data for generating the network graph. It can be either a torch.Tensor or a dictionary containing input data.
        :param file_nme: Name of the file to save the generated graph (default is "default_name.onnx").
        :param input_names: Names of the input nodes in the generated ONNX file (default is ['inputs']).
        :param output_names: Names of the output nodes in the generated ONNX file (default is same as input_names).
        :param export_params: If True, exports parameters along with the network graph (default is True).
        :param visualize_in_browser: If True, opens the network graph in a browser using Netron (default is False).
        :param width: Width of the displayed network graph (default is 1400).
        :param height: Height of the displayed network graph (default is 700).
        :param verbosity: Verbosity level of Netron (0 is silent, 1 is minimal, 2 is verbose) (default is 0).
        :param kwargs: Additional keyword arguments to pass to torch.onnx.export().

        :return: If visualize_in_browser is False, returns an IFrame object displaying the network graph,
                otherwise the graph will be shown in the browser.

        """

        input = self._to_device(data=input, device=self.device)

        if isinstance(input, Dict):
            input = (input, input)
        else:
            raise TypeError('Invalid type for attribute input. Only torch.Tensor or Dict are allowed.')

        input_names = input_names if input_names is not None else ['inputs']
        output_names = output_names if output_names is not None else ['outputs']

        torch.onnx.export(self, input, file_nme, input_names=input_names, output_names=output_names,
                          export_params=export_params, **kwargs)

        netron_url = "http://localhost:8080/"
        netron.start(file_nme, browse=visualize_in_browser, verbosity=verbosity)

        if not visualize_in_browser:
            return IFrame(netron_url, width=width, height=height)

    def summary(self,
                input_data: torch.Tensor | Dict[str, torch.Tensor] | List[torch.Tensor],
                batch_size: int = -1,
                verbose: bool = True):
        """
        This method provides a custom summary of the neural network model, including details about each layer's
        input and output shapes, parameter count, and memory consumption. It also estimates the number
        of MAC (Multiply-Accumulate) operations.

        :param input_data: Input data to the model. It could be a torch.Tensor, a list of torch.Tensors, or a dictionary of torch.Tensors.
        :param batch_size: Batch size for the input data. Default is -1, which will be inferred from the input data if possible.
        :param verbose: If True, prints the summary; otherwise, suppresses printing.

        :returns: None

        """

        def register_hook(module):

            def hook(my_module, input, output):
                class_name = str(my_module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                if len(input) > 0:
                    summary[m_key]["input_shape"] = list(input[0].size())
                    summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    if hasattr(output, "size"):
                        summary[m_key]["output_shape"] = list(output.size())
                        summary[m_key]["output_shape"][0] = batch_size
                    elif hasattr(output, "last_hidden_state"):
                        summary[m_key]["output_shape"] = list(output.last_hidden_state.size())
                        summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(my_module, "weight") and hasattr(my_module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(my_module.weight.size())))
                    summary[m_key]["trainable"] = my_module.weight.requires_grad
                if hasattr(my_module, "bias") and hasattr(my_module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(my_module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, torch.nn.Sequential)
                    and not isinstance(module, torch.nn.ModuleList)
                    and not (module == self)
            ):
                hooks.append(module.register_forward_hook(hook))

        if isinstance(input_data, torch.Tensor):
            batch_size = input_data.size()[0]
        elif isinstance(input_data, List):
            batch_size = input_data[0].size()[0]
        elif isinstance(input_data, Dict):
            batch_size = list(input_data.values())[0].size()[0]
        else:
            raise TypeError("Inconsistent input_data Type. The input_data could be either a list of torch "
                            "Tensors or a dictionary. This function utilizes the forward pass of your model, "
                            "so ensure that your model is compatible with one of these types of data_input.")

        input_data = self._to_device(input_data, self.device)

        summary = OrderedDict()
        hooks = []

        self.apply(register_hook)

        _ = self(input_data)

        for h in hooks:
            h.remove()

        total_params = 0
        total_output = 0
        trainable_params = 0

        separator_len = 50

        if verbose:
            line_new = "{:>25}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
            separator_len = len(line_new)
            print("-" * separator_len)
            print(line_new)
            print("=" * separator_len)
            for layer in summary:
                # input_shape, output_shape, trainable, nb_params
                line_new = "{:>25}  {:>25} {:>15}".format(
                    layer,
                    str(summary[layer].get("output_shape", '-')),
                    "{0:,}".format(summary[layer]["nb_params"]),
                )
                total_params += summary[layer]["nb_params"]
                total_output += np.prod(summary[layer].get("output_shape", 0))
                if "trainable" in summary[layer]:
                    if summary[layer]["trainable"]:
                        trainable_params += summary[layer]["nb_params"]
                print(line_new)

        # Assume 4 bytes/number (float on cuda).
        if isinstance(input_data, torch.Tensor):
            total_input_size = abs(np.prod(input_data.size()) * 4. / (1024 ** 2.))
        elif isinstance(input_data, List):
            total_input_size = abs(sum(np.prod(v.size()) for v in input_data) * 4. / (1024 ** 2.))
        else:
            total_input_size = abs(sum(np.prod(v.size()) for v in input_data.values()) * 4. / (1024 ** 2.))

        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        macs, _ = profile(self, inputs=(input_data,), verbose=False)
        macs = clever_format([macs], "%.3f")

        print("=" * separator_len)
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("-" * separator_len)
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("-" * separator_len)
        print(f"Number of MAC operations: {macs}")
        print("=" * separator_len)


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
        self.batch_result_str = None

    def __print_batch_results(self,
                              current_epoch: int,
                              tot_num_epochs: int,
                              current_iteration: int,
                              iterations_per_epoch: int,
                              criterion_name: str,
                              loss: torch.Tensor,
                              metrics_results: Dict[str, torch.Tensor]):
        """
        Print the results of the current batch.

        :param current_epoch: The current epoch index.
        :param tot_num_epochs: The total number of epochs.
        :param current_iteration: The current iteration index within the epoch.
        :param iterations_per_epoch: The total number of iterations within an epoch.
        :param criterion_name: The name of the criterion used for training.
        :param loss: The loss tensor for the current batch.
        :param metrics_results: A dictionary containing the results of metrics for the current batch.

        """
        batch_result_str = f"\rEpoch: {current_epoch + 1}/{tot_num_epochs} Iterations: {current_iteration + 1}/{iterations_per_epoch} - {criterion_name}: {loss.item():.10f}"
        for metric_name, metric_val in metrics_results.items():
            batch_result_str += f" - {metric_name}: {metric_val:.10f}"

        self.batch_result_str = batch_result_str
        sys.stdout.write(batch_result_str)
        sys.stdout.flush()

    def __print_epoch_results(self,
                              train_results: Dict[str, float],
                              val_results: Dict[str, float],
                              current_epoch: int,
                              tot_num_epochs: int,
                              training_time: float,
                              verbose: int):
        """
        Print the results of the current epoch.

        :param train_results: A dictionary containing the results of metrics for the training set.
        :param val_results: A dictionary containing the results of metrics for the validation set.
        :param current_epoch: The current epoch index.
        :param tot_num_epochs: The total number of epochs.
        :param training_time: The time taken for training the epoch.
        :param verbose: The verbosity level. If 2, results are printed in a tabular format; otherwise, results are printed in a classic format.

        """

        sys.stdout.write("\r" + " " * (len(self.batch_result_str)) + "\r")
        sys.stdout.flush()

        if verbose == 2:
            time_str = f"Time: {training_time:.4f}s"
            epoch_str = f"Epoch {current_epoch + 1}/{tot_num_epochs}"
            epoch_width = len(epoch_str)
            max_metric_name_length = max(len(metric_name) for metric_name in train_results.keys())
            num_separator = max_metric_name_length + 50
            out_str = f"\r{'=' * num_separator}\n"
            sys.stdout.write(f"{out_str}")
            sys.stdout.flush()
            out_str = f"\r{epoch_str}{time_str:>{num_separator - epoch_width - 1}}\n"
            out_str += "=" * num_separator + "\n"
            out_str += f"\r{'Metric/Loss':<{max_metric_name_length + 10}}{'Training':<25}{'Validation':<25}\n"
            out_str += "-" * num_separator + "\n"
            for (train_key, train_value), (val_key, val_value) in zip(train_results.items(), val_results.items()):
                train_formatted = "{:.10f}".format(train_value)
                val_formatted = "{:.10f}".format(val_value)
                out_str += f"\r{train_key.capitalize():<{max_metric_name_length + 10}}{train_formatted:<25}{val_formatted:<25}\n"
        else:
            out_str = f"\r Epoch: {current_epoch + 1}/{tot_num_epochs} Time: {training_time:.4f}s"
            for key, value in train_results.items():
                out_str += f" - {key}: {value:.10f}"
            for key, value in val_results.items():
                out_str += f" - {'val_' + key}: {value:.10f}"
            out_str += "\n\n\n"

        sys.stdout.write(out_str)
        sys.stdout.flush()

    @staticmethod
    def __apply_function(input_data: Dict[str, torch.Tensor] | torch.Tensor, func: Callable) -> (Dict[str, torch.Tensor] |
                                                                                                 torch.Tensor):
        """
        Apply a given function to the input data.

        :params input_data: Input data to which the function will be applied.
        :params func: A callable function to be applied to the input data.

        :return:  Output data after applying the function.

        """
        output_data = {}

        if isinstance(input_data, dict):
            for key, value in input_data.items():
                output_data[key] = func(value, dim=-1).detach().cpu()
        elif torch.is_tensor(input_data):
            output_data = func(input_data, dim=-1).detach().cpu()
        else:
            raise ValueError("Input data should be a tensor or a dictionary of tensors.")

        return output_data

    @staticmethod
    def __merge_batch_predictions(data_list: list[Dict[str, torch.Tensor]] | list[torch.Tensor]) -> (Dict[str, torch.Tensor] |
                                                                                                     torch.Tensor):
        """
        Merge a list of dictionaries or tensors along the batch dimension.

        :params data_list (list): A list containing dictionaries or tensors.

        :return: Merged predictions. If the input is a list of dictionaries, returns a dictionary where the values
                are tensors concatenated along the batch dimension. If the input is a list of tensors, returns
                a single tensor obtained by concatenating all tensors along the batch dimension.

        """

        if isinstance(data_list[0], dict):
            merged_dict = {}
            for key in data_list[0].keys():
                merged_dict[key] = torch.cat([d[key] for d in data_list], dim=0)
            predictions = merged_dict
        elif isinstance(data_list[0], torch.Tensor):
            predictions = torch.cat(data_list, dim=0)
        else:
            raise ValueError("Input data should be either a list of dictionaries with tensors or a list of tensors.")
        return predictions

    def validate(self,
                 data_loader: torch.utils.data.DataLoader,
                 criterion: Criterion | MultiHeadCriterion,
                 metrics: Optional[List[Metric | MultiHeadMetric]] = None,
                 aggregate_loss_on_dataset: bool = True) -> Dict[str, float] | Tuple[Dict[str, float], torch.Tensor]:

        """
        Validate the model on the given data loader.

        :param data_loader: DataLoader containing data.
        :param criterion: Criterion or MultyHeadCriterion object specifying the loss function(s) to optimize.
        :param metrics: Optional list of Metric or MultyHeadMetric objects for evaluation during training.
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

        results = {}
        aggregated_losses = torch.tensor([], device='cpu')

        self.eval()
        with torch.no_grad():
            for iteration, (inputs, labels) in enumerate(data_loader):
                inputs, labels = self._to_device(inputs, self.device), self._to_device(labels, self.device)
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
            if isinstance(metric, SingleHeadMetric):
                results[metric.name] = metric.get_result()
            elif isinstance(metric, MultiHeadMetric):
                metric_results = metric.get_result()
                for head_metric, head_metric_result in metric_results.items():
                    results[head_metric] = head_metric_result

            metric.reset_state()

        return results

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            criterion: Criterion | MultiHeadCriterion,
            optimizer: OptimizerWrapper,
            num_epochs: int,
            metrics: Optional[List[Metric | MultiHeadMetric]] = None,
            early_stopper: Optional[EarlyStopper | MultipleEarlyStoppers] = None,
            aggregate_loss_on_dataset: bool = True,
            verbose: int = 1,
            interaction_with_wandb: bool = False,
            interaction_function_with_wandb: Optional[Callable] = None) -> Dict[str, Dict[str, List[Any]]]:
        """
        Train the model.


        :param train_loader: Training data loader.
        :param val_loader: Validation data loader.
        :param criterion: Criterion or MultyHeadCriterion object specifying the loss function(s) to optimize.
        :param optimizer: An instance of OptimizerWrapper, specifying the optimizer for training.
        :param num_epochs: Number of training epochs.
        :param metrics: Optional list of Metric or MultyHeadMetric objects for evaluation during training.
        :param early_stopper: An object of type EarlyStopper, if None training goes on until num_epochs has been done
        :param aggregate_loss_on_dataset: If True, the reduce strategy is applied over all the loss of the samples of the dataset.
                                          Otherwise, the reduce strategy is applied over all the batches to get a partial loss for each batch,
                                          then the partial losses are reduced to get a unique loss for the epoch.
                                          In the first case, the result is more accurate, but more RAM is used.
                                          In the second case, the result is less accurate due to numerical approximation, and less RAM is used.
        :param verbose: If 0 the training loop is silent if 1 print the training process results and if 2 the training result are printed in a tabular way.
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
            if isinstance(metric, SingleHeadMetric):
                train_history[metric.name] = []
                val_history[metric.name] = []
            elif isinstance(metric, MultiHeadMetric):
                train_history[metric.name] = []
                val_history[metric.name] = []
                for head_metric in metric.metrics_functions.values():
                    train_history[head_metric.name] = []
                    val_history[head_metric.name] = []
            else:
                raise TypeError('Inconsistent type for metric parameter. '
                                'Only Metric or MultyHeadMetric object allowed.')

        iterations_per_epoch = len(train_loader)

        if early_stopper and early_stopper.restore_weights:
            early_stopper.create_directory()

        try:
            for epoch in range(num_epochs):

                log_params = {}
                start_time = time.time()
                self.train()

                for iteration, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = self._to_device(inputs, self.device), self._to_device(labels, self.device)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss = criterion.reduction_function(loss)
                    loss.backward()
                    optimizer.step()

                    metrics_value = {}

                    for metric in metrics:
                        if isinstance(metric, SingleHeadMetric):
                            metrics_value[metric.name] = metric(outputs, labels)
                        elif isinstance(metric, MultiHeadMetric):
                            metric_results = metric(outputs, labels)
                            for head_metric_key, head_metric_result in metric_results.items():
                                metrics_value[head_metric_key] = head_metric_result

                    if verbose > 0:
                        self.__print_batch_results(current_epoch=epoch,
                                                   tot_num_epochs=num_epochs,
                                                   current_iteration=iteration,
                                                   iterations_per_epoch=iterations_per_epoch,
                                                   criterion_name=criterion.name,
                                                   loss=loss,
                                                   metrics_results=metrics_value)

                train_results = self.validate(data_loader=train_loader,
                                              criterion=criterion,
                                              metrics=metrics,
                                              aggregate_loss_on_dataset=aggregate_loss_on_dataset)

                val_results = self.validate(data_loader=val_loader,
                                            criterion=criterion,
                                            metrics=metrics,
                                            aggregate_loss_on_dataset=aggregate_loss_on_dataset)

                end_time = time.time()

                for key, value in train_results.items():
                    train_history[key].append(value)
                for key, value in val_results.items():
                    val_history[key].append(value)

                if interaction_with_wandb:
                    for (train_key, train_value), (val_key, val_value) in zip(train_results.items(), val_results.items()):
                        log_params['train_' + 'loss' if criterion.name is train_key else 'train_' + train_key] = train_value
                        log_params['val_' + 'loss' if criterion.name is train_key else 'val_' + train_key] = val_value

                    additional_interaction_data = interaction_function_with_wandb(self) if interaction_function_with_wandb is not None else {}
                    wandb.log({**log_params, **additional_interaction_data})

                if verbose > 0:
                    self.__print_epoch_results(train_results=train_results,
                                               val_results=val_results,
                                               current_epoch=epoch,
                                               tot_num_epochs=num_epochs,
                                               training_time=end_time - start_time,
                                               verbose=verbose)

                if early_stopper is not None:
                    stop_flag = early_stopper(val_history, self)
                    if stop_flag:
                        if verbose > 0:
                            early_stopper.get_message()
                        break
                    if not stop_flag:
                        if verbose > 0:
                            early_stopper.get_message()

        finally:
            if early_stopper and early_stopper.restore_weights:
                early_stopper.delete_directory()
        if verbose > 0:
            print("\nTrain Completed.")

        return {'train': train_history, 'val': val_history}

    def predict(self,
                data: torch.utils.data.DataLoader[torch.Tensor] | torch.utils.data.DataLoader[Dict[str, torch.Tensor]],
                model_output_function_transformation: Callable = torch.round) -> torch.Tensor:
        """
        Generate predictions for the given input data using the trained model.

        :param data: Input data for which predictions are to be generated. It can be either a tensor or a dictionary
                    containing input data (e.g., features or images).
        :param model_output_function_transformation: A callable function applied to the model's output for post-processing
                    predictions. Default is torch.round, which is commonly used for binary classification tasks.

        :return: Tensor containing the predicted labels.

        """

        self.eval()
        predicted_labels_list = []

        for batch_data, _ in data:
            batch_data_device = self._to_device(data=batch_data, device=self.device)
            batch_output = self(batch_data_device)
            batch_output = TrainableModule.__apply_function(batch_output, model_output_function_transformation)
            predicted_labels_list.append(batch_output)

        predicted_labels = TrainableModule.__merge_batch_predictions(predicted_labels_list)

        return predicted_labels
