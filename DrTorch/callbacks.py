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


import os
from typing import List, Dict

import torch
import numpy as np
import shutil


class EarlyStopper:
    """
    Implements early stopping based on a monitored metric during model training.

    :param monitor: The metric to monitor for early stopping.
    :param patience: The number of epochs with no improvement after which training will be stopped.
    :param delta: The minimum change in the monitored metric to qualify as an improvement.
    :param mode: One of {'min', 'max'}. In 'min' mode, training stops when the monitored metric stops decreasing;
                 in 'max' mode, it stops when the metric stops increasing.
    :param restore_weights: If True, restore the model weights to the best weights when early stopping is activated.
    :param folder_name: The name of the folder to store weights history.
                        A hidden directory is created within the project directory to store weights history.
                        If the fitting process is unexpectedly interrupted, in some cases, the directory may not be
                        automatically deleted, requiring manual intervention to ensure its removal

    Methods:
    - __call__(history_values: List[float], model_ptr) -> bool:
      Call method to check if early stopping criteria are met.
    - create_directory() -> None:
      Create a hidden directory to store weights history.
    - delete_directory() -> None:
      Delete the directory and reset the counter.
    - get_message() -> str:
      Get a message indicating that early stopping is activated, optionally with a note about restoring the best weights.
    """

    def __init__(self,
                 monitor: str,
                 patience: int = 1,
                 delta: float = 0.0,
                 mode: str = 'min',
                 restore_weights: bool = False,
                 folder_name: str = '.weights_memory'):
        """
        Initialize an EarlyStopper object to monitor a specified metric or loss during training and perform early stopping.

        :param monitor: The metric to monitor for early stopping.
        :param patience: The number of epochs with no improvement after which training will be stopped.
        :param delta: The minimum change in the monitored metric to qualify as an improvement.
        :param mode: One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has
                     stopped decreasing; in 'max' mode, it will stop when the quantity monitored has stopped increasing.
        :param restore_weights: If True, restore the model weights to the best weights when early stopping is activated.
        :param folder_name: The name of the folder to store weights history.
                            A hidden directory is created within the project directory to store weights history.
                            If the fitting process is unexpectedly interrupted, in some cases, the directory may not be
                            automatically deleted, requiring manual intervention to ensure its removal

        """

        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.restore_weights = restore_weights
        self.counter = 0
        self.hidden_directory = os.path.join(os.getcwd(), folder_name)

    def __call__(self, history_values: Dict[str, List[float]], model_ptr) -> bool: # todo l'aggiunta del tipo di dato per mode_pt Trainable_module  crea un problema di importing circolari
        """
        Call method to check if early stopping criteria are met.

        :param history_values: A list containing the historical values of the monitored metric.
        :param model_ptr: The drTorch model whose weights are being monitored.

        :return: True if early stopping criteria are met, False otherwise.

        """

        stop_flag = False
        self.counter += 1
        weights_path = os.path.join(self.hidden_directory, f'model_{self.counter}')

        metric_to_monitor = history_values[self.monitor]

        if len(metric_to_monitor) > self.patience:
            value_to_compare = metric_to_monitor[-self.patience - 1]
            list_to_compare = metric_to_monitor[-self.patience:]
            if self.mode == 'min':
                stop_flag = value_to_compare < np.min(list_to_compare) + self.delta
            else:
                stop_flag = value_to_compare > np.max(list_to_compare) - self.delta

            previous_weights_index = self.counter - self.patience
            previous_weights_path = os.path.join(self.hidden_directory, f'model_{previous_weights_index}')
            if self.restore_weights:
                if stop_flag:
                    model_ptr.load_state_dict(torch.load(previous_weights_path))
                else:
                    torch.save(model_ptr.state_dict(), weights_path)
                    os.remove(previous_weights_path)
        else:
            torch.save(model_ptr.state_dict(), weights_path)
        return stop_flag

    def create_directory(self) -> None:
        """
        Create a hidden directory to store weights history.
        the directory was renamed by default as '.weights_memory'

        :return: None

        """

        os.makedirs(self.hidden_directory)

    def delete_directory(self) -> None:
        """
        Delete the directory and reset the counter.

        :return: None

        """

        self.counter = 0
        shutil.rmtree(self.hidden_directory)

    def get_message(self) -> str:
        """
        Get a message indicating that early stopping is activated, optionally with a note about restoring the best weights.

        :return: A string message.

        """

        msg = "Early stopping activated"
        if self.restore_weights:
            msg += ", best weights reloaded"
        return msg


class MultipleEarlyStoppers:
    """
    Implements multiple early stoppers based on different monitored metrics during model training.
    This class allows you to freeze separate parts of the network based on the different metrics being monitored.

    Args:
        stoppers (Dict[str, EarlyStopper]): A dictionary containing instances of EarlyStopper
            with their associated names as keys.
        layers_to_freeze (Dict[str, List[str]]): A dictionary specifying the layers to freeze
            for each early stopper, with the early stopper names as keys and lists of layer names
            as values.
        restore_weights (bool, optional): Whether to restore the model weights to the best weights
            when early stopping is activated. Defaults to False.
        folder_name (str, optional): The name of the folder to store weights history. A hidden
            directory is created within the project directory to store weights history.
            Defaults to '.weights_memory'.

    Attributes:
        stoppers (Dict[str, EarlyStopper]): A dictionary containing instances of EarlyStopper
            with their associated names as keys.
        layers_to_freeze (Dict[str, List[str]]): A dictionary specifying the layers to freeze
            for each early stopper, with the early stopper names as keys and lists of layer names
            as values.
        restore_weights (bool): Whether to restore the model weights to the best weights
            when early stopping is activated.
        hidden_directory (str): The path to the hidden directory to store weights history.
        stop_flags (Dict[str, bool]): A dictionary storing the flags indicating whether early
            stopping criteria are met for each early stopper.
    """

    def __init__(self,
                 stoppers: Dict[str, EarlyStopper],
                 layers_to_freeze: Dict[str, List[str]],
                 restore_weights: bool = False,
                 folder_name: str = '.weights_memory'):
        """
        Initializes a MultipleEarlyStoppers object to implement multiple early stoppers based on monitored metrics
        during model training.

       :params stoppers: A dictionary containing early stoppers, where keys are the names of
                         the metrics being monitored and values are EarlyStopper instances.
       :params layers_to_freeze: A dictionary containing layers to freeze for each head, where
                                 keys are the names of the heads and values are lists of layer names.
       :params restore_weights: If True, restores the model weights to the best weights when early
                                stopping is activated. Defaults to False.
       :params folder_name: The name of the folder to store weights history. Defaults to '.weights_memory'.

        """

        self.stoppers = stoppers
        self.layers_to_freeze = layers_to_freeze
        self.restore_weights = restore_weights

        self.hidden_directory = os.path.join(os.getcwd(), folder_name)
        self.stop_flags = {stopper.monitor: False for stopper in stoppers.values()}

        for stopper in self.stoppers.values():
            stopper.hidden_directory = os.path.join(self.hidden_directory, stopper.monitor)
            stopper.restore_weights = restore_weights

    def __call__(self, history_values: Dict[str, List[float]], model_ptr) -> bool:
        """
        Checks if the early stopping criteria are met based on the provided history values.

        :params history_values : A list containing the historical values of the monitored metric.
        :params model_ptr: The eights are being monitored.

        :returns: True if the early stopping criteria are met for all early stoppers, False otherwise.

        """

        out_str = ''
        model_weights_copy = model_ptr.state_dict()

        for head_name, stopper in self.stoppers.items():

            if not self.stop_flags[stopper.monitor] and stopper(history_values, model_ptr):
                self.stop_flags[stopper.monitor] = True

                for layer_name, param in model_ptr.named_parameters():
                    if layer_name in self.layers_to_freeze[head_name]:
                        param.requires_grad = False
                        out_str += (f'{layer_name} has been freezed because {stopper.monitor} '
                                    f'is no more {"decreasing" if stopper.mode == "min" else "increasing"}\n')
                    else:
                        if self.restore_weights:
                            param.data.copy_(model_weights_copy[layer_name])

        if out_str != '':
            print('\n' + out_str)
            if not all(self.stop_flags.values()):
                monitor_not_activated = [monitor for monitor, flag in self.stop_flags.items() if not flag]
                monitor_not_activated = ", ".join(monitor_not_activated)
                monitor_not_activated = ' and'.join(monitor_not_activated.rsplit(',', 1))
                print(f'The training is continuing. Stopping criteria not reached for {monitor_not_activated}\n')

        return all(self.stop_flags.values())

    def create_directory(self) -> None:
        """
        Create a hidden directory to store weights history.
        the directory was renamed by default as '.weights_memory'

        :return: None

        """
        for stopper in self.stoppers.values():
            stopper.create_directory()

    def delete_directory(self) -> None:
        """
        Delete the directory and reset the counter.

        :return: None

        """

        for stopper in self.stoppers.values():
            stopper.counter = 0

        shutil.rmtree(self.hidden_directory)

    def get_message(self) -> str:
        """
        Get a message indicating that early stopping is activated, optionally with a note about restoring the best weights.

        :return: A string message.

        """

        msg = "All the Early Stopper has been activated"
        if self.restore_weights:
            msg += ", best weights reloaded"
        return msg
