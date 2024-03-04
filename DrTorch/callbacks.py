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
from typing import List

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
                 monitor:str,
                 patience:int = 1,
                 delta:float = 0.0,
                 mode: str = 'min',
                 restore_weights:bool = False,
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
        self.folder_name = folder_name
        self.counter = 0
        self.hidden_directory = os.path.join(os.getcwd(), self.folder_name)

    def __call__(self, history_values: List[float], model_ptr) -> bool:
        """
        Call method to check if early stopping criteria are met.

        :param history_values: A list containing the historical values of the monitored metric.
        :param model_ptr: The drTorch model whose weights are being monitored.

        :return: True if early stopping criteria are met, False otherwise.

        """

        stop_flag = False
        self.counter += 1
        weights_path = os.path.join(self.hidden_directory, f'model_{self.counter}')

        if len(history_values) > self.patience:
            value_to_compare = history_values[-self.patience - 1]
            list_to_compare = history_values[-self.patience:]
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
            msg += ", best weights reloaded."
        return msg
