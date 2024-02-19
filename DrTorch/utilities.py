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

from abc import ABC, abstractmethod
from typing import Any, Union, Callable, List, Dict, Optional

import pandas as pd
import torch

from matplotlib import pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 label_df: pd.DataFrame,
                 data_pipeline: Callable,
                 label_pipeline: Callable):
        """
        CustomDataset class for PyTorch.

        :param data_df: The input DataFrame containing data.
        :param label_df: The input DataFrame containing labels.
        :param data_pipeline :A function or transformation to be applied to each data sample.
        :param label_pipeline: A function or transformation to be applied to each label sample.
        """
        self.data_df = data_df
        self.label_df = label_df
        self.data_pipeline = data_pipeline
        self.label_pipeline = label_pipeline

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        :return: The number of samples in the dataset.
        """

        return len(self.data_df)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        Retrieves a specific sample from the dataset.

        :param idx: The index of the desired sample.

        :return: A tuple containing the processed data and label samples.

        """
        data_sample = self.data_df.iloc[idx]
        label_sample = self.label_df.iloc[idx]
        return self.data_pipeline(data_sample), self.label_pipeline(label_sample)


class DataLoaderStrategy(ABC):
    """
    Abstract base class for creating data loaders.

    :param logic: A callable function or logic used for configuring data loader parameters.

    """

    def __init__(self,
                 logic: Callable = None):
        """
        Constructor for DataLoaderStrategy.

        :param logic: A callable function or logic used for configuring data loader parameters.

        """

        self.logic = logic

    @abstractmethod
    def create(self,
               data: Union[pd.DataFrame | torch.Tensor],
               labels: Union[pd.DataFrame | torch.Tensor],
               shuffle: bool,
               device: str = 'cpu',
               num_workers: int = 0,
               **kwargs: Dict):
        """
        Abstract method to be implemented by subclasses. Defines the strategy for creating a data loader.

        :param data: The input data.
        :param labels: Labels associated with the data.
        :param shuffle: Flag indicating whether to shuffle the data.
        :param device: Device on which the dataloader will be installed.
        :param num_workers: Number of parallel worker processes to use for loading data.
        :param kwargs: Additional keyword arguments for configuring the data loader.

        :return: Implementation-specific data loader object.

        """
        pass

    def get_dataloader_params(self,
                              param: str,
                              total_hyperparameters: Optional[Dict]):
        """
        Get data loader parameters based on the specified logic or callable.

        :param param: The parameter to retrieve.
        :param total_hyperparameters: A dictionary containing total hyperparameters for the data loader.

        :return: The value of the specified parameter based on the configured logic.

        :raises ValueError: If the specified parameter is not defined in the logic.

        """

        if self.logic:
            return self.logic(param, total_hyperparameters)
        raise ValueError(f"param {param} not defined")


class DataLoaderFromTensorStrategy(DataLoaderStrategy):
    """
    DataLoaderFromTensorStrategy is a strategy class for creating PyTorch DataLoader instances from input tensors.

    It inherits from DataLoaderStrategy, which is part of a broader design for flexible DataLoader creation.

    Usage:
    strategy = DataLoaderFromTensorStrategy()
    data_loader = strategy.create(data, labels, shuffle=True, batch_size=32)

    Parameters:
    - data (torch.Tensor): The input data tensor.
    - labels (torch.Tensor): The tensor containing corresponding labels for the input data.
    - shuffle (bool): Flag indicating whether to shuffle the data during each epoch.
    - batch_size (int, optional): The batch size for the DataLoader. Default is 32.

    Returns:
    torch.utils.data.DataLoader: A PyTorch DataLoader instance configured based on the provided parameters.
    """

    def create(self,
               data: torch.Tensor,
               labels: torch.Tensor,
               shuffle: bool,
               device: str = 'cpu',
               num_workers: int = 0,
               batch_size: int = 32) -> torch.utils.data.DataLoader:

        """
        Create a PyTorch DataLoader instance from input data and labels.

        :param data: The input data tensor.
        :param labels: The tensor containing corresponding labels for the input data.
        :param shuffle: Flag indicating whether to shuffle the data during each epoch.
        :param device: Device on which the dataloader will be installed.
        :param num_workers: Number of parallel worker processes to use for loading data.
        :param batch_size: The batch size for the DataLoader. Default is 32.

        :return:  A PyTorch DataLoader instance configured based on the provided parameters.

        """

        torch_dataset = torch.utils.data.TensorDataset(data, labels)
        if device == 'cuda' and torch.cuda.is_available():
            dataloader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     pin_memory=True,
                                                     num_workers=num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader


class DataLoaderFromPipelineStrategy(DataLoaderStrategy):
    """
    DataLoader strategy implementation using a custom data pipeline.

    Methods:
    - create(data, labels, shuffle, device='cpu', pin_memory=False, num_workers: int = 1, batch_size: int = 32,
             data_pipeline=None, label_pipeline=None) -> torch.utils.data.DataLoader:
        Create a DataLoader instance based on the provided data and labels, with optional data and label pipelines.

    Attributes:
    - None
    """

    def create(self,
               data: Optional[torch.Tensor | pd.DataFrame],
               labels: Optional[torch.Tensor | pd.DataFrame],
               shuffle: bool,
               device:str = 'cpu',
               pin_memory:bool = False,
               num_workers: int = 1,
               batch_size: int = 32,
               data_pipeline: Callable = None,
               label_pipeline: Callable = None):
        """

         Parameters:
         :param data: The data to be loaded.
         :param labels: The labels corresponding to the data.
         :param shuffle: Whether to shuffle the data during loading.
         :param device: The device on which to load the data ('cpu' or 'cuda').
         :param pin_memory: Whether to pin memory for faster GPU transfer.
         :param num_workers: Number of parallel processes to use for data loading.
         :param batch_size: Size of the batches to load.
         :param data_pipeline: A data preprocessing pipeline.
         :param label_pipeline: A label preprocessing pipeline.

         Returns:
         :return: torch.utils.data.DataLoader instance based on the provided parameters.

         Example:
         # Assuming `data` and `labels` are your data and labels, and `pipeline` is your data pipeline
         dataloader_from_strategy_builder = DataLoaderFromPipelineStrategy()
         dataloader = dataloader_from_strategy_builder.create(data, labels, shuffle=True, device='cuda', data_pipeline=pipeline)

         """

        dataset = Dataset(data_df=data,
                          label_df=labels,
                          data_pipeline=data_pipeline,
                          label_pipeline=label_pipeline)

        if device == 'cuda' and torch.cuda.is_available():
            dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     pin_memory=pin_memory,
                                                     num_workers=num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


def plot_history(history: Dict[str, List[float]], patience: int = None) -> None:
    """
    Plot training and validation history for each metric.

    Parameters:
    :param history: A dictionary containing training and validation metrics history.
    :param patience: Number of epochs for patience circle (if None, no circle will be plotted).

    Notes:
    - Assumes a dictionary structure with 'train' and 'val' keys, each containing metrics as subkeys.
    - This function is specially designed to work with the fitting function defined inside the modules.py file

    Example:
    # Assuming `history` contains your training and validation history
    plot_history(your_history_variable, patience=5)
    """

    metrics = list(history['train'].keys())  # Assuming all metrics are present in the 'train' field
    num_metrics = len(metrics)

    if num_metrics > 1:
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

        for i, metric in enumerate(metrics):
            print(metric)
            epochs = range(1, len(history['train'][metric]) + 1)
            axes[i].plot(epochs, history['train'][metric], label='Training')
            axes[i].plot(epochs, history['val'][metric], label='Validation')

            if patience is not None:
                idx_last_patience = len(history['val'][metric]) - patience

                max_value = history['val'][metric][idx_last_patience - 1]
                axes[i].scatter(idx_last_patience, max_value, color='red', marker='o',
                                label=f'Best model obtained with patient = {patience}')

            axes[i].set_title(f'{metric} History')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].grid(True)

    else:
        metric = metrics[0]
        epochs = range(1, len(history['train'][metric]) + 1)
        plt.plot(epochs, history['train'][metric], label='Training')
        plt.plot(epochs, history['val'][metric], label='Validation')

        if patience is not None:
            idx_last_patience = len(history['val'][metric]) - patience

            max_value = history['val'][metric][idx_last_patience - 1]
            plt.scatter(idx_last_patience, max_value, color='red', marker='o',
                        label=f'Best model obtained with patient = {patience}')

        plt.title(f'{metric} History')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()

        plt.grid(True)

    plt.tight_layout()
    plt.show()
