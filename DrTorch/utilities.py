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
                 data: pd.DataFrame | Dict,
                 labels: pd.DataFrame | Dict,
                 sample_preprocess_f: Optional[Callable] = None,
                 label_preprocess_f: Optional[Callable] = None):
        """
        CustomDataset class for PyTorch.

        :param data: The input data.
        :param labels: The input labels.
        :param sample_preprocess_f: A function or transformation to be applied to each data sample, but once by once.
        :param label_preprocess_f: A function or transformation to be applied to each label sample but once by once.
        """
        self.data = data
        self.labels = labels
        self.sample_preprocess_f = sample_preprocess_f
        self.label_preprocess_f = label_preprocess_f

    @staticmethod
    def __get_n_element(d: Dict) -> int:
        """
        Private static method that recursively examines a dictionary and returns the length of the first tensor
        found at the last level.

        :params d : The dictionary to examine.
        :returns: The length of the first tensor found at the last level of the dictionary.

        Example:
            d = {'input_ids': torch.tensor([[1, 2, 3],
                                            [4, 5, 6]]),
                'attention_mask': torch.tensor([[1, 1, 0],
                                                [1, 0, 0]])}
            Dataset.__get_n_element(d)
            >> 2
        """

        for v in d.values():
            if isinstance(v, dict):
                return Dataset.__get_n_element(v)
            elif isinstance(v, torch.Tensor):
                return v.shape[0]

    @staticmethod
    def __generate_sample(data_dict: Dict, idx: int) -> Dict:
        """
        Private static method to generate a sample from a data dictionary.

        Parameters:
        :params data_dict : The dictionary of data from which to generate the sample.
        :params idx : The index of the sample to generate.

        :returns:  The generated sample.

        Example:
            data_dict = {'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]), 'attention_mask': torch.tensor([[1, 1, 0], [1, 0, 0]])}
            Dataset.__generate_sample(data_dict, 1)
            >>{'input_ids': tensor([4, 5, 6]), 'attention_mask': tensor([1, 0, 0])}

        """
        sample = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                sample[key] = Dataset.generate_data_sample(value, idx)
            else:
                sample[key] = value[idx] if isinstance(value, torch.Tensor) and idx < value.shape[0] else None
        return sample

    def __len__(self) -> int:
        if isinstance(self.data, pd.DataFrame):
            n_element = len(self.data)
        elif isinstance(self.data, Dict) or hasattr(self.data, 'keys'):
            n_element = self.__get_n_element(self.data)
        else:
            print(type(self.data))
            raise TypeError(
                "Type inconsistency for the 'data' attribute. Only pd.Dataframe Dict or Dict like are allowed."
                "The __len__() function is unable to compute the number of elements contained in the Dataset.")
        return n_element

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        Retrieves a specific sample from the dataset.

        :param idx: The index of the desired sample.

        :return: A tuple containing the processed data and label samples.

        """

        if isinstance(self.data, pd.DataFrame):
            data_sample = self.data.iloc[idx]
        elif isinstance(self.data, Dict) or hasattr(self.data, 'keys'):
            data_sample = self.__generate_sample(self.data, idx)
        else:
            raise TypeError(
                "Type inconsistency for the 'data' attribute. Only pd.Dataframe Dict or Dict like are allowed.")

        if isinstance(self.labels, pd.DataFrame):
            label_sample = self.labels.iloc[idx]
        elif isinstance(self.labels, Dict) or hasattr(self.labels, 'keys'):
            label_sample = self.__generate_sample(self.labels, idx)
        else:
            raise TypeError(
                "Type inconsistency for the 'label' attribute. Only pd.Dataframe, Dict or Dict like are allowed.")

        if self.sample_preprocess_f is not None:
            data_sample = self.sample_preprocess_f(data_sample)
        if self.label_preprocess_f is not None:
            label_sample = self.label_preprocess_f(label_sample)

        return data_sample, 0  # label_sample


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
               num_workers: int = 1,
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
               num_workers: int = 1,
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
    - create(data, labels, shuffle, data_preprocess_f=None, labels_preprocess_f=None, device='cpu',
             pin_memory=False, num_workers=1, batch_size=32, data_pipeline=None, label_pipeline=None) -> torch.utils.data.DataLoader:
        Create a DataLoader instance based on the provided data and labels, with optional data and label pipelines.

    Attributes:
    - None
    """

    def create(self,
               data: pd.DataFrame,
               labels: pd.DataFrame,
               shuffle: bool,
               data_preprocess_f: Optional[Callable] = None,
               labels_preprocess_f: Optional[Callable] = None,
               device: str = 'cpu',
               pin_memory: bool = False,
               num_workers: int = 1,
               batch_size: int = 32,
               sample_preprocess_f: Optional[Callable] = None,
               label_preprocess_f: Optional[Callable] = None):
        """
        Create a DataLoader instance based on the provided data and labels, with optional data and label pipelines.

        Parameters:
        :param data: The data to be loaded.
        :param labels: The labels corresponding to the data.
        :param shuffle: Whether to shuffle the data during loading.
        :param data_preprocess_f: A function to preprocess all the data before loading.
                                  For instance, this function could tokenize all input data in a single operation.
        :param labels_preprocess_f: A function to preprocess  all the labels before loading.
                                    For instance, this function could one hot encode all the labels in a single operation.
        :param device: The device on which to load the data ('cpu' or 'cuda').
        :param pin_memory: Whether to pin memory for faster GPU transfer.
        :param num_workers: Number of parallel processes to use for data loading.
        :param batch_size: Size of the batches to load.
        :param sample_preprocess_f: A function used to preprocess individual samples.
                                     This function enables preprocessing operations to be applied on individual inputs.
                                     It is particularly useful when dealing with large datasets, such as images,
                                     where it may not be feasible to preprocess or load the entire dataset at once.
                                     For instance, this function could load an image when the 'data' attribute
                                     refers to a DataFrame containing paths to images.

        :param label_preprocess_f: A function designed to preprocess individual labels.
                                    This function allows for preprocessing operations to be applied on individual labels.
                                    It is beneficial when it is impractical to preprocess or load the entire label set
                                    simultaneously due to memory constraints.
                                    For example, this function could apply a transformation to a single label,
                                    such as categorical encoding.

        Returns:
        :return: torch.utils.data.DataLoader instance based on the provided parameters.

        Example:
        # Assuming `data` and `labels` are your data and labels, and `pipeline` is your data pipeline
        dataloader_from_strategy_builder = DataLoaderFromPipelineStrategy()
        dataloader = dataloader_from_strategy_builder.create(data, labels, shuffle=True, device='cuda', data_pipeline=pipeline)

        """

        if data_preprocess_f is not None:
            data = data_preprocess_f(data)

        if labels_preprocess_f is not None:
            labels = labels_preprocess_f(labels)

        dataset = Dataset(data=data,
                          labels=labels,
                          sample_preprocess_f=sample_preprocess_f,
                          label_preprocess_f=label_preprocess_f)

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
