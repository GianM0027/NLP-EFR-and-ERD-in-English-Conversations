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

from typing import Optional, Callable, List, Any, Dict, Tuple
from sympy import Float

from abc import ABC, abstractmethod

import torch
import functools
import numpy as np


class Metric(ABC):
    """
    Abstract base class for implementing evaluation metrics.

    Attributes:
        name (str): Name of the metric.

    Methods:
        __call__(self, predicted_classes, target_classes, accumulate_statistic=False):
            Computes the metric based on predicted and target classes.
        update_state(self, predicted_classes, target_classes):
            Updates the internal state of the metric.
        reset_state(self, *args, **kwargs):
            Resets the internal state of the metric.
        get_result(self, *args, **kwargs):
            Computes and returns the final result of the metric.

    """

    def __init__(self,
                 name: str = "default_name"):
        self.name = name

    @abstractmethod
    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):
        """
         Computes the metric based on predicted and target classes.

         :param predicted_classes: Predicted class labels.
         :param target_classes: True class labels.
         :param accumulate_statistic: Whether to accumulate intermediate statistics for later retrieval.

         :return: Computed metric value.

         """
        pass

    @abstractmethod
    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor):
        """
        Updates the internal state of the metric.

        :param predicted_classes: Predicted class labels.
        :param target_classes: True class labels.

        """
        pass

    @abstractmethod
    def reset_state(self,
                    *args: List[Any],
                    **kwargs: Dict[str, Any]):
        """
        Resets the internal state of the metric.

        :param args: Additional arguments if needed.
        :param kwargs: Additional keyword arguments if needed.

        """
        pass

    @abstractmethod
    def get_result(self,
                   *args: List[Any],
                   **kwargs: Dict[str, Any]):
        """
        Computes and returns the final result of the metric.

        :param args: Additional arguments if needed.
        :param kwargs: Additional keyword arguments if needed.

        :return: Final computed metric result.

        """
        pass

    def __str__(self) -> str:
        """
        Get the name of the metric as a string.

        :return: Name of the metric.

        """
        return self.name


class SingleHeadMetric(Metric, ABC):
    """
    Abstract base class for implementing evaluation metrics for single-head models.

    Attributes:
    - num_classes (int): The number of classes for which the metric is computed.
    - name (str): Name of the metric.
    - pred_transform (Callable): A transformation function for mapping the model output into a suitable metric input.
    - target_transform (Callable): A transformation function for mapping the target labels into a suitable metric input.

    Methods:
    - __init__(num_classes: int, name: str = "default_name", pred_transform: Optional[Callable] = None,
               target_transform: Optional[Callable] = None):
        Initialize a SingleHeadMetric object.
    """

    def __init__(self,
                 num_classes: int,
                 name: str = "default_name",
                 pred_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize a SingleHeadMetric object.

        :param num_classes: The number of classes for which the metric is computed.
        :param name: Name of the metric.
        :param pred_transform: A transformation function for mapping the model output into a suitable metric input.
                               If None, defaults to rounding for binary classification and argmax for multiclass.
        :param target_transform: A transformation function for mapping the target labels into a suitable metric input.
                                 If None and num_classes is not 2, defaults to argmax.

        """

        super().__init__(name)
        self.num_classes = num_classes

        if pred_transform is None:
            if self.num_classes == 2:
                pred_transform = torch.round
            else:
                pred_transform = functools.partial(torch.argmax, dim=-1)

        if target_transform is None and self.num_classes != 2:
            target_transform = functools.partial(torch.argmax, dim=-1)

        self.pred_transform = pred_transform
        self.target_transform = target_transform


class MultyHeadMetric(Metric):
    """
    Multi-head metric implementation for evaluating multiple metrics on different subsets of predictions and targets.

    Attributes:
    - metrics_functions (dict): A dictionary containing metric functions for each head.
    - metric_weights (list): Optional list of weights for each metric when aggregating results.
    - aggregate_metrics_function (callable): Optional function for aggregating individual head metrics into a single score.

    Methods:
    - __init__(self, name: str, metrics_functions: Dict[str, Callable], metric_weights: Optional[List[int]] = None,
               aggregate_metrics_function: Optional[Callable] = None): Constructor method to initialize the multi-head metric.
    - __call__(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor, accumulate_statistic: bool = False) -> Dict:
               Main method to compute the metric values based on the predicted and target classes.
    - update_state(self, predicted_classes: Dict[str, torch.Tensor], target_classes: Dict[str, torch.Tensor]) -> None:
               Method to update the state of individual metrics with new predicted and target classes.
    - reset_state(self) -> None: Method to reset the state of all individual metrics.
    - get_result(self) -> Dict[str, Float]: Method to retrieve the results of all individual metrics.

    TODO ADD EXEMPLE
    """

    def __init__(self,
                 name: str,
                 metrics_functions: Dict[str, SingleHeadMetric],
                 metric_weights: Optional[List[int]] = None,
                 aggregate_metrics_function: Optional[Callable] = None):
        """
        Initializes the multi-head metric with the provided metrics functions and optional parameters.

        :param name: The name of the multi-head metric.
        :param metrics_functions: A dictionary mapping head keys to metric constructor functions.
        :param metric_weights: Optional list of weights for each metric when aggregating results.
        :param aggregate_metrics_function: Optional function for aggregating individual head metrics into a single score.

        """

        super().__init__(name)
        self.metrics_functions = metrics_functions
        self.metric_weights = metric_weights

        if aggregate_metrics_function is not None:
            self.aggregate_metrics_function = aggregate_metrics_function

    def __call__(self,
                 predicted_classes: Dict[str, torch.Tensor],
                 target_classes: Dict[str, torch.Tensor],
                 accumulate_statistic: bool = False) -> Dict:
        """
        Computes the metric values based on the predicted and target classes.

        :param predicted_classes: Predicted classes for each head.
        :param target_classes: Target classes for each head.
        :param accumulate_statistic: Whether to accumulate statistics over multiple calls.

        :return: Dictionary containing the metric values for each head, and the aggregated metric value if
                 an aggregate_metrics_function is specified.

        """

        results = {}

        for head_key, metric in self.metrics_functions.items():
            results[head_key] = metric(predicted_classes=predicted_classes[head_key],
                                       target_classes=target_classes[head_key])

        return self.__organize_output(results)

    def update_state(self,
                     predicted_classes: Dict[str, torch.Tensor],
                     target_classes: Dict[str, torch.Tensor]) -> None:
        """
        Updates the state of individual metrics with new predicted and target classes.

        :param predicted_classes: Dictionary containing predicted classes for each head.
        :param target_classes: Dictionary containing target classes for each head.

        """

        for head_key, metric in self.metrics_functions.items():
            metric.update_state(predicted_classes=predicted_classes[head_key],
                                target_classes=target_classes[head_key])

    def reset_state(self) -> None:
        """
        Resets the state of all individual metrics.

        """

        for metric in self.metrics_functions.values():
            metric.reset_state()

    def get_result(self) -> Dict[str, Float]:
        """
        Retrieves the results of all individual metrics.

        :return: Dictionary containing the metric values for each head, and the aggregated metric value if
                 an aggregate_metrics_function is specified.

        """

        results = {}

        for head_key, metric in self.metrics_functions.items():
            results[head_key] = metric.get_result()

        return self.__organize_output(results)

    def __organize_output(self, results: Dict[str, float]) -> Dict:
        """
        Organizes the output of the metric functions.

        :param results: Dictionary containing the metric values for each head.

        :return: Dictionary containing the metric values for each head, and the aggregated metric value if
                 an aggregate_metrics_function is specified.

        """
        if self.aggregate_metrics_function is not None:
            if self.metric_weights is not None:
                weighted_metric_results = torch.tensor(self.metric_weights) * torch.tensor(list(results.values()))
                results[self.name] = self.aggregate_metrics_function(weighted_metric_results).items()
            else:
                results[self.name] = self.aggregate_metrics_function(torch.tensor(list(results.values()))).items()

        results = {self.metrics_functions[key].name if key in self.metrics_functions.keys()
                   else key: results[key] for key in results.keys()}

        return results


class F1_Score(SingleHeadMetric):
    """
    F1 Score metric implementation for multiclass classification tasks.

    Attributes:
        name (str): Name of the metric.
        mode (str): Computation mode for F1 Score ('none', 'binary','macro', 'micro').
        pos_label: Used when mode='binary' to select the class you want to consider.
        num_classes (int): Number of classes in the classification task.
        classes_to_exclude (list[int] or np.ndarray[int]): Classes to exclude from the computation.
        classes_to_consider (np.ndarray[int]): Classes to consider for computation.
        tps (np.ndarray): True positives for each class.
        fps (np.ndarray): False positives for each class.
        fns (np.ndarray): False negatives for each class.

    Methods:
        __call__(self, predicted_classes, target_classes, accumulate_statistic=False):
            Computes the F1 Score based on predicted and target classes.
        update_state(self, predicted_classes, target_classes):
            Updates the internal state of the F1 Score metric.
        reset_state(self, *args, **kwargs):
            Resets the internal state of the F1 Score metric.
        get_result(self, *args, **kwargs):
            Computes and returns the final F1 Score result.
        __str__(self) -> str:
            Returns the name of the metric as a string.

    """

    def __init__(self,
                 name: str = 'default_name',
                 mode: str = 'macro',
                 pos_label: int = 1,
                 num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,

                 **parent_params: Dict[str, Any]):
        """
        Initialize the F1 Score metric.

        :param name: Name of the metric.
        :param mode: Computation mode for F1 Score ('none', 'macro', 'micro').
        :param pos_label: Used when mode='binary' to select the class you want to consider.
        :param num_classes: Number of classes. Required for computing F1 Score.
        :param classes_to_exclude: Classes to exclude from the computation. Defaults to None.

        **parent_params: Additional parameters to be passed to the parent class.

        """

        super().__init__(name=name, num_classes=num_classes, **parent_params)

        self.mode = mode
        self.pos_label = pos_label
        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(self.num_classes)[
            ~np.isin(np.arange(self.num_classes), self.classes_to_exclude)]
        self.tps = np.zeros((self.num_classes,))
        self.fps = np.zeros((self.num_classes,))
        self.fns = np.zeros((self.num_classes,))

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):
        """
        Compute the F1 Score based on predicted and target classes.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :param accumulate_statistic: Whether to accumulate internal statistics.

        :return: Computed F1 Score.

        """

        tps, fps, fns = self.update_state(predicted_classes, target_classes)
        if not accumulate_statistic:
            self.reset_state()

        eps = np.finfo(float).eps
        denominators = 2 * tps + fps + fns
        f1s = 2 * tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[self.classes_to_consider]
        elif self.mode == 'binary':
            result = f1s[self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(f1s[self.classes_to_consider])
        elif self.mode == 'micro':
            result = 2 * np.sum(tps[self.classes_to_consider]) / np.sum(denominators[self.classes_to_consider])
        else:
            raise ValueError("Undefined mode specified, available modes are 'none','macro' and 'micro'")

        return result

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> Tuple[np.array, np.array, np.array]:
        """
        Update the internal state of the F1 Score metric.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.

        :return: Tuple containing true positives, false positives, and false negatives.

        """

        predicted_classes = self.pred_transform(predicted_classes)
        target_classes = self.target_transform(target_classes)

        predicted_classes = predicted_classes.cpu().numpy()
        target_classes = target_classes.cpu().numpy()

        mask = ~np.isin(target_classes, self.classes_to_exclude)

        predicted_classes = predicted_classes[mask]
        target_classes = target_classes[mask]

        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for predicted_id, target_id in zip(predicted_classes, target_classes):
            confusion_matrix[predicted_id, target_id] += 1

        tps = np.diag(confusion_matrix)
        fps = np.sum(confusion_matrix, axis=1) - tps
        fns = np.sum(confusion_matrix, axis=0) - tps

        self.tps += tps
        self.fps += fps
        self.fns += fns

        return tps, fps, fns

    def reset_state(self) -> None:
        """
        Reset the internal state of the F1 Score metric.

        :return: None

        """

        self.tps = np.zeros((self.num_classes,))
        self.fps = np.zeros((self.num_classes,))
        self.fns = np.zeros((self.num_classes,))

    def get_result(self) -> float:
        """
        Compute and return the final F1 Score result.

        :return: Computed F1 Score.

        """

        eps = np.finfo(float).eps
        denominators = 2 * self.tps + self.fps + self.fns
        f1s = 2 * self.tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[self.classes_to_consider]
        elif self.mode == 'binary':
            result = f1s[self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(f1s[self.classes_to_consider])
        elif self.mode == 'micro':
            result = 2 * np.sum(self.tps[self.classes_to_consider]) / np.sum(denominators[self.classes_to_consider])
        else:
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary', 'macro' and 'micro'")

        return result
