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
        self.name = name

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

    """

    def __init__(self,
                 name: str,
                 metrics_functions: Dict[str, Callable],
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
        self.metrics_functions = {}
        self.metric_weights = metric_weights

        """
        for head_key, metric_params in metrics_functions:
            metric_constructor = metric_params.pop('metric_constructor')
            self.metrics_functions[head_key] = metric_constructor(**metric_params)
        """

        if aggregate_metrics_function is not None:
            self.aggregate_metrics_function = aggregate_metrics_function

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False) -> Dict:
        """
        Computes the metric values based on the predicted and target classes.

        :param predicted_classes: Predicted classes for each head.
        :param target_classes: Target classes for each head.
        :param accumulate_statistic: Whether to accumulate statistics over multiple calls.

        :return: Dictionary containing the metric values for each head.

        """

        results = {}

        for head_key, metric in self.metrics_functions:
            results[head_key] = metric(predicted_classes=predicted_classes[head_key],
                                       target_classes=target_classes[head_key])

        if self.aggregate_metrics_function is not None:
            if self.metric_weights is not None:
                results[self.name] = self.aggregate_metrics_function(self.metric_weights, results.values())
            else:
                results[self.name] = self.aggregate_metrics_function(results.values())

        return results

    def update_state(self,
                     predicted_classes: Dict[str, torch.Tensor],
                     target_classes: Dict[str, torch.Tensor]) -> None:
        """
        Updates the state of individual metrics with new predicted and target classes.

        :param predicted_classes: Dictionary containing predicted classes for each head.
        :param target_classes: Dictionary containing target classes for each head.

        """

        for head_key, metric in self.metrics_functions:
            metric.update_state(predicted_classes=predicted_classes[head_key],
                                target_classes=target_classes[head_key])

    def reset_state(self) -> None:
        """
        Resets the state of all individual metrics.

        """

        for _, metric in self.metrics_functions:
            metric.reset_state()

    def get_result(self) -> Dict[str, Float]:
        """
        Retrieves the results of all individual metrics.

        :return: Dictionary containing the results of all individual metrics.

        """

        results = {}
        for head_key, metric in self.metrics_functions:
            results[head_key] = metric.get_result()
        if self.aggregate_metrics_function is not None:
            if self.metric_weights is not None:
                results[self.name] = self.aggregate_metrics_function(self.metric_weights, results.values())
            else:
                results[self.name] = self.aggregate_metrics_function(results.values())

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
                 mode: str = 'macro',
                 pos_label: int = 1,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 **parent_params: Dict[str, Any]):
        """

        :param mode: Computation mode for F1 Score ('none', 'macro', 'micro').
        :param pos_label: Used when mode='binary' to select the class you want to consider.
        :param classes_to_exclude: Classes to exclude from the computation.

        """

        super().__init__(**parent_params)

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


class F1_Score_Multi_Labels:
    """
        F1_Score_Multi_Labels class implements the F1 Score metric for multi-label classification tasks.

        Args:
            name (str): Name of the metric.
            num_classes (int): Number of classes in the classification task.
            num_labels (int): Number of labels associated with each sample.
            mode (str, optional): Computation mode for F1 Score ('none', 'macro', 'binary','micro'). Defaults to 'macro'.
            pos_label (int): Used when mode='binary' to select the class you want to consider.
            compute_mean (bool, optional): Whether to compute the mean of F1 Scores. Defaults to True.
            classes_to_exclude (list or np.ndarray, optional): Classes to exclude from the computation.

        Attributes:
            name (str): Name of the metric.
            mode (str): Computation mode for F1 Score ('none', 'macro', 'micro').
            num_classes (int): Number of classes in the classification task.
            classes_to_exclude (list or np.ndarray): Classes to exclude from the computation.
            classes_to_consider (np.ndarray): Classes to consider based on exclusion.
            num_labels (int): Number of labels associated with each sample.
            compute_mean (bool): Whether to compute the mean of F1 Scores.
            tps (np.ndarray): True positives count.
            fps (np.ndarray): False positives count.
            fns (np.ndarray): False negatives count.

        Methods:
            update_state(predicted_classes: torch.Tensor, target_classes: torch.Tensor) -> tuple[np.array, np.array, np.array]:
                Update the internal state of the F1 Score metric.
            get_result() -> float:
                Compute and return the final F1 Score result.
            reset_state() -> None:
                Reset the internal state of the F1 Score metric.
            __str__() -> str:
                Get the name of the metric as a string.
            set_mode(compute_mean: bool) -> None:
                Set the computation mode for mean.
            __call__(predicted_classes: torch.Tensor, target_classes: torch.Tensor, accumulate_statistic: bool = False) -> float:
                Update the state, compute F1 Scores, and return the result.

        Raises:
            ValueError: If an undefined mode is specified.

        Notes:
            This class is designed to works if each label has the same number of classes

        Example:
            ```python
            f1_metric = F1_Score_Multi_Labels(name='F1_Score', num_classes=10, num_labels=5)
            result = f1_metric(predicted_classes, target_classes)
            ```

        """

    def __init__(self,
                 name: str,
                 num_classes: int,
                 num_labels: int,
                 mode: str = 'macro',
                 pos_label: int = 1,
                 compute_mean: bool = True,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None):
        """

        :param name: Name of the metric.
        :param num_classes:  Number of classes in the classification task.
        :param num_labels: Number of labels.
        :param mode: Computation mode for F1 Score ('none', 'macro', 'micro', 'binary').
        :param pos_label: Used when mode='binary' to select the class you want to consider.
        :param compute_mean: flag to compute the mean over the different labels.
        :param classes_to_exclude: Classes to exclude from the computation.

        """

        self.name = name
        self.mode = mode
        self.pos_label = pos_label
        self.num_classes = num_classes
        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(num_classes)[~np.isin(np.arange(num_classes), self.classes_to_exclude)]
        self.num_labels = num_labels
        self.compute_mean = compute_mean

        self.tps = np.zeros((self.num_labels, self.num_classes))
        self.fps = np.zeros((self.num_labels, self.num_classes))
        self.fns = np.zeros((self.num_labels, self.num_classes))

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> Tuple[np.array, np.array, np.array]:
        """
        Update the internal state of the F1 Score metric.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.

        :return: Tuple containing true positives, false positives, and false negatives.

        """

        tps = np.zeros((self.num_labels, self.num_classes))
        fps = np.zeros((self.num_labels, self.num_classes))
        fns = np.zeros((self.num_labels, self.num_classes))

        for i in range(self.num_labels):
            current_pred_class = predicted_classes[:, i].cpu().detach().numpy().astype(int)
            current_target_classes = target_classes[:, i].cpu().detach().numpy().astype(int)

            mask = ~np.isin(current_target_classes, self.classes_to_exclude)

            current_pred_class = current_pred_class[mask]
            current_target_classes = current_target_classes[mask]

            confusion_matrix = np.zeros((self.num_classes, self.num_classes))
            for predicted_id, target_id in zip(current_pred_class, current_target_classes):
                confusion_matrix[predicted_id, target_id] += 1

            tps[i, :] = np.diag(confusion_matrix)
            fps[i, :] = np.sum(confusion_matrix, axis=1) - tps[i]
            fns[i, :] = np.sum(confusion_matrix, axis=0) - tps[i]

        self.tps += tps
        self.fps += fps
        self.fns += fns

        return tps, fps, fns

    def get_result(self) -> float:
        """
        Compute and return the final F1 Score result.

        :return: Computed F1 Score.

        """

        eps = np.finfo(float).eps

        denominators = 2 * self.tps + self.fps + self.fns
        f1s = 2 * self.tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[:, self.classes_to_consider]
        elif self.mode == 'binary':
            result = f1s[:, self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(f1s[:, self.classes_to_consider], axis=1)
        elif self.mode == 'micro':
            result = 2 * np.sum(self.tps[:, self.classes_to_consider], axis=1) / np.sum(
                denominators[:, self.classes_to_consider], axis=1)
        else:
            raise ValueError("Undefined mode specified, available modes are 'none','macro' and 'micro'")

        if self.compute_mean:
            if self.mode == 'none':
                result = np.mean(result, axis=0)
            else:
                result = np.mean(result)

        return result

    def reset_state(self) -> None:
        """
         Reset the internal state of the F1 Score metric.

        :return: None
        """

        self.tps = np.zeros((self.num_labels, self.num_classes))
        self.fps = np.zeros((self.num_labels, self.num_classes))
        self.fns = np.zeros((self.num_labels, self.num_classes))

    def __str__(self) -> str:
        """
        Get the name of the metric as a string.

        :return: Name of the metric.

        """

        return self.name

    def set_mode(self, compute_mean_flag: bool):
        self.compute_mean = compute_mean_flag

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):

        """
        Compute for each label the F1 Score based on predicted and target classes.


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
            result = f1s[:, self.classes_to_consider]
        elif self.mode == 'binary':
            result = f1s[:, self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(f1s[:, self.classes_to_consider], axis=1)
        elif self.mode == 'micro':
            result = 2 * np.sum(tps[:, self.classes_to_consider], axis=1) / np.sum(
                denominators[:, self.classes_to_consider], axis=1)
        else:
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary', 'macro' and 'micro'")

        if self.compute_mean:
            if self.mode == 'none':
                result = np.mean(result, axis=0)
            else:
                result = np.mean(result)

        return result
