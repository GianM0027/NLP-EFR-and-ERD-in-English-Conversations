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

    Examples:
        # Example of how to instantiate and use MultyHeadMetric
        import torch
        from DrTorch.metrics import MultyHeadMetric, Recall, Precision

        # Sample predicted and target classes for different heads
        predicted_classes = {
            'head1': torch.tensor([0, 1, 2, 1, 0]),
            'head2': torch.tensor([1, 2, 1, 0, 1])
        }
        target_classes = {
            'head1': torch.tensor([0, 1, 2, 2, 1]),
            'head2': torch.tensor([2, 1, 0, 1, 2])
        }

        # Instantiate individual metrics for each head
        recall_metric = Recall(name='Recall', num_classes=3)
        precision_metric = Precision(name='Precision', num_classes=3)

        # Define metrics functions dictionary for each head
        metrics_functions = {'head1': recall_metric, 'head2': precision_metric}

        # Instantiate MultyHeadMetric with defined metrics functions
        multi_head_metric = MultyHeadMetric(name='MultiHeadMetric',
                                            metrics_functions=metrics_functions,
                                            aggregate_metrics_function=your_custom_aggregation_function)

        # Compute metric values for each head
        results = multi_head_metric(predicted_classes, target_classes)

        # Retrieve results
        print("Metric Results:", results)

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
                results[self.name] = self.aggregate_metrics_function(weighted_metric_results).item()
            else:
                results[self.name] = self.aggregate_metrics_function(torch.tensor(list(results.values()))).item()

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

    Examples:
        # Example of how to instantiate and use F1_Score metric
        import torch
        from your_module import F1_Score

        # Sample predicted and target classes
        predicted_classes = torch.tensor([0, 1, 2, 1, 0])
        target_classes = torch.tensor([0, 1, 2, 2, 1])

        # Instantiate F1_Score metric
        f1_metric = F1_Score(name='F1_macro', mode='macro', num_classes=3)

        # Calculate F1 Score
        f1_score = f1_metric(predicted_classes, target_classes)

        print("F1 Score:", f1_score)

    """

    def __init__(self,
                 name: str = 'F1_macro',
                 mode: str = 'macro',
                 pos_label: int = 1,
                 num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the F1 Score metric.

        :param name: Name of the metric.
        :param mode: Computation mode for F1 Score ('none', 'binary', 'macro', 'micro').
        :param pos_label: Used when mode='binary' to select the class you want to consider.
        :param num_classes: Number of classes. Required for computing F1 Score.
        :param classes_to_exclude: Classes to exclude from the computation. Defaults to None.
        :param pred_transform:
        :param target_transform:

        **parent_params: Additional parameters to be passed to the parent class.

        """

        super().__init__(name=name,
                         num_classes=num_classes,
                         pred_transform=pred_transform,
                         target_transform=target_transform)

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
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary','macro' and 'micro'")

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


class Accuracy(SingleHeadMetric):
    """
    Accuracy metric implementation for multiclass classification tasks.

    Attributes:
        name (str): Name of the metric.
        num_classes (int): Number of classes in the classification task.
        classes_to_exclude (list[int] or np.ndarray[int]): Classes to exclude from the computation.
        classes_to_consider (np.ndarray[int]): Classes to consider for computation.
        numerator (float): Accumulator for the numerator in accuracy computation.
        denominator (float): Accumulator for the denominator in accuracy computation.

    Methods:
        __init__(self, name: str = 'Accuracy', num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
            Initializes the Accuracy metric.

        __call__(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor,
                 accumulate_statistic: bool = False) -> float:
            Computes the accuracy based on predicted and target classes.

        update_state(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor) -> Tuple[float, float]:
            Updates the internal state of the Accuracy metric.

        reset_state(self) -> None:
            Resets the internal state of the Accuracy metric.

        get_result(self) -> float:
            Computes and returns the final accuracy result.
     Examples:
        # Example of how to instantiate and use Accuracy metric
        import torch
        from your_module import Accuracy

        # Sample predicted and target classes
        predicted_classes = torch.tensor([0, 1, 2, 1, 0])
        target_classes = torch.tensor([0, 1, 2, 2, 1])

        # Instantiate Accuracy metric
        accuracy_metric = Accuracy(name='Accuracy', num_classes=3)

        # Calculate accuracy
        accuracy = accuracy_metric(predicted_classes, target_classes)

        print("Accuracy:", accuracy)

    """

    def __init__(self,
                 name: str = 'Accuracy',
                 num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the Accuracy metric.


        :params:name: Name of the metric.
        :params:num_classes: Number of classes in the classification task.
        :params:classes_to_exclude: Classes to exclude from the computation. Defaults to None.
        :params:pred_transform: A transformation function for mapping the model output into a suitable
                                             metric input. Defaults to None.
        :params:target_transform: A transformation function for mapping the target labels into a suitable
                                                    metric input. Defaults to None.

        """

        super().__init__(name=name,
                         num_classes=num_classes,
                         pred_transform=pred_transform,
                         target_transform=target_transform)

        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(self.num_classes)[
            ~np.isin(np.arange(self.num_classes), self.classes_to_exclude)]
        self.numerator = 0
        self.denominator = 0

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):
        """
        Compute the Accuracy metric based on predicted and target classes.

        :params predicted_classes: Predicted classes.
        :params target_classes: Target (ground truth) classes.
        :params accumulate_statistic: Whether to accumulate internal statistics.

        returns: Computed Accuracy Score.

        """

        numerator, denominator = self.update_state(predicted_classes, target_classes)
        if not accumulate_statistic:
            self.reset_state()

        accuracy = numerator / denominator
        return accuracy

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> Tuple[Float, Float]:
        """
        Update the internal state of the Accuracy metric.

        :params predicted_classes: Predicted classes.
        :params target_classes: Target (ground truth) classes.

        returns: True positives and false positives.

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

        numerator = np.sum(np.diag(confusion_matrix))
        denominator = np.sum(confusion_matrix, axis=None)

        self.numerator += numerator
        self.denominator += denominator

        return numerator, denominator

    def reset_state(self) -> None:
        """
        Reset the internal state of the Accuracy metric.

        :return: None

        """

        self.numerator = 0
        self.denominator = 0

    def get_result(self) -> float:
        """
        Compute and return the final Accuracy metric.

        :return: Computed Accuracy Score.

        """

        return self.numerator / self.denominator


class Recall(SingleHeadMetric):
    """
    Recall metric implementation for multiclass classification tasks.

    Attributes:
        name (str): Name of the metric.
        mode (str): Computation mode for Recall ('none', 'binary', 'macro', 'micro').
        pos_label (int): Used when mode='binary' to select the class you want to consider.
        num_classes (int): Number of classes in the classification task.
        classes_to_exclude (list[int] or np.ndarray[int]): Classes to exclude from the computation.
        pred_transform (Optional[Callable]): A transformation function for mapping the model output into a suitable
                                              metric input.
        target_transform (Optional[Callable]): A transformation function for mapping the target labels into a suitable
                                                metric input.
        tps (int): True positives.
        fns (int): False negatives.

    Methods:
        __init__(self, name: str = 'Accuracy', num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
            Initialize the Recall metric.
        __call__(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor,
                 accumulate_statistic: bool = False) -> float:
            Compute the Recall Score based on predicted and target classes.
        update_state(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor) -> Tuple[Float, Float]:
            Update the internal state of the Recall Score metric.
        reset_state(self) -> None:
            Reset the internal state of the Recall metric.
        get_result(self) -> float:
            Compute and return the final Recall metric.
    Examples:
        # Example of how to instantiate and use Recall metric
        import torch
        from your_module import Recall

        # Sample predicted and target classes
        predicted_classes = torch.tensor([0, 1, 2, 1, 0])
        target_classes = torch.tensor([0, 1, 2, 2, 1])

        # Instantiate Recall metric
        recall_metric = Recall(name='Recall', num_classes=3)

        # Calculate Recall Score
        recall = recall_metric(predicted_classes, target_classes)

        print("Recall Score:", recall)

    """

    def __init__(self,
                 name: str = 'Recall',
                 mode: str = 'macro',
                 pos_label: int = 1,
                 num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the Recall metric.


        :params name: Name of the metric.
        :params num_classes: Number of classes in the classification task.
        :param mode: Computation mode for F1 Score ('none', 'binary', 'macro', 'micro').
        :param pos_label: Used when mode='binary' to select the class you want to consider.
        :params classes_to_exclude: Classes to exclude from the computation.
        :params pred_transform: A transformation function for mapping the model output into a suitable
                                metric input.
        :params target_transform: A transformation function for mapping the target labels into a suitable
                                  metric input.
        """
        super().__init__(name=name,
                         num_classes=num_classes,
                         pred_transform=pred_transform,
                         target_transform=target_transform)

        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(self.num_classes)[
            ~np.isin(np.arange(self.num_classes), self.classes_to_exclude)]
        self.mode = mode
        self.pos_label = pos_label
        self.tps = np.zeros((self.num_classes,))
        self.fns = np.zeros((self.num_classes,))

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):
        """
        Compute the Recall Score based on predicted and target classes.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :param accumulate_statistic: Whether to accumulate internal statistics.

        :return: Computed Recall Score.

        """

        tps, fns = self.update_state(predicted_classes, target_classes)

        if not accumulate_statistic:
            self.reset_state()

        eps = np.finfo(float).eps
        denominators = tps + fns + eps

        recalls = tps / denominators

        if self.mode == 'none':
            result = recalls[self.classes_to_consider]
        elif self.mode == 'binary':
            result = recalls[self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(recalls)
        elif self.mode == 'micro':
            result = np.sum(tps[self.classes_to_consider]) / (np.sum(tps[self.classes_to_consider] + fns[self.classes_to_consider]))
        else:
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary','macro' and 'micro'")

        return result

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> Tuple[np.array, np.array]:
        """
        Update the internal state of the Recall Score metric.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.

        :return: Tuple containing true positives and false negatives.

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
        fns = np.sum(confusion_matrix, axis=0) - tps

        self.tps += tps
        self.fns += fns

        return tps, fns

    def reset_state(self) -> None:
        """
        Reset the internal state of the Recall metric.

        :return: None

        """

        self.tps = np.zeros((self.num_classes,))
        self.fns = np.zeros((self.num_classes,))

    def get_result(self) -> float:
        """
        Compute and return the final Recall metric.

        :return: Computed Recall Score.

        """

        eps = np.finfo(float).eps
        denominators = 2 * self.tps + self.fns + eps
        recalls = self.tps / denominators

        if self.mode == 'none':
            result = recalls[self.classes_to_consider]
        elif self.mode == 'binary':
            result = recalls[self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(recalls)
        elif self.mode == 'micro':
            result = np.sum(self.tps[self.classes_to_consider]) / (np.sum(self.tps[self.classes_to_consider]) + np.sum(self.fns[self.classes_to_consider]))
        else:
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary','macro' and 'micro'")

        return result


class Precision(SingleHeadMetric):
    """
    Precision metric implementation for multiclass classification tasks.

    Attributes:
        name (str): Name of the metric.
        mode (str): Computation mode for Recall ('none', 'binary', 'macro', 'micro').
        pos_label (int): Used when mode='binary' to select the class you want to consider.
        num_classes (int): Number of classes in the classification task.
        classes_to_exclude (list[int] or np.ndarray[int]): Classes to exclude from the computation.
        pred_transform (Optional[Callable]): A transformation function for mapping the model output into a suitable
                                             metric input.
        target_transform (Optional[Callable]): A transformation function for mapping the target labels into a suitable
                                               metric input.
        tps (int): True positives.
        fps (int): False positives.

    Methods:
        __init__(self, name: str = 'Accuracy', num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
            Initialize the Precision metric.
        __call__(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor,
                 accumulate_statistic: bool = False) -> float:
            Compute the Precision Score based on predicted and target classes.
        update_state(self, predicted_classes: torch.Tensor, target_classes: torch.Tensor) -> Tuple[Float, Float]:
            Update the internal state of the Precision Score metric.
        reset_state(self) -> None:
            Reset the internal state of the Precision metric.
        get_result(self) -> float:
            Compute and return the final Precision metric.

    Examples:
        # Example of how to instantiate and use Precision metric
        import torch
        from your_module import Precision

        # Sample predicted and target classes
        predicted_classes = torch.tensor([0, 1, 2, 1, 0])
        target_classes = torch.tensor([0, 1, 2, 2, 1])

        # Instantiate Precision metric
        precision_metric = Precision(name='Precision', num_classes=3)

        # Calculate Precision Score
        precision = precision_metric(predicted_classes, target_classes)

        print("Precision Score:", precision)

    """

    def __init__(self,
                 name: str = 'Precision',
                 mode: str = 'macro',
                 pos_label: int = 1,
                 num_classes: int = None,
                 classes_to_exclude: Optional[List[int] | np.ndarray[int]] = None,
                 pred_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the Recall metric.

        :param name: Name of the metric.
        :param num_classes: Number of classes in the classification task.
        :param mode: Computation mode for F1 Score ('none', 'binary', 'macro', 'micro').
        :param pos_label: Used when mode='binary' to select the class you want to consider.
        :param classes_to_exclude: Classes to exclude from the computation.
        :param pred_transform: A transformation function for mapping the model output into a suitable metric input.
        :param target_transform: A transformation function for mapping the target labels into a suitable metric input.

        """

        super().__init__(name=name,
                         num_classes=num_classes,
                         pred_transform=pred_transform,
                         target_transform=target_transform)

        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(self.num_classes)[
            ~np.isin(np.arange(self.num_classes), self.classes_to_exclude)]

        self.mode = mode
        self.pos_label = pos_label
        self.tps = np.zeros((self.num_classes,))
        self.fps = np.zeros((self.num_classes,))

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):
        """
        Compute the Recall Score based on predicted and target classes.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :param accumulate_statistic: Whether to accumulate internal statistics.

        :return: Computed Precision Score.

        """

        tps, fps = self.update_state(predicted_classes, target_classes)

        if not accumulate_statistic:
            self.reset_state()

        eps = np.finfo(float).eps
        denominator = tps + fps + eps

        precision = tps / denominator

        if self.mode == 'none':
            result = precision[self.classes_to_consider]
        elif self.mode == 'binary':
            result = precision[self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(precision)
        elif self.mode == 'micro':
            result = np.sum(tps[self.classes_to_consider]) / (np.sum(tps[self.classes_to_consider] + fps[self.classes_to_consider]))
        else:
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary','macro' and 'micro'")

        return result

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> Tuple[np.array, np.array]:
        """
        Update the internal state of the Precision Score metric.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.

        :return: Tuple containing true positives and false positive.

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

        self.tps += tps
        self.fps += fps

        return tps, fps

    def reset_state(self) -> None:
        """
        Reset the internal state of the Precision metric.

        :return: None

        """

        self.tps = np.zeros((self.num_classes,))
        self.fps = np.zeros((self.num_classes,))

    def get_result(self) -> float:
        """
        Compute and return the final Precision metric.

        :return: Computed Precision Score.

        """
        eps = np.finfo(float).eps
        denominators = 2 * self.tps + self.fps + eps
        recalls = self.tps / denominators

        if self.mode == 'none':
            result = recalls[self.classes_to_consider]
        elif self.mode == 'binary':
            result = recalls[self.pos_label]
        elif self.mode == 'macro':
            result = np.mean(recalls)
        elif self.mode == 'micro':
            result = np.sum(self.tps[self.classes_to_consider]) / (np.sum(self.tps[self.classes_to_consider] + self.fps[self.classes_to_consider]))
        else:
            raise ValueError("Undefined mode specified, available modes are 'none', 'binary','macro' and 'micro'")

        return result
