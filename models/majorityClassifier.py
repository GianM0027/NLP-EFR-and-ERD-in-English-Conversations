import numpy as np
import pandas as pd


class MajorityClassifier:
    """
    Majority Classifier.

    This classifier predicts the majority emotion and trigger values observed in the training data for each instance
    in the test data.

    Attributes:
        major_emotion (str): The majority emotion value observed in the training data.
        major_trigger (str): The majority trigger value observed in the training data.

    Methods:
        fit(train_df: pd.DataFrame) -> None:
            Fits the majority classifier to the training data by determining the majority emotion and trigger values
            observed in the training data.

        predict(test_df: pd.DataFrame) -> (list[list[str]], list[list[int]]):
            Performs predictions based on the majority emotion and trigger values observed in the training data.

    """

    def __init__(self):
        self.major_emotion = None
        self.major_trigger = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fits the majority classifier to the training data by determining the majority emotion and trigger values
        observed in the training data.


        :params train_df: The training DataFrame containing the 'emotions' and 'triggers' columns.

        Returns: None

        """

        flatten_emotions = [item for sublist in train_df["emotions"] for item in sublist]
        flatten_triggers = [item for sublist in train_df["triggers"] for item in sublist]

        emotion_values, emotion_counts = np.unique(flatten_emotions, return_counts=True)
        triggers_values, triggers_counts = np.unique(flatten_triggers, return_counts=True)

        self.major_emotion = emotion_values[np.argmax(emotion_counts)]
        self.major_trigger = int(triggers_values[np.argmax(triggers_counts)])

    def predict(self, test_df: pd.DataFrame) -> (list[list[str]], list[list[int]]):
        """
        Performs predictions based on the majority emotion and trigger values observed in the training data.

        :params test_df: The test DataFrame on which to compute the predictions.

        :returns:
            tuple: A tuple containing:
                - emotion_predictions (list[list[str]]): A list of lists containing the predicted emotion values (strings).
                - triggers_predictions (list[list[int]]): A list of lists containing the predicted trigger values (0 or 1).
        """
        emotion_predictions = []
        triggers_predictions = []

        for index, row in test_df.iterrows():
            # compute how many emotions and triggers to predict
            len_emotions = len(row["emotions"])

            emotions = [self.major_emotion for _ in range(len_emotions)]
            triggers = [self.major_trigger for _ in range(len_emotions)]

            # append the predictions
            emotion_predictions.append(emotions)
            triggers_predictions.append(triggers)

        return emotion_predictions, triggers_predictions
