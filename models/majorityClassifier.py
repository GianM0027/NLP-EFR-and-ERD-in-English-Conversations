import numpy as np
import pandas as pd


class MajorityClassifier:
    """
    Majority classifier
    """

    def __init__(self):
        self.major_emotion = None
        self.major_trigger = None

    def fit(self, train_df: pd.DataFrame) -> None:
        flatten_emotions = [item for sublist in train_df["emotions"] for item in sublist]
        flatten_triggers = [item for sublist in train_df["triggers"] for item in sublist]

        emotion_values, emotion_counts = np.unique(flatten_emotions, return_counts=True)
        triggers_values, triggers_counts = np.unique(flatten_triggers, return_counts=True)

        self.major_emotion = emotion_values[np.argmax(emotion_counts)]
        self.major_trigger = triggers_values[np.argmax(triggers_counts)]

    def predict(self, test_df: pd.DataFrame) -> (list[list[str]], list[list[int]]):
        """
        Performs a prediction based on the majority of values for columns "emotions" and "triggers"

        :param test_df: test set on which compute the predictions
        :return: a tuple where the first element is a lists of lists containing the predicted emotion (strings), the
        second element is a list of lists of 0 or 1s (int).
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
