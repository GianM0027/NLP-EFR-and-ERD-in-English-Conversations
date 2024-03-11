import numpy as np
import pandas as pd


class RandomClassifier:
    """
    Random Classifier.

    This classifier performs random predictions for the 'emotions' and 'triggers' columns based on the provided list
    of possible emotion values.

    Attributes:
        emotions (list[str]): A list of possible emotion values.
        n_emotions (int): The number of possible emotion values.

    Methods:
        predict(test_df: pd.DataFrame) -> (list[list[str]], list[list[int]]):
            Performs random predictions for the 'emotions' and 'triggers' columns in the test data.

    """

    def __init__(self, emotions: list[str]):
        self.emotions = emotions
        self.n_emotions = len(emotions)

    def predict(self, test_df: pd.DataFrame) -> (list[list[str]], list[list[int]]):
        """
        Performs a random prediction for columns "emotions" and "triggers"

        :param test_df: test set on which compute the predictions
        :return: a tuple where the first element is a lists of lists containing emotions randomly sampled (strings), the
        second element is a list of lists of 0 and 1s (int).
        """
        emotion_predictions = []
        triggers_predictions = []

        for index, row in test_df.iterrows():
            len_emotions = len(row["emotions"])

            rand_emotions = [self.emotions[np.random.randint(0, self.n_emotions)] for _ in range(len_emotions)]

            rand_triggers = [float(np.random.randint(0, 2)) for _ in range(len_emotions)]

            emotion_predictions.append(rand_emotions)
            triggers_predictions.append(rand_triggers)

        return emotion_predictions, triggers_predictions
