import numpy as np
import pandas as pd


class RandomClassifier:
    """
    Random classifier
    """

    def __init__(self, classes: list[str]):
        self.classes = classes
        self.num_classes = len(classes)

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
            # compute how many emotions and triggers to predict
            len_emotions = len(row["emotions"])

            # randomly select classes from self.classes list
            rand_emotions = [self.classes[np.random.randint(0, self.num_classes)] for _ in range(len_emotions)]

            # randomly select 0 and 1s
            rand_triggers = [np.random.randint(0, 2) for _ in range(len_emotions)]

            # append the random predictions
            emotion_predictions.append(rand_emotions)
            triggers_predictions.append(rand_triggers)

        return emotion_predictions, triggers_predictions
