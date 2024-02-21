from typing import Tuple, List

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from transformers import BertModel, BertTokenizer



def replace_nan_with_zero(lst: List) -> List:
    """
    Takes a list with NaN values and converts them to zero.

    :param lst: original list
    :return: the list with all the NaNs converted to zero
    """
    return [0.0 if np.isnan(x) else x for x in lst if np.isscalar(x)]


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given a dataframe, return a (80, 10, 10) random split of its rows in 3 new dataframes

    :param df: original dataframe to split
    :return: train, validation and test sets (80, 10, 10), in this order.
    """
    df['index'] = df.index

    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42, replace=False)

    # Calculate the indexes for splitting
    train_end = int(len(df_shuffled) * 0.8)
    val_end = int(len(df_shuffled) * 0.9)

    # Split the data
    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]

    # Riporta la colonna 'index' come indice
    train_df.set_index('index', inplace=True, drop=True)
    val_df.set_index('index', inplace=True, drop=True)
    test_df.set_index('index', inplace=True, drop=True)

    return train_df, val_df, test_df


def plot_emotion_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Plot the distribution of emotions in the three datasets

    :param train_df: training set
    :param val_df: validation set
    :param test_df: test set
    :return: None
    """
    plt.figure(figsize=(18, 6))
    plt.suptitle("Emotions occurrence in the datasets")

    datasets = [train_df, val_df, test_df]
    titles = ['Training Set', 'Validation Set', 'Test Set']

    for i, dataset in enumerate(datasets):
        plt.subplot(1, 3, i + 1)
        flatten_emotions = [item for sublist in dataset["emotions"] for item in sublist]
        emotion_values, emotion_counts = np.unique(flatten_emotions, return_counts=True)
        plt.bar(emotion_values, emotion_counts)
        plt.title(titles[i])
        plt.grid()
        plt.xticks(rotation=45)  # Rotate labels to avoid overlap
        plt.ylabel('Counts')

    plt.tight_layout()
    plt.show()


def display_utterance(dataframe: pd.DataFrame, utterance_id: str | int):
    """
    Display the data relate to a specific utterance id

    :param dataframe: A pandas dataframe that contain the data
    :param utterance_id: Utterance id related to the speach that you want to show
    :return: None
    """
    table_data = []
    for column_value in dataframe.loc[utterance_id]:
        table_data.append(column_value)

    new_df = pd.DataFrame(table_data).transpose()
    new_df.columns = dataframe.columns
    print(utterance_id.replace('_', ' ').capitalize())
    display(new_df)
    print()


def download_bert_initializers(bert_path: os.path) -> (BertModel, BertTokenizer):
    """
    Downloads the BERT model and tokenizer of 'bert-base-uncased' and saves them to a specified directory.
    This function checks if the directory exists, creates it if it does not, downloads the model and tokenizer,
    and saves them in the specified directory for future utilization.

    :param bert_path: The directory path where the BERT model and tokenizer should be saved.
    :return: a tuple containing the downloaded BertModel and BertTokenizer instances.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    tokenizer.save_pretrained(bert_path)
    model.save_pretrained(bert_path)

    print(f"BERT model and tokenizer have been saved to {bert_path}")

    return model, tokenizer


def retrieve_bert_initializers(bert_path: os.path) -> (BertModel, BertTokenizer):
    """
    Retrieves the BERT model and tokenizer from a specified directory.
    This function loads the BERT model and tokenizer that were previously saved in a specified directory,
    and returns them for future use.

    :param bert_path: The directory path where the BERT model and tokenizer should be retrieved.
    :return: a tuple containing the loaded BertModel and BertTokenizer instances.
    """
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)

    return model, tokenizer


