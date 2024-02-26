from typing import Tuple, List, Optional

import os
from IPython.core.display_functions import display

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pandas as pd
import numpy as np

from transformers import BertModel, BertTokenizer


def replace_nan_with_zero(lst: List) -> List:
    """
    Takes a list with NaN values and converts them to zero.

    :param lst: original list
    :return: the list with all the NaNs converted to zero
    """
    return [0.0 if np.isnan(x) else x for x in lst if np.isscalar(x)]


def create_wordcloud(df: pd.DataFrame, my_class_index: str = 'WORD', f_sizes: tuple[int, int] = (10, 5)) -> None:
    """
    Generates and displays a word cloud based on the specified DataFrame and column.

    :param df: The input DataFrame containing text data.
    :param my_class_index: The column name in the DataFrame that contains the text data.
      Defaults to 'WORD'.
    :param f_sizes: A tuple representing the size of the generated plot.

    :return: This function displays the generated word cloud using Matplotlib.

    Example:
    ```python
    import pandas as pd
    from my_wordcloud_module import create_wordcloud

    # Assuming 'df' is a DataFrame with a column named 'WORD' containing text data
    create_wordcloud(df, my_class_index='WORD', f_sizes=(12, 6))
    ```

    Note:
    - Ensure that the 'wordcloud' and 'matplotlib.pyplot' libraries are installed.
    - You can install them using the following:
      ```
      pip install wordcloud
      pip install matplotlib
      ```
    """
    text = " ".join(df[my_class_index].sum())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=f_sizes)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given a dataframe, return a (80, 10, 10) random split of its rows in 3 new dataframes

    :param df: original dataframe to split
    :return: train, validation and test sets (80, 10, 10), in this order.
    """
    df['index'] = df.index

    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, replace=False)

    # Calculate the indexes for splitting
    train_end = int(len(df_shuffled) * 0.8)
    val_end = int(len(df_shuffled) * 0.9)

    # Split the data
    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]

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


def plot_triggers_per_emotion(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    For each emotion, plot how many triggers it activates

    :param train_df: training set
    :param val_df: validation set
    :param test_df: test set
    :return: None
    """
    plt.figure(figsize=(18, 6))
    plt.suptitle("Emotions related to trigger's activation")

    datasets = [train_df, val_df, test_df]
    titles = ['Training Set', 'Validation Set', 'Test Set']

    for i, dataset in enumerate(datasets):
        plt.subplot(1, 3, i + 1)
        flatten_emotions = [item for sublist in dataset["emotions"] for item in sublist]
        flatten_triggers = [item for sublist in dataset["triggers"] for item in sublist]

        count_dict = {emotion: 0 for emotion in np.unique(flatten_emotions)}
        for idx in range(len(flatten_triggers)):
            if flatten_triggers[idx] == 1:
                count_dict[flatten_emotions[idx]] += 1

        plt.bar(count_dict.keys(), count_dict.values())
        plt.title(titles[i])
        plt.grid()
        plt.xticks(rotation=45)  # Rotate labels to avoid overlap
        plt.ylabel('Counts')

    plt.tight_layout()
    plt.show()


def display_dialogue(dataframe: pd.DataFrame, utterance_id: str | int) -> None:
    """
    Display the data relate to a specific utterance id

    :param dataframe: A pandas dataframe that contain the data
    :param utterance_id: Utterance id related to the speach that you want to show

    :return: None

    """

    table_data = []

    df_portion = dataframe.loc[utterance_id] if type(utterance_id) is str else dataframe.iloc[utterance_id]

    for column_value in df_portion:
        table_data.append(column_value)

    new_df = pd.DataFrame(table_data).transpose()
    new_df.columns = dataframe.columns
    if type(utterance_id) is str:
        print(utterance_id.replace('_', ' ').capitalize())
    else:
        print('Utterance_' + str(utterance_id))
    display(new_df)
    print()


def produce_speaker_emotion_distribution(dataframe) -> pd.DataFrame:
    """
     Produce emotion distribution for each speaker based on the provided dataframe.

     This function extracts emotion data associated with each speaker from the input dataframe and creates a new dataframe
     representing the distribution of emotions for each speaker.

     :param dataframe: A pandas DataFrame containing the data. It should have columns 'speakers' and 'emotions', where
                       'speakers' contain lists of speaker identifiers and 'emotions' contain lists of emotions associated
                       with each speaker.
     :return: A pandas DataFrame representing the emotion distribution for each speaker.
              The DataFrame has speakers as rows and emotions as columns. Each cell represents the count of occurrences
              of an emotion for a particular speaker. If an emotion didn't occur for a speaker, the cell value will be 0.
     """

    tmp_dict = {'speakers': [element for current_list in dataframe['speakers'] for element in current_list],
                'emotions': [element for current_list in dataframe['emotions'] for element in current_list]
                }
    tmp_df = pd.DataFrame(tmp_dict)
    emotions = tmp_df['emotions'].unique().tolist()
    tpm_df = tmp_df.pivot_table(index='speakers', columns='emotions', aggfunc='size', fill_value=0)

    return tpm_df.sort_values(by=emotions, ascending=[False]*len(emotions))


def concat_with_sep(string_list):
    """
    Concatenates a list of strings with a separator.

    :params string_list (list): A list of strings to be concatenated.

    :returns: A string obtained by joining all the strings in the input list with the separator '[SEP]'.

    Example:
        >>> concat_with_sep(['a', 'b', 'c'])
        'a [SEP] b [SEP] c'

    """
    return " [SEP] ".join(string_list)


def create_classes_weights(list_of_label_index: list[int], list_of_index_to_exclude: Optional[List] = None) -> np.array:
    """
    Calculate class weights for a list of classes.

    This function takes a 2D list of classes (e.g., part-of-speech tags)
    and calculates class weights to use during the training process
    based on the formula n_samples / (n_classes * np.bincount(y)).


    list_of_label_index A list of indexes that represent the classes,
    list_of_index_to_exclude: List of elements to be dropped from consideration when calculating weights.

    :return: An array of class weights corresponding to the input classes. The weights are inversely
             proportional to the class occurrences in the input data. The weights for the 'list_of_index_to_exclude' classes
             to drop are set to 0.
    """

    if list_of_index_to_exclude is None:
        list_of_index_to_exclude = []

    occurrences = np.bincount(list_of_label_index)
    tmp_occurrences = occurrences.copy()
    tmp_occurrences[list_of_index_to_exclude] = 0
    class_weights = np.sum(tmp_occurrences) / ((len(occurrences) - len(list_of_index_to_exclude)) * occurrences)
    class_weights[list_of_index_to_exclude] = 0
    return class_weights


def download_bert_initializers(bert_path: os.path) -> Tuple[BertModel, BertTokenizer]:
    """
    Downloads the BERT model and tokenizer of 'local-bert' and saves them to a specified directory.
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


def retrieve_bert_initializers(bert_path: os.path) -> Tuple[BertModel, BertTokenizer]:
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
