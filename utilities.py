from typing import Tuple, List, Optional, Dict

import os
from IPython.core.display_functions import display

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pandas as pd
import numpy as np
import torch

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

    return tpm_df.sort_values(by=emotions, ascending=[False] * len(emotions))


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


def find_max_encoded_utterance(tokenizer: BertTokenizer, data: pd.Series) -> int:
    """
    Find the maximum length of encoded utterances in the given data.

    :param tokenizer: The tokenizer object used for encoding.

    :param data: A  pd.Series of utterances.

    :return: The maximum length of encoded utterances.
    """

    tokenized_batch = tokenizer.batch_encode_plus(data.sum(),
                                                  padding=True,
                                                  return_tensors='pt')
    return tokenized_batch["input_ids"].shape[1]


def pad_utterances(sequences: List[torch.Tensor], pad_token_id):
    """
    Pad a list of sequences with a given pad token ID.

    :param sequences: A list of PyTorch tensors representing sequences.
    :param pad_token_id: The ID of the padding token.

    :return: A PyTorch tensor containing padded sequences.

    """

    max_list_length = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padding_needed = max_list_length - seq.size(0)
        if padding_needed > 0:
            padding_tensor = torch.full((padding_needed, seq.size(1)), pad_token_id)
            padded_seq = torch.cat([seq, padding_tensor], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences)


def remove_redundant_cls(input_ids, attention_mask, token_type_ids):
    pad = torch.zeros(input_ids.shape[0]-1)

    input_ids[1:, 0] = pad
    attention_mask[1:, 0] = pad
    token_type_ids[1:, 0] = pad

    return input_ids, attention_mask, token_type_ids

def tokenize_data(data: pd.Series, max_tokenized_length, tokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenize a pandas Series of text data.

    :params data: A pandas Series containing text data.

    :returns: A dictionary containing tokenized input, attention masks, and token type IDs.

    """

    input_ids_list = []
    attention_masks_list = []
    token_type_ids_list = []

    for text_list in data:
        tokenized_utterances = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_list,
                                                           padding="max_length",
                                                           max_length=max_tokenized_length,
                                                           return_tensors='pt')

        input_ids, attention_mask, token_type_ids = remove_redundant_cls(tokenized_utterances['input_ids'],
                                                                         tokenized_utterances['attention_mask'],
                                                                         tokenized_utterances['token_type_ids'])

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)

    padded_input_ids = pad_utterances(input_ids_list, tokenizer.pad_token_id)
    padded_attention_masks = pad_utterances(attention_masks_list, 0)
    padded_token_type_ids = pad_utterances(token_type_ids_list, 0)

    output = {'input_ids': padded_input_ids,
              'attention_mask': padded_attention_masks,
              'token_type_ids': padded_token_type_ids
              }

    return output


def preprocess_labels(labels: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """
    Preprocesses emotion and trigger labels.

    :param labels: A Dataframe containing emotion and trigger labels.

    :return: A dictionary containing torch tensors for encoded emotions and triggers.

    """

    emotions, triggers = labels['emotions'], labels['triggers']
    max_length = max(emotions.apply(len).max(), triggers.apply(len).max())

    emotions_padded = emotions.apply(lambda em_list: em_list + ['z_emotion'] * (max_length - len(em_list)))
    triggers_padded = triggers.apply(lambda t_list: t_list + [2] * (max_length - len(t_list)))

    encoded_emotions_tensor = torch.tensor(pd.get_dummies(emotions_padded.sum()).values.astype(float))
    encoded_triggers_tensor = torch.tensor(pd.get_dummies(triggers_padded.sum()).values.astype(float))

    encoded_emotions_tensor = encoded_emotions_tensor.view(-1, max_length, 8)
    encoded_triggers_tensor = encoded_triggers_tensor.view(-1, max_length, 3)

    return {'emotions': encoded_emotions_tensor, 'triggers': encoded_triggers_tensor}


def create_directories(paths) -> None:
    """
    Creates al the directories listed in paths (excluding files at the end of it, if present)

    :param paths: directories to create
    :return: None
    """
    for path in paths:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
