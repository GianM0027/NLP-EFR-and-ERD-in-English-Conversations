from typing import Tuple, List, Optional, Dict, Union

import transformers
from IPython.core.display_functions import display
import os

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix


import pandas as pd
import numpy as np
import torch

from transformers import BertModel, BertTokenizer, LongformerModel, LongformerTokenizer

import DrTorch.metrics
from DrTorch.metrics import F1_Score


class Sentence_F1:
    """
       A class to compute the F1 score for sentences/dialogues.

       Attributes:
       - f1_obj (F1_Score): An instance of the F1_Score class used for computing F1 score.
       - key_to_consider (str): The key to consider for F1 score computation.
       - f1_scores_per_dialogue (list): A list to store F1 scores computed for each dialogue.
       - name (str): Name of the F1 score object.
    """
    def __init__(self, f1_obj: F1_Score, key_to_consider: str):
        """
        Initialize Sentence_F1 class.

        :params  f1_obj: An instance of the F1_Score class.
        :params  key_to_consider: The key to consider for F1 score computation.

        """

        self.f1_obj = f1_obj
        self.key_to_consider = key_to_consider
        self.f1_scores_per_dialogue = []
        self.name = self.f1_obj.name

    def __call__(self, predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False) -> float:

        """
        Compute the mean F1 score for predicted and target classes.


        :params predicted_classes (torch.Tensor): Predicted classes.
        :params target_classes (torch.Tensor): Target (true) classes.
        :params accumulate_statistic (bool): Whether to accumulate statistics or not.

        Returns: Mean F1 score over the dialogues.

        """

        f1_scores_per_dialogue = self.update_state(predicted_classes, target_classes)

        if not accumulate_statistic:
            self.reset_state()

        mean_f1 = sum(f1_scores_per_dialogue) / len(f1_scores_per_dialogue)

        return mean_f1

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> np.array:
        """
        Update the internal state of F1 score metric.


        :params  predicted_classes: Predicted classes.
        :params  target_classes: Target (true) classes.

        Returns: List of F1 scores per dialogue.

        """

        f1_scores_per_dialogue = []

        for pred, target in zip(predicted_classes, target_classes):
            Single_dialogue_pred = torch.unsqueeze(pred, dim=0)
            Single_dialogue_target = torch.unsqueeze(target, dim=0)
            f1_scores_per_dialogue.append(
                self.f1_obj(predicted_classes=Single_dialogue_pred, target_classes=Single_dialogue_target))

        self.f1_scores_per_dialogue += f1_scores_per_dialogue

        return f1_scores_per_dialogue

    def reset_state(self) -> None:
        """
        Reset the internal state of the F1 Score metric.

        :return: None

        """

        self.f1_scores_per_dialogue = []

    def get_result(self) -> float:
        """
        Compute and return the final F1 Score result.

        :return: Computed F1 Score.

        """

        mean_f1 = sum(self.f1_scores_per_dialogue) / len(self.f1_scores_per_dialogue)

        return mean_f1


def replace_nan_with_zero(lst: List) -> List[float]:
    """
    Takes a list with NaN values and converts them to zero.

    :param lst: original list.

    :return: the list with all the NaNs converted to zero

    """

    return [0.0 if pd.isna(x) else x for x in lst]


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

    :param df: original dataframe to split.

    :return: train, validation and test sets (80, 10, 10), in this order.

    """

    df['index'] = df.index

    df_shuffled = df.sample(frac=1, replace=False)

    train_end = int(len(df_shuffled) * 0.8)
    val_end = int(len(df_shuffled) * 0.9)

    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]

    train_df.set_index('index', inplace=True, drop=True)
    val_df.set_index('index', inplace=True, drop=True)
    test_df.set_index('index', inplace=True, drop=True)

    return train_df, val_df, test_df


def plot_emotion_distribution(train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              column_name: str,
                              title: str) -> None:
    """
    Plot the distribution of emotions in the three datasets

    :param train_df: training set.
    :param val_df: validation set.
    :param test_df: test set.
    :param column_name: name of the column whose distributions you want to plot.
    :param title: title.

    :return: None

    """

    plt.figure(figsize=(18, 6))
    plt.suptitle(title)

    datasets = [train_df, val_df, test_df]
    titles = ['Training Set', 'Validation Set', 'Test Set']

    for i, dataset in enumerate(datasets):
        plt.subplot(1, 3, i + 1)
        flatten_values = [item for sublist in dataset[column_name] for item in sublist]
        values, counts = np.unique(flatten_values, return_counts=True)
        plt.bar(values, counts)
        plt.title(titles[i])
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

        plt.bar(list(count_dict.keys()), list(count_dict.values()))
        plt.title(titles[i])
        plt.xticks(rotation=45)
        plt.ylabel('Counts')

    plt.tight_layout()
    plt.show()


def plot_all_distributions(df): # TODO CANCELLA LA FUNZIONE QUANDO ABBIAMO FINITO TUTTO
    """
    FUNZIONE DA CANCELLARE, L'HO FATTA SOLO PER PRODURRE UN GRAFICO PER IL REPORT E LA TENGO QUA PER SICUREZZA
    Args:
        df:

    Returns:

    """
    plt.figure(figsize=(18, 6))

    # emotion distribution
    plt.subplot(1, 3, 1)
    flatten_values = [item for sublist in df["emotions"] for item in sublist]
    values, counts = np.unique(flatten_values, return_counts=True)
    plt.bar(values, counts)
    plt.title("Emotion distribution", fontsize = 15)
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=45)  # Rotate labels to avoid overlap
    plt.ylabel('Counts', fontsize = 12)
    plt.xlabel('Emotions', fontsize = 12)

    # positive triggers per emotion
    plt.subplot(1, 3, 2)
    flatten_emotions = [item for sublist in df["emotions"] for item in sublist]
    flatten_triggers = [item for sublist in df["triggers"] for item in sublist]

    count_dict = {emotion: 0 for emotion in np.unique(flatten_emotions)}
    for idx in range(len(flatten_triggers)):
        if flatten_triggers[idx] == 1:
            count_dict[flatten_emotions[idx]] += 1

    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.title("Positive triggers per emotion", fontsize = 15)
    plt.xlabel('Emotions', fontsize = 12)
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.xticks(rotation=45)

    # trigger distribution
    plt.subplot(1, 3, 3)
    flatten_values = [item for sublist in df["triggers"] for item in sublist]
    values, counts = np.unique(flatten_values, return_counts=True)
    plt.bar(values, counts)
    plt.title("Trigger distribution", fontsize = 15)
    plt.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.xticks([0, 1])  # Set x-ticks to only include 0 and 1
    plt.xlabel('Trigger value', fontsize = 12)
    plt.show()

    plt.tight_layout()
    plt.show()


def display_dialogue(dataframe: pd.DataFrame, utterance_id: str | int, verbose = True) -> None:
    """
    Display the data related to a specific utterance id

    :param dataframe: A pandas dataframe that contain the data
    :param utterance_id: Utterance id related to the speach that you want to show
    :param verbose: flag to print the number of utterance

    :return: None

    """

    table_data = []

    df_portion = dataframe.loc[utterance_id] if type(utterance_id) is str else dataframe.iloc[utterance_id]

    for column_value in df_portion:
        table_data.append(column_value)

    new_df = pd.DataFrame(table_data).transpose()
    new_df.columns = dataframe.columns
    if verbose:
        if type(utterance_id) is str:
            print(utterance_id.replace('_', ' ').capitalize())
        else:
            print('Utterance_' + str(utterance_id))
    display(new_df)
    print()


def produce_speaker_emotion_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
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


def download_bert_initializers(bert_constructor: type[BertModel | LongformerModel],
                               bert_tokenizer_constructor: type[BertTokenizer | LongformerTokenizer],
                               bert_path: os.path,
                               version:str = 'bert-base-uncased') -> Tuple[BertModel, BertTokenizer]:
    """
    Downloads the BERT model and tokenizer of 'local-bert' and saves them to a specified directory.
    This function checks if the directory exists, creates it if it does not, downloads the model and tokenizer,
    and saves them in the specified directory for future utilization.

    :param bert_path: The directory path where the BERT model and tokenizer should be saved.
    :param version: The bert Version.
    :param bert_constructor: Constructor for the Bert model.
    :param bert_tokenizer_constructor: Constructor for the Bert Tokenizer.
    :return: a tuple containing the downloaded BertModel and BertTokenizer instances.

    """

    model = bert_constructor.from_pretrained(version)
    tokenizer = bert_tokenizer_constructor.from_pretrained(version)

    tokenizer.save_pretrained(bert_path)
    model.save_pretrained(bert_path)

    print(f"BERT model and tokenizer have been saved to {bert_path}")

    return model, tokenizer


def retrieve_bert_initializers(bert_constructor: type[BertModel | LongformerModel],
                               bert_tokenizer_constructor: type[BertTokenizer | LongformerTokenizer],
                               bert_path: os.path) -> Tuple[BertModel, BertTokenizer] | Tuple[LongformerModel, LongformerTokenizer]:
    """
    Retrieves the BERT model and tokenizer from a specified directory.
    This function loads the BERT model and tokenizer that were previously saved in a specified directory,
    and returns them for future use.

    :param bert_constructor: Constructor for the Bert model.
    :param bert_tokenizer_constructor: Constructor for the Bert Tokenizer.
    :param bert_path: The directory path where the BERT model and tokenizer should be retrieved.

    :return: a tuple containing the loaded BertModel and BertTokenizer instances.

    """

    model = bert_constructor.from_pretrained(bert_path)
    tokenizer = bert_tokenizer_constructor.from_pretrained(bert_path)

    return model, tokenizer


def find_max_encoded_utterance_len(tokenizer: BertTokenizer, data: pd.Series) -> Tuple[int, int]:
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


def plot_distribution_of_dialogue_lengths(df: pd.DataFrame, tokenizer: BertTokenizer, figsize, vert=False, title = 'Distribution of Dialogue Lengths') -> None:
    """
    Plot the distribution (boxplot) of the utterance length (tokenized)

    :param df: dataframe to plot
    :param tokenizer: bert tokenizer to use
    :param vert: flag to set the plot vertical

    :returns: None
    """
    tokenized_utterances = tokenizer.batch_encode_plus(df["utterances"].sum(), padding=False)
    lengths = [len(utterance) for utterance in tokenized_utterances["input_ids"]]

    plt.figure(figsize=figsize)
    plt.boxplot(lengths, vert=vert)
    max_length = max(lengths)

    if vert:
        plt.yticks(np.arange(0, max_length + 1, 5))
        plt.xticks([])
    else:
        plt.xticks(np.arange(0, max_length + 1, 5))
        plt.yticks([])
    plt.xlabel('Length of Dialogues')
    plt.title(title)
    plt.grid(True)

    plt.show()

def remove_longest_utterances(data: pd.DataFrame, tokenizer: BertTokenizer, threshold: int = 65) -> pd.DataFrame:
    """
    removes all the dialogues which contain an utterance above a certain threshold

    :param data: dataframe from which data is retrieved
    :param tokenizer: tokenizer used to compute the tokenized sentences
    :param threshold: value after which a dialogue is excluded

    :return: a copy of the original data without the long utterances
    """
    indexes_to_drop = [
        idx
        for idx, dialogue in data["utterances"].items()
        if any(
            len(tokenizer.encode_plus(utterance, padding=False)['input_ids']) > threshold
            for utterance in dialogue
        )
    ]
    print(f"{len(indexes_to_drop)} rows dropped\n")
    return data.drop(indexes_to_drop)


def plot_number_of_trigger_frequency(data: list) -> None:
    """
    Plot the distribution of the number of triggers per dialogue

    :param data: the list containig the number of trigger per dialogue

    :return: None

    """

    plt.hist(data, bins=range(min(data), max(data) + 2), edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], range(0, 10))
    plt.title('N of Triggers per Dialogue')
    plt.grid(False)
    plt.show()


def plot_trigger_position_reversed(data: list) -> None:
    """
    Plot the distribution of the number of triggers per dialogue

    :param data: the list containig the position of each trigger in the dataset
    expressed by the distance to the end of the dialoge in utternces

    :return: None

    """

    plt.hist(data, bins=range(min(data), 12),
             edgecolor='black')  # 12 is set to display only the relevant part of the histogram
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
               ['i', 'i -1', 'i -2', 'i -3', 'i -4', 'i -5', 'i -6', 'i -7', 'i -8', 'i -9', 'i -10'])
    plt.title('Position of Triggers')
    plt.grid(False)
    plt.show()


def pad_utterances(sequences: List[torch.Tensor], pad_token_id: int):
    """
    Pad a list of sequences with a given pad token ID.

    :param sequences: A list of PyTorch tensors representing sequences.
    :param pad_token_id: The ID of the padding token.

    :return: A PyTorch tensor containing padded sequences.

    """
    max_dialogue_length = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padding_needed = max_dialogue_length - seq.size(0)
        if padding_needed > 0:
            padding_tensor = torch.full((padding_needed, seq.size(1)), pad_token_id)
            padded_seq = torch.cat([seq, padding_tensor], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences)


def tokenize_data_big_bertOne(data: pd.Series, max_tokenized_length: int, tokenizer: transformers.BartTokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenize a pandas Series of text data.

    :params data: A pandas Series containing text data.
    :params max_tokenized_length: The maximum lang of the tokenized sentence.
    :params tokenizer: Bert tokenizer.

    :returns: A dictionary containing tokenized input, attention masks, and token type IDs.

    """

    input_ids_list = []
    attention_masks_list = []

    for text_list in data:
        tokenized_utterances = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_list,
                                                           padding="max_length",
                                                           max_length=max_tokenized_length,
                                                           return_tensors='pt')

        input_ids, attention_mask = remove_redundant_cls_big_bertOne(tokenized_utterances['input_ids'],
                                                          tokenized_utterances['attention_mask'])

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

    padded_input_ids = pad_utterances(input_ids_list, tokenizer.pad_token_id)
    padded_attention_masks = pad_utterances(attention_masks_list, 0)

    output = {'input_ids': padded_input_ids,
              'attention_mask': padded_attention_masks}

    return output

def remove_redundant_cls_big_bertOne(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Removes the redundant [CLS] token from the input tensors.


    :params input_ids (torch.Tensor): Tensor containing the input token IDs.
    :params attention_mask (torch.Tensor): Tensor containing the attention mask.
    :params token_type_ids (torch.Tensor): Tensor containing the token type IDs.

    :returns:
        tuple: A tuple containing the modified input tensors:
            - input_ids (torch.Tensor): Tensor containing the input token IDs with the first [CLS] token removed.
            - attention_mask (torch.Tensor): Tensor containing the attention mask with the first [CLS] token removed.
            - token_type_ids (torch.Tensor): Tensor containing the token type IDs with the first [CLS] token removed.

    """

    pad = torch.zeros(input_ids.shape[0] - 1)

    input_ids[1:, 0] = pad
    attention_mask[1:, 0] = pad

    return input_ids, attention_mask


def remove_redundant_cls(input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
    """
    Removes the redundant [CLS] token from the input tensors.


    :params input_ids (torch.Tensor): Tensor containing the input token IDs.
    :params attention_mask (torch.Tensor): Tensor containing the attention mask.
    :params token_type_ids (torch.Tensor): Tensor containing the token type IDs.

    :returns:
        tuple: A tuple containing the modified input tensors:
            - input_ids (torch.Tensor): Tensor containing the input token IDs with the first [CLS] token removed.
            - attention_mask (torch.Tensor): Tensor containing the attention mask with the first [CLS] token removed.
            - token_type_ids (torch.Tensor): Tensor containing the token type IDs with the first [CLS] token removed.

    """

    pad = torch.zeros(input_ids.shape[0] - 1)

    input_ids[1:, 0] = pad
    attention_mask[1:, 0] = pad
    token_type_ids[1:, 0] = pad

    return input_ids, attention_mask, token_type_ids


def tokenize_data(data: pd.Series, max_tokenized_length: int, tokenizer: transformers.BartTokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenize a pandas Series of text data.

    :params data: A pandas Series containing text data.
    :params max_tokenized_length: The maximum lang of the tokenized sentence.
    :params tokenizer: Bert tokenizer.

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


def tokenize_data_Bertone(data: pd.Series, max_tokenized_length: int, tokenizer: transformers.BartTokenizer) -> Dict[
    str, torch.Tensor]:
    """
    Tokenize a pandas Series of text data.

    :params data: A pandas Series containing text data.
    :params max_tokenized_length: The maximum lang of the tokenized sentence.
    :params tokenizer: Bert tokenizer.

    :returns: A dictionary containing tokenized input, attention masks, and token type IDs.

    """

    input_ids_list = []
    attention_masks_list = []

    for text_list in data:
        tokenized_utterances = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_list,
                                                           padding="max_length",
                                                           max_length=max_tokenized_length,
                                                           return_tensors='pt')

        input_ids, attention_mask = remove_redundant_cls_Bertone(tokenized_utterances['input_ids'],
                                                                 tokenized_utterances['attention_mask'])

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

    padded_input_ids = pad_utterances(input_ids_list, tokenizer.pad_token_id)
    padded_attention_masks = pad_utterances(attention_masks_list, 0)

    output = {'input_ids': padded_input_ids,
              'attention_mask': padded_attention_masks}

    return output


def remove_redundant_cls_Bertone(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Removes the redundant [CLS] token from the input tensors.


    :params input_ids (torch.Tensor): Tensor containing the input token IDs.
    :params attention_mask (torch.Tensor): Tensor containing the attention mask.
    :params token_type_ids (torch.Tensor): Tensor containing the token type IDs.

    :returns:
        tuple: A tuple containing the modified input tensors:
            - input_ids (torch.Tensor): Tensor containing the input token IDs with the first [CLS] token removed.
            - attention_mask (torch.Tensor): Tensor containing the attention mask with the first [CLS] token removed.
            - token_type_ids (torch.Tensor): Tensor containing the token type IDs with the first [CLS] token removed.

    """

    pad = torch.zeros(input_ids.shape[0] - 1)

    input_ids[1:, 0] = pad
    attention_mask[1:, 0] = pad

    return input_ids, attention_mask
def create_directories(paths: List[os.PathLike]) -> None:
    """
    Creates al the directories listed in paths (excluding files at the end of it, if present)

    :param paths: directories to create.
    :return: None
    """
    for path in paths:
        if os.path.isfile(path):
            directory = os.path.dirname(path)
        else:
            directory = path
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def compute_unrolled_f1(emotion_f1: DrTorch.metrics.F1_Score,
                        trigger_f1: DrTorch.metrics.F1_Score,
                        targets_emotions: pd.Series,
                        predictions_emotions: pd.Series,
                        targets_triggers: pd.Series,
                        predictions_triggers: pd.Series,
                        emotion_to_index: dict[str, int]):
    """
    Computes the F1 scores for emotions and triggers across the entire dataset after unrolling the predictions and targets.


    :params emotion_f1: A function that computes the F1 score for emotions.
    :params trigger_f1: A function that computes the F1 score for triggers.
    :params df: The DataFrame containing the dataset.
    :params emotion_to_index: A dictionary mapping emotions to their corresponding indices.

    returns: - unrolled_emotion_f1: The F1 score for emotions after unrolling.
             - unrolled_trigger_f1: The F1 score for triggers after unrolling.
    """

    unrolled_predicted_emotions = [emotion_to_index[emotion] for emotion in predictions_emotions.sum()]
    unrolled_target_emotion = [emotion_to_index[emotion] for emotion in targets_emotions.sum()]

    unrolled_predicted_triggers = predictions_triggers.sum()
    unrolled_target_triggers = targets_triggers.sum()

    unrolled_emotion_f1 = emotion_f1(predicted_classes=torch.tensor(unrolled_predicted_emotions, dtype=torch.int),
                                     target_classes=torch.tensor(unrolled_target_emotion, dtype=torch.int))

    unrolled_trigger_f1 = trigger_f1(predicted_classes=torch.tensor(unrolled_predicted_triggers, dtype=torch.int),
                                     target_classes=torch.tensor(unrolled_target_triggers, dtype=torch.int))

    return unrolled_emotion_f1, unrolled_trigger_f1


def compute_f1_per_dialogues(emotion_f1: DrTorch.metrics.F1_Score,
                             trigger_f1: DrTorch.metrics.F1_Score,
                             targets_emotions: pd.Series,
                             predictions_emotions: pd.Series,
                             targets_triggers: pd.Series,
                             predictions_triggers: pd.Series,
                             emotion_to_index: dict[str, int]):
    """
    Computes the F1 scores for emotions and triggers per dialogue in the dataset.


    :params emotion_f1: A function that computes the F1 score for emotions.
    :params trigger_f1: A function that computes the F1 score for triggers.
    :params df: The DataFrame containing the dataset.
    :params emotion_to_index: A dictionary mapping emotions to their corresponding indices.

    returns: DataFrame: A DataFrame containing the F1 scores for emotions and triggers per dialogue.
    """

    sequences_emotions_f1 = {}
    sequences_triggers_f1 = {}
    for dialog_id in targets_emotions.index:
        sequences_emotions_f1[dialog_id] = emotion_f1(predicted_classes=torch.tensor([emotion_to_index[e] for e in predictions_emotions[dialog_id]], dtype=torch.int),
                                                      target_classes=torch.tensor([emotion_to_index[e] for e in targets_emotions[dialog_id]], dtype=torch.int))

        sequences_triggers_f1[dialog_id] = trigger_f1(predicted_classes=torch.tensor([predictions_triggers[dialog_id]], dtype=torch.int),
                                                      target_classes=torch.tensor([targets_triggers[dialog_id]], dtype=torch.int))

    return pd.DataFrame(data={'emotion_f1': list(sequences_emotions_f1.values()),
                              'trigger_f1': list(sequences_triggers_f1.values())},
                        index=list(sequences_emotions_f1.keys()))


def reshape_loss_input(x: torch.Tensor):
    return x.view(-1, x.shape[-1])


def plot_trigger_frequency_and_position_reversed(df: pd.DataFrame) -> None:
    """
    Plot the frequency and reversed position of triggers

    :param df: DataFrame containing triggers data

    :return: None
    """
    plt.figure(figsize=(18, 6))
    plt.suptitle("Frequency and Reversed Position of Triggers")

    triggers_f = []
    triggers_pos = []

    for tr in df:
        triggers_f.append(int(sum(tr)))
        for i in range(len(tr)):
            if tr[i] == 1:
                triggers_pos.append(len(tr) - i - 1)

    plt.subplot(1, 2, 1)
    plt.hist(triggers_f, bins=range(min(triggers_f), max(triggers_f) + 2), edgecolor='black')
    plt.xlabel('Number of Triggers')
    plt.ylabel('Frequency')
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], range(0, 10))
    plt.title('Trigger Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(triggers_pos, bins=range(min(triggers_pos), 12), edgecolor='black')
    plt.xlabel('Position (Reversed)')
    plt.ylabel('Frequency')
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
               ['i', 'i -1', 'i -2', 'i -3', 'i -4', 'i -5', 'i -6', 'i -7', 'i -8', 'i -9', 'i -10'])
    plt.title('Trigger Position (Reversed)')

    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(target_emotions: pd.Series,
                          pred_emotions: pd.Series,
                          emotion_to_index_map: dict[str, int],
                          target_triggers: pd.Series,
                          pred_triggers: pd.Series,
                          title: str):

    target_emotion_flattened = [emotion_to_index_map[emotion] for emotion in target_emotions.sum()]
    pred_emotions_flattened = [emotion_to_index_map[emotion] for emotion in pred_emotions.sum()]
    target_triggers_flatten = target_triggers.sum()
    pred_triggers_flatten = pred_triggers.sum()

    emotion_conf_matrix = confusion_matrix(target_emotion_flattened, pred_emotions_flattened)[:7,:7]
    trigger_conf_matrix = confusion_matrix(target_triggers_flatten, pred_triggers_flatten)[:2,:2]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    emotion_labels = list(emotion_to_index_map.keys())[:-1]

    im1 = axes[0].imshow(emotion_conf_matrix, cmap='Blues', interpolation='nearest')
    axes[0].set_title('Confusion Matrix - Emotions')
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')
    axes[0].set_xticks(np.arange(len(emotion_labels)))
    axes[0].set_yticks(np.arange(len(emotion_labels)))
    axes[0].set_xticklabels(emotion_labels, rotation=45)
    axes[0].set_yticklabels(emotion_labels)
    for i in range(len(emotion_to_index_map)-1):
        for j in range(len(emotion_to_index_map)-1):
            axes[0].text(j, i, emotion_conf_matrix[i, j], ha='center', va='center', color='black')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(trigger_conf_matrix, cmap='Blues', interpolation='nearest')
    axes[1].set_title('Confusion Matrix - Triggers')
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('True labels')
    axes[1].set_xticks(np.arange(2))
    axes[1].set_yticks(np.arange(2))
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, trigger_conf_matrix[i, j], ha='center', va='center', color='black')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
