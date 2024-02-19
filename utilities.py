import numpy as np
import pandas as pd


def replace_nan_with_zero(lst: list) -> list:
    """
    Takes a list with NaN values and convert them to zero

    :param lst: original list
    :return: the list with all the NaNs converted to zero
    """
    return [0 if (x is not float or np.isnan(x)) else x for x in lst]


def split_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Given a dataframe, return a (80, 10, 10) random split of its rows in 3 new dataframes

    :param df: original dataframe to split
    :return: train, validation and test sets (80, 10, 10), in this order.
    """
    df_rows = df.shape[0]
    indexes = np.arange(0, df_rows)
    np.random.shuffle(indexes)

    train_end = int(df_rows * 0.8)
    val_end = int(df_rows * 0.9)

    train_indexes = indexes[:train_end]
    val_indexes = indexes[train_end:val_end]
    test_indexes = indexes[val_end:]

    return df.iloc[train_indexes, :], df.iloc[val_indexes, :], df.iloc[test_indexes, :]

