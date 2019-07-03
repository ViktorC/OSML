import pickle

import numpy as np
import pandas as pd


class Model:
    """
    An abstract class representing a statistical model or machine learning algorithm.
    """
    def __init__(self, epsilon):
        if epsilon is None or epsilon <= 0:
            raise ValueError
        self.epsilon = epsilon
        self._observation_types = None
        self._label_type = None
        self._label_name = ''

    def _validate_observations(self, observations_df):
        if len(observations_df.dtypes) == 0 or len(observations_df.index) == 0 \
                or any(not np.issubdtype(dtype, np.number) for dtype in observations_df.dtypes) \
                or not observations_df.dtypes.equals(self._observation_types):
            raise ValueError

    def _validate_labels(self, labels_sr):
        if len(labels_sr.index) == 0 or not np.issubdtype(labels_sr.dtype, np.number) \
                or labels_sr.dtype != self._label_type:
            raise ValueError

    def _validate_observations_and_labels(self, observations_df, labels_sr):
        self._validate_observations(observations_df)
        self._validate_labels(labels_sr)
        if len(observations_df.index) != len(labels_sr.index):
            raise ValueError

    def _fit(self, observations_df: pd.DataFrame, labels_sr: pd.Series) -> None:
        pass

    def fit(self, observations_df, labels_sr):
        """
        It fits the model to the data and establishes the parameter estimates, if any.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
            labels_sr: A series of target values where each element is the label of the corresponding row in the data
            frame of observations. It can be None if the model is unsupervised.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels (unless no labels
            are provided).
        """
        self._observation_types = observations_df.dtypes
        if labels_sr is not None:
            self._label_type = labels_sr.dtype
            self._label_name = labels_sr.name
            self._validate_observations_and_labels(observations_df, labels_sr)
        else:
            self._validate_observations(observations_df)
        return self._fit(observations_df, labels_sr)


def save(model, file_path):
    """
    It serializes the model into a binary format and writes it to the specified file path.

    Args:
        model: The model to serialize.
        file_path: The path to the file to which the serialized model is to be written.
    """
    with open(file_path, mode='wb') as file:
        pickle.dump(model, file)


def load(file_path):
    """
    It reads a binary file and deserializes a model from its contents.

    Args:
        file_path: The path to the file containing the serialized model.

    Returns:
        The saved machine learning model.
    """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)


def is_continuous(data_type):
    """
    It determines whether the data type is a sub-type of the float data type.

    Args:
        data_type: The data type.

    Returns:
        Whether the data type is continuous/real.
    """
    return np.issubdtype(data_type, np.floating)


def is_categorical(data_type):
    """
    It determines whether the data type is a sub-type of the integer data type.

    Args:
        data_type: The data type.

    Returns:
        Whether the data type is categorical.
    """
    return np.issubdtype(data_type, np.integer)
