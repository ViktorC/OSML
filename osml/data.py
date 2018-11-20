import numpy as np
import pandas as pd


class DataSet:
    """An abstract class representing a data set."""
    def get_training_observations(self):
        """Returns a data frame of the training observations.

        Returns:
            A pandas data frame of training data instances without their labels.
        """
        pass

    def get_test_observations(self):
        """Returns a data frame of the test observations.

        Returns:
            A pandas data frame of test data instances without their labels.
        """
        pass

    def get_training_labels(self):
        """Returns a series of the training observation labels.

        Returns:
            A series of values where each element is the predicted label of the corresponding training observation.
        """
        pass

    def get_test_labels(self):
        """Returns a series of the test observation labels.

        Returns:
            A series of values where each element is the predicted label of the corresponding test observation.
        """
        pass


def shuffle(observations_df, labels_sr):
    """Shuffles the rows of the observations data frame and the values of the label series.

    It applies the same permutation to the indices of both the data frame and the series. Therefore, observations
    and labels indexed by the same number will still share a common index.

    Args:
        observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
        labels_sr: A series of target values where each element is the label of the corresponding row in the data
        frame of observations.
    """
    indices = np.arange(observations_df.values.shape[0])
    np.random.shuffle(indices)
    for column in observations_df.columns:
        observations_df[column] = observations_df[column].values[indices]
    labels_sr[:] = labels_sr.values[indices]


def split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices):
    """Splits the observations and labels into training and test observations and labels.

    It also optionally shuffles the observation-label pairs and can also reset their indices after the split.

    Args:
        observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
        labels_sr: A series of target values where each element is the label of the corresponding row in the data
        frame of observations.
        test_share: The proportion of the instances that should be used for testing.
        shuffle_data: Whether the observation-label pairs are to be randomly shuffled.
        reset_indices: Whether the indices of the two data frames and the two series are to be reset after the split.

    Return:
        The training observations data frame, the test observations data frame, the training labels series, and the
        test labels series.

    Raises:
        ValueError: If the test share is not greater than 0 or not less than 1.
    """
    if test_share <= 0 or test_share >= 1:
        raise ValueError
    if shuffle_data:
        shuffle(observations_df, labels_sr)
    total_instances = len(observations_df.index)
    test_instances = int(total_instances * test_share)
    training_observations_df = observations_df.iloc[test_instances:, :]
    test_observations_df = observations_df.iloc[:test_instances, :]
    training_labels_sr = labels_sr.iloc[test_instances:]
    test_labels_sr = labels_sr.iloc[:test_instances]
    if reset_indices:
        training_observations_df = training_observations_df.reset_index(drop=True)
        test_observations_df = test_observations_df.reset_index(drop=True)
        training_labels_sr = training_labels_sr.reset_index(drop=True)
        test_labels_sr = test_labels_sr.reset_index(drop=True)
    return training_observations_df, test_observations_df, training_labels_sr, test_labels_sr


class BostonDataSet(DataSet):
    """The Boston house pricing data set."""
    def __init__(self, data_path, shuffle_data=True, reset_indices=True, test_share=.3, label_column_idx=13):
        complete_df = pd.read_csv(data_path, index_col=False)
        complete_df['chas'] = complete_df['chas'].astype(np.int_)
        labels_sr = pd.Series(complete_df[complete_df.columns[label_column_idx]])
        observations_df = complete_df.drop(complete_df.columns[label_column_idx], axis=1)
        self.training_observations_df, self.test_observations_df, self.training_labels_sr, self.test_labels_sr =\
            split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices)

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr


class ExamDataSet(DataSet):
    """
    A data set consisting of observations of two exam scores and a binary label denoting whether the student has been
    admitted to college.
    """
    def __init__(self, obs_path, label_path, shuffle_data=True, reset_indices=True, test_share=.3):
        observations_df = pd.DataFrame(np.loadtxt(obs_path))
        labels_sr = pd.Series(np.loadtxt(label_path), dtype=np.int_)
        self.training_observations_df, self.test_observations_df, self.training_labels_sr, self.test_labels_sr = \
            split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices)

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr


class IrisDataSet(DataSet):
    """The iris flower data set."""
    def __init__(self, data_path, shuffle_data=True, reset_indices=True, test_share=.3, label_column_idx=4):
        complete_df = pd.read_csv(data_path, index_col=False)
        complete_df['species'] = complete_df['species'].replace('setosa', 0)
        complete_df['species'] = complete_df['species'].replace('versicolor', 1)
        complete_df['species'] = complete_df['species'].replace('virginica', 2)
        complete_df['species'] = complete_df['species'].astype(np.int_)
        labels_sr = pd.Series(complete_df[complete_df.columns[label_column_idx]])
        observations_df = complete_df.drop(complete_df.columns[label_column_idx], axis=1)
        self.training_observations_df, self.test_observations_df, self.training_labels_sr, self.test_labels_sr = \
            split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices)

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr


class TitanicDataSet(DataSet):
    """The Titanic data set."""
    def __init__(self, data_path, shuffle_data=True, reset_indices=True, test_share=.3, label_column_idx=0):
        complete_df = pd.read_csv(data_path, index_col=False)
        complete_df = complete_df.drop('Name', axis=1)
        complete_df['Sex'] = complete_df['Sex'].replace('male', 0)
        complete_df['Sex'] = complete_df['Sex'].replace('female', 1)
        complete_df['Sex'] = complete_df['Sex'].astype(np.int_)
        complete_df['Pclass'] = complete_df['Pclass'].astype(np.int_)
        complete_df['Age'] = complete_df['Age'].astype(np.float_)
        complete_df['Siblings/Spouses Aboard'] = complete_df['Siblings/Spouses Aboard'].astype(np.float_)
        complete_df['Parents/Children Aboard'] = complete_df['Parents/Children Aboard'].astype(np.float_)
        labels_sr = pd.Series(complete_df[complete_df.columns[label_column_idx]])
        observations_df = complete_df.drop(complete_df.columns[label_column_idx], axis=1)
        self.training_observations_df, self.test_observations_df, self.training_labels_sr, self.test_labels_sr = \
            split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices)

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr


class MushroomDataSet(DataSet):
    """The mushrooms data set."""
    def __init__(self, data_path, shuffle_data=True, reset_indices=True, test_share=.3, label_column_idx=0):
        complete_df = pd.read_csv(data_path, index_col=False)
        complete_df = complete_df.applymap(lambda e: ord(e))
        complete_df['type'] = complete_df['type'].replace(101, 0)
        complete_df['type'] = complete_df['type'].replace(112, 1)
        labels_sr = pd.Series(complete_df[complete_df.columns[label_column_idx]])
        observations_df = complete_df.drop(complete_df.columns[label_column_idx], axis=1)
        self.training_observations_df, self.test_observations_df, self.training_labels_sr, self.test_labels_sr = \
            split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices)

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr
