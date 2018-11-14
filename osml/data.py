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

    @staticmethod
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
        observations_df[:] = observations_df.values[indices]
        labels_sr[:] = labels_sr.values[indices]


class BostonHousingDataSet(DataSet):
    """The Boston house pricing data set."""
    def __init__(self, data_path, shuffle=True, test_share=.3, label_column_idx=13):
        if test_share <= 0 or test_share >= 1:
            raise ValueError
        complete_df = pd.read_csv(data_path, index_col=False)
        labels_sr = pd.Series(complete_df[complete_df.columns[label_column_idx]])
        observations_df = complete_df.drop(complete_df.columns[label_column_idx], axis=1)
        if shuffle:
            DataSet.shuffle(observations_df, labels_sr)
        total_instances = len(observations_df.index)
        test_instances = int(total_instances * test_share)
        self.training_observations_df = observations_df.iloc[test_instances:, :]
        self.test_observations_df = observations_df.iloc[:test_instances, :]
        self.training_labels_sr = labels_sr.iloc[test_instances:]
        self.test_labels_sr = labels_sr.iloc[:test_instances]

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
    def __init__(self, obs_path, label_path, shuffle=True,
                 test_share=.3):
        if test_share <= 0 or test_share >= 1:
            raise ValueError
        observations_df = pd.DataFrame(np.loadtxt(obs_path))
        labels_sr = pd.Series(np.loadtxt(label_path))
        if shuffle:
            DataSet.shuffle(observations_df, labels_sr)
        total_instances = len(observations_df.index)
        test_instances = int(total_instances * test_share)
        self.training_observations_df = observations_df.iloc[test_instances:, :]
        self.test_observations_df = observations_df.iloc[:test_instances, :]
        self.training_labels_sr = labels_sr.iloc[test_instances:]
        self.test_labels_sr = labels_sr.iloc[:test_instances]

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr