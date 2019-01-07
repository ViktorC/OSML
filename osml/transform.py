import numpy as np
import pandas as pd
import osml.model as osm


class TransformativeModel(osm.Model):
    """
    An abstract class representing a data transformation model.
    """
    def __init__(self, epsilon=1e-8):
        super(TransformativeModel, self).__init__(epsilon)

    def _transform(self, observations_df: pd.DataFrame, labels_sr: pd.Series) -> np.array:
        pass

    def transform(self, observations_df, labels_sr):
        """
        Transforms the observation sample using the information learnt about the data set. This method requires the
        'fit' method to have been invoked.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
            labels_sr: A series of target values where each element is the label of the corresponding row in the data
            frame of observations.

        Returns:
            The transformed data frame of observations.
        """
        self._validate_observations_and_labels(observations_df, labels_sr)
        return pd.DataFrame(self._transform(observations_df, labels_sr))


class Centering(TransformativeModel):
    """
    A preprocessor that centers observations according to the mean vector of the data set it is fitted on.
    """
    def __init__(self):
        super(Centering, self).__init__()
        self._mean = None

    def _fit(self, observations_df, labels_sr):
        self._mean = observations_df.values.mean(axis=0)

    def _transform(self, observations_df, labels_sr):
        return observations_df.values - self._mean


class Standardization(TransformativeModel):
    """
    A preprocessor that centers and standardizes observations according to the mean vector and standard deviation of
    the data set it is fitted on.
    """
    def __init__(self):
        super(Standardization, self).__init__()
        self._mean = None
        self._sd = None

    def _fit(self, observations_df, labels_sr):
        observations = observations_df.values
        self._mean = observations.mean(axis=0)
        self._sd = observations.std(axis=0)

    def _transform(self, observations_df, labels_sr):
        return (observations_df.values - self._mean) / (self._sd + self.epsilon)


class PrincipalComponentAnalysis(TransformativeModel):
    """
    A principal component analysis preprocessor.
    """
    def __init__(self, standardize=False, relative_variance_to_retain=1., dimensions_to_retain=None):
        super(PrincipalComponentAnalysis, self).__init__()
        if not 0 < relative_variance_to_retain <= 1:
            raise ValueError
        if dimensions_to_retain is not None and dimensions_to_retain <= 0:
            raise ValueError
        self.standardize = standardize
        self.relative_variance_to_retain = relative_variance_to_retain
        self.dimensions_to_retain = dimensions_to_retain
        self._mean = None
        self._sd = None
        self._eigen_vector_matrix = None

    def _fit(self, observations_df, labels_sr):
        observations = observations_df.values
        self._mean = observations.mean(axis=0)
        if self.standardize:
            self._sd = observations.std(axis=0)
            observations = (observations - self._mean) / (self._sd + self.epsilon)
        else:
            observations -= self._mean
        covariance_matrix = np.dot(observations.transpose(), observations)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        # Sort the eigen vectors by their eigen values in descending order
        sorted_indices = np.flip(np.argsort(eigen_values))
        if self.dimensions_to_retain is not None:
            sorted_indices = sorted_indices[:min(self.dimensions_to_retain, len(sorted_indices))]
            self._eigen_vector_matrix = eigen_vectors[:, sorted_indices]
        else:
            total_eigen_values = eigen_values.sum()
            cutoff = 1
            retained_eigen_values = 0
            for eigen_value in eigen_values[sorted_indices]:
                retained_eigen_values += eigen_value
                if float(retained_eigen_values) / total_eigen_values >= self.relative_variance_to_retain:
                    break
                cutoff += 1
            self._eigen_vector_matrix = eigen_vectors[:, sorted_indices[:cutoff]]

    def _transform(self, observations_df, labels_sr):
        observations = observations_df.values - self._mean
        if self.standardize:
            observations /= (self._sd + self.epsilon)
        return np.dot(observations, self._eigen_vector_matrix)
