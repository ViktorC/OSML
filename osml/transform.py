import numpy as np
import pandas as pd
import osml.model as osm


class TransformativeModel(osm.Model):
    """
    An abstract class representing a data transformation model.
    """
    def __init__(self, epsilon=1e-8):
        super(TransformativeModel, self).__init__(epsilon)

    def _transform(self, observations_df: pd.DataFrame) -> np.array:
        pass

    def transform(self, observations_df):
        """
        Transforms the observation sample using the information learnt about the data set. This method requires the
        'fit' method to have been invoked.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.

        Returns:
            The transformed data frame of observations.
        """
        self._validate_observations(observations_df)
        return pd.DataFrame(self._transform(observations_df))


class Centering(TransformativeModel):
    """
    A preprocessor that centers observations according to the mean vector of the data set it is fitted on.
    """
    def __init__(self):
        super(Centering, self).__init__()
        self._mean = None

    def _fit(self, observations_df, labels_sr):
        self._mean = observations_df.values.mean(axis=0)

    def _transform(self, observations_df):
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

    def _transform(self, observations_df):
        return (observations_df.values - self._mean) / (self._sd + self.epsilon)


class BoxCoxTransformation(TransformativeModel):
    """
    A Box-Cox transformation to attempt to make data more normally distributed.
    """
    def __init__(self):
        super(BoxCoxTransformation, self).__init__()
        self._lambda1 = None
        self._lambda2 = None

    def _fit(self, observations_df, labels_sr):
        observations = observations_df.values
        self._lambda2 = -observations.min(axis=0) + self.epsilon
        self._lambda1 = np.zeros(self._lambda2.shape)

    def _transform(self, observations_df):
        observations = observations_df.values
        return np.where(self._lambda1 == 0,
                        np.log(observations + self._lambda2),
                        (np.power(observations + self._lambda2, self._lambda1) - 1) / self._lambda1)


def _select_k_top_eigen_vectors(eigen_values, eigen_vectors, k):
    sorted_indices = np.flip(np.argsort(eigen_values))
    sorted_indices = sorted_indices[:min(k, len(sorted_indices))]
    return eigen_values[sorted_indices], eigen_vectors[:, sorted_indices]


def _select_min_required_eigen_vectors(eigen_values, eigen_vectors, relative_variance_to_retain):
    sorted_indices = np.flip(np.argsort(eigen_values))
    total_eigen_values = eigen_values.sum()
    cutoff = 1
    retained_eigen_values = 0
    for eigen_value in eigen_values[sorted_indices]:
        retained_eigen_values += eigen_value
        if float(retained_eigen_values) / total_eigen_values >= relative_variance_to_retain:
            break
        cutoff += 1
    sorted_indices = sorted_indices[:cutoff]
    return eigen_values[sorted_indices], eigen_vectors[:, sorted_indices]


def _select_eigen_vectors(eigen_values, eigen_vectors, dimensions_to_retain, relative_variance_to_retain):
    if dimensions_to_retain is not None:
        return _select_k_top_eigen_vectors(eigen_values, eigen_vectors, dimensions_to_retain)
    return _select_min_required_eigen_vectors(eigen_values, eigen_vectors, relative_variance_to_retain)


class PrincipalComponentAnalysis(TransformativeModel):
    """
    A principal component analysis preprocessor.

    https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c
    """
    def __init__(self, standardize=False, whiten=False, relative_variance_to_retain=1., dimensions_to_retain=None):
        super(PrincipalComponentAnalysis, self).__init__()
        if not 0 < relative_variance_to_retain <= 1:
            raise ValueError
        if dimensions_to_retain is not None and dimensions_to_retain <= 0:
            raise ValueError
        self.standardize = standardize
        self.whiten = whiten
        self.relative_variance_to_retain = relative_variance_to_retain
        self.dimensions_to_retain = dimensions_to_retain
        self._mean = None
        self._sd = None
        self._eigen_vector_matrix = None
        self._eigen_values = None

    def _fit(self, observations_df, labels_sr):
        observations = observations_df.values
        self._mean = observations.mean(axis=0)
        prepped_observations = observations - self._mean
        if self.standardize:
            self._sd = observations.std(axis=0)
            prepped_observations /= (self._sd + self.epsilon)
        covariance_matrix = (prepped_observations.T @ prepped_observations) / (len(observations_df.index) - 1)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        self._eigen_values, self._eigen_vector_matrix = _select_eigen_vectors(eigen_values, eigen_vectors,
                                                                              self.dimensions_to_retain,
                                                                              self.relative_variance_to_retain)

    def _transform(self, observations_df):
        prepped_observations = observations_df.values - self._mean
        if self.standardize:
            prepped_observations /= (self._sd + self.epsilon)
        transformed_observations = prepped_observations @ self._eigen_vector_matrix
        if self.whiten:
            transformed_observations /= np.sqrt(self._eigen_values + self.epsilon)
        return transformed_observations


class LinearDiscriminantAnalysis(TransformativeModel):
    """
    A linear discriminant analysis preprocessor.

    https://sebastianraschka.com/Articles/2014_python_lda.html
    """
    def __init__(self, relative_variance_to_retain=1., dimensions_to_retain=None):
        super(LinearDiscriminantAnalysis, self).__init__()
        if not 0 < relative_variance_to_retain <= 1:
            raise ValueError
        if dimensions_to_retain is not None and dimensions_to_retain <= 0:
            raise ValueError
        self.relative_variance_to_retain = relative_variance_to_retain
        self.dimensions_to_retain = dimensions_to_retain
        self._eigen_vector_matrix = None

    def _fit(self, observations_df, labels_sr):
        n_features = len(observations_df.columns)
        n_observations = len(observations_df.index)
        total_mean = observations_df.values.mean(axis=0)
        within_class_covariance_matrix = np.zeros((n_features, n_features))
        between_class_covariance_matrix = np.zeros((n_features, n_features))
        unique_classes = labels_sr.unique()
        n_classes = unique_classes.shape[0]
        for klass in unique_classes:
            class_observations = observations_df.values[labels_sr[labels_sr == klass].index]
            class_mean = class_observations.mean(axis=0)
            within_class_deviance = class_observations - class_mean
            within_class_covariance_matrix += within_class_deviance.T @ within_class_deviance
            between_class_deviance = class_mean - total_mean
            between_class_covariance_matrix += np.outer(between_class_deviance, between_class_deviance)
        # LDA assumes the covariance of features to be the same across the different classes.
        within_class_covariance_matrix /= (n_observations - n_classes)
        between_class_covariance_matrix /= n_classes
        lda_matrix = np.linalg.inv(within_class_covariance_matrix) @ between_class_covariance_matrix
        eigen_values, eigen_vectors = np.linalg.eig(lda_matrix)
        _, self._eigen_vector_matrix = _select_eigen_vectors(eigen_values, eigen_vectors, self.dimensions_to_retain,
                                                             self.relative_variance_to_retain)

    def _transform(self, observations_df):
        return observations_df.values @ self._eigen_vector_matrix
