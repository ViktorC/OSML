import math

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

    def fit_transform(self, observations_df, labels_sr):
        """
        It fits the model to the data and establishes the parameter estimates, if any, and transforms the observations
        using the parameter estimates.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
            labels_sr: A series of target values where each element is the label of the corresponding row in the data
            frame of observations. It can be None if the model is unsupervised.

        Returns:
            The transformed data frame of observations.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels.
        """
        self.fit(observations_df, labels_sr)
        return self.transform(observations_df)


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
            within_class_residuals = class_observations - class_mean
            within_class_covariance_matrix += within_class_residuals.T @ within_class_residuals
            between_class_residuals = class_mean - total_mean
            between_class_covariance_matrix += np.outer(between_class_residuals, between_class_residuals)
        # LDA assumes the covariance of features to be the same across the different classes.
        within_class_covariance_matrix /= (n_observations - n_classes)
        between_class_covariance_matrix /= n_classes
        lda_matrix = np.linalg.inv(within_class_covariance_matrix) @ between_class_covariance_matrix
        eigen_values, eigen_vectors = np.linalg.eig(lda_matrix)
        _, self._eigen_vector_matrix = _select_eigen_vectors(eigen_values, eigen_vectors, self.dimensions_to_retain,
                                                             self.relative_variance_to_retain)

    def _transform(self, observations_df):
        return observations_df.values @ self._eigen_vector_matrix


class BoxCoxTransformation(TransformativeModel):
    """
    A Box-Cox transformation to make individual features more normally distributed.
    """
    def __init__(self, iterations=20, init_bracket_width=1, min_bracket_width=1e-4, min_gradient=1e-4):
        super(BoxCoxTransformation, self).__init__()
        if iterations <= 0:
            raise ValueError
        if init_bracket_width <= 0:
            raise ValueError
        if min_bracket_width <= 0:
            raise ValueError
        if init_bracket_width <= min_bracket_width:
            raise ValueError
        if min_gradient < 0:
            raise ValueError
        self.iterations = iterations
        self.init_bracket_width = init_bracket_width
        self.min_bracket_width = min_bracket_width
        self.min_gradient = min_gradient
        self._lambda1 = None
        self._lambda2 = None

    @staticmethod
    def _box_cox_transform(observations, lambda1, lambda2):
        return np.where(np.isclose(np.zeros(observations.shape), np.zeros(observations.shape) + lambda1),
                        np.log(observations + lambda2),
                        (np.power(observations + lambda2, lambda1) - 1) / lambda1)

    @staticmethod
    def _box_cox_log_likelihood(observations, lambda1, lambda2):
        # Apply the Box-Cox transform.
        transformed_observations = BoxCoxTransformation._box_cox_transform(observations, lambda1, lambda2)
        # The log likelihood of the transformed observations.
        n_observations = transformed_observations.shape[0]
        transformed_log_likelihood = -n_observations / 2 * (np.log(transformed_observations.var(axis=0)) + 1)
        # But we need the log likelihood of the transformed observations in terms of the original observations to make
        # sure the likelihoods are comparable across different values of lambda.
        #   PDFy = fy(y) * dy; y = g(x)
        #   fy(y) = fx(g(x))
        #   dy = g(x + dx) - g(x)
        #   PDFy = fy(y) * dy = fx(g(x)) * (g(x + dx) - g(x)) = fx(g(x)) * dx * (g(x + dx) - g(x)) / dx
        #   g'(x) = lim d -> 0 : (g(x + dx) - g(x)) / dx
        #   PDFy = fy(y) * dy = fx(g(x)) * g'(x) * dx
        # Note that we are using the log likelihood here.
        log_box_cox_gradient = (lambda1 - 1) * np.log(observations + lambda2).sum(axis=0)
        return transformed_log_likelihood + log_box_cox_gradient

    def _box_cox_log_likelihood_gradient(self, observations, transformed_observations):
        mean = transformed_observations.mean(axis=0)
        residuals = transformed_observations - mean
        variance = np.square(residuals).mean(axis=0)
        # This is the derivative of the univariate Gaussian log likelihood function (the features are assumed to be
        # linearly independent) w.r.t. the Box-Cox-transformed observations.
        d_log_likelihood_wrt_transformed_observations = -residuals / variance
        # The derivative of the log likelihood w.r.t. lambda1.
        d_log_likelihood_wrt_lambda1 = np.log(observations + self._lambda2)
        # The derivative of the Box-Cox transformation w.r.t. lambda1.
        d_box_cox_transform_wrt_lambda1 = np.where(
            np.logical_not(np.isclose(np.zeros(observations.shape), np.zeros(observations.shape) + self._lambda1)),
            (1 + np.power(observations + self._lambda2, self._lambda1) *
             (np.log(observations + self._lambda2) * self._lambda1 - 1)) /
            (np.square(self._lambda1) + self.epsilon),
            # When lambda1 = 0, add a term to the transformation function so that it becomes
            # ln(x + lambda2) + b * lambda1, to avoid an all 0 gradient w.r.t. lambda1 (and hence avoid getting
            # stuck at lambda1 = 0).
            np.ones(observations.shape))
        # Apply the chain rule.
        return (d_log_likelihood_wrt_lambda1 +
                d_box_cox_transform_wrt_lambda1 * d_log_likelihood_wrt_transformed_observations).sum(axis=0)

    phi = (1 + math.sqrt(5)) / 2

    @staticmethod
    def _calculate_breaking_points(a, b):
        golden_section = (b - a) / BoxCoxTransformation.phi
        c = b - golden_section
        d = a + golden_section
        return c, d

    def _calculate_box_cox_log_likelihood(self, observations, ascent_direction, step_size, ind):
        observations_ind = observations[:, ind]
        lambda1_ind = self._lambda1[ind] + step_size * ascent_direction[ind]
        lambda2_ind = self._lambda2[ind]
        return self._box_cox_log_likelihood(observations_ind, lambda1_ind, lambda2_ind)

    def _golden_section_line_search(self, observations, ascent_direction):
        step_sizes = np.zeros(self._lambda1.shape)
        for i in range(len(step_sizes)):
            a = 0.
            b = self.init_bracket_width
            c, d = self._calculate_breaking_points(a, b)
            while abs(c - d) >= self.min_bracket_width:
                log_likelihood_c = self._calculate_box_cox_log_likelihood(observations, ascent_direction, c, i)
                log_likelihood_d = self._calculate_box_cox_log_likelihood(observations, ascent_direction, d, i)
                if log_likelihood_c > log_likelihood_d:
                    b = d
                else:
                    a = c
                c, d = self._calculate_breaking_points(a, b)
            step_sizes[i] = (a + b) / 2
        return step_sizes

    def _fit(self, observations_df, labels_sr):
        observations = observations_df.values
        self._lambda2 = np.maximum(np.zeros((observations.shape[1])), -observations.min(axis=0)) + self.epsilon
        # Initialize lambda1 to random values in the range of [-5, 5).
        self._lambda1 = np.random.random(self._lambda2.shape) * 10 - 5
        # Maximize the log of the Gaussian likelihood function using gradient ascent.
        for i in range(self.iterations):
            transformed_observations = self._box_cox_transform(observations, self._lambda1, self._lambda2)
            gradient = self._box_cox_log_likelihood_gradient(observations, transformed_observations)
            if np.all(np.absolute(gradient) <= self.min_gradient):
                break
            step_sizes = self._golden_section_line_search(observations, gradient)
            self._lambda1 += gradient * step_sizes

    def _transform(self, observations_df):
        return self._box_cox_transform(observations_df.values, self._lambda1, self._lambda2)
