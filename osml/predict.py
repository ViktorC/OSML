import copy
import math
import multiprocessing as mp
import operator
import random
import typing

import numpy as np
import pandas as pd

import osml.model as osm


class PredictiveModel(osm.Model):
    """
    An abstract class representing a statistical model or machine learning algorithm.
    """
    def __init__(self, epsilon=1e-8):
        super(PredictiveModel, self).__init__(epsilon)

    def _predict(self, observations_df: pd.DataFrame) -> np.array:
        pass

    def _evaluate(self, predictions_sr: pd.Series, labels_sr: pd.Series) -> float:
        pass

    def _test(self, observations_df: pd.DataFrame, labels_sr: pd.Series) -> float:
        pass

    def predict(self, observations_df):
        """
        Predicts the labels of the observations.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.

        Returns:
            A series of values where each element is the predicted label of the corresponding observation.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels. Or if the types of
            the features are not the same as those of the data the model was fit to.
        """
        self._validate_observations(observations_df)
        predictions_sr = pd.Series(self._predict(observations_df), name=self._label_name, dtype=self._label_type)
        self._validate_labels(predictions_sr)
        return predictions_sr

    def evaluate(self, predictions_sr, labels_sr):
        """
        A function for evaluating the model's predictions according to some arbitrary metric.

        Args:
            predictions_sr: A series of predictions.
            labels_sr: A series of target values where each element is the label of the corresponding element in the
            series of predictions.

        Returns:
            A non-standardized numeric measure of the explanatory power of the predictions. It is not comparable across
            different model types (i.e. regression, classification, and binary classification).

        Raises:
            ValueError: If the predictions or labels are empty or do not match in length or do not have the expected
            data type.
        """
        if predictions_sr.count() != labels_sr.count():
            raise ValueError
        self._validate_labels(predictions_sr)
        self._validate_labels(labels_sr)
        return self._evaluate(predictions_sr, labels_sr)

    def test(self, observations_df, labels_sr):
        """
        Computes the prediction error of the model on a test data set in terms of the function the model minimizes.
        It is not applicable to models that do not optimize a well-defined loss function.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
            labels_sr: A series of target values where each element is the label of the corresponding row in the data
            frame of observations.

        Returns:
            A non-standardized numeric measure of prediction error. It is not comparable across different model types.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels. Or if the types of
            the features or the labels are not the same as those of the data the model was fit to.
        """
        self._validate_observations_and_labels(observations_df, labels_sr)
        return self._test(observations_df, labels_sr)


class RegressionModel(PredictiveModel):
    """
    An abstract base class for regression models.
    """
    def __init__(self):
        super(RegressionModel, self).__init__()

    def _validate_labels(self, labels_sr):
        super(RegressionModel, self)._validate_labels(labels_sr)
        if not osm.is_continuous(labels_sr.dtype):
            raise ValueError

    def _evaluate(self, predictions_sr, labels_sr):
        # Root mean squared error.
        return np.sqrt(np.square(predictions_sr.values - labels_sr.values).mean())


class ClassificationModel(PredictiveModel):
    """
    An abstract base class for classification models.
    """
    def __init__(self):
        super(ClassificationModel, self).__init__()

    def _validate_labels(self, labels_sr):
        super(ClassificationModel, self)._validate_labels(labels_sr)
        if not osm.is_categorical(labels_sr.dtype):
            raise ValueError

    def _evaluate(self, predictions_sr, labels_sr):
        # Accuracy.
        return len([p for i, p in predictions_sr.iteritems() if p == labels_sr[i]]) / predictions_sr.count()


class BinaryClassificationModel(PredictiveModel):
    """
    An abstract base class for binary classification models.
    """
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()

    def _fuzzy_predict(self, observations_df: pd.DataFrame) -> np.array:
        pass

    def _validate_labels(self, labels_sr):
        super(BinaryClassificationModel, self)._validate_labels(labels_sr)
        if not osm.is_categorical(labels_sr.dtype) or any(v != 0 and v != 1 for v in labels_sr.unique()):
            raise ValueError

    def _predict(self, observations_df):
        return np.rint(self._fuzzy_predict(observations_df))

    def _evaluate(self, predictions_sr, labels_sr):
        # F1 score.
        precision = self.evaluate_precision(predictions_sr, labels_sr)
        recall = self.evaluate_recall(predictions_sr, labels_sr)
        return 2 * precision * recall / (precision + recall)

    def fuzzy_predict(self, observations_df):
        """
        Makes fuzzy predictions about the observations belonging to the positive class.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.

        Returns:
            A series of values where each row is the predicted fuzzy positive class membership of the corresponding
            observation.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels. Or if the types of
            the features are not the same as those of the data the model was fit to.
        """
        self._validate_observations(observations_df)
        return pd.Series(self._fuzzy_predict(observations_df), dtype=np.float_)

    def evaluate_precision(self, predictions_sr, labels_sr):
        """
        Evaluates the precision of the predictions (true positive predictions / all positive predictions).

        Args:
            predictions_sr: A series of predictions.
            labels_sr: A series of target values where each element is the label of the corresponding element in the
            series of predictions.

        Returns:
            A scalar measure of the predictions' precision.

        Raises:
            ValueError: If the predictions or labels are empty or do not match in length or do not have the expected
            data type.
        """
        if predictions_sr.count() != labels_sr.count():
            raise ValueError
        self._validate_labels(predictions_sr)
        self._validate_labels(labels_sr)
        n_pos_predictions = predictions_sr.sum()
        n_true_pos_predictions = labels_sr[predictions_sr == 1].sum()
        return float(n_true_pos_predictions) / n_pos_predictions

    def evaluate_recall(self, predictions_sr, labels_sr):
        """
        Evaluates the recall of the predictions (true positive predictions / all positives).

        Args:
            predictions_sr: A series of predictions.
            labels_sr: A series of target values where each element is the label of the corresponding element in the
            series of predictions.

        Returns:
            A scalar measure of the predictions' recall.

        Raises:
            ValueError: If the predictions or labels are empty or do not match in length or do not have the expected
            data type.
        """
        if predictions_sr.count() != labels_sr.count():
            raise ValueError
        self._validate_labels(predictions_sr)
        self._validate_labels(labels_sr)
        n_pos_labels = labels_sr.sum()
        n_true_pos_predictions = predictions_sr[labels_sr == 1].sum()
        return float(n_true_pos_predictions) / n_pos_labels

    def evaluate_fall_out(self, predictions_sr, labels_sr):
        """
        Evaluates the fall out of the predictions (false positive predictions / all negatives).

        Args:
            predictions_sr: A series of predictions.
            labels_sr: A series of target values where each element is the label of the corresponding element in the
            series of predictions.

        Returns:
            A scalar measure of the predictions' fall out.

        Raises:
            ValueError: If the predictions or labels are empty or do not match in length or do not have the expected
            data type.
        """
        if predictions_sr.count() != labels_sr.count():
            raise ValueError
        self._validate_labels(predictions_sr)
        self._validate_labels(labels_sr)
        n_neg_labels = len(labels_sr[labels_sr == 0])
        n_false_pos_predictions = predictions_sr[labels_sr != 1].sum()
        return float(n_false_pos_predictions) / n_neg_labels

    def evaluate_receiver_operating_characteristic(self, fuzzy_predictions_sr, labels_sr, threshold_increment=.01):
        """
        Computes the receiver operating characteristic curve using a preset threshold increment.

        Args:
            fuzzy_predictions_sr: A series of fuzzy class membership predictions.
            labels_sr: A series of target values where each element is the label of the corresponding element in the
            series of predictions.
            threshold_increment: The value by which the predictions threshold should be incremented to calculate the
            ROC curve.

        Returns:
            A list of tuples containing the threshold, recall, and fall-out values; and a scalar representing the area
            under the curve.

        Raises:
            ValueError: If the threshold increment is not between 0 and 1.
        """
        if not 0 < threshold_increment < 1:
            raise ValueError
        roc = []
        auc = .0
        last_recall = None
        last_fall_out = None
        for threshold in np.arange(0., 1. + threshold_increment, threshold_increment):
            predictions_sr = pd.Series(np.where(fuzzy_predictions_sr.values >= threshold, 1, 0),
                                       dtype=labels_sr.dtype)
            recall = self.evaluate_recall(predictions_sr, labels_sr)
            fall_out = self.evaluate_fall_out(predictions_sr, labels_sr)
            roc.append((threshold, recall, fall_out))
            if last_recall and last_fall_out:
                auc += (recall + last_recall) / 2 * (last_fall_out - fall_out)
            last_recall = recall
            last_fall_out = fall_out
        return roc, auc


def _bias_trick(observations_df):
    observations = observations_df.values
    obs_ext = np.ones((observations.shape[0], observations.shape[1] + 1))
    obs_ext[:, :-1] = observations
    return obs_ext


class LinearRegression(RegressionModel):
    """
    A linear regression model fitted using ordinary least squares.
    """
    def __init__(self):
        super(LinearRegression, self).__init__()
        self._beta = None

    def _fit(self, observations_df, labels_sr):
        # Take the partial derivatives of the sum of squared errors w.r.t. both parameters of the model (weights and
        # bias), set both derivatives to 0, and solve the resulting system of linear equations. Whether the found roots
        # really minimize the sum of squared errors is not verified here.
        # NOTE: Using the bias trick reduces the parameters to a single vector and significantly simplifies the
        # analytic solution.
        observations = _bias_trick(observations_df)
        observations_trans = observations.T
        self._beta = np.linalg.inv(observations_trans @ observations) @ (observations_trans @ labels_sr.values)

    def _predict(self, observations_df):
        return _bias_trick(observations_df) @ self._beta

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean()


class LinearRidgeRegression(LinearRegression):
    """
    A linear ridge regression model fitted using ordinary least squares.
    """
    def __init__(self, alpha=1e-2):
        super(LinearRidgeRegression, self).__init__()
        if alpha is None or alpha < 0:
            raise ValueError
        self.alpha = alpha

    def _fit(self, observations_df, labels_sr):
        observations = _bias_trick(observations_df)
        observations_trans = observations.T
        self._beta = np.linalg.inv((observations_trans @ observations) + self.alpha) @ \
            (observations_trans @ labels_sr.values)

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean() + \
               self.alpha * np.square(self._beta).sum()


class LinearLassoRegression(LinearRegression):
    """
    A Least Absolute Shrinkage and Selection Operator regression model fitted using coordinate descent.
    """
    def __init__(self, alpha=1e-2, iterations=100):
        super(LinearLassoRegression, self).__init__()
        if alpha is None or alpha < 0:
            raise ValueError
        if iterations is None or iterations <= 0:
            raise ValueError
        self.alpha = alpha
        self.iterations = iterations

    def _soft_threshold(self, p):
        if p < -self.alpha:
            return p + self.alpha
        if -self.alpha <= p <= self.alpha:
            return 0
        if p > self.alpha:
            return p - self.alpha

    def _exact_line_search(self, observations, labels, i):
        obs_i = observations[:, i]
        obs_not_i = np.delete(observations, i, axis=1)
        beta_not_i = np.delete(self._beta, i)
        obs_i_trans = obs_i.T
        p = obs_i_trans @ labels - obs_i_trans @ (obs_not_i @ beta_not_i)
        z = obs_i_trans @ obs_i
        # As the absolute value function is not differentiable at 0, use soft thresholding.
        return self._soft_threshold(p) / z

    def _fit(self, observations_df, labels_sr):
        observations = _bias_trick(observations_df)
        labels = labels_sr.values
        self._beta = np.zeros((observations.shape[1], ))
        prev_loss = float('inf')
        for i in range(self.iterations):
            # Optimize the parameters analytically dimension by dimension (exact line search).
            for j in range(observations.shape[1]):
                # Solve for the (sub)derivative of the loss function with respect to the jth parameter.
                self._beta[j] = self._exact_line_search(observations, labels, j)
            # If a full cycle of optimization does not reduce the loss, the model has converged.
            loss = self.test(observations_df, labels_sr)
            if loss < prev_loss:
                prev_loss = loss
            else:
                break

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean() + \
               self.alpha * np.abs(self._beta).sum()


def _logistic_function(x):
    return np.reciprocal(np.exp(-x) + 1)


class LogisticRegression(BinaryClassificationModel):
    """
    A logistic regression model fitted using the Newton-Raphson method.
    """
    def __init__(self, iterations=100, min_gradient=1e-7):
        super(LogisticRegression, self).__init__()
        if iterations is None or iterations < 1:
            raise ValueError
        if min_gradient is None or min_gradient < 0:
            raise ValueError
        self.iterations = iterations
        self.min_gradient = min_gradient
        self._beta = None

    def _gradient(self, observations_trans, predictions, labels):
        return observations_trans @ (predictions - labels)

    def _hessian(self, observations, observations_trans, predictions):
        return (observations_trans * (predictions * (1 - predictions))) @ observations

    def _fit(self, observations_df, labels_sr):
        # Use Newton's method to numerically approximate the root of the log likelihood function's derivative.
        observations = _bias_trick(observations_df)
        observations_trans = observations.T
        labels = labels_sr.values
        self._beta = np.zeros(observations.shape[1], )
        for i in range(self.iterations):
            predictions = _logistic_function(observations @ self._beta)
            first_derivative = self._gradient(observations_trans, predictions, labels)
            if np.all(np.absolute(first_derivative) <= self.min_gradient):
                break
            # If the number of parameters is N, the Hessian will take the shape of an N by N matrix.
            second_derivative = self._hessian(observations, observations_trans, predictions)
            self._beta = self._beta - np.linalg.inv(second_derivative) @ first_derivative

    def _fuzzy_predict(self, observations_df):
        return _logistic_function(_bias_trick(observations_df) @ self._beta)

    def _test(self, observations_df, labels_sr):
        labels = labels_sr.values
        observations = _bias_trick(observations_df)
        predictions = np.minimum(_logistic_function(observations @ self._beta) + self.epsilon, 1 - self.epsilon)
        return -((np.log(predictions) * labels + np.log(1 - predictions) * (1 - labels)).mean())


class LogisticRidgeRegression(LogisticRegression):
    """
    A logistic regression model with L2 regularization.
    """
    def __init__(self, iterations=100, min_gradient=1e-7, alpha=1e-2):
        super(LogisticRidgeRegression, self).__init__(iterations, min_gradient)
        if alpha is None or alpha < 0:
            raise ValueError
        self.alpha = alpha

    def _gradient(self, observations_trans, predictions, labels):
        return observations_trans @ (predictions - labels) + self.alpha / 2 * self._beta

    def _hessian(self, observations, observations_trans, predictions):
        return ((observations_trans * (predictions * (1 - predictions))) @ observations) + self.alpha / 2

    def _test(self, observations_df, labels_sr):
        return super(LogisticRidgeRegression, self)._test(observations_df, labels_sr) + \
               self.alpha * np.square(self._beta).sum()


class NaiveBayes(ClassificationModel):
    """
    A naive Bayes classification model.
    """
    def __init__(self, laplace_smoothing=1.):
        super(NaiveBayes, self).__init__()
        if laplace_smoothing is None or laplace_smoothing < 0:
            raise ValueError
        self.laplace_smoothing = laplace_smoothing
        self._classes = None
        self._class_priors = None
        self._feature_statistics = None

    def _fit(self, observations_df, labels_sr):
        self._classes = labels_sr.unique()
        self._class_priors = labels_sr.value_counts().astype(np.float_) / labels_sr.count()
        indices_by_label = {}
        for label_value in self._classes:
            indices_by_label[label_value] = [i for i, v in enumerate(labels_sr) if v == label_value]
        self._feature_statistics = []
        for column in observations_df.columns:
            feature_values_sr = observations_df[column]
            if osm.is_categorical(feature_values_sr.dtype):
                self._feature_statistics.append(self.CategoricalFeatureStatistics(feature_values_sr, indices_by_label,
                                                                                  self.laplace_smoothing))
            elif osm.is_continuous(feature_values_sr.dtype):
                self._feature_statistics.append(self.ContinuousFeatureStatistics(feature_values_sr, indices_by_label))
            else:
                raise ValueError

    def _predict(self, observations_df):
        predictions = np.zeros((len(observations_df.index)))
        for i, row in enumerate(observations_df.itertuples()):
            highest_class_probability = -1
            prediction = None
            probability_of_evidence = 1
            for j, feature_stats in enumerate(self._feature_statistics):
                probability_of_evidence *= feature_stats.get_probability(row[j + 1])
            for label_class in self._classes:
                class_prior = self._class_priors[label_class]
                likelihood = 1
                for j, feature_stats in enumerate(self._feature_statistics):
                    likelihood *= feature_stats.get_likelihood(row[j + 1], label_class)
                class_probability = likelihood * class_prior / probability_of_evidence
                if class_probability > highest_class_probability:
                    highest_class_probability = class_probability
                    prediction = label_class
            predictions[i] = prediction
        return predictions

    def _test(self, observations_df, labels_sr):
        return None

    class FeatureStatistics:
        """
        An abstract base class for feature statistics.
        """
        def get_probability(self, value):
            """
            A function that returns the probability of the feature taking on the specified value.

            Args:
                value: The value whose probability is to be established.

            Returns:
                The probability of the specified value for the feature.
            """
            pass

        def get_likelihood(self, value, label_value):
            """
            A function that returns the probability of the feature having the specified value given the specified
            label value.

            Args:
                value: The feature value.
                label_value: The label value.

            Returns:
                The conditional probability of the feature value given the label value.
            """
            pass

    class ContinuousFeatureStatistics:
        """
        A class for continuous feature statistics.
        """
        def __init__(self, values_sr, indices_by_label):
            self.mean = values_sr.mean()
            self.variance = values_sr.var()
            self.mean_and_variance_by_class = {}
            for label_value, indices in indices_by_label.items():
                relevant_values_sr = values_sr.reindex(indices)
                self.mean_and_variance_by_class[label_value] = (relevant_values_sr.mean(), relevant_values_sr.var())

        @staticmethod
        def calculate_probability(value, mean, var):
            # Gaussian probability distribution function.
            return math.exp(-((value - mean) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)

        def get_probability(self, value):
            return self.calculate_probability(value, self.mean, self.variance)

        def get_likelihood(self, value, label_value):
            mean_and_variance_tuple = self.mean_and_variance_by_class[label_value]
            return self.calculate_probability(value, mean_and_variance_tuple[0], mean_and_variance_tuple[1])

    class CategoricalFeatureStatistics:
        """
        A class for categorical feature statistics.
        """
        def __init__(self, values_sr, indices_by_label, laplace_smoothing):
            self.laplace_smoothing = laplace_smoothing
            self.value_counts = values_sr.value_counts()
            self.total_value_count = values_sr.count()
            self.likelihoods = {}
            for label_value, indices in indices_by_label.items():
                self.likelihoods[label_value] = (values_sr.reindex(indices).value_counts(), len(indices))

        @staticmethod
        def calculate_probability(value, value_counts, total_value_count, laplace_smoothing):
            # Laplace smoothed probability mass function.
            value_count = laplace_smoothing
            if value in value_counts.index:
                value_count += value_counts[value]
            return float(value_count) / (total_value_count + value_counts.count() * laplace_smoothing)

        def get_probability(self, value):
            return self.calculate_probability(value, self.value_counts, self.total_value_count, self.laplace_smoothing)

        def get_likelihood(self, value, label_value):
            likelihood_tuple = self.likelihoods[label_value]
            return self.calculate_probability(value, likelihood_tuple[0], likelihood_tuple[1], self.laplace_smoothing)


class DiscriminantAnalysis(ClassificationModel):
    """
    A discriminant analysis classification model base class.

    https://web.stanford.edu/class/stats202/content/lec9.pdf
    """
    def __init__(self):
        super(DiscriminantAnalysis, self).__init__()
        self._classes = None
        self._class_priors = None
        self._mean_by_class = None

    def _init_covariance_estimates(self, n_features: int) -> None:
        pass

    def _update_covariance_estimates(self, covariance_matrix: np.array, klass: str, n_observations: int):
        pass

    def _finalize_covariance_estimates(self, total_n_observations: int) -> None:
        pass

    def _calculate_relative_probability(self, observation_sr: pd.Series, klass: str) -> float:
        pass

    def _fit(self, observations_df, labels_sr):
        total_n_observations = len(observations_df.index)
        n_features = len(observations_df.columns)
        self._classes = labels_sr.unique()
        self._class_priors = labels_sr.value_counts().astype(np.float_) / labels_sr.count()
        self._mean_by_class = {}
        self._init_covariance_estimates(n_features)
        for klass in self._classes:
            respective_observations_df = observations_df.reindex(labels_sr[labels_sr == klass].index)
            n_observations = len(respective_observations_df.index)
            class_mean = respective_observations_df.mean(axis=0).values
            self._mean_by_class[klass] = class_mean
            class_residuals = respective_observations_df.values - class_mean
            class_covariance_matrix = class_residuals.T @ class_residuals
            self._update_covariance_estimates(class_covariance_matrix, klass, n_observations)
        self._finalize_covariance_estimates(total_n_observations)

    def _predict(self, observations_df):
        predictions = np.zeros((len(observations_df.index)))
        for i, row in observations_df.iterrows():
            highest_relative_probability = None
            prediction = None
            for klass in self._classes:
                relative_probability = self._calculate_relative_probability(row, klass)
                if highest_relative_probability is None or relative_probability > highest_relative_probability:
                    highest_relative_probability = relative_probability
                    prediction = klass
            predictions[i] = prediction
        return predictions

    def _test(self, observations_df, labels_sr):
        return None


class LinearDiscriminantAnalysis(DiscriminantAnalysis):
    """
    A linear discriminant analysis classification model.
    """
    def __init__(self):
        super(LinearDiscriminantAnalysis, self).__init__()
        self._inverse_common_covariance_matrix = None

    def _init_covariance_estimates(self, n_features):
        self._inverse_common_covariance_matrix = np.zeros((n_features, n_features))

    def _update_covariance_estimates(self, covariance_matrix, klass, n_observations):
        self._inverse_common_covariance_matrix += covariance_matrix

    def _finalize_covariance_estimates(self, total_n_observations):
        self._inverse_common_covariance_matrix = np.linalg.inv(self._inverse_common_covariance_matrix /
                                                               (total_n_observations - len(self._classes)))

    def _calculate_relative_probability(self, observation_sr, klass):
        mean = self._mean_by_class[klass]
        return np.log(self._class_priors[klass]) - \
            .5 * (mean @ self._inverse_common_covariance_matrix) @ mean.T + \
            (observation_sr.values @ self._inverse_common_covariance_matrix) @ mean.T


class QuadraticDiscriminantAnalysis(DiscriminantAnalysis):
    """
    A quadratic discriminant analysis classification model.
    """
    def __init__(self):
        super(QuadraticDiscriminantAnalysis, self).__init__()
        self._inverse_covariance_matrix_by_class = None

    def _init_covariance_estimates(self, n_features):
        self._inverse_covariance_matrix_by_class = {}

    def _update_covariance_estimates(self, covariance_matrix, klass, n_observations):
        self._inverse_covariance_matrix_by_class[klass] = np.linalg.inv(covariance_matrix / (n_observations - 1))

    def _calculate_relative_probability(self, observation_sr, klass):
        observation = observation_sr.values
        mean = self._mean_by_class[klass]
        inverse_covariance_matrix = self._inverse_covariance_matrix_by_class[klass]
        return np.log(self._class_priors[klass]) - \
            .5 * (mean @ inverse_covariance_matrix) @ mean.T + \
            (observation @ inverse_covariance_matrix) @ mean.T - \
            .5 * ((observation @ inverse_covariance_matrix) @ observation.T -
                  np.log(np.linalg.det(np.linalg.inv(inverse_covariance_matrix))))


class KNearestNeighbors(PredictiveModel):
    """
    An abstract weighted k-nearest neighbors model.

    https://epub.ub.uni-muenchen.de/1769/1/paper_399.pdf
    """
    def __init__(self, k, rbf_gamma):
        super(KNearestNeighbors, self).__init__()
        if k is None or k < 1:
            raise ValueError
        if rbf_gamma <= 0:
            raise ValueError
        self.k = k
        self.rbf_gamma = rbf_gamma
        self._observations_mean = None
        self._observations_sd = None
        self._observations = None
        self._labels = None

    def _distances(self, observation):
        # Simple Euclidian distance.
        return np.sqrt(np.square(self._observations - observation).sum(axis=1))

    def _weight(self, distance):
        # Radial basis function kernel.
        return math.exp(-self.rbf_gamma * (distance ** 2))

    def _select_weighted_nearest_neighbors(self, observation):
        distances = self._distances(observation)
        k_plus_1_nn_indices = np.argpartition(distances, self.k + 1)[:self.k + 1]
        k_plus_1_nn = pd.Series(distances[k_plus_1_nn_indices], index=k_plus_1_nn_indices)
        k_plus_1_nn = k_plus_1_nn.sort_values(ascending=False)
        weighted_nearest_neighbor_labels = {}
        distance_to_normalize_by = None
        for i, value in k_plus_1_nn.iteritems():
            if distance_to_normalize_by is None:
                distance_to_normalize_by = value
                continue
            weight = self._weight(value / (distance_to_normalize_by + self.epsilon))
            label = self._labels[i]
            if label in weighted_nearest_neighbor_labels:
                weighted_nearest_neighbor_labels[label] += weight
            else:
                weighted_nearest_neighbor_labels[label] = weight
        return weighted_nearest_neighbor_labels

    def _fit(self, observations_df, labels_sr):
        if len(observations_df.index) < self.k + 1:
            raise ValueError
        self._observations = observations_df.values
        self._labels = labels_sr.values

    def _test(self, observations_df, labels_sr):
        return None


class KNearestNeighborsRegression(KNearestNeighbors, RegressionModel):
    """
    A weighted k-nearest neighbors regression model.
    """
    def __init__(self, k=7, rbf_gamma=1):
        KNearestNeighbors.__init__(self, k, rbf_gamma)
        RegressionModel.__init__(self)

    def _predict(self, observations_df):
        predictions = np.zeros((len(observations_df.index)), )
        for i, row in observations_df.iterrows():
            weighted_labels_sum = 0
            weight_sum = 0
            for key, value in self._select_weighted_nearest_neighbors(row.values).items():
                weighted_labels_sum += key * value
                weight_sum += value
            predictions[i] = weighted_labels_sum / weight_sum
        return predictions


class KNearestNeighborsClassification(KNearestNeighbors, ClassificationModel):
    """
    A weighted k-nearest neighbors classification model.
    """
    def __init__(self, k=7, rbf_gamma=1):
        KNearestNeighbors.__init__(self, k, rbf_gamma)
        ClassificationModel.__init__(self)

    def _predict(self, observations_df):
        predictions = np.zeros((len(observations_df.index)), )
        for i, row in observations_df.iterrows():
            nearest_neighbors = self._select_weighted_nearest_neighbors(row.values)
            sorted_nearest_neighbors = sorted(nearest_neighbors.items(), key=operator.itemgetter(1), reverse=True)
            predictions[i] = sorted_nearest_neighbors[0][0]
        return predictions


def _identity_function(x):
    return x


class DecisionTree(PredictiveModel):
    """
    An abstract decision tree base class.
    """
    def __init__(self, max_depth, min_observations, min_impurity, min_impurity_reduction, random_feature_selection,
                 feature_sample_size_function):
        super(DecisionTree, self).__init__()
        if random_feature_selection is None or not isinstance(random_feature_selection, bool):
            raise ValueError
        self.max_depth = max_depth
        self.min_observations = max(1, min_observations) if min_observations is not None else 1
        self.min_impurity = max(0, min_impurity) if min_impurity is not None else 0
        self.min_impurity_reduction = min_impurity_reduction if min_impurity_reduction is not None else float('-inf')
        self.random_feature_selection = random_feature_selection
        self.feature_sample_size_function = feature_sample_size_function if feature_sample_size_function is not None \
            else _identity_function
        self._root = None

    def _setup(self, labels_sr: pd.Series) -> None:
        pass

    def _compute_impurity(self, labels_sr: pd.Series) -> float:
        pass

    def _compute_leaf_node_value(self, labels_sr: pd.Series) -> float:
        pass

    def _compute_aggregate_impurity(self, list_of_labels_sr):
        total_labels = 0
        aggregate_impurity = 0
        for labels_sr in list_of_labels_sr:
            count = labels_sr.count()
            total_labels += count
            if count > 1:
                impurity = self._compute_impurity(labels_sr)
                aggregate_impurity += count * impurity
        return aggregate_impurity / total_labels

    def _test_categorical_split(self, column_sr, labels_sr):
        impurity = None
        indices_by_feature_value = None
        unique_values = column_sr.unique()
        if len(unique_values) > 1:
            list_of_labels_sr = []
            indices_by_feature_value = {}
            for value in unique_values:
                indices = column_sr[column_sr == value].index
                indices_by_feature_value[value] = indices
                list_of_labels_sr.append(labels_sr.reindex(indices))
            impurity = self._compute_aggregate_impurity(list_of_labels_sr)
        return impurity, indices_by_feature_value

    def _test_continuous_split(self, column_sr, labels_sr):
        impurity = None
        indices_by_feature_value = None
        sorted_column_sr = column_sr.sort_values()
        sorted_labels_sr = labels_sr.reindex(sorted_column_sr.index)
        for ind, value in enumerate(sorted_column_sr):
            if ind < len(sorted_column_sr) - 1:
                label = sorted_labels_sr.iat[ind]
                next_ind = ind + 1
                next_value = sorted_column_sr.iat[next_ind]
                next_label = sorted_labels_sr.iat[next_ind]
                if value == next_value or label == next_label:
                    continue
                list_of_labels_sr = [sorted_labels_sr.iloc[:next_ind], sorted_labels_sr.iloc[next_ind:]]
                split_point_impurity = self._compute_aggregate_impurity(list_of_labels_sr)
                if impurity is None or split_point_impurity < impurity:
                    impurity = split_point_impurity
                    split_point = (value + next_value) / 2
                    indices_by_feature_value = {split_point: sorted_column_sr.index[next_ind:],
                                                None: sorted_column_sr.index[:next_ind]}
        return impurity, indices_by_feature_value

    def _select_split(self, observations_df, labels_sr, features):
        best_feature = None
        best_feature_categorical = None
        best_indices_by_feature_value = None
        best_impurity = None
        if self.random_feature_selection:
            n_features = len(features)
            features_to_consider = random.sample(range(n_features), int(self.feature_sample_size_function(n_features)))
        else:
            features_to_consider = None
        for i, feature in enumerate(features):
            if self.random_feature_selection and i not in features_to_consider:
                continue
            feature_categorical = False
            column_sr = observations_df[feature]
            if osm.is_categorical(column_sr.dtype):
                feature_categorical = True
                impurity, indices_by_feature_value = self._test_categorical_split(column_sr, labels_sr)
            elif osm.is_continuous(column_sr.dtype):
                impurity, indices_by_feature_value = self._test_continuous_split(column_sr, labels_sr)
            else:
                raise ValueError
            # If all the observations have the same value for the selected feature, we cannot split by it.
            if impurity is None:
                continue
            if best_impurity is None or impurity < best_impurity:
                best_feature = feature
                best_feature_categorical = feature_categorical
                best_indices_by_feature_value = indices_by_feature_value
                best_impurity = impurity
        return best_feature, best_feature_categorical, best_indices_by_feature_value, best_impurity

    def _build_tree(self, observations_df, labels_sr, features, depth):
        # Instantiate the node and calculate its value.
        node = self.DecisionTreeNode()
        node.value = self._compute_leaf_node_value(labels_sr)
        node.instances = labels_sr.count()
        # Check termination conditions.
        if len(features) == 0 or node.instances <= self.min_observations:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        node.impurity = self._compute_impurity(labels_sr)
        if node.impurity <= self.min_impurity:
            return node
        # Select the best feature to split by.
        best_feature, best_feature_categorical, best_indices_by_feature_value, best_impurity = \
            self._select_split(observations_df, labels_sr, features)
        # Check split metric reduction termination condition (and whether any split points could be established at all).
        if best_impurity is None or node.impurity - best_impurity <= self.min_impurity_reduction:
            return node
        # If the best feature is categorical, remove it from the list of available features
        if best_feature_categorical:
            features.remove(best_feature)
        # Split the observations and the labels and recurse to build the child nodes and their sub-trees.
        node.feature = best_feature
        node.children_by_feature_value = {}
        for key, value in best_indices_by_feature_value.items():
            instances = len(value)
            if instances == 0:
                continue
            child = self._build_tree(observations_df.reindex(value), labels_sr.reindex(value), features, depth + 1)
            node.children_by_feature_value[key] = child
        if best_feature_categorical:
            features.append(best_feature)
        return node

    def _fit(self, observations_df, labels_sr):
        self._setup(labels_sr)
        self._root = self._build_tree(observations_df, labels_sr, observations_df.columns.tolist(), 0)

    def _predict(self, observations_df):
        predictions = np.zeros((len(observations_df.index)), )
        for i, row in observations_df.iterrows():
            node = self._root
            terminate_early = False
            while node.feature is not None and not terminate_early:
                feature_value = row[node.feature]
                feature_type = observations_df[node.feature].dtype
                if osm.is_categorical(feature_type):
                    if feature_value in node.children_by_feature_value:
                        node = node.children_by_feature_value[feature_value]
                    else:
                        terminate_early = True
                elif osm.is_continuous(feature_type):
                    split_point = [key for key in node.children_by_feature_value.keys() if key is not None][0]
                    if feature_value >= split_point:
                        node = node.children_by_feature_value[split_point]
                    else:
                        node = node.children_by_feature_value[None]
                else:
                    raise ValueError
            predictions[i] = node.value
        return predictions

    def _test(self, observations_df, labels_sr):
        return None

    class DecisionTreeNode:
        """
        A class representing a decision tree node.
        """
        def __init__(self):
            self.value = None
            self.instances = None
            self.impurity = None
            self.feature = None
            self.children_by_feature_value = None


class DecisionTreeClassification(DecisionTree, ClassificationModel):
    """
    A decision tree classifier model.
    """
    def __init__(self, max_depth=None, min_observations=1, min_entropy=1e-5, min_info_gain=0.,
                 random_feature_selection=False, feature_sample_size_function=math.sqrt):
        DecisionTree.__init__(self, max_depth, min_observations, min_entropy, min_info_gain, random_feature_selection,
                              feature_sample_size_function)
        ClassificationModel.__init__(self)
        self._classes = None

    def _setup(self, labels_sr):
        self._classes = labels_sr.unique()

    def _compute_impurity(self, labels_sr):
        # Shannon entropy.
        entropy = 0
        for klass in self._classes:
            count = labels_sr[labels_sr == klass].count()
            if count > 0:
                proportion = float(count) / labels_sr.count()
                entropy -= proportion * math.log(proportion, 2)
        return entropy

    def _compute_leaf_node_value(self, labels_sr):
        return labels_sr.mode()[0]


class DecisionTreeRegression(DecisionTree, RegressionModel):
    """
    A decision tree classifier model.
    """
    def __init__(self, max_depth=None, min_observations=1, min_variance=1e-5, min_variance_reduction=0.,
                 random_feature_selection=False, feature_sample_size_function=math.sqrt):
        DecisionTree.__init__(self, max_depth, min_observations, min_variance, min_variance_reduction,
                              random_feature_selection, feature_sample_size_function)
        RegressionModel.__init__(self)

    def _compute_impurity(self, labels_sr):
        # Variance (i.e. MSE).
        return labels_sr.var()

    def _compute_leaf_node_value(self, labels_sr):
        return labels_sr.mean()


class SupportVectorMachine(BinaryClassificationModel):
    """
    A support vector machine binary classifier model.
    """
    pass


class MultiBinaryClassification(ClassificationModel):
    """
    A meta model that uses multiple binary classifiers for multi-class classification.
    """
    def __init__(self, base_binary_classifier, number_of_processes=1):
        super(MultiBinaryClassification, self).__init__()
        if not isinstance(base_binary_classifier, BinaryClassificationModel):
            raise ValueError
        if number_of_processes is None or number_of_processes < 1:
            raise ValueError
        self.base_binary_classifier = base_binary_classifier
        self.number_of_processes = number_of_processes
        self._classes = None
        self._binary_classifiers = None

    @staticmethod
    def _build_binary_classifier(base_binary_classifier, observations_df, labels_sr, klass):
        classifier = copy.deepcopy(base_binary_classifier)
        binary_labels_sr = labels_sr.apply(lambda label: 1 if label == klass else 0)
        classifier.fit(observations_df, binary_labels_sr)
        return classifier

    @staticmethod
    def _make_class_probability_prediction(binary_classifier, observations_df):
        return binary_classifier.fuzzy_predict(observations_df)

    def _fit(self, observations_df, labels_sr):
        self._classes = labels_sr.unique()
        base = self.base_binary_classifier
        if self.number_of_processes > 1:
            with mp.Pool(min(len(self._classes), self.number_of_processes)) as pool:
                futures = [pool.apply_async(self._build_binary_classifier, args=(base, observations_df, labels_sr, c))
                           for c in self._classes]
                self._binary_classifiers = [f.get() for f in futures]
        else:
            self._binary_classifiers = [self._build_binary_classifier(base, observations_df, labels_sr, c)
                                        for c in self._classes]

    def _predict(self, observations_df):
        if self.number_of_processes > 1:
            with mp.Pool(min(len(self._classes), self.number_of_processes)) as pool:
                futures = [pool.apply_async(self._make_class_probability_prediction, args=(c, observations_df))
                           for c in self._binary_classifiers]
                class_probability_predictions = [f.get() for f in futures]
        else:
            class_probability_predictions = [self._make_class_probability_prediction(c, observations_df)
                                             for c in self._binary_classifiers]
        class_probability_predictions_df = pd.concat(class_probability_predictions, axis=1)
        prediction_class_indices_sr = class_probability_predictions_df.idxmax(axis=1)
        return prediction_class_indices_sr.apply(lambda e: self._classes[e]).values

    def _test(self, observations_df, labels_sr):
        return None


class BootstrapAggregating(PredictiveModel):
    """
    A meta model that trains multiple models on random subsets of the training data set sampled with replacement.
    """
    def __init__(self, base_model, number_of_models, sample_size_function, number_of_processes):
        if base_model is None or not isinstance(base_model, PredictiveModel):
            raise ValueError
        if number_of_models is None or number_of_models < 1:
            raise ValueError
        if number_of_processes is None or number_of_processes < 1:
            raise ValueError
        super(BootstrapAggregating, self).__init__()
        self.base_model = base_model
        self.number_of_models = number_of_models
        self.sample_size_function = sample_size_function if sample_size_function is not None else _identity_function
        self.number_of_processes = min(number_of_models, number_of_processes)
        self._models = None

    def _aggregate_predictions(self, list_of_predictions_sr: typing.List[pd.Series]) -> pd.Series:
        pass

    @staticmethod
    def _build_model(base_model, sample_size, observations_df, labels_sr):
        model = copy.deepcopy(base_model)
        random.seed()
        indices = [observations_df.index[random.randint(0, len(observations_df.index) - 1)] for _ in range(sample_size)]
        observations_sample_df = observations_df.reindex(indices).reset_index(drop=True)
        labels_sample_sr = labels_sr.reindex(indices).reset_index(drop=True)
        model.fit(observations_sample_df, labels_sample_sr)
        return model

    @staticmethod
    def _make_prediction(model, observations_df):
        return model.predict(observations_df)

    def _fit(self, observations_df, labels_sr):
        sample_size = self.sample_size_function(len(observations_df.index))
        if self.number_of_processes > 1:
            with mp.Pool(self.number_of_processes) as pool:
                futures = [pool.apply_async(self._build_model,
                                            args=(self.base_model, sample_size, observations_df, labels_sr))
                           for _ in range(self.number_of_models)]
                self._models = [f.get() for f in futures]
        else:
            self._models = [self._build_model(self.base_model, sample_size, observations_df, labels_sr)
                            for _ in range(self.number_of_models)]

    def _predict(self, observations_df):
        if self.number_of_processes > 1:
            with mp.Pool(self.number_of_processes) as pool:
                futures = [pool.apply_async(self._make_prediction, args=(model, observations_df))
                           for model in self._models]
                return self._aggregate_predictions([f.get() for f in futures])
        else:
            return self._aggregate_predictions([self._make_prediction(model, observations_df)
                                                for model in self._models])

    def _test(self, observations_df, labels_sr):
        return None


class BootstrapAggregatingClassification(BootstrapAggregating, ClassificationModel):
    """
    A bootstrap aggregating classification meta model.
    """
    def __init__(self, base_model, number_of_models, sample_size_function=None, number_of_processes=mp.cpu_count()):
        if not isinstance(base_model, ClassificationModel):
            raise ValueError
        BootstrapAggregating.__init__(self, base_model, number_of_models, sample_size_function, number_of_processes)
        ClassificationModel.__init__(self)

    def _aggregate_predictions(self, list_of_predictions_sr):
        return pd.concat(list_of_predictions_sr, axis=1).apply(lambda r: r.value_counts().index[0], axis=1,
                                                               result_type='reduce').values


class BootstrapAggregatingRegression(BootstrapAggregating, RegressionModel):
    """
    A bootstrap aggregating regression meta model.
    """
    def __init__(self, base_model, number_of_models, sample_size_function=None, number_of_processes=mp.cpu_count()):
        if not isinstance(base_model, RegressionModel):
            raise ValueError
        BootstrapAggregating.__init__(self, base_model, number_of_models, sample_size_function, number_of_processes)
        RegressionModel.__init__(self)

    def _aggregate_predictions(self, list_of_predictions_sr):
        return pd.concat(list_of_predictions_sr, axis=1).apply(lambda r: r.mean(), axis=1, result_type='reduce').values


class GradientBoosting(PredictiveModel):
    """
    A gradient boosting abstract meta model class.
    """
    def __init__(self, base_model, number_of_models, sampling_factor, min_gradient, max_step_size,
                 step_size_decay_factor, armijo_factor):
        super(GradientBoosting, self).__init__()
        if not isinstance(base_model, RegressionModel):
            raise ValueError
        if number_of_models is None or number_of_models < 1:
            raise ValueError
        if sampling_factor is None or not 0 < sampling_factor <= 1:
            raise ValueError
        if min_gradient is None or min_gradient < 0:
            raise ValueError
        if max_step_size is None or max_step_size <= 0:
            raise ValueError
        if step_size_decay_factor is None or not 0 < step_size_decay_factor < 1:
            raise ValueError
        if armijo_factor is None or not 0 < armijo_factor < 1:
            raise ValueError
        self.base_model = base_model
        self.number_of_models = number_of_models
        self.sampling_factor = sampling_factor
        self.min_gradient = min_gradient
        self.max_step_size = max_step_size
        self.step_size_decay_factor = step_size_decay_factor
        self.armijo_factor = armijo_factor
        self._weighted_models = None

    def _loss(self, raw_predictions_sr: pd.Series, labels_sr: pd.Series) -> float:
        pass

    def _d_loss_wrt_raw_predictions(self, raw_predictions_sr: pd.Series, labels_sr: pd.Series) -> pd.Series:
        pass

    def _raw_predict(self, observations_df):
        predictions = np.zeros((len(observations_df.index)), )
        for (model, weight) in self._weighted_models:
            predictions += model.predict(observations_df).values * weight
        return predictions

    def _build_model(self, observations_df, labels_sr):
        model = copy.deepcopy(self.base_model)
        if self.sampling_factor == 1:
            model.fit(observations_df, labels_sr)
        else:
            observations_sample_df = observations_df.sample(int(len(observations_df.index) * self.sampling_factor))
            model.fit(observations_sample_df, labels_sr.reindex(observations_sample_df.index))
        return model

    def _backtracking_line_search(self, base_predictions_sr, gradient_sr, descent_direction_sr, labels_sr):
        # Backtracking line search using the Armijo-Goldstein termination condition.
        base_loss = self._loss(base_predictions_sr, labels_sr)
        # The expected change of loss w.r.t. the predictions as per the first order Taylor-expansion of the loss
        # function around the base predictions (down-scaled by a constant between 0 and 1).
        expected_change = self.armijo_factor * (gradient_sr.values @ descent_direction_sr.values)
        step_size = self.max_step_size
        # Keep reducing the step size until it yields a reduction in loss that matches or exceeds our expectations for
        # a change this big in the independent variable.
        while self._loss(base_predictions_sr + step_size * descent_direction_sr, labels_sr) > \
                base_loss + step_size * expected_change:
            step_size *= self.step_size_decay_factor
        return step_size

    def _fit(self, observations_df, labels_sr):
        self._weighted_models = []
        predictions_sr = pd.Series(np.zeros((len(observations_df.index), )))
        for i in range(self.number_of_models):
            gradient_sr = self._d_loss_wrt_raw_predictions(predictions_sr, labels_sr)
            if np.all(np.absolute(gradient_sr.values) <= self.min_gradient):
                break
            model = self._build_model(observations_df, -gradient_sr)
            residual_predictions_sr = model.predict(observations_df)
            weight = self._backtracking_line_search(predictions_sr, gradient_sr, residual_predictions_sr, labels_sr)
            self._weighted_models.append((model, weight))
            predictions_sr += residual_predictions_sr * weight

    def _test(self, observations_df: pd.DataFrame, labels_sr: pd.Series):
        return self._loss(pd.Series(self._raw_predict(observations_df)), labels_sr)


class GradientBoostingBinaryClassification(GradientBoosting, BinaryClassificationModel):
    """
    A gradient boosting binary classification meta model minimizing the log loss function.
    """
    def __init__(self, base_model, number_of_models, sampling_factor=1., min_gradient=1e-7, max_step_size=1000.,
                 step_size_decay_factor=.7, armijo_factor=.7):
        GradientBoosting.__init__(self, base_model, number_of_models, sampling_factor, min_gradient, max_step_size,
                                  step_size_decay_factor, armijo_factor)
        BinaryClassificationModel.__init__(self)

    def _loss(self, raw_predictions_sr, labels_sr):
        # Binary cross entropy loss.
        labels = labels_sr.values
        raw_predictions = raw_predictions_sr.values
        squashed_predictions = np.minimum(_logistic_function(raw_predictions) + self.epsilon, 1 - self.epsilon)
        return (-labels * np.log(squashed_predictions) - (1 - labels) * np.log(1 - squashed_predictions)).mean()

    def _d_loss_wrt_raw_predictions(self, raw_predictions_sr, labels_sr):
        return pd.Series((_logistic_function(raw_predictions_sr.values) - labels_sr.values) / labels_sr.count())

    def _fuzzy_predict(self, observations_df):
        return _logistic_function(self._raw_predict(observations_df))


class GradientBoostingRegression(GradientBoosting, RegressionModel):
    """
    A gradient boosting regression meta model minimizing the MSE loss function.
    """
    def __init__(self, base_model, number_of_models, sampling_factor=1., min_gradient=1e-7, max_step_size=1000.,
                 step_size_decay_factor=.7, armijo_factor=.7):
        GradientBoosting.__init__(self, base_model, number_of_models, sampling_factor, min_gradient, max_step_size,
                                  step_size_decay_factor, armijo_factor)
        RegressionModel.__init__(self)

    def _loss(self, raw_predictions_sr, labels_sr):
        # Mean squared error.
        return np.square(raw_predictions_sr.values - labels_sr.values).mean()

    def _d_loss_wrt_raw_predictions(self, raw_predictions_sr, labels_sr):
        return 2 / labels_sr.count() * (raw_predictions_sr - labels_sr)

    def _predict(self, observations_df):
        return self._raw_predict(observations_df)


class BaggedTreesClassification(BootstrapAggregatingClassification):
    """
    A bagged trees classification model.
    """
    def __init__(self, number_of_models, sample_size_function=None, max_depth=None, min_observations=1,
                 min_entropy=1e-5, min_info_gain=0., number_of_processes=mp.cpu_count()):
        super(BaggedTreesClassification, self)\
            .__init__(DecisionTreeClassification(max_depth=max_depth, min_observations=min_observations,
                                                 min_entropy=min_entropy, min_info_gain=min_info_gain,
                                                 random_feature_selection=False),
                      number_of_models, sample_size_function, number_of_processes)


class BaggedTreesRegression(BootstrapAggregatingRegression):
    """
    A bagged trees regression model.
    """
    def __init__(self, number_of_models, sample_size_function=None, max_depth=None, min_observations=1,
                 min_variance=1e-5, min_variance_reduction=0., number_of_processes=mp.cpu_count()):
        super(BaggedTreesRegression, self)\
            .__init__(DecisionTreeRegression(max_depth=max_depth, min_observations=min_observations,
                                             min_variance=min_variance, min_variance_reduction=min_variance_reduction,
                                             random_feature_selection=False),
                      number_of_models, sample_size_function, number_of_processes)


class RandomForestClassification(BootstrapAggregatingClassification):
    """
    A random forest classification model.
    """
    def __init__(self, number_of_models, sample_size_function=None, max_depth=None, min_observations=1,
                 min_entropy=1e-5, min_info_gain=0., feature_sample_size_function=math.sqrt,
                 number_of_processes=mp.cpu_count()):
        super(RandomForestClassification, self)\
            .__init__(DecisionTreeClassification(max_depth=max_depth, min_observations=min_observations,
                                                 min_entropy=min_entropy, min_info_gain=min_info_gain,
                                                 random_feature_selection=True,
                                                 feature_sample_size_function=feature_sample_size_function),
                      number_of_models, sample_size_function, number_of_processes)


class RandomForestRegression(BootstrapAggregatingRegression):
    """
    A random forest regression model.
    """
    def __init__(self, number_of_models, sample_size_function=None, max_depth=None, min_observations=1,
                 min_variance=1e-5, min_variance_reduction=0., feature_sample_size_function=math.sqrt,
                 number_of_processes=mp.cpu_count()):
        super(RandomForestRegression, self)\
            .__init__(DecisionTreeRegression(max_depth=max_depth, min_observations=min_observations,
                                             min_variance=min_variance, min_variance_reduction=min_variance_reduction,
                                             random_feature_selection=True,
                                             feature_sample_size_function=feature_sample_size_function),
                      number_of_models, sample_size_function, number_of_processes)


class BoostedTreesBinaryClassification(GradientBoostingBinaryClassification):
    """
    A gradient boosting binary classification model using regression decision trees.
    """
    def __init__(self, number_of_models, max_depth=None, min_observations=1, min_variance=0., min_variance_reduction=0.,
                 random_feature_selection=False, feature_sample_size_function=math.sqrt, sampling_factor=1.,
                 min_gradient=1e-7, max_step_size=1000., step_size_decay_factor=.7, armijo_factor=.7):
        super(BoostedTreesBinaryClassification, self)\
            .__init__(DecisionTreeRegression(max_depth=max_depth, min_observations=min_observations,
                                             min_variance=min_variance, min_variance_reduction=min_variance_reduction,
                                             random_feature_selection=random_feature_selection,
                                             feature_sample_size_function=feature_sample_size_function),
                      number_of_models, sampling_factor, min_gradient, max_step_size, step_size_decay_factor,
                      armijo_factor)


class BoostedTreesRegression(GradientBoostingRegression):
    """
    A gradient boosting regression model using regression decision trees.
    """
    def __init__(self, number_of_models, max_depth=None, min_observations=1, min_variance=0., min_variance_reduction=0.,
                 random_feature_selection=False, feature_sample_size_function=math.sqrt, sampling_factor=1.,
                 min_gradient=1e-7, max_step_size=1000., step_size_decay_factor=.7, armijo_factor=.7):
        super(BoostedTreesRegression, self)\
            .__init__(DecisionTreeRegression(max_depth=max_depth, min_observations=min_observations,
                                             min_variance=min_variance, min_variance_reduction=min_variance_reduction,
                                             random_feature_selection=random_feature_selection,
                                             feature_sample_size_function=feature_sample_size_function),
                      number_of_models, sampling_factor, min_gradient, max_step_size, step_size_decay_factor,
                      armijo_factor)
