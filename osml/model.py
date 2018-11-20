import math
import numpy as np
import pandas as pd


class Model:
    """An abstract class representing a statistical model or machine learning algorithm."""
    def __init__(self):
        self._observation_types = None
        self._label_type = None
        self._label_name = ''

    def _validate_observations(self, observations_df):
        if len(observations_df.dtypes) == 0 or len(observations_df.index) == 0\
                or any(not np.issubdtype(dtype, np.number) for dtype in observations_df.dtypes)\
                or not observations_df.dtypes.equals(self._observation_types):
            raise ValueError

    def _validate_labels(self, labels_sr):
        if len(labels_sr.index) == 0 or not np.issubdtype(labels_sr.dtype, np.number)\
                or labels_sr.dtype != self._label_type:
            raise ValueError

    def _validate_observations_and_labels(self, observations_df, labels_sr):
        self._validate_observations(observations_df)
        self._validate_labels(labels_sr)
        if len(observations_df.index) != len(labels_sr.index):
            raise ValueError

    def _fit(self, observations_df, labels_sr):
        pass

    def _predict(self, observations_df):
        pass

    def _test(self, observations_df, labels_sr):
        pass

    def _evaluate(self, predictions_sr, labels_sr):
        pass

    def fit(self, observations_df, labels_sr):
        """Fits the model to the data.

        It learns a model to map observations to labels from the provided data.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
            labels_sr: A series of target values where each element is the label of the corresponding row in the data
            frame of observations.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels.
        """
        self._observation_types = observations_df.dtypes
        self._label_type = labels_sr.dtype
        self._label_name = labels_sr.name
        self._validate_observations_and_labels(observations_df, labels_sr)
        return self._fit(observations_df, labels_sr)

    def predict(self, observations_df):
        """Predicts the labels of the observations.

        Args:
            observations_df: A 2D frame of data points where each column is a feature and each row is an observation.

        Returns:
            A series of values where each element is the predicted label of the corresponding observation.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels. Or if the types of
            the features are not the same as those of the data the model was fit to.
        """
        self._validate_observations(observations_df)
        predictions = pd.Series(self._predict(observations_df), name=self._label_name, dtype=self._label_type)
        self._validate_labels(predictions)
        return predictions

    def test(self, observations_df, labels_sr):
        """Computes the prediction error of the model on a test data set in terms of the function the model minimizes.

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

    def evaluate(self, predictions_sr, labels_sr):
        """A function for evaluating the accuracy of the model's predictions. It can be either the minimized function
        or some other metric of the model's explanatory power based on its predictions.

        Args:
            predictions_sr: A series of predictions.
            labels_sr: A series of target values where each element is the label of the corresponding row in the data
            frame of observations.

        Returns:
            A non-standardized numeric measure of prediction error. It is not comparable across different model types.

        Raises:
            ValueError: If the number of observations is 0 or it does not match the number of labels. Or if the types of
            the features or the labels are not the same as those of the data the model was fit to.
        """
        self._validate_labels(predictions_sr)
        self._validate_labels(labels_sr)
        return self._evaluate(predictions_sr, labels_sr)


def _is_continuous(data_type):
    return np.issubdtype(data_type, np.floating)


def _is_categorical(data_type):
    return np.issubdtype(data_type, np.integer)


class RegressionModel(Model):
    """An abstract base class for regression models."""
    def __init__(self):
        super(RegressionModel, self).__init__()

    def _validate_labels(self, labels_sr):
        super(RegressionModel, self)._validate_labels(labels_sr)
        if not _is_continuous(labels_sr.dtype):
            raise ValueError

    def _evaluate(self, predictions_sr, labels_sr):
        # Root mean squared error.
        return np.sqrt(np.square(predictions_sr.values - labels_sr.values).mean())


class ClassificationModel(Model):
    """An abstract base class for classification models."""
    def __init__(self):
        super(ClassificationModel, self).__init__()

    def _validate_labels(self, labels_sr):
        super(ClassificationModel, self)._validate_labels(labels_sr)
        if not _is_categorical(labels_sr.dtype):
            raise ValueError

    def _evaluate(self, predictions_sr, labels_sr):
        # The number of correct predictions per the total number of predictions.
        return len([p for i, p in enumerate(predictions_sr) if p == labels_sr[i]]) / len(predictions_sr)


class BinaryClassificationModel(ClassificationModel):
    """An abstract base class for binary classification models."""
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()

    def _validate_labels(self, labels_sr):
        super(BinaryClassificationModel, self)._validate_labels(labels_sr)
        if not _is_categorical(labels_sr.dtype) or any(v != 0 and v != 1 for v in labels_sr.unique()):
            raise ValueError


def _bias_trick(observations_df):
    observations = observations_df.values
    obs_ext = np.ones((observations.shape[0], observations.shape[1] + 1))
    obs_ext[:, :-1] = observations
    return obs_ext


class LinearRegression(RegressionModel):
    """A linear regression model using ordinary least squares."""
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.beta = None

    def _fit(self, observations_df, labels_sr):
        # Take the partial derivatives of the sum of squared errors w.r.t. both parameters of the model (weights and
        # bias), set both derivatives to 0, and solve the resulting system of linear equations. Whether the found roots
        # really minimize the sum of squared errors is not verified here.
        # NOTE: Using the bias trick reduces the parameters to a single vector and significantly simplifies the
        # analytic solution.
        observations = _bias_trick(observations_df)
        observations_trans = observations.transpose()
        self.beta = np.linalg.inv(observations_trans @ observations) @ (observations_trans @ labels_sr.values)

    def _predict(self, observations_df):
        return _bias_trick(observations_df) @ self.beta

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean()


class LogisticRegression(BinaryClassificationModel):
    """A logistic regression model using the Newton-Raphson method."""
    def __init__(self, iterations=100):
        super(LogisticRegression, self).__init__()
        self.iterations = iterations
        self.beta = None

    def _logistic_function(self, observations):
        return np.reciprocal(np.exp(-observations @ self.beta) + 1)

    def _fit(self, observations_df, labels_sr):
        # Use Newton's method to numerically approximate the root of the log likelihood function's derivative.
        observations = _bias_trick(observations_df)
        observations_trans = observations.transpose()
        labels = labels_sr.values
        self.beta = np.zeros(observations.shape[1], )
        for i in range(self.iterations):
            predictions = self._logistic_function(observations)
            first_derivative = observations_trans @ (predictions - labels)
            # If the number of parameters is N, the Hessian will take the shape of an NxN matrix.
            second_derivative = (observations_trans * (predictions * (1 - predictions))) @ observations
            self.beta = self.beta - np.linalg.inv(second_derivative) @ first_derivative

    def _predict(self, observations_df):
        return np.rint(self._logistic_function(_bias_trick(observations_df)))

    def _test(self, observations_df, labels_sr):
        labels = labels_sr.values
        predictions = self._logistic_function(_bias_trick(observations_df))
        return (np.log(predictions) * labels + np.log(1 - predictions) * (1 - labels)).mean()


class NaiveBayes(ClassificationModel):
    """A naive Bayes multinomial classification model."""
    def __init__(self, laplace_smoothing=1.):
        super(Model, self).__init__()
        self.laplace_smoothing = laplace_smoothing
        self.epsilon = 1e-8
        self.classes = None
        self.class_priors = None
        self.feature_statistics = None

    def _fit(self, observations_df, labels_sr):
        self.classes = labels_sr.unique()
        self.class_priors = labels_sr.value_counts().astype(np.float_) / labels_sr.count()
        indices_by_label = {}
        for label_value in self.classes:
            indices_by_label[label_value] = [i for i, v in enumerate(labels_sr) if v == label_value]
        self.feature_statistics = []
        for column in observations_df.columns:
            feature_values_sr = observations_df[column]
            if _is_categorical(feature_values_sr.dtype):
                self.feature_statistics.append(self.CategoricalFeatureStatistics(feature_values_sr, indices_by_label,
                                                                                 self.laplace_smoothing))
            elif _is_continuous(feature_values_sr.dtype):
                self.feature_statistics.append(self.ContinuousFeatureStatistics(feature_values_sr, indices_by_label))
            else:
                raise ValueError

    def _predict(self, observations_df):
        predictions = []
        for i, row in observations_df.iterrows():
            highest_class_probability = -1
            prediction = None
            probability_of_evidence = 1
            for j, feature_stats in enumerate(self.feature_statistics):
                probability_of_evidence *= feature_stats.get_probability(row[j])
            for label_class in self.classes:
                class_prior = self.class_priors[label_class]
                likelihood = 1
                for j, feature_stats in enumerate(self.feature_statistics):
                    likelihood *= feature_stats.get_likelihood(row[j], label_class)
                class_probability = likelihood * class_prior / probability_of_evidence
                if class_probability > highest_class_probability:
                    highest_class_probability = class_probability
                    prediction = label_class
            predictions.append(prediction)
        return predictions

    def _test(self, observations_df, labels_sr):
        return (1 + self.epsilon) / (self.evaluate(self.predict(observations_df), labels_sr) + self.epsilon) - 1

    class FeatureStatistics:
        """An abstract base class for feature statistics."""
        def get_probability(self, value):
            """A function that returns the probability of the feature taking on the specified value.

            Args:
                value: The value whose probability is to be established.

            Returns:
                The probability of the specified value for the feature.
            """
            pass

        def get_likelihood(self, value, label_value):
            """A function that returns the probability of the feature having the specified value given the specified
            label value.

            Args:
                value: The feature value.
                label_value: The label value.

            Returns:
                The conditional probability of the feature value given the label value.
            """
            pass

    class ContinuousFeatureStatistics:
        """A class for continuous feature statistics."""
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
        """A class for categorical feature statistics."""
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
            if value in value_counts.index:
                value_count = value_counts[value] + laplace_smoothing
            else:
                value_count = laplace_smoothing
            return float(value_count) / (total_value_count + value_counts.count() * laplace_smoothing)

        def get_probability(self, value):
            return self.calculate_probability(value, self.value_counts, self.total_value_count, self.laplace_smoothing)

        def get_likelihood(self, value, label_value):
            likelihood_tuple = self.likelihoods[label_value]
            return self.calculate_probability(value, likelihood_tuple[0], likelihood_tuple[1], self.laplace_smoothing)

