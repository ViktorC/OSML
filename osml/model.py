import math
import numpy as np
import operator
import pandas as pd


class Model:
    """An abstract class representing a statistical model or machine learning algorithm."""
    def __init__(self, epsilon=1e-8):
        self._observation_types = None
        self._label_type = None
        self._label_name = ''
        self._epsilon = epsilon

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

    def _fit(self, observations_df: pd.DataFrame, labels_sr: pd.Series) -> None:
        pass

    def _predict(self, observations_df: pd.DataFrame) -> pd.Series:
        pass

    def _test(self, observations_df: pd.DataFrame, labels_sr: pd.Series) -> float:
        return (1 + self._epsilon) / (self.evaluate(self.predict(observations_df), labels_sr) + self._epsilon) - 1

    def _evaluate(self, predictions_sr: pd.Series, labels_sr: pd.Series) -> float:
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
        # The reciprocal of the root mean squared error plus one.
        return 1 / (np.sqrt(np.square(predictions_sr.values - labels_sr.values).mean()) + 1)


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
        return len([p for i, p in predictions_sr.iteritems() if p == labels_sr[i]]) / predictions_sr.count()


class BinaryClassificationModel(ClassificationModel):
    """An abstract base class for binary classification models."""
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()

    def _validate_labels(self, labels_sr):
        super(BinaryClassificationModel, self)._validate_labels(labels_sr)
        if not _is_categorical(labels_sr.dtype) or any(v != 0 and v != 1 for v in labels_sr.unique()):
            raise ValueError


class MultiBinaryClassification(ClassificationModel):
    """A wrapper class that uses multiple binary classifiers for multi-class classification."""
    def __init__(self, binary_classification, *args):
        super(MultiBinaryClassification, self).__init__()
        self.binary_classification = binary_classification
        self.args = args
        self.classes = None
        self.binary_classifiers = None

    def _fit(self, observations_df, labels_sr):
        self.classes = labels_sr.unique()
        self.binary_classifiers = []
        for klass in self.classes:
            classifier = self.binary_classification(*self.args)
            binary_labels_sr = labels_sr.apply(lambda label: 1 if label == klass else 0)
            classifier.fit(observations_df, binary_labels_sr)
            self.binary_classifiers.append(classifier)

    def _predict(self, observations_df):
        predictions = []
        label_sr = pd.Series([1])
        for i, row in observations_df.iterrows():
            most_confident_prediction = None
            lowest_loss = None
            for j, classifier in enumerate(self.binary_classifiers):
                loss = classifier.test(pd.DataFrame(index=[0], columns=row.index,
                                                    data=row.values.reshape(1, len(row.index))), label_sr)
                if lowest_loss is None or loss < lowest_loss:
                    lowest_loss = loss
                    most_confident_prediction = self.classes[j]
            predictions.append(most_confident_prediction)
        return predictions


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
            try:
                self.beta = self.beta - np.linalg.inv(second_derivative) @ first_derivative
            except np.linalg.LinAlgError:
                # The values of the Hessian are so small that its determinant is practically 0, therefore, it is not
                # invertible. No further second order optimization is possible.
                break

    def _predict(self, observations_df):
        return np.rint(self._logistic_function(_bias_trick(observations_df)))

    def _test(self, observations_df, labels_sr):
        labels = labels_sr.values
        predictions = self._logistic_function(_bias_trick(observations_df))
        predictions[predictions == 0] = self._epsilon
        predictions[predictions == 1] = 1 - self._epsilon
        return -((np.log(predictions) * labels + np.log(1 - predictions) * (1 - labels)).mean())


class NaiveBayes(ClassificationModel):
    """A naive Bayes multinomial classification model."""
    def __init__(self, laplace_smoothing=1.):
        super(NaiveBayes, self).__init__()
        self.laplace_smoothing = laplace_smoothing
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
        for row in observations_df.itertuples():
            highest_class_probability = -1
            prediction = None
            probability_of_evidence = 1
            for j, feature_stats in enumerate(self.feature_statistics):
                probability_of_evidence *= feature_stats.get_probability(row[j + 1])
            for label_class in self.classes:
                class_prior = self.class_priors[label_class]
                likelihood = 1
                for j, feature_stats in enumerate(self.feature_statistics):
                    likelihood *= feature_stats.get_likelihood(row[j + 1], label_class)
                class_probability = likelihood * class_prior / probability_of_evidence
                if class_probability > highest_class_probability:
                    highest_class_probability = class_probability
                    prediction = label_class
            predictions.append(prediction)
        return predictions

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


class KNearestNeighbors(Model):
    """An abstract weighted k-nearest neighbors model."""
    def __init__(self, k, standardize):
        super(KNearestNeighbors, self).__init__()
        if k <= 0:
            raise ValueError
        self.k = k
        self.standardize = standardize
        self.observations_mean = None
        self.observations_sd = None
        self.preprocessed_observations = None
        self.labels = None

    def _fit(self, observations_df, labels_sr):
        if len(observations_df.index) < self.k + 1:
            raise ValueError
        if self.standardize:
            self.observations_mean = observations_df.values.mean(axis=0)
            self.observations_sd = np.sqrt(np.square(observations_df.values - self.observations_mean).mean(axis=0))
            self.preprocessed_observations = (observations_df.values - self.observations_mean) / self.observations_sd
        else:
            self.preprocessed_observations = observations_df.values
        self.labels = labels_sr.values

    def _distances(self, observation):
        # Simple Euclidian distance.
        return np.sqrt(np.square(self.preprocessed_observations - observation).sum(axis=1))

    def _weight(self, distance):
        # Gaussian kernel function.
        return math.exp(-(distance ** 2) / 2) / math.sqrt(2 * math.pi)

    def _select_weighted_nearest_neighbors(self, observation):
        if self.standardize:
            preprocessed_observation = (observation - self.observations_mean) / self.observations_sd
        else:
            preprocessed_observation = observation
        distances = self._distances(preprocessed_observation)
        k_plus_1_nn_indices = np.argpartition(distances, self.k + 1)
        k_plus_1_nn = pd.Series(distances[k_plus_1_nn_indices], index=k_plus_1_nn_indices)
        k_plus_1_nn = k_plus_1_nn.sort_values()
        weighted_nearest_neighbor_labels = {}
        distance_to_normalize_by = None
        for i, value in k_plus_1_nn.iteritems():
            if distance_to_normalize_by is None:
                distance_to_normalize_by = value
                continue
            weight = self._weight(value / (distance_to_normalize_by + self._epsilon))
            label = self.labels[i]
            if label in weighted_nearest_neighbor_labels:
                weighted_nearest_neighbor_labels[label] += weight
            else:
                weighted_nearest_neighbor_labels[label] = weight
        return weighted_nearest_neighbor_labels


class KNearestNeighborsRegression(KNearestNeighbors, RegressionModel):
    """A weighted k-nearest neighbors regression model."""
    def __init__(self, k=7, standardize=True):
        KNearestNeighbors.__init__(self, k, standardize)
        RegressionModel.__init__(self)

    def _predict(self, observations_df):
        predictions = []
        for i, row in observations_df.iterrows():
            weighted_labels_sum = 0
            weight_sum = 0
            for key, value in self._select_weighted_nearest_neighbors(row.values).items():
                weighted_labels_sum += key * value
                weight_sum += value
            predictions.append(weighted_labels_sum / weight_sum)
        return predictions


class KNearestNeighborsClassification(KNearestNeighbors, ClassificationModel):
    """A weighted k-nearest neighbors classification model."""
    def __init__(self, k=7, standardize=True):
        KNearestNeighbors.__init__(self, k, standardize)
        ClassificationModel.__init__(self)

    def _predict(self, observations_df):
        predictions = []
        for i, row in observations_df.iterrows():
            nearest_neighbors = self._select_weighted_nearest_neighbors(row.values)
            sorted_nearest_neighbors = sorted(nearest_neighbors.items(), key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_nearest_neighbors[0][0])
        return predictions


class DecisionTree(Model):
    """An abstract decision tree base class."""
    def __init__(self):
        super(DecisionTree, self).__init__()

    def _fit(self, observations_df, labels_sr):
        pass
