import copy
import math
import multiprocessing as mp
import numpy as np
import operator
import pandas as pd
import random
import typing


class Model:
    """An abstract class representing a statistical model or machine learning algorithm."""
    def __init__(self, epsilon=1e-8):
        if epsilon is None or epsilon <= 0:
            raise ValueError
        self._observation_types = None
        self._label_type = None
        self._label_name = ''
        self._epsilon = epsilon

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
        predictions = []
        label_sr = pd.Series([1])
        for i, row in observations_df.iterrows():
            most_confident_prediction = None
            lowest_loss = None
            for j, classifier in enumerate(self._binary_classifiers):
                loss = classifier.test(pd.DataFrame(index=[0], columns=row.index,
                                                    data=row.values.reshape(1, len(row.index))), label_sr)
                if lowest_loss is None or loss < lowest_loss:
                    lowest_loss = loss
                    most_confident_prediction = self._classes[j]
            predictions.append(most_confident_prediction)
        return predictions


def _bias_trick(observations_df):
    observations = observations_df.values
    obs_ext = np.ones((observations.shape[0], observations.shape[1] + 1))
    obs_ext[:, :-1] = observations
    return obs_ext


class LinearRegression(RegressionModel):
    """A linear regression model fitted using ordinary least squares."""
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
        observations_trans = observations.transpose()
        self._beta = np.linalg.inv(observations_trans @ observations) @ (observations_trans @ labels_sr.values)

    def _predict(self, observations_df):
        return _bias_trick(observations_df) @ self._beta

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean()


class RidgeRegression(LinearRegression):
    """A linear ridge regression model fitted using ordinary least squares."""
    def __init__(self, alpha=1e-2):
        super(RidgeRegression, self).__init__()
        if alpha is None or alpha < 0:
            raise ValueError
        self.alpha = alpha

    def _fit(self, observations_df, labels_sr):
        observations = _bias_trick(observations_df)
        observations_trans = observations.transpose()
        self._beta = np.linalg.inv((observations_trans @ observations) + self.alpha) @ \
            (observations_trans @ labels_sr.values)

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean() + \
               self.alpha * np.square(self._beta).sum()


class LassoRegression(LinearRegression):
    """A Least Absolute Shrinkage and Selection Operator regression model fitted using coordinate descent."""
    def __init__(self, alpha=1e-2, iterations=100):
        super(LassoRegression, self).__init__()
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

    def _fit(self, observations_df, labels_sr):
        observations = _bias_trick(observations_df)
        labels = labels_sr.values
        self._beta = np.zeros((observations.shape[1], ))
        prev_loss = float('inf')
        for i in range(self.iterations):
            # Optimize the parameters analytically dimension by dimension.
            for j in range(observations.shape[1]):
                # Solve for the (sub)derivative of the loss function with respect to the jth parameter.
                obs_j = observations[:, j]
                obs_not_j = np.delete(observations, j, axis=1)
                beta_not_j = np.delete(self._beta, j)
                obs_j_trans = obs_j.transpose()
                p = obs_j_trans @ labels - obs_j_trans @ (obs_not_j @ beta_not_j)
                z = obs_j_trans @ obs_j
                # As the absolute value function is not differentiable at 0, use soft thresholding.
                self._beta[j] = self._soft_threshold(p) / z
            # If a full cycle of optimization does not reduce the loss, the model has converged.
            loss = self.test(observations_df, labels_sr)
            if loss < prev_loss:
                prev_loss = loss
            else:
                break

    def _test(self, observations_df, labels_sr):
        return np.square(self._predict(observations_df) - labels_sr.values).mean() + \
               self.alpha * np.abs(self._beta).sum()


class LogisticRegression(BinaryClassificationModel):
    """A logistic regression model fitted using the Newton-Raphson method."""
    def __init__(self, iterations=100, min_gradient=1e-7):
        super(LogisticRegression, self).__init__()
        if iterations is None or iterations < 1:
            raise ValueError
        if min_gradient is None or min_gradient < 0:
            raise ValueError
        self.iterations = iterations
        self.min_gradient = min_gradient
        self._beta = None

    def _logistic_function(self, observations):
        return np.reciprocal(np.exp(-observations @ self._beta) + 1)

    def _fit(self, observations_df, labels_sr):
        # Use Newton's method to numerically approximate the root of the log likelihood function's derivative.
        observations = _bias_trick(observations_df)
        observations_trans = observations.transpose()
        labels = labels_sr.values
        self._beta = np.zeros(observations.shape[1], )
        for i in range(self.iterations):
            predictions = self._logistic_function(observations)
            first_derivative = observations_trans @ (predictions - labels)
            if np.all(first_derivative <= self.min_gradient):
                break
            # If the number of parameters is N, the Hessian will take the shape of an N by N matrix.
            second_derivative = (observations_trans * (predictions * (1 - predictions))) @ observations
            self._beta = self._beta - np.linalg.inv(second_derivative) @ first_derivative

    def _predict(self, observations_df):
        return np.rint(self._logistic_function(_bias_trick(observations_df)))

    def _test(self, observations_df, labels_sr):
        labels = labels_sr.values
        observations = _bias_trick(observations_df)
        predictions = np.minimum(self._logistic_function(observations) + self._epsilon, 1 - self._epsilon)
        return -((np.log(predictions) * labels + np.log(1 - predictions) * (1 - labels)).mean())


class NaiveBayes(ClassificationModel):
    """A naive Bayes multinomial classification model."""
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
            if _is_categorical(feature_values_sr.dtype):
                self._feature_statistics.append(self.CategoricalFeatureStatistics(feature_values_sr, indices_by_label,
                                                                                  self.laplace_smoothing))
            elif _is_continuous(feature_values_sr.dtype):
                self._feature_statistics.append(self.ContinuousFeatureStatistics(feature_values_sr, indices_by_label))
            else:
                raise ValueError

    def _predict(self, observations_df):
        predictions = []
        for row in observations_df.itertuples():
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
        if k is None or k < 1:
            raise ValueError
        if not isinstance(standardize, bool):
            raise ValueError
        self.k = k
        self.standardize = standardize
        self._observations_mean = None
        self._observations_sd = None
        self._preprocessed_observations = None
        self._labels = None

    def _fit(self, observations_df, labels_sr):
        if len(observations_df.index) < self.k + 1:
            raise ValueError
        if self.standardize:
            self._observations_mean = observations_df.values.mean(axis=0)
            self._observations_sd = np.sqrt(np.square(observations_df.values - self._observations_mean).mean(axis=0))
            self._preprocessed_observations = (observations_df.values - self._observations_mean) / self._observations_sd
        else:
            self._preprocessed_observations = observations_df.values
        self._labels = labels_sr.values

    def _distances(self, observation):
        # Simple Euclidian distance.
        return np.sqrt(np.square(self._preprocessed_observations - observation).sum(axis=1))

    def _weight(self, distance):
        # Gaussian kernel function.
        return math.exp(-(distance ** 2) / 2) / math.sqrt(2 * math.pi)

    def _select_weighted_nearest_neighbors(self, observation):
        if self.standardize:
            preprocessed_observation = (observation - self._observations_mean) / self._observations_sd
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
            label = self._labels[i]
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
    def __init__(self, max_depth, min_observations, min_impurity, min_impurity_reduction, random_feature_selection,
                 feature_sample_size_function):
        super(DecisionTree, self).__init__()
        if random_feature_selection is None or not isinstance(random_feature_selection, bool):
            raise ValueError
        self.max_depth = max_depth
        self.min_observations = max(2, min_observations) if min_observations is not None else 2
        self.min_impurity = max(0, min_impurity) if min_impurity is not None else 0
        self.min_impurity_reduction = min_impurity_reduction if min_impurity_reduction is not None else float('-inf')
        self.random_feature_selection = random_feature_selection
        self.feature_sample_size_function = feature_sample_size_function if feature_sample_size_function is not None \
            else lambda n: n
        self._root = None

    def _setup(self, labels_sr: pd.Series) -> None:
        pass

    def _compute_impurity(self, labels_sr: pd.Series) -> float:
        pass

    def _compute_leaf_node_value(self, labels_sr: pd.Series) -> float:
        pass

    def _compute_aggregate_impurity(self, list_of_labels_sr):
        total_labels = 0
        for labels_sr in list_of_labels_sr:
            total_labels += labels_sr.count()
        aggregate_impurity = 0
        for labels_sr in list_of_labels_sr:
            count = labels_sr.count()
            if count > 0:
                impurity = self._compute_impurity(labels_sr)
                if math.isnan(impurity):
                    impurity = 0
                aggregate_impurity += count * impurity / total_labels
        return aggregate_impurity

    def _build_tree(self, observations_df, labels_sr, features, depth):
        # Instantiate the node and calculate its value.
        node = self.DecisionTreeNode()
        node.value = self._compute_leaf_node_value(labels_sr)
        # Check termination conditions.
        if len(features) == 0 or labels_sr.count() <= self.min_observations:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        current_impurity = self._compute_impurity(labels_sr)
        if current_impurity <= self.min_impurity:
            return node
        # Select the best feature to split by.
        best_feature = None
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
            column_sr = observations_df[feature]
            indices_by_feature_value = {}
            uniques = column_sr.unique()
            # If all the observations have the same value for the selected feature, we cannot split by it.
            if len(uniques) == 1:
                continue
            impurity = None
            if _is_categorical(column_sr.dtype):
                list_of_labels_sr = []
                for value in uniques:
                    indices = column_sr[column_sr == value].index
                    indices_by_feature_value[value] = indices
                    list_of_labels_sr.append(labels_sr.reindex(indices))
                impurity = self._compute_aggregate_impurity(list_of_labels_sr)
            elif _is_continuous(column_sr.dtype):
                uniques.sort()
                for j, value in enumerate(uniques):
                    if j < len(uniques) - 1:
                        split_point = (value + uniques[j + 1]) / 2
                        ge_split_indices = column_sr[column_sr >= split_point].index
                        l_split_indices = column_sr[column_sr < split_point].index
                        list_of_labels_sr = [labels_sr.reindex(ge_split_indices), labels_sr.reindex(l_split_indices)]
                        split_point_impurity = self._compute_aggregate_impurity(list_of_labels_sr)
                        if impurity is None or split_point_impurity < impurity:
                            indices_by_feature_value = {split_point: ge_split_indices, None: l_split_indices}
                            impurity = split_point_impurity
            else:
                raise ValueError
            if best_impurity is None or impurity < best_impurity:
                best_feature = feature
                best_indices_by_feature_value = indices_by_feature_value
                best_impurity = impurity
        # Check split metric reduction termination condition (and whether any split points could be established at all).
        if best_impurity is None or current_impurity - best_impurity <= self.min_impurity_reduction:
            return node
        # Split the observations and the labels and recurse to build the child nodes and their sub-trees.
        features.remove(best_feature)
        node.feature = best_feature
        node.children_by_feature_value = {}
        for key, value in best_indices_by_feature_value.items():
            if len(value) == 0:
                child = self.DecisionTreeNode()
                child.value = node.value
            else:
                child = self._build_tree(observations_df.reindex(value), labels_sr.reindex(value), features, depth + 1)
            node.children_by_feature_value[key] = child
        features.append(best_feature)
        return node

    def _fit(self, observations_df, labels_sr):
        self._setup(labels_sr)
        self._root = self._build_tree(observations_df, labels_sr, observations_df.columns.tolist(), 0)

    def _predict(self, observations_df):
        predictions = []
        for i, row in observations_df.iterrows():
            node = self._root
            terminate_early = False
            while not terminate_early and node.feature is not None:
                feature_value = row[node.feature]
                feature_type = observations_df[node.feature].dtype
                if _is_categorical(feature_type):
                    if feature_value in node.children_by_feature_value:
                        node = node.children_by_feature_value[feature_value]
                    else:
                        terminate_early = True
                elif _is_continuous(feature_type):
                    split_point = [key for key in node.children_by_feature_value.keys() if key is not None][0]
                    if feature_value >= split_point:
                        node = node.children_by_feature_value[split_point]
                    else:
                        node = node.children_by_feature_value[None]
                else:
                    raise ValueError
            predictions.append(node.value)
        return predictions

    class DecisionTreeNode:
        """A class representing a decision tree node."""
        def __init__(self):
            self.value = None
            self.feature = None
            self.children_by_feature_value = None


class DecisionTreeClassification(DecisionTree, ClassificationModel):
    """A decision tree classifier model."""
    def __init__(self, max_depth=None, min_observations=2, min_entropy=1e-5, min_info_gain=0,
                 random_feature_selection=False, feature_sample_size_function=math.sqrt):
        DecisionTree.__init__(self, max_depth, min_observations, min_entropy, min_info_gain, random_feature_selection,
                              feature_sample_size_function)
        ClassificationModel.__init__(self)
        self._classes = None

    def _setup(self, labels_sr):
        self._classes = labels_sr.unique()

    def _compute_impurity(self, labels_sr):
        # Entropy.
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
    """A decision tree classifier model."""
    def __init__(self, max_depth=None, min_observations=2, min_variance=1e-2, min_variance_reduction=0,
                 random_feature_selection=False, feature_sample_size_function=math.sqrt):
        DecisionTree.__init__(self, max_depth, min_observations, min_variance, min_variance_reduction,
                              random_feature_selection, feature_sample_size_function)
        RegressionModel.__init__(self)

    def _compute_impurity(self, labels_sr):
        # Variance.
        return labels_sr.var()

    def _compute_leaf_node_value(self, labels_sr):
        return labels_sr.mean()


class BootstrapAggregating(Model):
    """A meta model that trains multiple models on random subsets of the training data set sampled with replacement."""
    def __init__(self, base_model, number_of_models, sample_size_function, number_of_processes):
        if base_model is None or not isinstance(base_model, Model):
            raise ValueError
        if number_of_models is None or number_of_models < 1:
            raise ValueError
        if number_of_processes is None or number_of_processes < 1:
            raise ValueError
        super(BootstrapAggregating, self).__init__()
        self.base_model = base_model
        self.number_of_models = number_of_models
        self.sample_size_function = sample_size_function if sample_size_function is not None else lambda n: n
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
            return [self._make_prediction(model, observations_df) for model in self._models]


class BootstrapAggregatingClassification(BootstrapAggregating, ClassificationModel):
    """A bootstrap aggregating classification meta model."""
    def __init__(self, base_model, number_of_models, sample_size_function=None, number_of_processes=mp.cpu_count()):
        if not isinstance(base_model, ClassificationModel):
            raise ValueError
        BootstrapAggregating.__init__(self, base_model, number_of_models, sample_size_function, number_of_processes)
        ClassificationModel.__init__(self)

    def _aggregate_predictions(self, list_of_predictions_sr):
        return pd.concat(list_of_predictions_sr, axis=1).apply(lambda r: r.value_counts().index[0], axis=1,
                                                               result_type='reduce')


class BootstrapAggregatingRegression(BootstrapAggregating, RegressionModel):
    """A bootstrap aggregating regression meta model."""
    def __init__(self, base_model, number_of_models, sample_size_function=None, number_of_processes=mp.cpu_count()):
        if not isinstance(base_model, RegressionModel):
            raise ValueError
        BootstrapAggregating.__init__(self, base_model, number_of_models, sample_size_function, number_of_processes)
        RegressionModel.__init__(self)

    def _aggregate_predictions(self, list_of_predictions_sr):
        return pd.concat(list_of_predictions_sr, axis=1).apply(lambda r: r.mean(), axis=1, result_type='reduce')


class RandomForestClassification(BootstrapAggregatingClassification):
    """A random forest classification model."""
    def __init__(self, number_of_models, sample_size_function=None, max_depth=None, min_observations=2,
                 min_entropy=1e-5, min_info_gain=0, feature_sample_size_function=math.sqrt,
                 number_of_processes=mp.cpu_count()):
        super(RandomForestClassification, self)\
            .__init__(DecisionTreeClassification(max_depth=max_depth, min_observations=min_observations,
                                                 min_entropy=min_entropy, min_info_gain=min_info_gain,
                                                 random_feature_selection=True,
                                                 feature_sample_size_function=feature_sample_size_function),
                      number_of_models, sample_size_function, number_of_processes)


class RandomForestRegression(BootstrapAggregatingRegression):
    """A random forest regression model."""
    def __init__(self, number_of_models, sample_size_function=None, max_depth=None, min_observations=2,
                 min_variance=1e-2, min_variance_reduction=0, feature_sample_size_function=math.sqrt,
                 number_of_processes=mp.cpu_count()):
        super(RandomForestRegression, self)\
            .__init__(DecisionTreeRegression(max_depth=max_depth, min_observations=min_observations,
                                             min_variance=min_variance, min_variance_reduction=min_variance_reduction,
                                             random_feature_selection=True,
                                             feature_sample_size_function=feature_sample_size_function),
                      number_of_models, sample_size_function, number_of_processes)
