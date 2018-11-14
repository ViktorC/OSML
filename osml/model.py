import numpy as np
import pandas as pd


class Model(object):
    """An abstract class representing a statistical model or machine learning algorithm."""
    def __init__(self):
        self._observation_types = None
        self._label_type = None
        self._label_name = ''

    def _validate_observations(self, observations_df):
        if len(observations_df.dtypes) == 0 or not observations_df.dtypes.equals(self._observation_types):
            raise ValueError

    def _validate_labels(self, labels_sr):
        if labels_sr.dtype != self._label_type:
            raise ValueError

    def _validate_observations_and_labels(self, observations_df, labels_sr):
        self._validate_observations(observations_df)
        self._validate_labels(labels_sr)
        if len(observations_df.index) != len(labels_sr.index):
            raise ValueError

    @staticmethod
    def _bias_trick(observations_df):
        observations = observations_df.values
        obs_ext = np.ones((observations.shape[0], observations.shape[1] + 1))
        obs_ext[:, :-1] = observations
        return obs_ext

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
        return pd.Series(self._predict(observations_df), name=self._label_name)

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


class LinearRegression(Model):
    """A linear regression model using ordinary least squares."""
    def __init__(self):
        super(Model, self).__init__()
        self.w = None

    @staticmethod
    def _mean_squared_error(predictions, labels):
        return np.square(predictions - labels).mean()

    def _fit(self, observations_df, labels_sr):
        # Take the partial derivatives of the sum of squared errors w.r.t. both parameters of the model (weights and
        # bias), set both derivatives to 0, and solve the resulting system of linear equations. Whether the found roots
        # really minimize the sum of squared errors is not verified here.
        # NOTE: Using the bias trick reduces the parameters to a single vector and significantly simplifies the
        # analytic solution.
        observations = Model._bias_trick(observations_df)
        observations_trans = observations.transpose()
        self.w = np.linalg.inv(observations_trans @ observations) @ (observations_trans @ labels_sr.values)

    def _predict(self, observations_df):
        return Model._bias_trick(observations_df) @ self.w

    def _test(self, observations_df, labels_sr):
        return LinearRegression._mean_squared_error(self._predict(observations_df), labels_sr.values)

    def _evaluate(self, predictions_sr, labels_sr):
        return LinearRegression._mean_squared_error(predictions_sr.values, labels_sr.values)


class LogisticRegression(Model):
    """A logistic regression model using the Newton-Raphson method."""
    def __init__(self, iterations=100, epsilon=1e-8):
        super(Model, self).__init__()
        self.iterations = iterations
        self.epsilon = epsilon
        self.w = None

    def _logistic_function(self, observations):
        return np.reciprocal(np.exp(-observations @ self.w) + 1)

    def _fit(self, observations_df, labels_sr):
        # Use Newton's method to numerically approximate the root of the log likelihood function's derivative.
        observations = Model._bias_trick(observations_df)
        observations_trans = observations.transpose()
        labels = labels_sr.values
        self.w = np.zeros(observations.shape[1], )
        for i in range(self.iterations):
            predictions = self._logistic_function(observations)
            first_derivative = observations_trans @ (predictions - labels)
            # If the number of parameters is N, the Hessian will take the shape of an NxN matrix.
            second_derivative = (observations_trans * (predictions * (1 - predictions))) @ observations
            self.w = self.w - np.linalg.inv(second_derivative) @ first_derivative

    def _predict(self, observations_df):
        return np.rint(self._logistic_function(Model._bias_trick(observations_df)))

    def _test(self, observations_df, labels_sr):
        labels = labels_sr.values
        predictions = self._logistic_function(Model._bias_trick(observations_df))
        return (-np.log(predictions) * labels - np.log(1 - predictions) * (1 - labels)).mean()

    def _evaluate(self, predictions_sr, labels_sr):
        return len([p for i, p in enumerate(predictions_sr) if p == labels_sr[i]]) / len(predictions_sr)
