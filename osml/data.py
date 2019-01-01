import datetime as dt
import numpy as np
import pandas as pd


class DataSet:
    """An abstract class representing a data set."""
    def get_training_observations(self) -> pd.DataFrame:
        """Returns a data frame of the training observations.

        Returns:
            A pandas data frame of training data instances without their labels.
        """
        pass

    def get_test_observations(self) -> pd.DataFrame:
        """Returns a data frame of the test observations.

        Returns:
            A pandas data frame of test data instances without their labels.
        """
        pass

    def get_training_labels(self) -> pd.Series:
        """Returns a series of the training observation labels.

        Returns:
            A series of values where each element is the predicted label of the corresponding training observation.
        """
        pass

    def get_test_labels(self) -> pd.Series:
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
    if test_share < 0 or test_share > 1:
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


def oversample(observations_df, labels_sr, shuffle_data=True):
    """Over-samples the data set to address class imbalance. It repeats rows of the data set until all classes are
    featured as many times as the originally most frequent class.

    Args:
        observations_df: A 2D frame of data points where each column is a feature and each row is an observation.
        labels_sr: A series of target values where each element is the label of the corresponding row in the data
        frame of observations.
        shuffle_data: Whether the observation-label pairs are to be randomly shuffled.

    Return:
        The observations data frame and the labels series.
    """
    oversampled_observations_df = None
    oversampled_labels_sr = None
    highest_count = None
    label_value_counts_sr = labels_sr.value_counts()
    for i, count in enumerate(label_value_counts_sr):
        indices = labels_sr[labels_sr == label_value_counts_sr.index[i]].index
        class_labels_sr = labels_sr.reindex(indices).reset_index(drop=True)
        class_observations_df = observations_df.reindex(indices).reset_index(drop=True)
        if not highest_count:
            highest_count = count
            oversampled_labels_sr = class_labels_sr
            oversampled_observations_df = class_observations_df
        else:
            diff = highest_count
            while diff > 0:
                batch = min(count, diff)
                oversampled_labels_sr = oversampled_labels_sr.append(class_labels_sr[:batch]).reset_index(drop=True)
                oversampled_observations_df = oversampled_observations_df.append(class_observations_df.head(batch))\
                    .reset_index(drop=True)
                diff -= batch
    if shuffle_data:
        shuffle(oversampled_observations_df, oversampled_labels_sr)
    return oversampled_observations_df, oversampled_labels_sr


class BostonDataSet(DataSet):
    """The Boston house pricing data set."""
    def __init__(self, data_path, shuffle_data=True, reset_indices=True, test_share=.3, label_column_idx=13):
        complete_df = pd.read_csv(data_path, index_col=False)
        complete_df = complete_df.astype(np.float_)
        complete_df['chas'] = complete_df['chas'].astype(np.byte)
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
        labels_sr = pd.Series(np.loadtxt(label_path), dtype=np.byte)
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
        complete_df['Sex'] = complete_df['Sex'].astype(np.byte)
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
        complete_df = pd.get_dummies(complete_df, columns=complete_df.columns, prefix=complete_df.columns,
                                     dtype=np.byte)
        complete_df = complete_df.drop('type_p', axis=1)
        complete_df = complete_df.drop('bruises_f', axis=1)
        complete_df = complete_df.drop('gill_attachment_a', axis=1)
        complete_df = complete_df.drop('gill_spacing_c', axis=1)
        complete_df = complete_df.drop('gill_size_b', axis=1)
        complete_df = complete_df.drop('stalk_shape_e', axis=1)
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


class IMDBDataSet(DataSet):
    """A data set of movie attributes as data points and user ratings as the labels based on an IMDb ratings history."""
    def __init__(self, data_path, shuffle_data=True, reset_indices=True, oversample_training_data=True,
                 test_share=.2, label_column_idx=0, min_director_occurrences=2, binary=False, positive_rating_cutoff=7):
        observations_df = pd.read_csv(data_path, encoding='ISO-8859-1')
        observations_df = observations_df.drop('Title', axis=1)
        observations_df = observations_df.drop('URL', axis=1)
        observations_df = observations_df.drop('Year', axis=1)
        observations_df.dropna(inplace=True)
        observations_df['Release Date'] = observations_df['Release Date']\
            .apply(lambda e: self.calculate_days_since_release(dt.datetime.strptime(e, "%Y-%m-%d")))
        observations_df['Date Rated'] = observations_df['Date Rated'] \
            .apply(lambda e: self.calculate_days_since_release(dt.datetime.strptime(e, "%Y-%m-%d")))
        observations_df.rename(columns={'Release Date': 'Title Age (years)', 'Date Rated': 'Rating Age (years)',
                                        'Num Votes': 'Votes (1k)'}, inplace=True)
        observations_df['Title Age (years)'] = observations_df['Title Age (years)'].astype(np.float_) / 365.
        observations_df['Rating Age (years)'] = observations_df['Rating Age (years)'].astype(np.float_) / 365.
        observations_df['Votes (1k)'] = observations_df['Votes (1k)'].astype(np.float_) / 1000.
        observations_df = pd.get_dummies(observations_df, columns=["Title Type"], prefix=["Type"], dtype=np.byte)
        genres_dummies_df = observations_df.pop('Genres').str.get_dummies(sep=', ')
        genres_dummies_df = genres_dummies_df.add_prefix('Genre_')
        genres_dummies_df = genres_dummies_df.astype(np.byte)
        observations_df = observations_df.join(genres_dummies_df)
        directors_dummies_df = observations_df.pop('Directors').str.get_dummies(sep=', ')
        directors_to_drop = [c for c in directors_dummies_df.columns
                             if directors_dummies_df[c].sum() < min_director_occurrences]
        directors_dummies_df = directors_dummies_df.drop(directors_to_drop, axis=1)
        directors_dummies_df = directors_dummies_df.add_prefix('Director_')
        directors_dummies_df = directors_dummies_df.astype(np.byte)
        observations_df = observations_df.join(directors_dummies_df)
        labels_sr = observations_df.pop(observations_df.columns[label_column_idx + 1])
        if binary:
            labels_sr = labels_sr.apply(lambda e: 1 if e >= positive_rating_cutoff else 0)
        self.training_observations_df, self.test_observations_df, self.training_labels_sr, self.test_labels_sr = \
            split_train_test(observations_df, labels_sr, test_share, shuffle_data, reset_indices)
        if oversample_training_data:
            self.training_observations_df, self.training_labels_sr = oversample(self.training_observations_df,
                                                                                self.training_labels_sr)
        self.training_observation_ids_sr = self.training_observations_df.pop('Const')
        self.test_observation_ids_sr = self.test_observations_df.pop('Const')

    @staticmethod
    def calculate_days_since_release(release_date):
        date_delta = dt.datetime.now() - release_date
        return max(0, date_delta.days)

    def build_observation(self, imdb_rating, runtime, num_of_votes, release_date, title_type, genres, directors):
        """Creates a single row data frame out of the observation.

        Args:
            imdb_rating: The IMDB rating of the title.
            runtime: The title's runtime in minutes.
            num_of_votes: The number of ratings the title has on IMDB.
            release_date: A date object representing the release date of the title.
            title_type: The type of the title.
            genres: A string of comma separated genres describing the title.
            directors: A string of the comma separated full names of the title's directors.

        Returns:
            A single row observation data frame.
        """
        observation_df = pd.DataFrame(index=[0], columns=self.training_observations_df.columns)
        observation_df = observation_df.fillna(0)
        observation_df = observation_df.astype(np.byte)
        float_features = (imdb_rating, float(runtime) / 60., float(num_of_votes) / 100000.,
                          float(self.calculate_days_since_release(release_date)) / 3650.)
        for i in range(4):
            column = observation_df.columns[i]
            observation_df[column] = observation_df[column].astype(np.float_)
            observation_df.at[0, column] = float_features[i]
        title_type_column = "Type_" + title_type
        if title_type_column in observation_df.columns:
            observation_df.at[0, title_type_column] = 1
        for genre in genres.split(', '):
            genre_column = 'Genre_' + genre
            if genre_column in observation_df.columns:
                observation_df.at[0, genre_column] = 1
        for director in directors.split(', '):
            director_column = 'Director_' + director
            if director_column in observation_df.columns:
                observation_df.at[0, director_column] = 1
        return observation_df

    def get_training_observation_ids(self):
        """Returns a series of IMDb title IDs with each element corresponding to a row of the training observations.

        Returns:
            A pandas series of IMDb IDs.
        """
        return self.training_observation_ids_sr

    def get_test_observation_ids(self):
        """Returns a series of IMDb title IDs with each element corresponding to a row of the test observations.

        Returns:
            A pandas series of IMDb IDs.
        """
        return self.test_observation_ids_sr

    def get_training_observations(self):
        return self.training_observations_df

    def get_test_observations(self):
        return self.test_observations_df

    def get_training_labels(self):
        return self.training_labels_sr

    def get_test_labels(self):
        return self.test_labels_sr
