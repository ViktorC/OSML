import numpy as np
import pytest
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import osml.data as osd
import osml.transform as ost


@pytest.mark.parametrize('data_set,tol', [
    (osd.BostonDataSet('data/boston/boston.csv'), 1e-5),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), 1e-5),
    (osd.IrisDataSet('data/iris/iris.csv'), 1e-5)
])
def test_centering(data_set, tol):
    centering = ost.Centering()
    centered_observations = centering.fit_transform(data_set.get_training_observations(), None)
    assert np.allclose(centered_observations.mean(axis=0), 0, atol=tol)


@pytest.mark.parametrize('data_set,mean_tol,sd_tol', [
    (osd.BostonDataSet('data/boston/boston.csv'), 1e-5, 1e-2),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), 1e-5, 1e-2),
    (osd.IrisDataSet('data/iris/iris.csv'), 1e-5, 1e-2)
])
def test_standardization(data_set, mean_tol, sd_tol):
    standardization = ost.Standardization()
    standardized_observations = standardization.fit_transform(data_set.get_training_observations(), None)
    assert np.allclose(standardized_observations.mean(axis=0), 0, atol=mean_tol)
    assert np.allclose(standardized_observations.std(axis=0), 1, atol=sd_tol)


@pytest.mark.parametrize('data_set,whiten,n_components,tol', [
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat', shuffle_data=False), False, 2, 1e-2),
    (osd.IrisDataSet('data/iris/iris.csv', shuffle_data=False), False, 4, 1e-2),
    (osd.IrisDataSet('data/iris/iris.csv', shuffle_data=False), True, 4, 1e-2),
    (osd.IrisDataSet('data/iris/iris.csv', shuffle_data=False), False, 2, 1e-2),
    (osd.IrisDataSet('data/iris/iris.csv', shuffle_data=False), True, 3, 1e-2)
])
def test_pca(data_set, whiten, n_components, tol):
    observations = data_set.get_training_observations()
    pca = ost.PrincipalComponentAnalysis(whiten=whiten, dimensions_to_retain=n_components)
    transformed_observations = pca.fit_transform(observations, None)
    sk_pca = PCA(whiten=whiten, n_components=n_components)
    sk_transformed_observations = sk_pca.fit_transform(observations.values)
    assert np.allclose(np.abs(transformed_observations.values), np.abs(sk_transformed_observations), atol=tol)


@pytest.mark.parametrize('data_set,n_components,tol', [
    (osd.IrisDataSet('data/iris/iris.csv', shuffle_data=False), 2, 5e-1)
])
def test_lda(data_set, n_components, tol):
    observations = data_set.get_training_observations()
    labels = data_set.get_training_labels()
    lda = ost.LinearDiscriminantAnalysis(dimensions_to_retain=n_components)
    transformed_observations = lda.fit_transform(observations, labels)
    sk_lda = LinearDiscriminantAnalysis(n_components=n_components)
    sk_transformed_observations = sk_lda.fit_transform(observations, labels)
    for label in labels.unique():
        relevant_obs = transformed_observations.values[labels == label, :]
        class_mean = relevant_obs.mean(axis=0)
        class_sd = relevant_obs.std(axis=0)
        std_relevant_obs = (relevant_obs - class_mean) / class_sd
        sk_relevant_obs = sk_transformed_observations[labels == label, :]
        sk_class_mean = sk_relevant_obs.mean(axis=0)
        sk_class_sd = sk_relevant_obs.std(axis=0)
        sk_std_relevant_obs = (sk_relevant_obs - sk_class_mean) / sk_class_sd
        assert np.allclose(np.abs(std_relevant_obs), np.abs(sk_std_relevant_obs), atol=tol)


@pytest.mark.parametrize('data_set,tol', [
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat', shuffle_data=False), 1e-2),
    (osd.IrisDataSet('data/iris/iris.csv', shuffle_data=False), 1e-2)
])
def test_box_cox_transform(data_set, tol):
    observations = data_set.get_training_observations()
    box_cox_transform = ost.BoxCoxTransformation()
    transformed_observations = box_cox_transform.fit_transform(observations, None)
    scp_transformed_observations = np.zeros(observations.shape)
    scp_lambdas = []
    for i, column in enumerate(observations.columns):
        observations_i = observations[column]
        scp_transformed_observations_i, lmb = boxcox(observations_i.values)
        scp_transformed_observations[:, i] = scp_transformed_observations_i
        scp_lambdas.append(lmb)
    assert np.allclose(transformed_observations.values, scp_transformed_observations, atol=tol)
