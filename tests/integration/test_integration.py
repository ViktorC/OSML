import pytest
import osml.data as osd
import osml.model as osm
import pandas as pd
import sklearn.linear_model as skm


@pytest.mark.parametrize('data_set,model,max_test_loss', [
    (osd.BostonHousingDataSet('data/boston/boston.csv'), osm.LinearRegression(), 40.),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.LogisticRegression(), .7)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    assert model.test(data_set.get_test_observations(), data_set.get_test_labels()) <= max_test_loss


@pytest.mark.parametrize('data_set,model,sk_model,min_accuracy_compared_to_sk', [
    (osd.BostonHousingDataSet('data/boston/boston.csv'), osm.LinearRegression(), skm.LinearRegression(), .98),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.LogisticRegression(),
        skm.LogisticRegression(solver='liblinear'), .9)
])
def test_model_accuracy_compared_to_sklearn(data_set, model, sk_model, min_accuracy_compared_to_sk):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values))
    sk_predictions.astype(predictions.dtype)
    model_accuracy = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_accuracy = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert sk_model_accuracy == 0 or model_accuracy / sk_model_accuracy >= min_accuracy_compared_to_sk
