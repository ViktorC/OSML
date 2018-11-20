import osml.data as osmld
import osml.model as osmlm
import pytest
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.naive_bayes as sknb


@pytest.mark.parametrize('data_set,model,max_test_loss', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), 40.),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(), .2),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), .2),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.NaiveBayes(), .35),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), .2)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    assert model.test(data_set.get_test_observations(), data_set.get_test_labels()) <= max_test_loss


@pytest.mark.parametrize('data_set,model,sk_model,min_accuracy_compared_to_sk', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), sklm.LinearRegression(), .98),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(),
        sklm.LogisticRegression(solver='liblinear'), .8),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), sknb.GaussianNB(), .98),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), sknb.MultinomialNB(), .98)
])
def test_model_accuracy_compared_to_sklearn(data_set, model, sk_model, min_accuracy_compared_to_sk):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_accuracy = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_accuracy = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert sk_model_accuracy == 0 or model_accuracy / sk_model_accuracy >= min_accuracy_compared_to_sk
