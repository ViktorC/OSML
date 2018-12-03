import osml.data as osmld
import osml.model as osmlm
import pytest
import pandas as pd
import sklearn.ensemble as ske
import sklearn.linear_model as sklm
import sklearn.naive_bayes as sknb
import sklearn.neighbors as sknn
import sklearn.tree as skt


@pytest.mark.parametrize('data_set,model,max_test_loss', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), 38.),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RidgeRegression(), 50.),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LassoRegression(), 42.),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(), .7),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(), .7),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.MultiBinaryClassification(osmlm.LogisticRegression(100), 3), .1),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), .2),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.NaiveBayes(), .4),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), .2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.KNearestNeighborsRegression(), 7.),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.KNearestNeighborsClassification(), .72),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.KNearestNeighborsClassification(), .2),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.DecisionTreeClassification(), .1),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.DecisionTreeClassification(), .3),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.DecisionTreeClassification(), .4),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.DecisionTreeRegression(), 7.),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BootstrapAggregatingClassification(
        osmlm.MultiBinaryClassification(osmlm.LogisticRegression(100)), 10), .1),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BootstrapAggregatingRegression(
        osmlm.LinearRegression(), 10), 6.5),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.RandomForestClassification(10), .2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RandomForestRegression(5), 6.2)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    test_loss = model.test(data_set.get_test_observations(), data_set.get_test_labels())
    assert test_loss <= max_test_loss


@pytest.mark.parametrize('data_set,model,sk_model,min_accuracy_compared_to_sk', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), sklm.LinearRegression(), .98),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RidgeRegression(), sklm.Ridge(), .98),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LassoRegression(), sklm.Lasso(), .9),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(),
        sklm.LogisticRegression(solver='liblinear'), .75),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(),
        sklm.LogisticRegression(solver='liblinear'), .9),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), sknb.GaussianNB(), .98),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), sknb.MultinomialNB(), .98),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.KNearestNeighborsRegression(),
        sknn.KNeighborsRegressor(n_neighbors=7, weights='distance'), .9),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'),
        osmlm.KNearestNeighborsClassification(), sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .7),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.KNearestNeighborsClassification(),
        sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .9),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.DecisionTreeClassification(),
        skt.DecisionTreeClassifier(), .95),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.DecisionTreeClassification(),
        skt.DecisionTreeClassifier(), .85),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.DecisionTreeClassification(), skt.DecisionTreeClassifier(), .85),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.DecisionTreeRegression(), skt.DecisionTreeRegressor(), .75),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.RandomForestClassification(number_of_models=10),
        ske.RandomForestClassifier(n_estimators=10), .9),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RandomForestRegression(number_of_models=10),
        ske.RandomForestRegressor(n_estimators=10), .75)
])
def test_model_accuracy_compared_to_sklearn(data_set, model, sk_model, min_accuracy_compared_to_sk):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_accuracy = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_accuracy = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert sk_model_accuracy == 0 or model_accuracy / sk_model_accuracy >= min_accuracy_compared_to_sk
