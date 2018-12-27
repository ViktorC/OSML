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
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(), 1.),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(), .8),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.GradientBoostingRegression(
        base_model=osmlm.LassoRegression(iterations=100), iterations=10), 41.5),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.BoostedTreesBinaryClassification(iterations=5), .6)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    test_loss = model.test(data_set.get_test_observations(), data_set.get_test_labels())
    assert test_loss <= max_test_loss


@pytest.mark.parametrize('data_set,model,max_inaccuracy', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), 7.),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RidgeRegression(), 7.5),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LassoRegression(), 7.5),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(), .35),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(), .3),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), .2),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.NaiveBayes(), .4),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), .2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.KNearestNeighborsRegression(), 7.),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.KNearestNeighborsClassification(), .4),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.KNearestNeighborsClassification(), .2),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.DecisionTreeClassification(), .01),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.DecisionTreeClassification(), .3),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.DecisionTreeClassification(), .3),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.DecisionTreeRegression(), 6.5),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.MultiBinaryClassification(
        base_binary_classifier=osmlm.LogisticRegression(iterations=100), number_of_processes=3), .2),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BaggedTreesClassification(number_of_models=10), .2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BaggedTreesRegression(number_of_models=5), 6.4),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.RandomForestClassification(number_of_models=10), .2),
    (osmld.IMDBDataSet('data/imdb/ratings.csv'), osmlm.RandomForestClassification(number_of_models=16), .8),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RandomForestRegression(number_of_models=5), 6.2),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BoostedTreesClassification(iterations=10, number_of_processes=3),
        .2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BoostedTreesRegression(iterations=5), 6.5)
])
def test_model_accuracy(data_set, model, max_inaccuracy):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    inaccuracy = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert inaccuracy <= max_inaccuracy


@pytest.mark.parametrize('data_set,model,sk_model,min_accuracy_compared_to_sk,min_abs_accuracy_diff', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), sklm.LinearRegression(), .98, .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RidgeRegression(), sklm.Ridge(), .98, .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LassoRegression(), sklm.Lasso(), .9, .05),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(),
        sklm.LogisticRegression(solver='liblinear'), .9, .05),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), sknb.GaussianNB(), .98, .05),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), sknb.MultinomialNB(), .98, .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.KNearestNeighborsRegression(),
        sknn.KNeighborsRegressor(n_neighbors=7, weights='distance'), .8, .05),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.KNearestNeighborsClassification(),
     sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .6, .15),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.KNearestNeighborsClassification(),
        sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .9, .05),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.DecisionTreeClassification(),
        skt.DecisionTreeClassifier(), .7, .05),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.DecisionTreeClassification(),
        skt.DecisionTreeClassifier(), .7, .05),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.DecisionTreeClassification(), skt.DecisionTreeClassifier(), .7,
        .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.DecisionTreeRegression(), skt.DecisionTreeRegressor(), .7,
        .05),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BaggedTreesClassification(number_of_models=10),
        ske.BaggingClassifier(n_estimators=10), .7, .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BaggedTreesRegression(number_of_models=10),
        ske.BaggingRegressor(n_estimators=10), .7, .05),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.RandomForestClassification(number_of_models=10),
        ske.RandomForestClassifier(n_estimators=10), .7, .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RandomForestRegression(number_of_models=10),
        ske.RandomForestRegressor(n_estimators=10), .7, .05),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BoostedTreesClassification(iterations=10, number_of_processes=3),
        ske.GradientBoostingClassifier(n_estimators=10), .7, .05),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BoostedTreesRegression(iterations=5),
        ske.GradientBoostingRegressor(n_estimators=5), .7, .05)
])
def test_model_accuracy_compared_to_sklearn(data_set, model, sk_model, min_accuracy_compared_to_sk,
                                            min_abs_accuracy_diff):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_accuracy = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_accuracy = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert abs(model_accuracy - sk_model_accuracy) < min_abs_accuracy_diff or model_accuracy == 0 \
        or sk_model_accuracy / model_accuracy >= min_accuracy_compared_to_sk
