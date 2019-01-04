import osml.data as osmld
import osml.model as osmlm
import pandas as pd
import pytest
import sklearn.ensemble as ske
import sklearn.linear_model as sklm
import sklearn.naive_bayes as sknb
import sklearn.neighbors as sknn
import sklearn.tree as skt


@pytest.mark.parametrize('data_set,model,max_test_loss', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), 40.),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRidgeRegression(), 52.),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearLassoRegression(), 46.),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(), 1.),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(), .8),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRidgeRegression(), 1.),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.GradientBoostingRegression(
        base_model=osmlm.LinearLassoRegression(iterations=100), number_of_models=10), 44),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.BoostedTreesBinaryClassification(number_of_models=5),
        .6)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    test_loss = model.test(data_set.get_test_observations(), data_set.get_test_labels())
    assert test_loss <= max_test_loss


@pytest.mark.parametrize('data_set,model,max_error', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), 7.2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRidgeRegression(), 7.6),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearLassoRegression(), 7.6),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.KNearestNeighborsRegression(), 7.2),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.DecisionTreeRegression(), 6.8),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BaggedTreesRegression(number_of_models=5), 6.6),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RandomForestRegression(number_of_models=5), 6.6),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BoostedTreesRegression(number_of_models=5), 6.6)
])
def test_regression_model_error(data_set, model, max_error):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    error = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert error <= max_error


@pytest.mark.parametrize('data_set,model,min_accuracy', [
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), .8),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.NaiveBayes(), .6),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), .8),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.KNearestNeighborsClassification(), .6),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.KNearestNeighborsClassification(), .8),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.DecisionTreeClassification(), .9),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.DecisionTreeClassification(), .6),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.DecisionTreeClassification(), .6),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.MultiBinaryClassification(
        base_binary_classifier=osmlm.LogisticRegression(iterations=100), number_of_processes=3), .8),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BaggedTreesClassification(number_of_models=10), .8),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.RandomForestClassification(number_of_models=10), .8),
    (osmld.IMDBDataSet('data/imdb/ratings.csv'), osmlm.RandomForestClassification(number_of_models=16), .15),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BoostedTreesClassification(number_of_models=8,
                                                                               number_of_processes=3), .8)
])
def test_classification_accuracy(data_set, model, min_accuracy):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    accuracy = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert accuracy >= min_accuracy


@pytest.mark.parametrize('data_set,model,min_f1_score', [
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.LogisticRegression(), .6),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(), .6),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRidgeRegression(), .6),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.BoostedTreesBinaryClassification(number_of_models=5),
        .8)
])
def test_binary_classification_f1_score(data_set, model, min_f1_score):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    f1_score = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert f1_score >= min_f1_score


@pytest.mark.parametrize('data_set,model,sk_model,max_error_compared_to_sk,min_abs_error_diff', [
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRegression(), sklm.LinearRegression(), 1.05, .1),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearRidgeRegression(), sklm.Ridge(), 1.05, .1),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.LinearLassoRegression(), sklm.Lasso(), 1.15, .15),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.KNearestNeighborsRegression(),
        sknn.KNeighborsRegressor(n_neighbors=7, weights='distance'), 1.25, .1),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.DecisionTreeRegression(),
        skt.DecisionTreeRegressor(), 1.25, .5),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BaggedTreesRegression(number_of_models=10),
        ske.BaggingRegressor(n_estimators=10), 1.25, .5),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.RandomForestRegression(number_of_models=10),
        ske.RandomForestRegressor(n_estimators=10), 1.25, .5),
    (osmld.BostonDataSet('data/boston/boston.csv'), osmlm.BoostedTreesRegression(number_of_models=5),
        ske.GradientBoostingRegressor(n_estimators=5), 1.25, .5)
])
def test_regression_error_compared_to_sklearn(data_set, model, sk_model, max_error_compared_to_sk, min_abs_error_diff):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_error = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_error = model.evaluate(sk_predictions, data_set.get_test_labels())
    print(model_error)
    print(sk_model_error)
    assert abs(model_error - sk_model_error) < min_abs_error_diff or \
        model_error / sk_model_error <= max_error_compared_to_sk


@pytest.mark.parametrize('data_set,model,sk_model,min_accuracy_compared_to_sk,min_abs_accuracy_diff', [
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.NaiveBayes(), sknb.GaussianNB(), .95, .05),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.NaiveBayes(), sknb.MultinomialNB(), .95, .05),
    (osmld.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osmlm.KNearestNeighborsClassification(),
     sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .8, .1),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.KNearestNeighborsClassification(),
        sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .8, .1),
    (osmld.MushroomDataSet('data/mushroom/mushroom.csv'), osmlm.DecisionTreeClassification(),
        skt.DecisionTreeClassifier(), .8, .1),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.DecisionTreeClassification(),
        skt.DecisionTreeClassifier(), .8, .1),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.DecisionTreeClassification(), skt.DecisionTreeClassifier(), .8, .1),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.BaggedTreesClassification(number_of_models=10),
        ske.BaggingClassifier(n_estimators=10), .8, .1),
    (osmld.IrisDataSet('data/iris/iris.csv'), osmlm.RandomForestClassification(number_of_models=10),
        ske.RandomForestClassifier(n_estimators=10), .8, .1)
])
def test_classification_accuracy_compared_to_sklearn(data_set, model, sk_model, min_accuracy_compared_to_sk,
                                                     min_abs_accuracy_diff):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_accuracy = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_accuracy = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert abs(model_accuracy - sk_model_accuracy) < min_abs_accuracy_diff or \
        model_accuracy / sk_model_accuracy >= min_accuracy_compared_to_sk


@pytest.mark.parametrize('data_set,model,sk_model,min_f1_score_compared_to_sk,min_abs_f1_score_diff', [
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.LogisticRegression(),
        sklm.LogisticRegression(solver='liblinear'), .9, .1),
    (osmld.TitanicDataSet('data/titanic/titanic.csv'), osmlm.BoostedTreesBinaryClassification(number_of_models=10),
        ske.GradientBoostingClassifier(n_estimators=10), .8, .1)
])
def test_binary_classification_f1_score_compared_to_sklearn(data_set, model, sk_model, min_f1_score_compared_to_sk,
                                                            min_abs_f1_score_diff):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_f1_score = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_f1_score = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert abs(model_f1_score - sk_model_f1_score) < min_abs_f1_score_diff or \
        model_f1_score / sk_model_f1_score >= min_f1_score_compared_to_sk
