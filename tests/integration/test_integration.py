import osml.data as osd
import osml.model as osm
import pandas as pd
import pytest
import sklearn.discriminant_analysis as skd
import sklearn.ensemble as ske
import sklearn.linear_model as sklm
import sklearn.naive_bayes as sknb
import sklearn.neighbors as sknn
import sklearn.tree as skt


@pytest.mark.parametrize('data_set,model,max_test_loss', [
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearRegression(), 40.),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearRidgeRegression(), 52.),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearLassoRegression(), 46.),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.LogisticRegression(), 1.),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.LogisticRegression(), .8),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.LogisticRidgeRegression(), 1.),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.GradientBoostingRegression(
        base_model=osm.LinearLassoRegression(iterations=100), number_of_models=10), 44),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osm.BoostedTreesBinaryClassification(number_of_models=5), .6)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    test_loss = model.test(data_set.get_test_observations(), data_set.get_test_labels())
    assert test_loss <= max_test_loss


@pytest.mark.parametrize('data_set,model,max_error', [
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearRegression(), 7.2),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearRidgeRegression(), 7.6),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearLassoRegression(), 7.6),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.KNearestNeighborsRegression(), 7.2),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.DecisionTreeRegression(), 6.8),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.BaggedTreesRegression(number_of_models=5), 6.6),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.RandomForestRegression(number_of_models=5), 6.6),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.BoostedTreesRegression(number_of_models=5), 6.6)
])
def test_regression_model_error(data_set, model, max_error):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    error = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert error <= max_error


@pytest.mark.parametrize('data_set,model,min_accuracy', [
    (osd.IrisDataSet('data/iris/iris.csv'), osm.NaiveBayes(), .8),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.NaiveBayes(), .6),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osm.NaiveBayes(), .8),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.LinearDiscriminantAnalysis(), .8),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.LinearDiscriminantAnalysis(), .6),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.QuadraticDiscriminantAnalysis(), .8),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.QuadraticDiscriminantAnalysis(), .6),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.KNearestNeighborsClassification(), .8),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.KNearestNeighborsClassification(), .6),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osm.DecisionTreeClassification(), .9),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.DecisionTreeClassification(), .6),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.DecisionTreeClassification(), .6),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.MultiBinaryClassification(
        base_binary_classifier=osm.LogisticRegression(iterations=100), number_of_processes=3), .8),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.BaggedTreesClassification(number_of_models=10), .8),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.RandomForestClassification(number_of_models=10), .8),
    (osd.IMDBDataSet('data/imdb/ratings.csv'), osm.RandomForestClassification(number_of_models=16), .15)
])
def test_classification_accuracy(data_set, model, min_accuracy):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    accuracy = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert accuracy >= min_accuracy


@pytest.mark.parametrize('data_set,model,min_f1_score', [
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.LogisticRegression(), .6),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.LogisticRegression(), .6),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.LogisticRidgeRegression(), .6),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osm.BoostedTreesBinaryClassification(number_of_models=5), .8)
])
def test_binary_classification_f1_score(data_set, model, min_f1_score):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    f1_score = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert f1_score >= min_f1_score


@pytest.mark.parametrize('data_set,model,sk_model,max_error_compared_to_sk,min_abs_error_diff', [
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearRegression(), sklm.LinearRegression(), 1.05, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearRidgeRegression(), sklm.Ridge(), 1.05, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.LinearLassoRegression(), sklm.Lasso(), 1.15, .15),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.KNearestNeighborsRegression(),
     sknn.KNeighborsRegressor(n_neighbors=7, weights='distance'), 1.25, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.DecisionTreeRegression(),
     skt.DecisionTreeRegressor(), 1.25, .5),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.BaggedTreesRegression(number_of_models=10),
     ske.BaggingRegressor(n_estimators=10), 1.25, .5),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.RandomForestRegression(number_of_models=10),
     ske.RandomForestRegressor(n_estimators=10), 1.25, .5),
    (osd.BostonDataSet('data/boston/boston.csv'), osm.BoostedTreesRegression(number_of_models=5),
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
    (osd.IrisDataSet('data/iris/iris.csv'), osm.NaiveBayes(), sknb.GaussianNB(), .95, .05),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osm.NaiveBayes(), sknb.MultinomialNB(), .95, .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.LinearDiscriminantAnalysis(), skd.LinearDiscriminantAnalysis(),
     .9, .075),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.LinearDiscriminantAnalysis(),
     skd.LinearDiscriminantAnalysis(), .9, .075),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.QuadraticDiscriminantAnalysis(), skd.QuadraticDiscriminantAnalysis(),
     .9, .1),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.QuadraticDiscriminantAnalysis(),
     skd.QuadraticDiscriminantAnalysis(), .8, .1),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.KNearestNeighborsClassification(),
     sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .8, .1),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osm.KNearestNeighborsClassification(),
     sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), .8, .1),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osm.DecisionTreeClassification(),
     skt.DecisionTreeClassifier(), .8, .1),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.DecisionTreeClassification(),
     skt.DecisionTreeClassifier(), .8, .1),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.DecisionTreeClassification(), skt.DecisionTreeClassifier(), .8, .1),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.BaggedTreesClassification(number_of_models=10),
     ske.BaggingClassifier(n_estimators=10), .8, .1),
    (osd.IrisDataSet('data/iris/iris.csv'), osm.RandomForestClassification(number_of_models=10),
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
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.LogisticRegression(),
     sklm.LogisticRegression(solver='liblinear'), .9, .1),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osm.BoostedTreesBinaryClassification(number_of_models=10),
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
