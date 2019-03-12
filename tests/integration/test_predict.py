import pandas as pd
import pytest
import sklearn.discriminant_analysis as skd
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import sklearn.naive_bayes as sknb
import sklearn.neighbors as sknn
import sklearn.svm as sks
import sklearn.tree as skt

import osml.data as osd
import osml.predict as osp


@pytest.mark.parametrize('data_set,model,max_test_loss', [
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearRegression(), 27.5),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearRidgeRegression(), 50.5),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearLassoRegression(), 31.),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.LogisticRegression(), 0.32),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.LogisticRegression(), 0.44),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.LogisticRidgeRegression(), 0.58),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.SupportVectorMachine(), 17.1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.GradientBoostingRegression(
        base_model=osp.LinearLassoRegression(iterations=100), number_of_models=10), 42.5),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osp.BoostedTreesBinaryClassification(number_of_models=5), .52)
])
def test_model_loss(data_set, model, max_test_loss):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    test_loss = model.test(data_set.get_test_observations(), data_set.get_test_labels())
    assert test_loss <= max_test_loss


@pytest.mark.parametrize('data_set,model,max_error', [
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearRegression(), 5.3),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearRidgeRegression(), 6.3),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearLassoRegression(), 5.6),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.KNearestNeighborsRegression(), 7.9),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.DecisionTreeRegression(), 5.8),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.BaggedTreesRegression(number_of_models=5), 4.4),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.RandomForestRegression(number_of_models=5), 4.9),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.BoostedTreesRegression(number_of_models=5), 4.2)
])
def test_regression_model_error(data_set, model, max_error):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    error = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert error <= max_error


@pytest.mark.parametrize('data_set,model,min_accuracy', [
    (osd.IrisDataSet('data/iris/iris.csv'), osp.NaiveBayes(), .95),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.NaiveBayes(), .78),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osp.NaiveBayes(), .93),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.LinearDiscriminantAnalysis(), .93),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.LinearDiscriminantAnalysis(), .79),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.QuadraticDiscriminantAnalysis(), .91),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.QuadraticDiscriminantAnalysis(), .87),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.KNearestNeighborsClassification(), .95),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.KNearestNeighborsClassification(), .7),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osp.DecisionTreeClassification(), 1.),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.DecisionTreeClassification(), .91),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.DecisionTreeClassification(), .76),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.MultiBinaryClassification(
        base_binary_classifier=osp.LogisticRegression(iterations=100), number_of_processes=3), .95),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.BaggedTreesClassification(number_of_models=10), .91),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.RandomForestClassification(number_of_models=10), .91),
    (osd.IMDBDataSet('data/imdb/ratings.csv'), osp.RandomForestClassification(number_of_models=16), .21)
])
def test_classification_accuracy(data_set, model, min_accuracy):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    accuracy = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert accuracy >= min_accuracy


@pytest.mark.parametrize('data_set,model,min_f1_score', [
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.LogisticRegression(), .8),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.LogisticRegression(), .68),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.LogisticRidgeRegression(), .7),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.SupportVectorMachine(tol=1e-2, c=.1, kernel='lin'), .66),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.SupportVectorMachine(), .54),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osp.BoostedTreesBinaryClassification(number_of_models=5), 1.)
])
def test_binary_classification_f1_score(data_set, model, min_f1_score):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    f1_score = model.evaluate(model.predict(data_set.get_test_observations()), data_set.get_test_labels())
    assert f1_score >= min_f1_score


@pytest.mark.parametrize('data_set,model,sk_model,max_error_compared_to_sk,min_abs_error_diff', [
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearRegression(), skl.LinearRegression(), .8, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearRidgeRegression(), skl.Ridge(), .8, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.LinearLassoRegression(), skl.Lasso(), 1., .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.KNearestNeighborsRegression(),
     sknn.KNeighborsRegressor(n_neighbors=7, weights='distance'), 1.1, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.DecisionTreeRegression(),
     skt.DecisionTreeRegressor(), 1.1, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.BaggedTreesRegression(number_of_models=10),
     ske.BaggingRegressor(n_estimators=10), 1.1, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.RandomForestRegression(number_of_models=10),
     ske.RandomForestRegressor(n_estimators=10), 1.2, .1),
    (osd.BostonDataSet('data/boston/boston.csv'), osp.BoostedTreesRegression(number_of_models=5),
     ske.GradientBoostingRegressor(n_estimators=5), 1.1, .1)
])
def test_regression_error_compared_to_sklearn(data_set, model, sk_model, max_error_compared_to_sk, min_abs_error_diff):
    model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    sk_model.fit(data_set.get_training_observations(), data_set.get_training_labels())
    predictions = model.predict(data_set.get_test_observations())
    sk_predictions = pd.Series(sk_model.predict(data_set.get_test_observations().values), dtype=predictions.dtype)
    model_error = model.evaluate(predictions, data_set.get_test_labels())
    sk_model_error = model.evaluate(sk_predictions, data_set.get_test_labels())
    assert abs(model_error - sk_model_error) < min_abs_error_diff or \
        model_error / sk_model_error <= max_error_compared_to_sk


@pytest.mark.parametrize('data_set,model,sk_model,min_accuracy_compared_to_sk,min_abs_accuracy_diff', [
    (osd.IrisDataSet('data/iris/iris.csv'), osp.NaiveBayes(), sknb.GaussianNB(), 1.2, .05),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osp.NaiveBayes(), sknb.MultinomialNB(), 1.2, .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.LinearDiscriminantAnalysis(), skd.LinearDiscriminantAnalysis(),
     1.2, .05),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.LinearDiscriminantAnalysis(),
     skd.LinearDiscriminantAnalysis(), 1.2, .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.QuadraticDiscriminantAnalysis(), skd.QuadraticDiscriminantAnalysis(),
     1.2, .1),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.QuadraticDiscriminantAnalysis(),
     skd.QuadraticDiscriminantAnalysis(), 1.1, .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.KNearestNeighborsClassification(),
     sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), 1.2, .05),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'),
     osp.KNearestNeighborsClassification(), sknn.KNeighborsClassifier(n_neighbors=7, weights='distance'), 1.2, .05),
    (osd.MushroomDataSet('data/mushroom/mushroom.csv'), osp.DecisionTreeClassification(),
     skt.DecisionTreeClassifier(), 1.2, .05),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.DecisionTreeClassification(),
     skt.DecisionTreeClassifier(), 1., .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.DecisionTreeClassification(), skt.DecisionTreeClassifier(), 1.1, .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.BaggedTreesClassification(number_of_models=10),
     ske.BaggingClassifier(n_estimators=10), 1.2, .05),
    (osd.IrisDataSet('data/iris/iris.csv'), osp.RandomForestClassification(number_of_models=10),
     ske.RandomForestClassifier(n_estimators=10), 1.2, .05)
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
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.LogisticRegression(),
     skl.LogisticRegression(solver='liblinear'), 1.2, .05),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.SupportVectorMachine(tol=1e-2, c=.1, kernel='lin'),
     sks.SVC(tol=1e-2, C=.1, kernel='linear'), 1.2, .05),
    (osd.ExamDataSet('data/exam/ex4x.dat', 'data/exam/ex4y.dat'), osp.SupportVectorMachine(),
     sks.SVC(tol=1e-3, gamma=.1, C=1., kernel='rbf'), 1.2, .05),
    (osd.TitanicDataSet('data/titanic/titanic.csv'), osp.BoostedTreesBinaryClassification(number_of_models=10),
     ske.GradientBoostingClassifier(n_estimators=10), 1.1, .05)
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
