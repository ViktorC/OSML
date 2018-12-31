# OSML [![Build Status](https://travis-ci.org/ViktorC/OSML.svg?branch=master)](https://travis-ci.org/ViktorC/OSML)
A simple Python library of 'old school' machine learning algorithms such as linear regression, logistic regression, naive Bayes, k-nearest neighbors, decision trees, support vector machines, and meta models like bootstrap aggregating and gradient boosting. OSML uses the [pandas](https://pandas.pydata.org/) and [NumPy](http://www.numpy.org/) libraries for manipulating data and performing basic linear algebra operations.

## Usage
The library has an API similar to that of sklearn. Data sets generally contain four collections of data; a training data frame, a series of training labels, a test data frame, and a series of test labels. Each model has a `fit` function that takes a data frame of observations and a series of labels as its arguments. The `predict` function can be used on fitted models to make predictions about the labels of a data frame of observations. All models have an `evaluate` function as well that measures the fit of a series of predictions to a series of labels based on some metric. This metric is the root mean squared error for regression models, accuracy for classification models, and the F1 score for binary classification models. Finally, models that optimize a loss function also have a `test` method that measures the error of the predictions of the model according to the loss function.

Binary classification models can also be used to make predictions that are not rounded to the closest integer via the `predict_probabilities` function. Moreover, they can evaluate the precision and recall of predictions using the `evaluate_precision` and `evaluate_recall` methods. The `MultiBinaryClassification` meta-model can turn a binary classifier into a multi-nomial classifier by fitting a copy of the binary classifier instance to each class of the multi-nomial data set. Other meta-models such as `BootstrapAggregating` and `GradientBoosting` function as model ensembles using multiple instances of a model to enhance its predictive power.

The code snippet below demonstrates the usage of the library on a simple example of fitting a random forest to a classification data set and using it to make predictions on new observations.

```python
import osml.data as osd
import osml.model as osm

data_set = osd.IrisDataSet('data/iris/iris.csv')
model = osm.RandomForestClassification(number_of_models=50)
model.fit(data_set.get_training_observations(), data_set.get_training_observations())

predictions = model.predict(data_set.get_test_observations())
accuracy = model.evaluate(predictions, data_set.get_test_labels())
print('accuracy:', accuracy)
```

## Data
| Data Set | Default Label Type | Feature Types |
| -------- | ------------------ | ------------- |
| Boston   | CN                 | CN, BC        |
| Exam     | BC                 | CN            |
| Iris     | CT                 | CN            |
| Titanic  | BC                 | CN, BC, OC    |
| Mushroom | BC                 | BC            |
| IMDb     | CT                 | CN, BC        |

CN = Continuous  
CT = Categorical  
BC = Binary Categorical  
OC = Ordinal Categorical  
NC = Nominal Categorical  

## Models
| Model                                | Prediction Type | Supported Feature Types | Optimization Method                              | Meta                     |
| ------------------------------------ | --------------- | ----------------------- | ----------------------------------------------- | ------------------------ |
| LinearRegression                     | CN              | CN, BC, OC              | Analytic                                        | :heavy_multiplication_x: |
| LinearRidgeRegression                | CN              | CN, BC, OC              | Analytic                                        | :heavy_multiplication_x: |
| LinearLassoRegression                | CN              | CN, BC, OC              | Coordinate Descent with Exact Line Search        | :heavy_multiplication_x: |
| LogisticRegression                   | BC              | CN, BC, OC              | Newton-Raphson                                  | :heavy_multiplication_x: |
| LogisticRidgeRegression              | BC              | CN, BC, OC              | Newton-Raphson                                  | :heavy_multiplication_x: |
| NaiveBayes                           | CT              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| KNearestNeighborsRegression          | CN              | CN, BC, OC              | -                                                | :heavy_multiplication_x: |
| KNearestNeighborsClassification      | CT              | CN, BC, OC              | -                                                | :heavy_multiplication_x: |
| DecisionTreeRegression               | CN              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| DecisionTreeClassification           | CT              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| BaggedTreesRegression                | CN              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| BaggedTreesClassification            | CT              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| RandomForestRegression               | CN              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| RandomForestClassification           | CT              | CN, BC, OC, NC          | -                                                | :heavy_multiplication_x: |
| BoostedTreesRegression               | CN              | CN, BC, OC, NC          | Gradient Descent with Backtracking Line Search | :heavy_multiplication_x: |
| BoostedTreesBinaryClassification     | BC              | CN, BC, OC, NC          | Gradient Descent with Backtracking Line Search | :heavy_multiplication_x: |
| BoostedTreesClassification           | CT              | CN, BC, OC, NC          | Gradient Descent with Backtracking Line Search | :heavy_multiplication_x: |
| MultiBinaryClassification            | CT              | -                       | -                                                | :heavy_check_mark:       |
| BootstrappingRegression              | CN              | -                       | -                                                | :heavy_check_mark:       |
| BootstrappingClassification          | CT              | -                       | -                                                | :heavy_check_mark:       |
| GradientBoostingRegression           | CN              | -                       | Gradient Descent with Backtracking Line Search | :heavy_check_mark:       |
| GradientBoostingBinaryClassification | BC              | -                       | Gradient Descent with Backtracking Line Search | :heavy_check_mark:       |
| GradientBoostingClassification       | CT              | -                       | Gradient Descent with Backtracking Line Search | :heavy_check_mark:       |
