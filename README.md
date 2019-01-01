# OSML [![Build Status](https://travis-ci.org/ViktorC/OSML.svg?branch=master)](https://travis-ci.org/ViktorC/OSML)
A simple Python library of 'old school' machine learning algorithms such as linear regression, logistic regression, naive Bayes, k-nearest neighbors, decision trees, support vector machines, and meta models like bootstrap aggregating and gradient boosting. OSML uses the [pandas](https://pandas.pydata.org/) and [NumPy](http://www.numpy.org/) libraries for manipulating data and performing basic linear algebra operations.

## Models
The primary components of the library are the machine learning models. They can be grouped into three major categories; regression, classification, and binary classification. Some of the models are parametric while others are rule based. There are also a number of meta-models that utilize multiple instances of other models.

The models only operate on numerical data. The possible data types explanatory variables may have are as below:
- Continuous (CN)
- Binary Categorical (BC)
- Ordinal Categorical (OC)
- Nominal Categorical (NC)  

All columns with `float` data type are assumed to represent continuous features, while all `int` columns correspond to categorical features. Every model supports continuous variables but not every model can handle all types of categorical variables. Both binary and ordinal variables work reasonably well with all algorithms, but only a few of them can handle nominal features. The tables below highlight some of the main characteristics of the algorithms offered by the library.

### Regression
| Model                       | Supported Feature Types | Meta                     | Optimization Method                            |
| --------------------------- | ----------------------- | ------------------------ | ---------------------------------------------- |
| LinearRegression            | CN, BC, OC              | :heavy_multiplication_x: | Analytic                                       |
| LinearRidgeRegression       | CN, BC, OC              | :heavy_multiplication_x: | Analytic                                       |
| LinearLassoRegression       | CN, BC, OC              | :heavy_multiplication_x: | Coordinate Descent with Exact Line Search      |
| KNearestNeighborsRegression | CN, BC, OC              | :heavy_multiplication_x: | -                                              |
| DecisionTreeRegression      | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| BaggedTreesRegression       | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| RandomForestRegression      | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| BoostedTreesRegression      | CN, BC, OC, NC          | :heavy_multiplication_x: | Gradient Descent with Backtracking Line Search |
| BootstrappingRegression     | -                       | :heavy_check_mark:       | -                                              |
| GradientBoostingRegression  | -                       | :heavy_check_mark:       | Gradient Descent with Backtracking Line Search |

Regression models predict continuous dependent variables. They can be used, for example, to predict house prices and temperatures. The data type of such predictions is always `float`.

### Binary Classification
| Model                                | Supported Feature Types | Meta                     | Optimization Method                            |
| ------------------------------------ | ----------------------- | ------------------------ | ---------------------------------------------- |
| LogisticRegression                   | CN, BC, OC              | :heavy_multiplication_x: | Newton-Raphson                                 |
| LogisticRidgeRegression              | CN, BC, OC              | :heavy_multiplication_x: | Newton-Raphson                                 |
| BoostedTreesBinaryClassification     | CN, BC, OC, NC          | :heavy_multiplication_x: | Gradient Descent with Backtracking Line Search |
| GradientBoostingBinaryClassification | -                       | :heavy_check_mark:       | Gradient Descent with Backtracking Line Search |

Binary classifiers predict binary dependent variables. The only two values the predictions can take on are 0 and 1. These values usually encode a true or false relationship with 1 meaning that the observation is predicted to belong to the class in question and 0 meaning it is not. They can be applied to problems such as the prediction of whether a mushroom is edible or whether a transaction fraudulent. The data type of these models' predictions is always `int`.

### Classification
| Classification Model            | Supported Feature Types | Meta                     | Optimization Method                            |
| ------------------------------- | ----------------------- | ------------------------ | ---------------------------------------------- |
| NaiveBayes                      | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| KNearestNeighborsClassification | CN, BC, OC              | :heavy_multiplication_x: | -                                              |
| DecisionTreeClassification      | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| BaggedTreesClassification       | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| RandomForestClassification      | CN, BC, OC, NC          | :heavy_multiplication_x: | -                                              |
| BoostedTreesClassification      | CN, BC, OC, NC          | :heavy_multiplication_x: | Gradient Descent with Backtracking Line Search |
| MultiBinaryClassification       | -                       | :heavy_check_mark:       | -                                              |
| BootstrappingClassification     | -                       | :heavy_check_mark:       | -                                              |
| GradientBoostingClassification  | -                       | :heavy_check_mark:       | Gradient Descent with Backtracking Line Search |

Classifiers predict categorical dependent variables. They are used to predict which class an observation belongs to. Examples include the prediction of which animal an image represents or which species of flower a specimen belongs to based on its petal measurements. Classification models make `int` predictions. All classifiers can be used as binary classifiers as well.

## Data
OSML also provides a number of different data sets the models can be tested on. It also includes functions for splitting observations and labels into training and test data sets, random shuffling data, and oversampling. The table below showcases the data sets including the types of their features and their labels.

| Data Set | Default Problem Type  | Feature Types |
| -------- | --------------------- | ------------- |
| Boston   | Regression            | CN, BC        |
| Exam     | Binary Classification | CN            |
| Iris     | Classification        | CN            |
| Titanic  | Binary Classification | CN, BC, OC    |
| Mushroom | Binary Classification | BC            |
| IMDb     | Classification        | CN, BC        |

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
