# OSML [![Build Status](https://travis-ci.org/ViktorC/OSML.svg?branch=master)](https://travis-ci.org/ViktorC/OSML)
A Python library of 'old school' machine learning algorithms such as linear regression, logistic regression, naive Bayes, k-nearest neighbors, decision trees, support vector machines, and meta models like bootstrap aggregating and gradient boosting. OSML uses the [pandas](https://pandas.pydata.org/) and [NumPy](http://www.numpy.org/) libraries for data manipulation and basic linear algebra operations.

## Models
The primary components of the library are the machine learning models. They can be grouped into two major categories, transformative and predictive models.

### Transformative

Transformative models can be fitted to data sets and be used to perform transformations on new data sets using the parameters learnt from the data sets they were fitted to. The most common use case of such models is pre-processing. Most transformative models use unsupervised learning The transformative models provided by OSML are as follows:

| Model                      | Objectives                                                                |
| -------------------------- | ------------------------------------------------------------------------- |
| Centering                  | zero mean                                                                 |
| Standardization            | zero mean, unit standard deviation                                        |
| PrincipalComponentAnalysis | linearly independent features, dimensionality reduction                   |
| LinearDiscriminantAnalysis | linearly independent features, class separation, dimensionality reduction |
| BoxCoxTransformation       | normal distribution                                                       |

### Predictive

The three main categories of predictive models are regression, classification, and binary classification. Some of the models are parametric while others are rule based, but they are all supervised machine learning models which means they require labels.

Regression models predict continuous dependent variables. They can be used, for example, to predict house prices and temperatures. The data type of such predictions is always `float`.

Binary classifiers predict binary dependent variables. The only two values the predictions can take on are 0 and 1 (after rounding to the closest integer). These values usually encode a true or false relationship with 1 meaning that the observation is predicted to belong to the class in question and 0 meaning it is not. The data type of these models' predictions is always `int`. Use cases of binary classifiers include, for example, the prediction of whether a mushroom is edible or whether a transaction is fraudulent. With some work, binary classifiers can be applied to multinomial classification problems as well. This is done by fitting a binary predictor to each class and when it comes to inference, running the observation through all the binary predictors and selecting the class whose predictor made the most confident positive prediction.

Classifiers predict categorical dependent variables. They are used to predict which class an observation belongs to. Examples include the prediction of which animal an image represents or which species of flower a specimen belongs to based on its petal measurements. Classification models make `int` predictions. All classifiers can be used as binary classifiers as well.

There are also a number of meta-models that utilize multiple instances of other models.

The models only operate on numerical data. The possible data types explanatory variables may have are as below:
- Continuous (CN)
- Binary Categorical (BC)
- Ordinal Categorical (OC)
- Nominal Categorical (NC)  

All columns with `float` data type are assumed to represent continuous features, while all `int` columns correspond to categorical features. Every model supports continuous variables but not every model can handle all types of categorical variables. Both binary and ordinal variables work reasonably well with most  algorithms, but only a few of them can handle nominal features. The tables below highlight some of the main characteristics of the algorithms offered by the library.

| Model                         | Supported Problem Types          | Supported Feature Types |
| ----------------------------- | -------------------------------- | ----------------------- |
| LinearRegression              | Regression                       | CN, BC, OC              |
| LinearRidgeRegression         | Regression                       | CN, BC, OC              |
| LinearLassoRegression         | Regression                       | CN, BC, OC              |
| LogisticRegression            | BinaryClassification             | CN, BC, OC              |
| LogisticRidgeRegression       | BinaryClassification             | CN, BC, OC              |
| NaiveBayes                    | Classification                   | CN, BC, OC, NC          |
| LinearDiscriminantAnalysis    | Classification                   | CN                      |
| QuadraticDiscriminantAnalysis | Classification                   | CN                      |
| KNearestNeighbors             | Classification, Regression       | CN, BC, OC              |
| DecisionTree                  | Classification, Regression       | CN, BC, OC, NC          |
| BaggedTrees                   | Classification, Regression       | CN, BC, OC, NC          |
| RandomForest                  | Classification, Regression       | CN, BC, OC, NC          |
| BoostedTrees                  | BinaryClassification, Regression | CN, BC, OC, NC          |

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
The library has an API similar to that of [sklearn](https://scikit-learn.org/stable/). Data sets generally contain four collections of data; a training data frame, a series of training labels, a test data frame, and a series of test labels. Each model has a `fit` function that takes a data frame of observations and a series of labels as its arguments. The `predict` function can be used on fitted models to make predictions about the labels of a data frame of observations. All models have an `evaluate` function as well that measures the fit of a series of predictions to a series of labels based on some metric. This metric is the root mean squared error for regression models, accuracy for classification models, and the F1 score for binary classification models. Finally, models that optimize a loss function also have a `test` method that measures the error of the predictions of the model according to the loss function.

Binary classification models can also be used to make predictions that are not rounded to the closest integer via the `predict_probabilities` function. Moreover, they can evaluate the precision, recall, and fall-out of predictions using the `evaluate_precision`, `evaluate_recall`, and `evaluate_fall_out` methods. They can also evaluate the receiver operating characteristic curve and the area under it using `evaluate_receiver_operating_characteristic`. The `MultiBinaryClassification` meta-model can turn a binary classifier into a multinomial classifier by fitting a copy of the binary classifier instance to each class of the multinomial data set. Other meta-models such as `BootstrapAggregating` and `GradientBoosting` function as model ensembles using multiple instances of a model to enhance its predictive power.

The code snippet below demonstrates the usage of the library on a simple example of fitting a random forest to a classification data set and using it to make predictions on new observations.

```python
import osml.data as osd
import osml.transform as ost
import osml.predict as osp

data_set = osd.IrisDataSet('data/iris/iris.csv')
transformer = ost.Standardization()
predictor = osp.RandomForestClassification(number_of_models=64)

predictor.fit(transformer.fit_transform(data_set.get_training_observations(), None), data_set.get_training_labels())

predictions = predictor.predict(transformer.transform(data_set.get_test_observations()))
accuracy = predictor.evaluate(predictions, data_set.get_test_labels())
print('accuracy:', accuracy)
```
