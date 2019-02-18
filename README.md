# OSML [![Build Status](https://travis-ci.org/ViktorC/OSML.svg?branch=master)](https://travis-ci.org/ViktorC/OSML)
A Python library of 'old school' machine learning algorithms such as linear regression, logistic regression, naive Bayes, k-nearest neighbors, decision trees, support vector machines, and meta models like bootstrap aggregating and gradient boosting. OSML uses the [pandas](https://pandas.pydata.org/) and [NumPy](http://www.numpy.org/) libraries for data manipulation and basic linear algebra operations.

## Models
The primary components of the library are the machine learning models. They can be grouped into two major categories, transformative and predictive models.

The models of the library only operate on numerical data. The possible data types explanatory variables may have are as below:
- __Continuous__ (CN)
- __Binary Categorical__ (BC)
- __Ordinal Categorical__ (OC)
- __Nominal Categorical__ (NC)  

All columns with `float` data type are assumed to represent continuous features, while all `int` columns correspond to categorical features. Every model supports continuous variables, but not all can handle categorical variables.

### Transformative

Transformative models can be fitted to data sets and be used to perform transformations on new data sets using the parameters learnt from the ones they were fitted to. All of these models assume the features to be continuous. The most common use case of such models is pre-processing. Most transformative models use __unsupervised__ learning, meaning that they do not require labelled training data. The transformative models provided by OSML are as follows.

| Model                      | Supervised | Objectives                                                                | Supported Feature Types |
| -------------------------- | ---------- | ------------------------------------------------------------------------- | ----------------------- |
| Centering                  | No         | zero mean                                                                 | CN                      |
| Standardization            | No         | zero mean, unit standard deviation                                        | CN                      |
| PrincipalComponentAnalysis | No         | linear independence, dimensionality reduction                             | CN                      |
| LinearDiscriminantAnalysis | Yes        | linear independence, class separation, dimensionality reduction           | CN                      |
| BoxCoxTransformation       | No         | univariate normal distribution                                            | CN                      |

### Predictive

All the predictive models provided by the library are __supervised__ machine learning algorithms. Despite this commonality, the algorithms are quite different in many ways. Predictors can be categorized based on several criteria. One such criterion is whether they are parameterized by a fixed number of parameters or not. __Parametric__ models depend on a finite number of parameters regardless of the number of observations in the training data. On the other hand, __non-parametric__ predictive models do not have a fixed number of parameters. However, in spite of what their name suggests, they do have parameters. In fact, the complexity (number of parameters) of non-parametric models grows with the number of observations in the training data.

Another criterion to group predictive algorithms by is the type of predictions they make. Based on this, there are three main categories of predictive models which are described below.

__Regression__ models predict continuous dependent variables. They can be used, for example, to predict house prices and temperatures. The data type of such predictions is always `float`.

__Binary classifiers__ predict binary dependent variables. The only two values the predictions can take on are 0 and 1 (after rounding to the closest integer). These values usually encode a true or false relationship with 1 meaning that the observation is predicted to belong to the class in question and 0 meaning it is not. The data type of these models' predictions is always `int`. Use cases of binary classifiers include, for example, the prediction of whether a mushroom is edible or whether a transaction is fraudulent. With some work, binary classifiers can be applied to multinomial classification problems as well. This is done by fitting a binary predictor to each class and when it comes to inference, running the observation through all the binary predictors and selecting the class whose predictor made the most confident positive prediction.

__Classifiers__ predict categorical dependent variables. They are used to predict which class an observation belongs to. Examples include the prediction of which animal an image represents or which species of flower a specimen belongs to based on its petal measurements. Classification models make `int` predictions. All classifiers can be used as binary classifiers as well.

It is worth noting that classifiers and binary classifiers can be further categorized into the following types.
- __Generative__ (G) 
- __Probabilistic Discriminative__ (PD)
- __Non-probabilistic Discriminative__ (NPD)

Generative classifiers model the joint probability _P(x ∩ y) = P(x | y) * P(y)_ i.e. the product of the marginal probability of the label being _y_ according to a PMF and the conditional probability of an observation taking on the value _x_ given that the label is _y_ according to a multivariate PDF (or the joint probability of the individual features being equal to the corresponding elements of the vector _x_ according to their univariate PDFs or PMFs). This allows for the Bayesian inference of the conditional probability of the label being _y_ given an observation _x_ as in _P(y | x) = P(x ∩ y) / P(x)_. The prediction of the generative model is then _arg<sub>y</sub>max f(y) = P(x ∩ y)_. Note that as _P(x)_ does not depend on _y_, it can be safely ignored

Unlike generative classifiers, probabilistic discriminative classifiers model the conditional probability _P(y | x)_ directly. However, not all discriminative models are probabilistic. Non-probabilistic discriminative classifiers do not estimate distributions but learn mapping functions between observations and labels one way or another.

Most predictive algorithms rely on certain assumptions about the data they are fitted to. For example, naive Bayes assumes that the features are independent (their covariance matrix is diagonal) and the continuous features are normally distributed, k-nearest neighbors assumes that the variance of a feature is proportional to its explanatory power, linear discriminant analysis assumes that all features are normally distributed and the covariance matrix of the features is the same across all classes, etc. It is good to know these assumptions and preprocess the training data accordingly using transformative models.

It is also possible to combine multiple models using a meta-model to improve their predictive power or to use the models for different problem types. For example, a meta-model can employ _n_ binary classifiers to function as a predictor for classification problems with _n_ classes. Meta-models can also utilize an ensemble of weak learners to create a more powerful predictor and help avoid overfitting. Examples of such ensemble algorithms include bootstrap aggregating and gradient boosting. Besides all three of these generic meta-models, OSML features the following predictive models.

| Model                         | Parametric | Supported Problem Types               | Supported Feature Types |
| ----------------------------- | ---------- | ------------------------------------- | ----------------------- |
| LinearRegression              | Yes        | Regression                            | CN, BC, OC              |
| LinearRidgeRegression         | Yes        | Regression                            | CN, BC, OC              |
| LinearLassoRegression         | Yes        | Regression                            | CN, BC, OC              |
| LogisticRegression            | Yes        | BinaryClassification (PD)             | CN, BC, OC              |
| LogisticRidgeRegression       | Yes        | BinaryClassification (PD)             | CN, BC, OC              |
| NaiveBayes                    | Yes        | Classification (G)                    | CN, BC, OC, NC          |
| LinearDiscriminantAnalysis    | Yes        | Classification (G)                    | CN                      |
| QuadraticDiscriminantAnalysis | Yes        | Classification (G)                    | CN                      |
| KNearestNeighbors             | No         | Classification (NPD), Regression      | CN, BC, OC              |
| DecisionTree                  | No         | Classification (NPD), Regression      | CN, BC, OC, NC          |
| BaggedTrees                   | No         | Classification (NPD), Regression      | CN, BC, OC, NC          |
| RandomForest                  | No         | Classification (NPD), Regression      | CN, BC, OC, NC          |
| BoostedTrees                  | No         | BinaryClassification (PD), Regression | CN, BC, OC, NC          |

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

Binary classification models can also be used to make predictions that are not rounded to the closest integer via the `fuzzy_predict` function. Moreover, they can evaluate the precision, recall, and fall-out of predictions using the `evaluate_precision`, `evaluate_recall`, and `evaluate_fall_out` methods. They can also evaluate the receiver operating characteristic curve and the area under it using `evaluate_receiver_operating_characteristic`. The `MultiBinaryClassification` meta-model can turn a binary classifier into a multinomial classifier by fitting a copy of the binary classifier instance to each class of the multinomial data set. Other meta-models such as `BootstrapAggregating` and `GradientBoosting` function as model ensembles using multiple instances of a model to enhance its predictive power.

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
