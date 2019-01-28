from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
from six.moves import urllib
import tensorflow as tf
## TF tutorial on building Linear model
'''
TensorFlow   Develop   Tutorials
TensorFlow Linear Model Tutorial

Contents
Setup
Reading The Census Data
Converting Data into Tensors
Selecting and Engineering Features for the Model

In this tutorial, we will use the tf.estimator API in TensorFlow to solve a binary classification problem: Given census data about a person such as age, gender, education and occupation (the features), we will try to predict whether or not the person earns more than 50,000 dollars a year (the target label). We will train a logistic regression model, and given an individual's information our model will output a number between 0 and 1, which can be interpreted as the probability that the individual has an annual income of over 50,000 dollars.

'''
### Setup ###
# Downlaoding the datasets
CSV_COLUMNS = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "gender",
            "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income_bracket"]
df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)

## SInce the task is a binary classifiction problem, we'll construct a label column 
# named 'label' whose value is 1 if imcome is over 50k, 0 otherwise.

train_labels = (df_train['income_bracket'].apply(lambda x: ">50K" in x)).astype(int)
test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)


'''
Next, take a loook at the dataframe and see which columns we can use to predict
the target label. The columns can be grouped into two types-categorical and 
continuous columns:
    A column is called categorical if its value can only be one of the categories in
    a finite set.For example, the native coungtry of a person or the education level are categorical columns. 

    A column is called continuous if its value can be any numerical value in a continuous range. For example, the capital gain of a person is a cojntitouds column.
'''
## Converting the Data into Tensors ###
'''
WHen building a tf.estimator model, the input data is specified by means of an 
input builder function. This builder function will not be called until it is later
passed to tf.estimator.Estimator methods such as train and evaluate. The purpose of 
this funciton is to construct the input data, which is represented in the form of tf.Tensor or tf.SparseTensor. In more detail, the input builder function returns
the following as a pair:
    1. feature_cols: a dict from feature column names to Tensors or SparseTensors
    2. label: a tensor containing the label column.

The keys of the feature_cols will be used to construct columns in the next section. Because we want to call the train and evaluate methods with different data, we define a method that returns an input function based on the given data. Note that the returned input function will be called while constructing the TensorFlow graph, not while running the graph. What it is returning is a representation of the input data as the fundamental unit of TensorFlow computations, a Tensor (or SparseTensor).

We use the tf.estimator.inputs.pandas_input_fn method to create an input function from pandas dataframes. Each continuous column in the train or test dataframe will be converted into a Tensor, which in general is a good format to represent dense data. For categorical data, we must represent the data as a SparseTensor. This data format is good for representing sparse data. Another more advanced way to represent input data would be to construct an Inputs And Readers that represents a file or other data source, and iterates through the file as TensorFlow runs the graph.
'''
def input_fn(data_file, num_epochs, shuffle):
    '''Input builder funciton'''
    df_data = pd.read_csv(
            tf.gfile.Open(data_file),
            names=CSV_COLUMNS, 
            skipinitialspace=True,
            engine='python',
            skiprows=1)

    # remove NaN elements
    df_data = df_data.dropna(how='any', axis=0)
    labels= df_data['income_bracket'].apply(lambda x: '>50K' in x).astype(int)
    return tf.estimator.inputs.pandas_input_fn(
            x=df_data,
            y=labels,
            batch_size=100,
            num_epochs=num_epochs,
            shuffle=shuffle,
            num_threads=5)


### Selecting and Engineering Features for the Model ###
'''
Selecting and crafting the right set of feature columns is key to learning an effective model. A feature column can be either one of the raw columns in the original dataframe (let's call them base feature columns), or any new columns created based on some transformations defined over one or multiple base columns (let's call them derived feature columns). Basically, "feature column" is an abstract concept of any raw or derived variable that can be used to predict the target label.
'''

## Base Categorical Feature Columns # 

'''
To define a feature column for a categorical feature, we can create a 
CategoricalColumn using the tf.feature_column API. If you know the set of all
possible feature values of a column and there are only a few of them, you can use
categorical_column_with_vocabjulary_list. Each key in the list will get assigned 
an auto-incremental ID starting from 0. For example, for the gender column we can 
assgin the feature string 'Female' to an integer ID of 0, and 'Male' to 1 by doing
'''

gender = tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', ['Female', 'Male'])

'''
whagt if we don't know the set of possible values in advance? Not a prob. We can
use categorivcal_column_with_hash_bucket instead:
'''
occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_backet_size=1000)
'''
What will happen is that each possible value in the feature column occupation will
be hashed to an integer ID as we encountre them in training. 
'''
'''
No matther which way we choose to define a SparseColumn, each feature string will
be mapped into an integer ID by looking up a fixed mapping or by hasing. 
Note that hashing collisions are possible, but may not significantly impact the model quality. Under the hood, the LinearModel class is responsible for managing the mapping and creating tf.Variable to store the model parameters (also known as model weights) for each feature ID. The model parameters will be learned through the model training process we'll go through later.
'''
# We'll do the similar trick to define the other categorical features:
education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            "Bachelors", "HS-grad", "11th", "Masters", "9th",
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
            "Preschool", "12th"
            ])

martial_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'martial_status', [
             "Married-civ-spouse", "Divorced", "Married-spouse-absent",
             "Never-married", "Separated", "Married-AF-spouse", "Widowed"
             ])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
            'Other-relative'
            ])

workclass = tf.features_column.categorical_column_with_vocabulary_list(
        'workclass', [
            "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
            "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
            ])

native_country = tf.feature_column.categorical_column_with_hash_bucket(
        'native_country', hash_bucket_size=1000)

### Base Continuous Feature Columns ###
'''
Similary, we can define a NumericColumn for each continuous feature column
that we want to use in the model:
'''
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_columnm.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

### Making Continuous Features Categorical through Bucketization ###

'''
Sometimes the relationship between a continuous feature and the label is not linear. As an hypothetical example, a person's income may grow with age in the early stage of one's career, then the growth may slow at some point, and finally the income decreases after retirement. In this scenario, using the raw age as a real-valued feature column might not be a good choice because the model can only learn one of the three cases:

    1. imcome always increases at some rate as age grows(positive correlation)
    2. imcome always decrease at some rate as age grows(negative correlation)
    3. imcome stays the same no matter at what age(no correlation)

If we want to lean the fine-grained correlation between income and each age group 
separately, we can leverage bucketization. Bucketization is a process of dividing 
the entire nrage of a continuous feature into a set of consecutive bins/buckets, 
and then depending on which bucket that value falls into. So, we can define a 
bucketized_column over age as:
'''
age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


'''
where th eboundaries is a list of bucket boundaries. In this case, there are 10
boundaries, resulting in 11 age goup buckets(from age 17 and below, 18-24, 25-29, ...
65 and over) 
'''
### Intersecting Multiple Columns with CrossedColumn ###

'''
Using each base feature column separately may not be enough to explain the data. For example, the correlation between education and the label (earning > 50,000 dollars) may be different for different occupations. Therefore, if we only learn a single model weight for education="Bachelors" and education="Masters", we won't be able to capture every single education-occupation combination (e.g. distinguishing between education="Bachelors" AND occupation="Exec-managerial" and education="Bachelors" AND occupation="Craft-repair"). To learn the differences between different feature combinations, we can add crossed feature columns to the model.
'''
education_x_occupation = tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000)

'''
We can also create a CrossedColumn over more than two columns. Each constituent 
column can be either a base feature colun that is categorical (SparseColumn), 
a bucketized real-valued feature column(BucketizedColumn), or even another 
CrossColumn. Here's an example:
'''
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

##### Defining the Logistic Regression Model #####
'''
After processing the input data and defining all the feature columns, we're now 
ready to put them all together and build a Logistic Regression model. In the 
previous section we've seen several types of base and derived feature columns, 
including:

    CategoricalColumn
    NumericColumn
    BucketizedColumn
    CrossedColumn

All of these are subclasses of the abstruct FeatureColumn class, and can be added
to the feature_columns field on a model:
'''
base_columns = [
        gender, native_country, education, occupation, workclass, relationship,
        age_buckets,
]

crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            ['native_country','occupation'], hash_bucket_size=1000)]

model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
        model_dir=model_dir, feature_columns=base_columns + crossed_columns)

'''
The model also automatically leaerns a bias term, which congtrols the prediction
one would make without observing any features. The learned model files will be 
stored in model_dir.
'''

### Training and Evaluating Our Model ###
'''
After adding all the features to the model, now let's look at how to actually train 
the model. Training a model is just a one-liner using the tf.estimator API:
'''
# set num_epochs to None to get infinite stream of data.
m.train(
        input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True), 
        steps=train_steps)

'''
After the model is trained, we can evaluate how good our model is at predicting 
the labels of the holdout data:
'''

results = m.evaluate(
        input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False), 
        steps=None)

print("model directory = %s"% model_dir)
for key in sorted(results):
    print('%s: %s' % (key, results[key]))

'''
The The first line of the output should be something like accuracy: 0.83557522, which means the accuracy is 83.6%. Feel free to try more features and transformations and see if you can do even better!
'''








































