# Crowd Sourced Mapping using Random Forest Classification

# SCOPE

This report aims to analyze and classify the data set containing NDVI
(Normalized Difference Vegetation Index) values between January 2014 and
July 2015. This report covers the automatic classification for land
cover which uses the Random Forest classification algorithm.
Furthermore, the RF classification is applied to limited features by
using the mechanism of the principal components of the above data
mentioned. Python Data Analysis software package is used for the
preliminary treatment of data and further classification of the data
set.

The dataset used for analysis was derived from geospatial data from two
sources: 1) Landsat time-series satellite imagery from the years
2014-2015, and 2) crowdsourced georeferenced polygons with land cover
labels obtained from OpenStreetMap. The crowdsourced polygons cover only
a small part of the image area and are used to extract training data
from the image for classifying the rest of the image.

The target attribute of this dataset is land cover class, which has 6
classes to classify impervious, farm, forest, grass, orchard, water and
has 10,545 cases.

The attributes used for classification has 28 features which includes
the maximum NDVI value of the class max\_ndvi and 27 NDVI values on
different dates between January 2014 and July 2015 denoted as
YYYYMMDD\_N (e.g., 20150720\_N).

# PRELIMINARY TREATMENT OF DATASET

The above dataset mentioned consists of 6 classes and the following are
the preliminary size of each class

  - Impervious – 969 cases

  - Farm – 1441 cases

  - Forest – 7431 cases

  - Grass – 446 cases

  - Orchard – 53 cases

  - Water – 205 cases

The above size shows that the Forest class has huge traces and Orchard
with the least number of traces which is negligible when compared to
Forest cases. The remained classes have lower number cases compared to
the same. This shows that the dataset is not balanced, so SMOTE
technique is used to rebalance the dataset for further classification.

Initially the dataset is divided into 6 different datasets with respect
to the classes and then split to 80%, 20% proportions for train and test
data for all datasets respectively. Merged the datasets denoted as TRAIN
and TEST and following are the results obtained

|            | \# Of cases in TRAIN | \# Of cases in TEST |
| ---------- | -------------------- | ------------------- |
| Impervious | 775                  | 194                 |
| Farm       | 1152                 | 289                 |
| Forest     | 5944                 | 1487                |
| Grass      | 356                  | 90                  |
| Orchard    | 42                   | 11                  |
| Water      | 164                  | 41                  |
| Total      | 8433                 | 2112                |

The above table shows the imbalance of the cases for all the classes, so
SMOTE is applied for further analysis and the following describes how
the SMOTE is applied for the TRAIN and TEST dataset.

# SMOTE FOR REBALANCING TRAIN AND TEST DATA SETS

The above proportionality shows that the resultant split data has
imbalanced proportionality of the high and low cases, to rebalance SMOTE
technique is used for further process.

SMOTE – Synthetic Minority Oversampling Technique, in general, SMOTE
works by selecting examples that are close in the feature space, drawing
a line between the examples in the feature space, and drawing a new
sample at a point along that line.

As the Water, Orchard, and Grass classes have smaller cases compared to
other classes, this may affect the classification after the SMOTE, so
these cases are removed from the current TRAIN and TEST set for better
classification results.

After applying SMOTE for our current TRAIN and TEST data sets, the
resultant data frames are rebalanced with equal cases of all classes.
The order of each class is 5944 cases for TRAIN and the resultant XTRAIN
has 17,832 cases, 1722 for TEST and the XTEST has 4461 cases. YTRAIN and
YTEST are the target attributes for both training and testing datasets
respectively.

# RANDOM FOREST

# RANDOM FOREST FUNCTION FOR PYTHON

In Python, RandomForestClassifier () function from the
‘sklearn.ensemble’ package is used to create a random forest model.
The following are the parameters used for the ZTRAIN and ZTEST data sets
for better results.

  - **n\_estimators:** Number of decision trees to be created in the
    model.

  - **criterion:** By default, the Gini index is used to split the
    branch while making a decision.

  - **max\_features:** Maximum features to be considered while the
    decision split is happening.

  - **min\_samples\_split:** Minimum samples required to split an
    internal node.

  - **random\_state:** Controls both the randomness of the bootstrapping
    of the samples used when building trees and the sampling of the
    features to consider when looking for the best split at each node

  - **max\_samples:** If bootstrap is True, the number of samples to
    draw from X to train each base estimator

  - **oob\_score:** Whether to use out-of-bag samples to estimate the
    generalization score. Only available if bootstrap=True.

> Methods involved for the above RandomForestClassifier function

  - **apply:** Apply trees in the forest to X, return leaf indices.

  - **decision\_path:** Return the decision path in the forest.

  - **fit:** Build a forest of trees from the training set (x, y).

  - **get\_params:** Get parameters for this estimator.

  - **predict:** Predict class for X.

  - **predict\_log\_proba:** Predict class log-probabilities for X.

  - **predict\_proba:** Predict class probabilities for X.

  - **score:** Return the mean accuracy on the given test data and
    labels.

  - **set\_params:** Set the parameters of this estimator.

# RANDOM FOREST CLASSIFICATION FOR XTRAIN

The above classification function is applied for the current XTRAIN and
XTEST datasets. The following are the parameters are used to create a RF
model,

  - > n\_estimators = 100, 200, 300, 400, 500

  - > criterion = “Gini” (default)

  - > max\_features = 5 (≈ √28, number of original features)

  - > min\_samples\_split = 3

  - > oob\_score = True

The above function creates a model with n\_estimators decision trees,
this model is computed to fit the XTRAIN, which is used to train the
dataset for classification, and predict function is used to test the
XTEST.

As this method is applied for multiple decision trees, the best
classification results are seen in the 400-decision tree model. So, this
model is considered a better model when the whole training set is used
for classification.

For this model, the accuracy score for XTEST data is 95.96% and the
XTRAIN data set is 100%, the computation time for the above
classification model is 19.39 seconds. The following is the confusion
matrix for XTRAIN and XTEST datasets. The OOB score for the current
model is 99.02%

<table>
<thead>
<tr class="header">
<th><p>RF(XTEST)</p>
<p>Ntrees = 400</p></th>
<th colspan='4'>Actual Classes</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<th rowspan='4'>Predicted classes</th>
<td>In Percentages/ Classes</td>
<td>Farm</td>
<td>Forest</td>
<td>Impervious</td>
</tr>
<tr class="even">
<td>Farm</td>
<td>92.33</td>
<td>3.96</td>
<td>3.69</td>
</tr>
<tr class="odd">
<td>Forest</td>
<td>0.87</td>
<td>98.92</td>
<td>0.2</td>
</tr>
<tr class="even">
<td>Impervious</td>
<td>1.34</td>
<td>2.01</td>
<td>96.64</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr class="header">
<th><p>RF(XTRAIN)</p>
<p>Ntrees = 400</p></th>
<th colspan='4'>Actual Classes</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<th rowspan='4'>Predicted classes</th>
<td>In Percentages/ Classes</td>
<td>Farm</td>
<td>Forest</td>
<td>Impervious</td>
</tr>
<tr class="even">
<td>Farm</td>
<td>100</td>
<td>0</td>
<td>0</td>
</tr>
<tr class="odd">
<td>Forest</td>
<td>0</td>
<td>100</td>
<td>0</td>
</tr>
<tr class="even">
<td>Impervious</td>
<td>0</td>
<td>0</td>
<td>100</td>
</tr>
</tbody>
</table>

PCA analysis is applied for current dataset to verify the results and
compare with the current model in further analysis.

# PRINCIPAL COMPONENT ANALYSIS FOR DATA SET

Principal Component Analysis, or PCA, is a dimensionality-reduction
method that is often used to reduce the dimensionality of large data
sets, by transforming a large set of variables into a smaller one that
still contains most of the information in the large set. Smaller data
sets are easier to explore and visualize and make analyzing data much
easier and faster for machine learning algorithms without extraneous
variables to process.

# STANDARDIZATION

Standardization aims to rescale the range of the continuous initial
variables so that each one of them contributes equally to the analysis.
If there are large differences between the ranges of initial variables,
those variables will be dominated on the smaller range variables and
this leads to biased results, so the standardization transforms the data
into comparable scales.

\[\text{rescaled}\left( \text{xi} \right) = \ \frac{xi - mean(x)}{standard\ deviation\ (x)}\]

# CORRELATION MATRIX COMPUTATION

The correlation matrix aims to compute the relationship between any 2
features, this results in a symmetric matrix of n features, since the
current DATA set has 400 features, the resultant correlation matrix
order will be 400 x 400 matrix and each element will be positive or
negative value ranges (0,1).

# EIGENVALUES AND VECTORS FOR CORRELATION

The percentage of Explained Variance is used to determine the principal
components of the data set, this can be computed from the cumulative sum
of eigenvalues (These eigenvalues and vectors are computed from the
above correlation matrix), the ‘r’ principal component least cumulative
sum of eigenvalues.

The following is the graphical representation of the PEV curve for the
current DATA set, the PEVs will be in increasing order, i.e., PEV (1) \<
PEV (2) \< PEV (3) \<… \< PEV (r) \< …\< PEV (28)

![Chart, line chart Description automatically
generated](media/image1.png)

At a 90% level of PEV, the least ‘r’ will be 20, this determines that
90% of the DATA set information can be derived from the initial 20
columns of the eigenvector matrix (Eigen vector-matrix will have the
same order as the correlation matrix). So, the 28 x 20 eigen
vector-matrix gives the first ‘r’ principal components of the current
data set, denoted as PCDATA.

# RECAST THE TRAINING AND TESTING DATA 

To attain certain features of the TRAIN and TEST data sets, the data
sets are computed for matrix multiplication with the above PCDATA
results recast data which has 90% information of the whole dataset. The
resultant TRAIN and TEST dataset has 20(r) features, and these are
denoted as ZTRAIN and ZTEST.

The order of the ZTRAIN dataset – 7871 x 20

The order of ZTEST dataset – 1970 x 20

Final Data Set = Data Set x Principal Component matrix (\~90%)

Again, the above-mentioned SMOTE technique is applied for both ZTRAIN
and ZTEST to rebalance the cases of each class and these datasets are
used for further classification.

# RANDOM FOREST FOR ZTRAIN DATASET

The above classification function is applied for the current XTRAIN and
XTEST datasets. The following are the parameters used to create an RF
model,

  - > n\_estimators = 100, 200, 300, 400, 500

  - > criterion = “Gini” (default)

  - > max\_features = 4 (≈ √20, r from the least PEV for the current
    > dataset)

  - > min\_samples\_split = 3

The above function creates a model with n\_estimators decision trees,
this model is computed to fit the ZTRAIN, which is used to train the
dataset for classification, and predict function is used to test the
ZTEST.

As this method is applied for multiple decision trees, the best
classification results are seen in the 400-decision tree model. So, this
model is considered better when the whole training set is used for
classification.

For this model, the accuracy score for ZTEST data is 93.97% and the
ZTRAIN data set is 100%, the computation time for the above
classification model is 25.31 seconds. The following is the confusion
matrix for ZTRAIN and ZTEST datasets. The OOB score for the model is
98.83%

<table>
<thead>
<tr class="header">
<th><p>RF(XTEST)</p>
<p>Ntrees = 400</p></th>
<th colspan='4'>Actual Classes</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<th rowspan='4'>Predicted classes</th>
<td>In Percentages/ Classes</td>
<td>Farm</td>
<td>Forest</td>
<td>Impervious</td>
</tr>
<tr class="even">
<td>Farm</td>
<td>91.05</td>
<td>7.06</td>
<td>1.88</td>
</tr>
<tr class="odd">
<td>Forest</td>
<td>0.87</td>
<td>98.85</td>
<td>0.26</td>
</tr>
<tr class="even">
<td>Impervious</td>
<td>5.24</td>
<td>2.76</td>
<td>91.99</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr class="header">
<th><p>RF(XTRAIN)</p>
<p>Ntrees = 400</p></th>
<th colspan='4'>Actual Classes</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<th rowspan='4'>Predicted classes</th>
<td>In Percentages/ Classes</td>
<td>Farm</td>
<td>Forest</td>
<td>Impervious</td>
</tr>
<tr class="even">
<td>Farm</td>
<td>100</td>
<td>0</td>
<td>0</td>
</tr>
<tr class="odd">
<td>Forest</td>
<td>0</td>
<td>100</td>
<td>0</td>
</tr>
<tr class="even">
<td>Impervious</td>
<td>0</td>
<td>0</td>
<td>100</td>
</tr>
</tbody>
</table>

# RF\* DISCUSSION

When the accuracy of the testing sets from the above model are compare,
we can see that the original training set has better results and the
computation time is faster that the training set after the PCA analysis,
this shows that the PCA is not required as the dataset has the minimal
number of features 28 after PCA is applied the dataset is having 20
features and the model shows more misclassification in the test dataset.
By considering the above information the RF\* can be the model for the
original dataset and this model is used for further analysis
information.

# FEATURE IMPORTANCE FOR RF\*

The feature/variable importance describes which features are relevant.
This is calculated as the decrease in node impurity weighted by the
probability of reaching that node. The importance is the built-in method
of random forest package. The feature's importance for the current model
with 400 decision trees is as follows

![Table Description automatically generated](media/image2.png)

The above plot determines the top feature importance that made in
decision trees for the current model, this shows that the ‘max\_ndvi’
feature has the highest importance for the current model with 16.89%.
