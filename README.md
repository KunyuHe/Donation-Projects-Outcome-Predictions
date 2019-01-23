Donation Projects Outcome Predictions
=======================================

## 1. Executive Summary


## 2. Introduction

**Directly from the project [Overview](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose):**

*DonorsChoose.org is an online charity that makes it easy to help students in need through school donations. At any time, thousands of teachers in K-12 schools propose projects requesting materials to enhance the education of their students. When a project reaches its funding goal, they ship the materials to the school.*

*The 2014 KDD Cup asks participants to help DonorsChoose.org identify projects that are exceptionally exciting to the business, at the time of posting. While all projects on the site fulfill some kind of need, certain projects have a quality above and beyond what is typical. By identifying and recommending such projects early, they will improve funding outcomes, better the user experience, and help more students receive the materials they need to learn.*

**For this project, the challenge would instead be predicting whether a proposed project would get fully funded.**

Getting fully funded is one of the requirements of being "exciting" to DonorsChoose.org, hence a precise prediction of whether future proposed projects can be fully funded would be an important building block for finding "exciting" projects.


## 3. Data

### a). Introduction to Data

**Directly from the project [Data Description](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data):**

*The data is provided in a relational format and split by dates. Any project posted prior to 2014-01-01 is in the training set (along with its funding outcomes). Any project posted after is in the test set. Some projects in the test set may still be live and are ignored in the scoring. We do not disclose which projects are still live to avoid leakage regarding the funding status.*

Basically there are five data sets, namely `projects`, `resources`, `essays`, `donations` and `outcomes`:

* `projects` - contains information about each project. This is provided for both the training and test set.
* `resources` - contains information about the resources requested for each project. This is provided for both the training and test set.
* `essays` - contains project text posted by the teachers. This is provided for both the training and test set.
* `donations` - contains information about the donations to each project. This is only provided for projects in the training set.
* `outcomes` - contains information about the outcomes of projects in the training set.

Since `outcomes` is only provided for the training set and access to test set is blocked by Kaggle. Information on whether a project posted after 2014-01-01 got fully funded is not available. **Hence we only use the original training set and split it into new training and test set.**

Notice that once we know the total price a project required, combined with total donations the project got, predicting `fully_funded` would be meaningless. So we won't use `donations` data set to train or test our model.

### b). ETL (Extract, Transform, Load)

**The ETL notebook for this project is available [here](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/eec51c21-c64d-44be-b8ae-89d37dfc5cbd/view?access_token=84b49773ba7003cf55cdd5450a8763cc8997afa9c8c52957a0d87ed4f649e4a2).**

In the ETL notebook, I load the data, which comes separately in `.csv` format from [IBM Cloud Object Storage](https://www.ibm.com/cloud/object-storage?S_PKG=AW&cm_mmc=Search_Google-_-Cloud_Cloud+Platform-_-WW_NA-_-+ibm++object++storage_Broad_&cm_mmca1=000016GC&cm_mmca2=10007090&cm_mmca7=9060146&cm_mmca8=aud-311016886972:kwd-346458796492&cm_mmca9=_k_CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE_k_&cm_mmca10=317209285867&cm_mmca11=b&mkwid=_k_CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE_k_|1445|530530&cvosrc=ppc.google.%2Bibm%20%2Bobject%20%2Bstorage&cvo_campaign=000016GC&cvo_crid=317209285867&Matchtype=b&gclid=CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE), as Pandas DataFrames and performs data preprocessing with Python's [Pandas](https://pandas.pydata.org/) library. The data cleaning part includes the following:

* Extracting relevant columns from the `outcomes` and `resources` data sets
* Merging all separate dataframes on primary key `projectid`
* Extracting the original training set based on dates the projects were posted *(only those posted before 01/01/2014)* and split it into new training and test set
* Dropping irrelevant columns and columns with more than 10% values missing
* Encoding some columns as binary dummies and converting some others to categoricals
* Dropping rows that still contain NA values

After preprocessing, the output dataframes *(`train` and `test`)* are stored in `.csv` format back to object storage as data assets named accordingly *(`Donation-Projects-Outcome-Prediction.data.train.csv` and `Donation-Projects-Outcome-Prediction.data.test.csv`)* for future use.

*(The naming convention comes from [Lightweight IBM Cloud Garage Method for Data Science](https://github.com/IBM/coursera/blob/master/coursera_capstone/Lightweight_IBM%20Cloud_Garage_Method_for_Data_Science_Romeo_Kienzler.pdf) by Romeo Kienzler)*


## 4. Methodology

To recognize potential fully funded projects from unsuccessful ones accurately, we use three methods to implement the classification:

### a) Random Forest

Random forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. *(Source: [Wikipedia](https://en.wikipedia.org/wiki/Random_forest))*

The randomness in random forests originates in two facts: each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set; in addition, when splitting a node during the construction of the tree, the split that is chosen is no longer the best split among all features, but is instead, the best split among a random subset of the features. *(Source: [scikit-learn User Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest))*

Random forest is considered a very handy and easy to use algorithm, because the number of hyperparameters is not that high, and they are straightforward to understand. Overfitting hardly happens in random forests, once there are enough trees in the forest.

**However, since random forest is based on bagging, to achieve good performance we need a large number of very deep trees, which makes the training process rather slow on large data sets.**


## 5. EDA (Exploratory Data Analysis)

**The EDA notebook for this project is available [here](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/d1f690c5-7b37-4dc9-9063-e4b371e3610d/view?access_token=3a8f94c265f0184cd98a4aa5aac7d335f8d42caf9688a10f59f35702b3fe67a2).**

In the EDA notebook, I perform some initial data explorations on the preprocessed data, output of the [ETL notebook](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/eec51c21-c64d-44be-b8ae-89d37dfc5cbd/view?projectid=fe8f6b6c-3d7e-4e63-bda2-d3d875f07abb&context=analytics) in data file `train.csv` on [IBM Cloud Object Storage](https://www.ibm.com/cloud/object-storage?S_PKG=AW&cm_mmc=Search_Google-_-Cloud_Cloud+Platform-_-WW_NA-_-ibm+cloud+object+storage_Exact_&cm_mmca1=000016GC&cm_mmca2=10007090&cm_mmca7=9021720&cm_mmca8=aud-311016886972:kwd-320507222281&cm_mmca9=_k_CjwKCAiAyfvhBRBsEiwAe2t_i1ZADYFVHn5C4dt5QQKrElVtqwWfFS08ZPk8slGBze9TyaULK38v3xoCIT0QAvD_BwE_k_&cm_mmca10=317209285666&cm_mmca11=e&mkwid=_k_CjwKCAiAyfvhBRBsEiwAe2t_i1ZADYFVHn5C4dt5QQKrElVtqwWfFS08ZPk8slGBze9TyaULK38v3xoCIT0QAvD_BwE_k_|1445|530573&cvosrc=ppc.google.ibm%20cloud%20object%20storage&cvo_campaign=000016GC&cvo_crid=317209285666&Matchtype=e&gclid=CjwKCAiAyfvhBRBsEiwAe2t_i1ZADYFVHn5C4dt5QQKrElVtqwWfFS08ZPk8slGBze9TyaULK38v3xoCIT0QAvD_BwE). I concentrate on how the proportion of fully funded projects, on average, changes across groups divided by several categorical variables and how some numerical variables differ across successful and failed donation projects. By EDA, I'm attempting to find those highly relevant to the outcome of donation projects.

Findings include:

* Total price requested, including and excluding optional tips, can be really strong signals for predicting project outcomes
* Word counts for need statement, short description and title of the projects differ greatly between successful and failed donation projects
* Number of items requsted, maximum, median, and minimum unit prices of the items requested can be good predictors of project outcomes
* Primary focus area of a project contains redundant information and can be represented by the project's primary focus subjects

*(Please check the full report for further reference)*


## 6. Feature Engineering

**The Feature Engineering notebook for this project is available [here](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/6d8c5d76-ea9f-4bf6-9b8a-3a0b09fb80cc/view?access_token=91df3b3a5ba72d70b2b60a9358fcdd1aa23098a6cc2acd4879fb39659013711d).**

In the notebook, I'll read preprocessed training and testing data, which comes as data outputs of the [ETL notebook](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/5a618e1d-b09e-4956-b0f0-ced1bbeae5a3/view?access_token=a7f6bc86b447a978fbf3c54457916f471e1fa4c4eccc9542d20086a6d63f6cb7) and stored on [IBM Cloud Object Storage](https://www.ibm.com/cloud/object-storage?S_PKG=AW&cm_mmc=Search_Google-_-Cloud_Cloud+Platform-_-WW_NA-_-+ibm++object++storage_Broad_&cm_mmca1=000016GC&cm_mmca2=10007090&cm_mmca7=9060146&cm_mmca8=aud-311016886972:kwd-346458796492&cm_mmca9=_k_CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE_k_&cm_mmca10=317209285867&cm_mmca11=b&mkwid=_k_CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE_k_|1445|530530&cvosrc=ppc.google.%2Bibm%20%2Bobject%20%2Bstorage&cvo_campaign=000016GC&cvo_crid=317209285867&Matchtype=b&gclid=CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE) in `.csv` format, and perform feature engineering based on findings in the [EDA notebook](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/ea878b70-71e0-406e-a118-7c8e898db8fb/view?access_token=031878c433b299b1fd3fbf7c0fb07b2ebc776d4b30ced9b8fe37e18a13890cec). The process includes:

* Create features including month, year of project posting and word counts for title, need statement and short description of the project
* Apply one-hot encoding on multinomial variables with multiple categories
* Drop our target, essay data and variables with too many categories from the features dataframe
* Use standard scaler to standardize the features dataframe and transform it into a Numpy feature matrix
* Extract the target variable *(`fully_funded`)*

Output data is stored in `.csv` fomat.


## 7. Modeling

**The Modeling notebook for this project is available [here](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/003188ac-fba2-4c1b-9aea-c48b6e441c96/view?access_token=92f6362d72dd16adde161360800ae262101094fc2115dada9552238264497196)*

In the modeling notebook for this project, I use both Random Forest and XGBoost to predict whether a donation project posted on DonorsChoose.org can get fully funded, based on information on resources they requested, essays they posted and their other characters. Training and test sets are split from projects posted before 01/01/2014. I used data outputs from the [Feature Engineering notebook](https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/6d8c5d76-ea9f-4bf6-9b8a-3a0b09fb80cc/view?access_token=91df3b3a5ba72d70b2b60a9358fcdd1aa23098a6cc2acd4879fb39659013711d) stored on [IBM Cloud Object Storage](https://www.ibm.com/cloud/object-storage?S_PKG=AW&cm_mmc=Search_Google-_-Cloud_Cloud+Platform-_-WW_NA-_-+ibm++object++storage_Broad_&cm_mmca1=000016GC&cm_mmca2=10007090&cm_mmca7=9060146&cm_mmca8=aud-311016886972:kwd-346458796492&cm_mmca9=_k_CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE_k_&cm_mmca10=317209285867&cm_mmca11=b&mkwid=_k_CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE_k_|1445|530530&cvosrc=ppc.google.%2Bibm%20%2Bobject%20%2Bstorage&cvo_campaign=000016GC&cvo_crid=317209285867&Matchtype=b&gclid=CjwKCAiAyfvhBRBsEiwAe2t_i-XCqy6aVw7VL5rPgPbazlACBDB8tL5qFioP_k0oLEF8dxisH8cTlBoClHoQAvD_BwE) in `.csv` format. The models are trained and tuned based on the training set, and evaluated based on the test set. I use test AUC as my evaluating metrics.

Models are first built with default parameters and tuned with grid search based on cross-validation. After tuning, random forest achieves a test accuracy of 74.679%, test AUC of 0.633. Most important features, according to random forest, include:

* Total price the project requested, including and excluding optional tip that donors give to DonorsChoose.org
* Number of items requsted, maximum, median, and minimum unit prices of the items
* Word count for need statement, short description and title of the projects
* Number of students reached
