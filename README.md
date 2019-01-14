Donation Projects Outcome Predictions
=======================================

## 1. Executive Summary

In this project, I use both Random Forest and XGBoost to predict whether a donation project posted on DonorsChoose.org can get fully funded, based on information on each project, resources they requested, project essays posted and their outcomes. Training and test sets are split from projects posted before 01/01/2014.

Models are first built with default parameters and tuned with grid search based on cross-validation. After tuning, random forest achieves a test accuracy of 74.813%, test AUC of 0.638. Most important features, according to random forest, include:

* Total price the project requested, including and excluding optional tip that donors give to DonorsChoose.org
* Number of items requsted, maximum, median, and minimum unit prices of the items
* Word count for need statement, short description and title of the projects
* Number of students reached


## 2. Introduction

**Directly from the project [Overview](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose):**

DonorsChoose.org is an online charity that makes it easy to help students in need through school donations. At any time, thousands of teachers in K-12 schools propose projects requesting materials to enhance the education of their students. When a project reaches its funding goal, they ship the materials to the school.

The 2014 KDD Cup asks participants to help DonorsChoose.org identify projects that are exceptionally exciting to the business, at the time of posting. While all projects on the site fulfill some kind of need, certain projects have a quality above and beyond what is typical. By identifying and recommending such projects early, they will improve funding outcomes, better the user experience, and help more students receive the materials they need to learn.

Successful predictions may require a broad range of analytical skills, from natural language processing on the need statements to data mining and classical supervised learning on the descriptive factors around each project.

**For this project, the challenge would instead be predicting whether a proposed project would get fully funded.**

**Getting fully funded is one of the requirements of being "exciting" to DonorsChoose.org, hence a precise prediction of whether future proposed projects can be fully funded would be an important building block for finding "exciting" projects.**


## 3. Data

**Directly from the project [Data Description](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data):**

The data is provided in a relational format and split by dates. Any project posted prior to 2014-01-01 is in the training set (along with its funding outcomes). Any project posted after is in the test set. Some projects in the test set may still be live and are ignored in the scoring. We do not disclose which projects are still live to avoid leakage regarding the funding status.

A detailed description of the data sets and variables can be found at the webpaged linked above.

Basically there are five data sets, namely `projects`, `resources`, `essays`, `donations` and `outcomes`:

* `projects` - contains information about each project. This is provided for both the training and test set.
* `resources` - contains information about the resources requested for each project. This is provided for both the training and test set.
* `essays` - contains project text posted by the teachers. This is provided for both the training and test set.
* `donations` - contains information about the donations to each project. This is only provided for projects in the training set.
* `outcomes` - contains information about the outcomes of projects in the training set.

**Since `outcomes` is only provided for the training set and access to test set is blocked by Kaggle. Information on whether a project posted after 2014-01-01 got fully funded is not available. Hence we only use the original training set and split it into new training and test set.**

**Notice that once we know the total price a project required, combined with total donations the project got, predicting `fully_funded` would be meaningless. So we won't use `donations` data set to train or test our model.**


## 4. Methodology

To recognize potential fully funded projects from unsuccessful ones accurately, we use three methods to implement the classification:

* __[Random Forest](https://en.wikipedia.org/wiki/Random_forest):__  ensemble learning method that operates by constructing a multitude of decision trees at training time, provided by [`scikit-learn`](https://scikit-learn.org/stable/)
* __[XGBoost](https://en.wikipedia.org/wiki/XGBoost):__ scalable, portable and distributed gradient boosting, provided by [`XGBoost`](https://xgboost.readthedocs.io/en/latest/)

## 5. Exploratory Data Analysis
