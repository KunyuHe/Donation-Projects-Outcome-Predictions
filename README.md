Donation Projects Outcome Predictions
=======================================

## 1. Executive Summary


## 2. Introduction

**Directly from the project [Overview](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose):**

DonorsChoose.org is an online charity that makes it easy to help students in need through school donations. At any time, thousands of teachers in K-12 schools propose projects requesting materials to enhance the education of their students. When a project reaches its funding goal, they ship the materials to the school.

The 2014 KDD Cup asks participants to help DonorsChoose.org identify projects that are exceptionally exciting to the business, at the time of posting. While all projects on the site fulfill some kind of need, certain projects have a quality above and beyond what is typical. By identifying and recommending such projects early, they will improve funding outcomes, better the user experience, and help more students receive the materials they need to learn.

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

### a) Random Forest

Random forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. *(Source: [Wikipedia](https://en.wikipedia.org/wiki/Random_forest))*

The randomness in random forests originates in two facts: each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set; in addition, when splitting a node during the construction of the tree, the split that is chosen is no longer the best split among all features, but is instead, the best split among a random subset of the features. *(Source: [scikit-learn User Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest))*

Random forest is considered a very handy and easy to use algorithm, because the number of hyperparameters is not that high, and they are straightforward to understand. Overfitting hardly happens in random forests, once there are enough trees in the forest.

**However, since random forest is based on bagging, to achieve good performance we need a large number of very deep trees, which makes the training process rather slow on large data sets.**

### b) XGBoost


## 5. Exploratory Data Analysis

Below are some examples from the [EDA section](https://render.githubusercontent.com/view/ipynb?commit=bfebd33c604a6a184b1aab5074da69e4598a35fa&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4b756e797548652f446f6e6174696f6e2d50726f6a656374732d4f7574636f6d652d50726564696374696f6e732f626665626433336336303461366131383462316161623530373464613639653435393861333566612f50726564696374696e672532304f7574636f6d65732532306f66253230446f6e6174696f6e25323050726f6a656374732e6970796e62&nwo=KunyuHe%2FDonation-Projects-Outcome-Predictions&path=Predicting+Outcomes+of+Donation+Projects.ipynb&repository_id=164807280&repository_type=Repository#Exploratory-Data-Analysis) of my final report:

### a). Project Counts across States

![](https://github.com/KunyuHe/Donation-Projects-Outcome-Predictions/blob/master/EDA_outputs/by_state.png)

We can see that **California, New York, North Carolina, Illinois, Texas** are the states with highest numbers of projects across the US.

### b). Project Outcome by Primary Focus Subject

![](https://github.com/KunyuHe/Donation-Projects-Outcome-Predictions/blob/master/EDA_outputs/subject_areas.png)

Here we can see the proportion of successful projects varies across promary focus subjects. **Sujects `Economics`, `Environmental Science`, `Music` and `Nutrition` have highest fully funded rates.**

### c). Project Outcome by Month of Year

![](https://github.com/KunyuHe/Donation-Projects-Outcome-Predictions/blob/master/EDA_outputs/month_of_year.png)

We observe that the **rate of fully funded projects are much higher (by nearly 8% on average) over the year from `August to December`, compared with the period from `January to July`.** Month of year could be a good predictor for whether a project would succeed.

### d). Project Outcome by Statistics of Resources Requested

![](https://github.com/KunyuHe/Donation-Projects-Outcome-Predictions/blob/master/EDA_outputs/resouces.png)

We observe that projects with **higher number of items requested, lower minimum, median and maximum price of requested items** are more likely to get fully funded. The differences are quiet significant.

*Please check the full report for further reference.*


## 6. Results

### a). Random Forest

Performance of random forest with default parameters on the test set is rather unsatisfying, with a test accuracy of 69.216%, and test AUC of 0.637.

After tuning by grid search, random forest classification achieved a test accuracy of 74.813%, test AUC of 0.638.

![](https://github.com/KunyuHe/Donation-Projects-Outcome-Predictions/blob/master/EDA_outputs/important_features.png)

However, random forest is quite slow on large data set and the performance is not satisfying. In a later version I would add XGBoost modeling.
