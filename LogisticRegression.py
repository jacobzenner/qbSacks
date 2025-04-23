# Linear Regression with NFL Data
# Determining when a sack will happen on a given play
# Jacob Zenner

#!pip install nfl_data_py
# !pip install pandas
# !pip install matplotlib
# !pip install seaborn
# !pip install numpy
# !pip install sklearn
# !pip install xgboost

import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, accuracy_score
from xgboost import XGBClassifier


# Raw NFL play-by-play data from 2021, 2022, and 2023 seasons
pbp = nfl.import_pbp_data([2019, 2020, 2021, 2022, 2023])

pbp_test = pbp[(pbp['pass'] == 1)]


# Cleaning the data to get only plays that are passing situations
pbp_clean = pbp[(pbp['pass'] == 1) & (pbp['play_type'] != "no_play")]

# size of pbp data decreases about 85000 rows
print(pbp.shape)
print(pbp_clean.shape)

# Plotting the portion of sacks vs non-sacks in the pbp data
sns.countplot(x=pbp_clean["sack"])
plt.show()

# Plotting the count of sacks for each specific down
sacks = pbp_clean[(pbp_clean['sack'] == 1)]
sns.countplot(x=sacks["down"])
plt.show()

# Plotting the count of sacks for different number of pass rushers
sacks = pbp_clean[(pbp_clean['sack'] == 1)]
sns.countplot(x=sacks["number_of_pass_rushers"])
plt.show()

# Plotting the count of sacks for number of defender in the box
sacks = pbp_clean[(pbp_clean['sack'] == 1)]
sns.countplot(x=sacks["defenders_in_box"])
plt.show()

#Creating a new variable that estimates a likely passing situation
pbp_clean['likely_pass'] = np.where(((pbp_clean['down'] == 3) & (pbp_clean['ydstogo'] >= 6)), 1,0)


# Removing all plays that do not have null values for categories
pre_df = pbp_clean[['game_id', 'play_id', 'season', 'name', 'down', 'ydstogo', 'yardline_100', 'game_seconds_remaining',
                    'defenders_in_box', 'number_of_pass_rushers', 'xpass', 'likely_pass', 'sack']]
df = pre_df.dropna()
df.isna().sum()


# Not really sure what we're doing past this point
df['down'] = df['down'].astype('category')
df_no_ids = df.drop(columns = ['game_id', 'play_id', 'name', 'season'])
df_no_ids = pd.get_dummies(df_no_ids, columns = ['down'])

# StratifiedShuffleSplit is used to split the data into training and test sets 
# while maintaining the proportion of classes. This is important to ensure that 
# both sets have similar distributions of the target variable (sack in this case).
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in sss.split(df_no_ids, df_no_ids['sack']):
    strat_train_set = df_no_ids.iloc[train_index]
    strat_test_set = df_no_ids.iloc[test_index]

# X_train consists of the features used for training the model, 
# and Y_train consists of the corresponding target variable (labels).
# Similarly, X_test and Y_test are the features and labels for the test set.
X_train = strat_train_set.drop(columns = ['sack'])
Y_train = strat_train_set['sack']
X_test = strat_test_set.drop(columns = ['sack'])
Y_test = strat_test_set['sack']

# Train the logistic regression model using the training data
LR = LogisticRegression()
LR.fit(X_train, Y_train)

# Make predictions on the test data
LR_pred = pd.DataFrame(LR.predict_proba(X_test), columns=['no_sack', 'sack'])['sack']

# Calculate accuracy
accuracy = accuracy_score(Y_test, LR_pred.round())

# Print the accuracy
print('Accuracy:', accuracy)

print('Brier Score: ', brier_score_loss(Y_test, LR_pred))

# Prediction using XGBoost
XGB = XGBClassifier(objective="binary:logistic", random_state=42)
XGB.fit(X_train, Y_train)

XGB_pred = pd.DataFrame(XGB.predict_proba(X_test), columns = ['no_sack', 'sack'])[['sack']]

print('Brier Score: ', brier_score_loss(Y_test, XGB_pred))


# Feature the importance in each prediction
sorted_idx = XGB.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], XGB.feature_importances_[sorted_idx])
plt.title("XGBClassifier Feature Importance")
plt.show()
