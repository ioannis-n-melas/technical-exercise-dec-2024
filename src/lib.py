"""
Column Name Description
CustomerID Unique customer ID.
Churn Flag indicating whether the customer churned (1) or not (0).
Tenure Tenure of the customer in the organization.
PreferredLoginDevice Preferred device used by the customer to login (e.g., mobile, web).
CityTier City tier classification (e.g., Tier 1, Tier 2, Tier 3).
WarehouseToHome Distance between the warehouse and the customer's home.
PreferredPaymentMode Preferred payment method used by the customer (e.g., credit card, debit
card, cash on delivery).
HourSpendOnApp Number of hours spent on the mobile application or website.
NumberOfDeviceRegistered Total number of devices registered to the customer's account.
PreferedOrderCat Preferred order category of the customer in the last month.
SatisfactionScore Customer's satisfaction score with the service.
NumberOfAddresses Total number of addresses added to the customer's account.
OrderAmountHikeFromLastYear Percentage increase in order value compared to last year.
CouponUsed Total number of coupons used by the customer in the last month.
OrderCount Total number of orders placed by the customer in the last month.
DaySinceLastOrder Number of days since the customer's last order.
CashbackAmount Average cashback received by the customer in the last month.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

# load data
df = pd.read_csv("../data/challenge_dataset.csv")

# drop duplicates
df.drop_duplicates().shape

# number of customers
df.CustomerID.nunique()

# deduplicate
df = df.drop_duplicates(subset="CustomerID", keep="first")

# set CustomerID as index
df = df.set_index("CustomerID")

# get dtypes
df.dtypes

# obj_cols
obj_cols = df.select_dtypes(include=["object"]).columns.values

# num_cols
num_cols = df.select_dtypes(include=["number"]).columns.values

# one hot encode obj_cols
df_obj = pd.get_dummies(df[obj_cols], drop_first=False)

# as int
df_obj = df_obj.astype(int)

# concat
df = pd.concat([df[num_cols], df_obj], axis=1)

# find cols with missing values
df.isnull().sum().sort_values(ascending=False)

# fill missing values with median
df = df.fillna(df.median())

# features
features = df.drop(columns=["Churn"]).columns.values

# response
response = "Churn"

# shuffle
df = df.sample(frac=1, random_state=42)

# split into train and test
train_df = df.iloc[: int(0.8 * len(df))]
test_df = df.iloc[int(0.8 * len(df)) :]

# scale only the features
scaler = StandardScaler()
train_df_scaled = scaler.fit_transform(train_df[features])
test_df_scaled = scaler.transform(test_df[features])

# to df
train_df_scaled = pd.DataFrame(train_df_scaled, columns=features, index=train_df.index)
test_df_scaled = pd.DataFrame(test_df_scaled, columns=features, index=test_df.index)

# assign response
train_df_scaled[response] = train_df[response]
test_df_scaled[response] = test_df[response]

# apply tSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_xy = tsne.fit_transform(train_df_scaled[features])

# assign to df
train_df_scaled["tsne_x"] = tsne_xy[:, 0]
train_df_scaled["tsne_y"] = tsne_xy[:, 1]

# plot
sns.scatterplot(x="tsne_x", y="tsne_y", hue=response, data=train_df_scaled)
plt.title("t-SNE of Customers")
plt.show()

### Supervised Analysis

# univariate
# iterate across features and perform mannwhitneyu test with response variable

samples_positive = train_df[train_df[response] == 1].index.values
samples_negative = train_df[train_df[response] == 0].index.values

univariate_results = []
for feature in features:
    stat, p = mannwhitneyu(
        train_df[feature].loc[samples_positive], train_df[feature].loc[samples_negative]
    )
    fold_change = (
        np.mean(train_df[feature].loc[samples_positive])
        / np.mean(train_df[feature].loc[samples_negative])
    )
    mean_positive = np.mean(train_df[feature].loc[samples_positive])
    mean_negative = np.mean(train_df[feature].loc[samples_negative])
    univariate_results.append((feature, p, fold_change, mean_positive, mean_negative))

# to df
univariate_results = pd.DataFrame(
    univariate_results,
    columns=["feature", "pvalue", "fold_change", "mean_positive", "mean_negative"],
)

# sort by p-value
univariate_results = univariate_results.sort_values(by="pvalue")

# assign phred score
univariate_results["phred"] = -np.log10(univariate_results["pvalue"])

# plot using Plotly
univariate_results['color'] = univariate_results['fold_change'].apply(lambda x: 'high in churn' if x > 1 else 'low in churn')


fig = px.bar(
    univariate_results,
    x="phred",
    y="feature",
    orientation='h',
    color='color',
    color_discrete_map={'high in churn': 'red', 'low in churn': 'blue'},
    labels={"color": "Churn Level"},
    width=800,
    height=800,
)
# add title
fig.update_layout(title="Univariate Analysis of Features")
fig.show()


# classification modelling

# RF classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
# GB classifier
# rf = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)


rf.fit(train_df_scaled[features], train_df_scaled[response])


# predict on train set
train_predictions = rf.predict_proba(train_df_scaled[features])[:, 1]

# to df
train_predictions = pd.DataFrame(train_predictions, columns=["prediction"], index=train_df_scaled.index)

# assign response
train_predictions[response] = train_df_scaled[response]

# plot ROC curve
fpr, tpr, thresholds = roc_curve(train_predictions[response], train_predictions["prediction"])
roc_auc = auc(fpr, tpr)


# predict on test set
test_predictions = rf.predict_proba(test_df_scaled[features])[:, 1]

# to df
test_predictions = pd.DataFrame(test_predictions, columns=["prediction"], index=test_df_scaled.index)

# assign response
test_predictions[response] = test_df_scaled[response]

# plot ROC curve
fpr, tpr, thresholds = roc_curve(test_predictions[response], test_predictions["prediction"])
roc_auc = auc(fpr, tpr)

# plot
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

# feature importance
feature_importance = pd.DataFrame(rf.feature_importances_, index=features, columns=["importance"])
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
feature_importance.plot(kind="bar", figsize=(10, 10))
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

### Find minimum set of non-redundant features

use_correlation = True 
if use_correlation:
    # calculate feature correlation matrix
    feature_correlation = train_df_scaled[features].corr()

    # recursively remove features with highest correlation until no features are removed, as long as the AUC doesn't decrease by more than 5%
    removed_features = []
    start_auc = roc_auc
    while True:

        most_correlated_feature = feature_correlation.drop(index=removed_features).drop(columns=removed_features).abs().sum(axis=1).sort_values(ascending=False).index[0]
        removed_features.append(most_correlated_feature)

        # fit RF classifier
        rf.fit(train_df_scaled[features].drop(columns=removed_features), train_df_scaled[response])
        # predict on test set
        test_predictions = rf.predict_proba(test_df_scaled[features].drop(columns=removed_features))[:, 1]
        # calculate AUC
        fpr, tpr, thresholds = roc_curve(test_df_scaled[response], test_predictions)
        new_auc = auc(fpr, tpr)

        if new_auc < 0.8:
            break

else: 
    # use feature importance to remove features
    # iterate across features, remove the least important feature until AUC drops to 0.8
    removed_features = []
    start_auc = roc_auc
    while True:

        weakest_feature = feature_importance.drop(index=removed_features).sort_values(by="importance").index[0]
        removed_features.append(weakest_feature)

        # fit RF classifier
        rf.fit(train_df_scaled[features].drop(columns=removed_features), train_df_scaled[response])

        # predict on test set
        test_predictions = rf.predict_proba(test_df_scaled[features].drop(columns=removed_features))[:, 1]
        # calculate AUC
        fpr, tpr, thresholds = roc_curve(test_df_scaled[response], test_predictions)
        new_auc = auc(fpr, tpr)

        if new_auc < 0.85:
            break

# kept features
kept_features = list(set(features) - set(removed_features))

# fit RF classifier on kept features
rf.fit(train_df_scaled[kept_features], train_df_scaled[response])

# predict on test set
test_predictions = rf.predict_proba(test_df_scaled[kept_features])[:, 1]

# calculate AUC
fpr, tpr, thresholds = roc_curve(test_df_scaled[response], test_predictions)
roc_auc = auc(fpr, tpr)

# plot
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

# feature importance
feature_importance = pd.DataFrame(rf.feature_importances_, index=kept_features, columns=["importance"])
feature_importance = feature_importance.sort_values(by="importance", ascending=False)
feature_importance.plot(kind="bar", figsize=(10, 10))
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

### sensitivity analysis 

# iterate across features, increase or decrease feature values by 20% and calculate predicted probabilities
sensitivity_results = []
# feature = 'Complain'
for feature in kept_features:
    # increase
    increased_feature = test_df_scaled[feature] + 0.5
    # predict
    increased_predictions = rf.predict_proba(test_df_scaled[kept_features].drop(columns=[feature]).assign(**{feature: increased_feature})[kept_features])[:, 1]
    # to df
    increased_predictions = pd.DataFrame(increased_predictions, columns=["prediction"], index=test_df_scaled.index)
    # assign response
    increased_predictions[response] = test_df_scaled[response]
    # assign feature name
    increased_predictions["feature"] = feature
    # assign direction
    increased_predictions["direction"] = "increased"

    # same for decreased
    decreased_feature = test_df_scaled[feature] - 0.5
    decreased_predictions = rf.predict_proba(test_df_scaled[kept_features].drop(columns=[feature]).assign(**{feature: decreased_feature})[kept_features])[:, 1]
    # to df
    decreased_predictions = pd.DataFrame(decreased_predictions, columns=["prediction"], index=test_df_scaled.index)
    # assign response
    decreased_predictions[response] = test_df_scaled[response]
    # assign feature name
    decreased_predictions["feature"] = feature
    # assign direction
    decreased_predictions["direction"] = "decreased"

    # append
    sensitivity_results.append(increased_predictions)
    sensitivity_results.append(decreased_predictions)

# to df
sensitivity_results = pd.concat(sensitivity_results)

# predictions on original data
original_predictions = rf.predict_proba(test_df_scaled[kept_features])[:, 1]
original_predictions = pd.DataFrame(original_predictions, columns=["prediction"], index=test_df_scaled.index)
original_predictions[response] = test_df_scaled[response]

# rename prediction column
sensitivity_results = sensitivity_results.rename(columns={"prediction": "prediction_perturbed"})

# reset index
sensitivity_results = sensitivity_results.reset_index()
original_predictions = original_predictions.reset_index()

# merge
sensitivity_results = pd.merge(sensitivity_results, original_predictions.drop(columns=["Churn"]), on=["CustomerID"])

# calculate prediction difference
sensitivity_results["prediction_diff"] = sensitivity_results["prediction_perturbed"] - sensitivity_results["prediction"]

# calculate absolute prediction difference
sensitivity_results["prediction_diff_abs"] = sensitivity_results["prediction_diff"].abs()

# sns barplot
sns.barplot(x="prediction_diff_abs", y="feature", data=sensitivity_results, orient="h")
plt.title("Sensitivity Analysis of Features")
plt.xlabel("Absolute Prediction Difference")
plt.ylabel("Feature")
plt.show()

# same but using prediction_diff
# figsize
plt.figure(figsize=(10, 10))
sns.barplot(x="prediction_diff", y="feature", data=sensitivity_results, orient="h", hue="direction")
plt.title("Sensitivity Analysis of Features")
plt.xlabel("Prediction Difference")
plt.ylabel("Feature")
plt.show()


