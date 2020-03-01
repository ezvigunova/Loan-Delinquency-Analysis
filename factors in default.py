from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
import seaborn as sns
import warnings  
from sklearn.ensemble import ExtraTreesClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

# DATA PREPARATION

loan_df = pd.read_csv("C:/Users/JZ/Desktop/working_file.csv",header=0,skip_blank_lines=True)
na_thresh = len(loan_df)*80/100
loan_df = loan_df.dropna(thresh=na_thresh, axis=1)
new_status_dict = {
    'Fully Paid': 'Fully Paid',
    'Charged Off': 'Charged Off',
    'Current': 'Current',
    'Default': 'Default',
    'Late (31-120 days)': 'Late',
    'In Grace Period': 'Late',
    'Late (16-30 days)': 'Late',
    'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid',
    'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
    'Issued': 'Issued'
}
loan_df['new_status'] = loan_df['loan_status'].map(new_status_dict)
pd.Series(pd.unique(loan_df['new_status'])).to_frame()
loan_df = loan_df[loan_df['loan_status'] != 'Issued']
loan_type_dict = {
    'Fully Paid': 'Good',
    'Charged Off': 'Bad',
    'Current': 'Good',
    'Default': 'Bad',
    'Late (31-120 days)': 'Bad',
    'In Grace Period': 'Bad',
    'Late (16-30 days)': 'Bad',
    'Does not meet the credit policy. Status:Fully Paid': 'Good',
    'Does not meet the credit policy. Status:Charged Off': 'Bad'
}

loan_df['loan_type'] = loan_df['loan_status'].map(loan_type_dict)
loan_df['score'] = loan_df.grade.map({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7})
target_dict = {
    'Good': 0,
    'Bad' : 1
}
loan_df['target'] = loan_df['loan_type'].map(target_dict)

#-------------------------------------------------------------------------#

# finding the correllation matrix and changing the categorical data to category for the plot.

df_LC = loan_df.filter(['loan_amnt','term','int_rate','installment','grade','sub_grade', 'home_ownership', 'out_prncp', 
                    'annual_inc','total_pymnt', 'loan_stat', 'verification_status','purpose','dti','delinq_2yrs','loan_status'])
df_LC.dtypes
plt.figure(figsize=(20,20))
sns.set_context("paper", font_scale=1)
sns.heatmap(df_LC.assign(grade=df_LC.grade.astype('category').cat.codes,
                         sub_g=df_LC.sub_grade.astype('category').cat.codes,
                         term=df_LC.term.astype('category').cat.codes,                        
                         ver =df_LC.verification_status.astype('category').cat.codes,
                        home=df_LC.home_ownership.astype('category').cat.codes,
                        purp=df_LC.purpose.astype('category').cat.codes).corr(), 
                         annot=True, cmap='bwr',vmin=-1, vmax=1, square=True, linewidths=0.5)

# loan amount and loan status 

plt.figure(figsize=(15,9))
ax = sns.boxplot(y="loan_status", x="loan_amnt", data=loan_df, showfliers=False)
ax = plt.xlabel('Loan Amount')
ax = plt.ylabel('Loan status')
ax = plt.title('Loan Amount - Loan Status')

#interest rate by grade

plt.figure(figsize=(10,6))
plot_data = loan_df.groupby('grade')['int_rate'].mean()
ax = sns.barplot(x=plot_data.index,y=plot_data.values,palette='OrRd')
ax = plt.xlabel('Loan Grade')
ax = plt.ylabel('Interest Rate')
ax = plt.title('Interest Rate Versus Loan Grade')

#interest rate by purpose

plt.figure(figsize=(13,7))
plot_data = loan_df.groupby('purpose')['loan_stat'].mean()
ax = sns.barplot(y=plot_data.index,x=plot_data.values,palette='BuPu')
ax = plt.ylabel('Loan Purpose')
ax = plt.xlabel('status')
ax = plt.title('Loan status by purposes')

# Random forest classification

dummies_df =pd.get_dummies(loan_df,columns=['term','grade','sub_grade','emp_length','home_ownership','pymnt_plan','purpose','addr_state','initial_list_status','application_type'])
dummies_df.dropna(inplace=True)
X_df = dummies_df.loc[:, ~(dummies_df.columns.isin(pd.Series(loan_df.select_dtypes(['object']).columns)))]
y = dummies_df['target'].values
X_df = X_df.loc[:, ~(X_df.columns.isin(['target','id', 'member_id','cut', 'issue_d']))]
X = X_df.values
forest = ExtraTreesClassifier(n_estimators=50,random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_

# Plot the feature importance

plot_data = pd.DataFrame({'features' : pd.Series(X_df.columns),'importance' : pd.Series(importances)})
plt.figure(figsize=(15,10))
plt.xticks(rotation=45)
plot_data = plot_data.sort_values('importance',ascending=False)
plot_data = plot_data[plot_data['importance'] > 0.01]
ax = sns.barplot(y=plot_data['importance'],x=plot_data['features'],)

# annual inc, empl_len, score
X=loan_df.iloc[:, [12, 145, 151]].values

# Make clusters
kmeans = KMeans(n_clusters=4)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
label = kmeans.labels_
loan_df["cluster"] = label;

# Plot clusters

X, y = make_blobs(n_samples=700, n_features=3, centers=4)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.set_xlabel('Annual income')
ax.set_ylabel('Employment length')
ax.set_zlabel('Credit score')


plt.figure(figsize=(10,5))
plot_data = loan_df.groupby(['cluster','loan_type']).size().unstack().T
r = range(4)

ax = plt.bar(r, plot_data.values[0], color='red', edgecolor='white',label='Bad')
ax = plt.bar(r, plot_data.values[1], bottom=plot_data.values[0], color='#7fd1b8', edgecolor='white',label='Good')
names = plot_data.columns
ax = plt.xticks(names)
ax = plt.legend(loc='upper right')
ax = plt.xlabel('Cluster')
ax = plt.ylabel('Number of loans')
ax = plt.title('Loan Type Distribution By Loan Grade')

    









