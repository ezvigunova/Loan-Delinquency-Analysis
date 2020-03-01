# Loan-Delinquency-Analysis

The dataset used for this project is a Lending Club loan history with personally identifiable information omitted. In this loan delinquency project, we will perform several analysis methods to determine the reasons certain loans get defaulted. We will explore various factors that could potentially predict if a loan applicant will be unable to pay their installments due to financial hardships. It would be necessary to look into relationships between the individual income, length of employment, living status, and existing balances on other credit accounts, number of credit inquiries and other variables that could negatively contribute to the future ability to repay the loans. 
  
Question 1: Is the history of financial hardships in the past an indicator of current or potential future issues?

In order to answer this question, it was necessary to define financial issues. Since the dataset contains a variety of loan statuses such as current, paid off, default, late and others, we decided to split them into two groups: good and bad. Bad loan statuses such as default and late would indicate a financial trouble. Then, we used a random forest classifier to assign all cases into these two groups so all existing loans would either be good or bad. After that, we used a feature importance function to show the most influential features that helped in grouping the cases into either good or bad ones, sorted by importance level. The result have shown that the most influential features in distinguishing bad and good ones were the history of collections, charge offs and recoveries, as well as current outstanding principal balance owed on the loan, total payment, loan amount, installment, interest rate, revolving balance and lastly, annual income. Knowing that these aspects can make or break a loan is very useful for determining potential future issues. Since we see that past financial troubles have the most significant impact on the current loan status, the conclusion is that a history of past financial issues can indeed predict non-payments, charge offs and defaults.

2. Does annual income affect individual ability to pay?

It is common for many banks and credit unions utilize annual income as one of the main factors that determine individual creditworthiness. It is usually assumed that higher income individuals always have enough money and can pay off their loans with no issues while low-income individuals mostly live paycheck to paycheck and struggle with making all required payments. If this assumption is true, it could mean that the majority of bad loans would belong to people with a low annual income. 
  
This dataset provides us with an opportunity to find out the truth – whether individual annual income has an effect on ability to pay off the loans. Hence, we first created the correlations table to find what variables are correlated. Particularly, we were interested in seeing if there is a strong correlation between the annual income, loan status and total payments. In the dataset, total payments variable indicates the loan amount that has already been paid off. Correlations table, however, has shown us a weak correlation between these variables, pointing that we need to do a further analysis.
  
Going further, we decided to do a linear regression analysis where we set the annual income, loan amount, installment, and outstanding balance as input variables and total payment received as an output variable.
The coefficient for annual income is 0.0004592570137176733
The coefficient for loan amount is 1.0476144928598519
The coefficient for outstanding balance is -0.9029898418452635
The intercept for our model is 25.058053249667864
The regression model score is 0.7951407660374087

The very low coefficient value of 0.0004 has demonstrated that annual income has practically no effect on the output variable. This means that for a one unit increase in annual income, there is only a 0.0004 increase in the total received. Loan amount, on the other hand, has more effect on the output variable. Hence, we can conclude that the relationship between annual income and the individual ability to pay is weak, so there must be a variety of other aspects that influence individual creditworthiness.

3. What borrowers get defaulted the most?

In order to better understand what people are at the highest risk of defaulting on their loans, it is necessary to create an average profile of the risky borrowers. In order to do that, we grouped all borrowers into 4 clusters by their annual income, credit score and employment length using k-means clustering algorithm. Each of the clusters represented a specific financial status, allowing us to group all borrowers into high risk, medium risk and low risk. These are the results of the clustering algorithm.
  
Cluster centers:

 0. [[5.60964662e+04 5.20452941e+00 2.72588235e+00] – med income, high length, high score (Purple)
 1. [6.20802928e+06 4.11111111e+00 2.27777778e+00] – high income, high length, high score (Yellow)
 2. [1.30614932e+05 4.67032150e+00 2.47660286e+00] - low income, med length, low score (Blue)
 3. [3.95058127e+05 4.73608838e+00 2.27986907e+00]] - med income, high length, low score (Red)
 
Hence, we determined that clusters 2 and 3 (blue and red on the plot) are considered high risk profiles while clusters 0 and 1 are the lowest risk. Then, we created a bar plot to see how many good and bad loans each category shows. Overall, we see that cluster 0 has the most defaults; however, it also has the largest number of loans. If we look at the ratio of good and bad loans, cluster 2 is doing worse than cluster 0. Cluster 3 that demonstrates a low credit score, has no defaults at all; however, we have very little data in this cluster as Lending Club rarely offers loans to people with a low credit score. We also have little to no data on cluster 1, possibly indicating that those with high credit score, high income and a stable career do not really need loans as they can afford major expenses. We have also created a bar plot to determine the distribution of bad and good loans by the borrower’s credit score that is represented by a grade in this dataset. This plot showed how the decrease in the credit score indicated an increase in the loan defaults.
  	
In conclusion, we can see that high credit score alone is not in any way a guarantee that a loan will be paid off in time and vice versa. Banks and credit unions should remember that there is a variety of aspects that impact borrower’s financial situation. In addition, as individual circumstances change, and so do their finances. When someone has all of the positives like high annual income, a stable career and high credit score, it is safe to assume they are a low risk borrower. However, when there is a mix of positive and negative aspects, it is nearly impossible to know what will happen in the future. 
