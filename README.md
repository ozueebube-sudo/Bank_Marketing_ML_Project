# Bank Marketing Machine Learning Project

## Project Overview

A bank wants to identify customers who are more likely to subscribe to a term deposit.

In this project, I used customer and campaign information to build a Logistic Regression model that predicts whether a customer is likely to subscribe.

The goal is to help the bank focus its marketing efforts on customers with a higher predicted likelihood of subscribing.

## Business Problem

Marketing campaigns can involve contacting many customers, but not every customer is equally likely to subscribe.

The question I explored was:

> Can customer and campaign information help identify customers who are more likely to subscribe to a term deposit?

## Dataset

The dataset used in this project is the **Bank Marketing dataset** from the UCI Machine Learning Repository.

It contains information about customers and their interactions with a bank's marketing campaigns.

The target variable is `y`:

- `yes` = customer subscribed to the term deposit
- `no` = customer did not subscribe

The dataset contains **41,188 observations and 21 columns** before duplicate removal.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## What I Did

The project followed these main steps:

1. Loaded and inspected the dataset.
2. Checked for duplicate and missing values.
3. Investigated `unknown` values in categorical variables.
4. Inspected the actual categories in the categorical columns.
5. Examined numerical variables and the target distribution.
6. Removed exact duplicate rows.
7. Excluded `duration` because it is only known after a call and would not be available when making a pre-call prediction.
8. Encoded the target variable.
9. Converted categorical variables using one-hot encoding.
10. Split the data into training and test sets.
11. Standardized the numerical features.
12. Checked correlations between predictor variables and noted possible multicollinearity.
13. Built an initial Logistic Regression model.
14. Evaluated the model using accuracy, a confusion matrix, and a classification report.
15. Compared the model with a simple baseline that predicted "No" for every customer.
16. Built a second Logistic Regression model using class balancing.
17. Compared the two Logistic Regression models.
18. Built a final pipeline that combines preprocessing and the Logistic Regression model.
19. Generated predicted outcomes and probabilities for customers.

## Model Evaluation

The first Logistic Regression model achieved about **89.8% accuracy**.

However, the dataset was imbalanced, with most customers not subscribing. The model had only about **21% recall for the "Yes" class**, meaning it missed many customers who actually subscribed.

I therefore trained a second Logistic Regression model using `class_weight="balanced"`.

The balanced model achieved about **83% accuracy**, but recall for the "Yes" class increased to about **65%**.

This shows the trade-off: the balanced model finds more actual subscribers, but it also makes more incorrect "Yes" predictions.

## Key Findings

- Most customers in the dataset did not subscribe to the term deposit.
- The original Logistic Regression model had high overall accuracy but struggled to identify customers who subscribed.
- Class balancing improved recall for the "Yes" class from **21% to about 65%**.
- The balanced model therefore identifies more potential subscribers, although it also produces more incorrect "Yes" predictions.
- The final pipeline can take new customer data and automatically apply the required preprocessing before making predictions.

## Business Implications

The model could help a bank prioritize customers who are more likely to subscribe to a term deposit.

Instead of contacting all customers equally, the bank could use the predicted probabilities to focus marketing efforts on customers with a higher predicted likelihood of subscribing.

However, the bank would need to consider the cost of contacting customers who are predicted as likely subscribers but ultimately do not subscribe.

## Limitations

- The model is based on historical data, so its performance may differ when used with future customers.
- The balanced model identifies more potential subscribers but also produces more incorrect "Yes" predictions.
- Some predictor variables were strongly correlated, particularly several economic variables. These possible multicollinearity issues were noted, but the variables were retained.
- The model shows associations between variables and predicted subscription likelihood. It does not prove that a particular variable causes a customer to subscribe.
- Model performance should be monitored if the model is used with new customer data.

## Conclusion

This project used Logistic Regression to predict whether a bank customer is likely to subscribe to a term deposit.

The original model achieved high accuracy but struggled to identify customers who subscribed. After applying class balancing, the model was able to identify more potential subscribers, increasing recall for the "Yes" class from 21% to about 65%.

The final model pipeline combines preprocessing and prediction, making it easier to apply the model to new customer data.

Overall, this project helped me understand how machine learning can be used to support customer targeting and marketing decisions.

## Dataset Source

**UCI Machine Learning Repository — Bank Marketing Dataset**

https://archive.ics.uci.edu/dataset/222/bank+marketing
