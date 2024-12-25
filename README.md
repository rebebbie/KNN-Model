# KNN-Model
Making a KNN model in Python, tested on the Iris dataset from sklearn.

For the Iris dataset, we notice the following:

Since the accuracy of KNN is heavily reliant on the hyperparameter K, I tested with values 5, 10, and 50 to see which ones return a high accuracy (A.K.A. low error) result. 

Here is the dataset:

![KNN](https://github.com/user-attachments/assets/9741bd3c-8a7d-4ec3-a1d2-b17afcfd480f)

When we predict using K=5, accuracy score is 97%

When we predict using K=10, accuracy score is 100%

When we predict using K=50, accuracy score is 93%

Evidently, it seems that this model works best when we set K=10. However, note that although the accuracy score is 100%, the testing method was very simple. I simply split the data set into 80% training data and 20% testing data, which means that an inaccurate prediction given K=10 could appear in the other data points. A more comprehensive testing method would be using K-fold cross validation, which would be a next step.

