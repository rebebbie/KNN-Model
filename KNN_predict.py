from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

cmap = ListedColormap(['#f94e2a','#8b5cf5','#64f92a'])

class Model:
    def __init__(self, k=1):
        """
        Initializes the KNN model and prompts user to input a value for K.
        Further steps from here would be to determine a suitable K hyperparameter 
        based on the data set. 
        """
        try:
            k_input = int(input("Enter the value of K: "))
            if k_input <= 0:
                raise ValueError("Please enter a positive integer.")
            self.k = k_input
            print("K is set to " + str(self.k))
        except ValueError:
            print("Invalid input. Automatically setting K=5")
            self.k = 5
		
    def fit_predict(self, X_train, y_train, X_test):
        """
        Fit the model on the training data so we can use KNN to predict outputs for test data.
        """
        self.X_train = X_train
        self.y_train = y_train

        predictions = []
        for X in X_test:
            # Calculate distances between the sample and all training points
            distances = [np.sqrt(np.sum((X - x_train)**2)) for x_train in self.X_train]

            # Get the indices of the top K nearest neighbors
            point_distances = np.argsort(distances)[:self.k]

            # Retrieve the labels of the nearest neighbors
            KNN = [self.y_train[i] for i in point_distances]

            # If labels are integers (classification), return majority vote
            if isinstance(self.y_train[0], (int, np.integer)):
                predictions.append(max(set(KNN), key=KNN.count))

            # If labels are floats (regression), return the average
            elif isinstance(self.y_train[0], (float, np.float_)):
                predictions.append(np.mean(KNN))
                
            else:
                raise ValueError("Data type is unsupported (needs to be integer or float).")

        return predictions


iris_data = load_iris()
X, y = iris_data.data, iris_data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

KNN_Model = Model()

predictions = KNN_Model.fit_predict(X_train, y_train, X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the KNN model is: " + str(round(accuracy, 2)))

plt.figure()
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()