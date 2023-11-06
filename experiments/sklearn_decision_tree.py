import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


#Reference model to compare to for adult dataset
#Create a greedy-rule list using successive decision trees of depth 1.

max_length = 5


dataset = pd.read_csv("data/adult.csv")

y = np.array(dataset["income"])
X = dataset.drop(columns=["income"])


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X,y)

#Predict the response for test dataset
y_pred = clf.predict(X)

print("Accuracy for adult dataset:",metrics.accuracy_score(y, y_pred))
