import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


#Reference model to compare to for adult dataset

max_depth = 10

dataset = pd.read_csv("data/folktable_unfiltered.csv")

y = np.array(dataset["Employed"])
X = dataset.drop(columns=["Employed"])


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=max_depth)

# Train Decision Tree Classifer
clf = clf.fit(X,y)

#Predict the response for test dataset
y_pred = clf.predict(X)
print("Depth :", clf.get_depth())

print("Accuracy for adult dataset:",metrics.accuracy_score(y, y_pred))
