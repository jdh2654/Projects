from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Loads the data from sklearn
cancerData = datasets.load_breast_cancer()

#Seperates data into the features and the output data
x = cancerData.data
y = cancerData.target

#creates training and testing data of the features and output, 30% of data is test data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3)

#Creates a random forest classifier, and trains it on the training data
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

#Creates a prediction based on the test feature data
prediction = rf.predict(X_test)

#Prints out 1's and 0's as the classes predicted outputs, based on the test data.
print("prediction values of the RF classifier on the test data:\n", prediction)

import pandas as pd
import seaborn as sns

#Finds the importance of each feature
featureImportance = pd.Series(rf.feature_importances_, index = cancerData.feature_names).sort_values(ascending = False)
print("Feature Importance:\n", featureImportance)

#Makes a bar chart of how important each feature is
sns.barplot(x = featureImportance, y = featureImportance.index)

#Adds labels to the bar chart
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.title("Importance of Features")
plt.show()