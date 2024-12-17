# Required Libraries
import pandas as pd
import numpy as np

# Extraction
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine_data = pd.read_csv(wine_url, header=None)

wine_quality_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_quality_data = pd.read_csv(wine_quality_url, sep=";")

# Initial look at the data
print(wine_data.head())
print(wine_quality_data.head())

# Transformation
# Assigning meaningful column names
wine_data.columns = ['class', 'alcohol', 'malic acid', 'ash',
                     'alcalinity of ash', 'magnesium', 'total phenols',
                     'flavonoids', 'nonflavonoid phenols', 'proanthocyanidins',
                     'color intensity', 'hue', 'OD280/OD315 of diluted wines',
                     'proline']

# Converting Class column into categorical datatype
wine_data['class'] = wine_data['class'].astype('category')

# Checking for any missing values in both datasets
print(wine_data.isnull().sum())
print(wine_quality_data.isnull().sum())

# Normalizing 'alcohol' column in the wine_data using Min-Max normalization
wine_data['alcohol'] = (wine_data['alcohol'] - wine_data['alcohol'].min()) / (wine_data['alcohol'].max() - wine_data['alcohol'].min())

# Creating an average quality column in wine_quality_data
wine_quality_data['average_quality'] = wine_quality_data[['fixed acidity', 'volatile acidity', 'citric acid',
                                                          'residual sugar', 'chlorides', 'free sulfur dioxide',
                                                          'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                                          'alcohol']].mean(axis = 1)

# Creating a 'quality_label' column based on 'average_quality'
wine_quality_data['quality_label'] = pd.cut(wine_quality_data['average_quality'], bins=[0, 5, 7, np.inf],
                                            labels = ['low', 'medium', 'high'])

# Loading
# Saving the transformed data as a csv file
wine_data.to_csv('wine_dataset.csv', index = False)
wine_quality_data.to_csv('wine_quality_dataset.csv', index = False)

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Matrix
corr = wine_quality_data.corr(numeric_only = True)

# Plot heatmap
plt.figure(figsize = (12, 10))
sns.heatmap(corr, annot = True, cmap = 'coolwarm')
plt.title('Correlation Matrix of Wine Quality Data')

# Save the figure
plt.savefig('correlation_matrix.png', dpi = 300, bbox_inches = 'tight')

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define predictors and target
X = wine_data.drop('class', axis = 1)
y = wine_data['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state = 42)

# Initialize the model
clf = RandomForestClassifier(random_state = 42)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score

# Initialize the model
clf = RandomForestClassifier(random_state = 42)

# Compute cross-validation score
scores = cross_val_score(clf, X, y, cv = 5)

print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

# Plot histogram
plt.figure(figsize = (10, 7))
wine_quality_data['quality'].plot(kind = 'hist', rwidth = 0.95,
                                      bins = np.arange(2.5, 9))

plt.title('Distribution of Wine Quality Ratings')
plt.xlabel('Quality Ratings')
plt.ylabel('Count')

# Adjust x-ticks
plt.xticks(np.arange(3, 9, step=1))

# Save the figure
plt.savefig('histogram_wine_quality_ratings.png', dpi = 300,
                                      bbox_inches = 'tight')
plt.show()
