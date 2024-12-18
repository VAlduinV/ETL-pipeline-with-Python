# Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def extract_data():
    """Extracts the wine and wine quality datasets."""
    wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    wine_quality_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine_data = pd.read_csv(wine_url, header=None)
    wine_quality_data = pd.read_csv(wine_quality_url, sep=";")
    return wine_data, wine_quality_data


def transform_data(wine_data, wine_quality_data):
    """Transforms and processes the datasets."""
    # Assign column names
    wine_data.columns = ['class', 'alcohol', 'malic acid', 'ash',
                         'alcalinity of ash', 'magnesium', 'total phenols',
                         'flavonoids', 'nonflavonoid phenols', 'proanthocyanidins',
                         'color intensity', 'hue', 'OD280/OD315 of diluted wines',
                         'proline']
    wine_data['class'] = wine_data['class'].astype('category')

    # Normalize alcohol column
    wine_data['alcohol'] = (wine_data['alcohol'] - wine_data['alcohol'].min()) / (
                wine_data['alcohol'].max() - wine_data['alcohol'].min())

    # Create average quality and quality label columns
    wine_quality_data['average_quality'] = wine_quality_data[['fixed acidity', 'volatile acidity', 'citric acid',
                                                              'residual sugar', 'chlorides', 'free sulfur dioxide',
                                                              'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                                              'alcohol']].mean(axis=1)
    wine_quality_data['quality_label'] = pd.cut(wine_quality_data['average_quality'], bins=[0, 5, 7, np.inf],
                                                labels=['low', 'medium', 'high'])
    return wine_data, wine_quality_data


def load_data(wine_data, wine_quality_data):
    """Saves transformed datasets to CSV files."""
    wine_data.to_csv('data/wine_dataset.csv', index=False)
    wine_quality_data.to_csv('data/wine_quality_dataset.csv', index=False)


def plot_correlation_matrix(wine_quality_data):
    """Generates and saves the correlation matrix heatmap."""
    corr = wine_quality_data.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Wine Quality Data')
    plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def model_training(wine_data):
    """Trains a RandomForest classifier and evaluates its accuracy."""
    X = wine_data.drop('class', axis=1)
    y = wine_data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Average cross-validation score:", scores.mean())


def plot_quality_histogram(wine_quality_data):
    """Plots and saves a histogram of wine quality ratings."""
    plt.figure(figsize=(10, 7))
    wine_quality_data['quality'].plot(kind='hist', rwidth=0.95, bins=np.arange(2.5, 9))
    plt.title('Distribution of Wine Quality Ratings')
    plt.xlabel('Quality Ratings')
    plt.ylabel('Count')
    plt.xticks(np.arange(3, 9, step=1))
    plt.savefig('images/histogram_wine_quality_ratings.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    wine_data, wine_quality_data = extract_data()
    print(wine_data.head())
    print(wine_quality_data.head())

    wine_data, wine_quality_data = transform_data(wine_data, wine_quality_data)
    print(wine_data.isnull().sum())
    print(wine_quality_data.isnull().sum())

    load_data(wine_data, wine_quality_data)
    plot_correlation_matrix(wine_quality_data)
    model_training(wine_data)
    plot_quality_histogram(wine_quality_data)
