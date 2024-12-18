
Wine Data Processing and Analysis
=================================

This project processes, transforms, and analyzes wine-related datasets using Python. It includes data extraction, transformation, modeling, and visualization.

Features
--------
- Extraction and processing of wine and wine-quality datasets.
- Transformation of raw data into meaningful structures.
- Visualization of data correlations and quality distributions.
- Training and evaluation of a RandomForest Classifier.

Functions Overview
------------------

1. **extract_data()**
   - Extracts the wine and wine-quality datasets from their respective URLs.
   - Returns: Two pandas DataFrames for wine data and wine quality data.

2. **transform_data(wine_data, wine_quality_data)**
   - Transforms raw datasets by assigning column names, normalizing data, and creating new features.
   - Parameters:
     - `wine_data`: Raw wine dataset DataFrame.
     - `wine_quality_data`: Raw wine quality dataset DataFrame.
   - Returns: Transformed DataFrames for wine data and wine quality data.

3. **load_data(wine_data, wine_quality_data)**
   - Saves the transformed datasets to CSV files for further use.
   - Parameters:
     - `wine_data`: Transformed wine dataset DataFrame.
     - `wine_quality_data`: Transformed wine quality dataset DataFrame.

4. **plot_correlation_matrix(wine_quality_data)**
   - Generates and saves a heatmap of the correlation matrix for the wine quality data.
   - Parameters:
     - `wine_quality_data`: Transformed wine quality dataset DataFrame.

5. **model_training(wine_data)**
   - Trains a RandomForest Classifier on the wine data and evaluates its performance.
   - Parameters:
     - `wine_data`: Transformed wine dataset DataFrame.

6. **plot_quality_histogram(wine_quality_data)**
   - Creates and saves a histogram of wine quality ratings.
   - Parameters:
     - `wine_quality_data`: Transformed wine quality dataset DataFrame.

Usage
-----

To use this project, ensure the following libraries are installed:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `sklearn`

Run the `main` script as follows:

.. code-block:: bash

   python main.py

License
-------

For more information, please refer to <https://unlicense.org>
