
# Covid Project - Predicting ICU Admission

## Overview
This project aims to predict ICU admission for COVID-19 patients using a variety of machine learning techniques. The dataset consists of patient metrics recorded during their hospital stay, with a focus on identifying key features that indicate a patient's likelihood of being admitted to the ICU.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Modeling](#modeling)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [License](#license)

## Project Setup
To run this project, you'll need to have Python installed along with the following libraries:
- pandas
- scikit-learn
- seaborn
- matplotlib

Clone the repository and install the required libraries:

```bash
git clone https://github.com/anthonygreco2019/Machine-Learning-Capstone-Project.git
cd Machine-Learning-Capstone-Project
pip install -r requirements.txt
```

## Data Preprocessing
1. **Loading the Dataset**: The dataset is loaded from a CSV file. 
   ```python
   df = pd.read_csv("path_to_csv/Covid_Project.csv")
   ```
2. **Encoding Categorical Variables**: Ordinal encoding is used for `WINDOW` and `AGE_PERCENTIL` categories.
   ```python
   oencW = OrdinalEncoder(categories=[['0-2', '2-4', '4-6', '6-12', 'ABOVE_12']])
   df['WINDOW'] = oencW.fit_transform(df[['WINDOW']])
   ```
3. **Handling Missing Values**: Missing values are filled using forward and backward filling based on patient visit identifiers.
   ```python
   df.fillna(method='ffill', inplace=True)
   df.fillna(method='bfill', inplace=True)
   ```

## Exploratory Data Analysis
We analyzed the correlation of various features with ICU admission. Features with a correlation coefficient less than 0.004 were dropped.

```python
corr_matrix = df.corr()['ICU']
not_Selected_Features = corr_matrix[corr_matrix['ICU'] < 0.004]
```

Visualizations were created to explore the distribution of ICU admissions across different windows:

```python
so.Plot(df, x="WINDOW", color="ICU").add(so.Bar(), so.Count(), so.Stack())
```

## Modeling
A Decision Tree classifier was used as the initial model to predict ICU admission. The dataset was split into training and testing sets, and the model was trained and evaluated.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Hyperparameter Tuning
A pipeline was created to standardize the data and perform PCA before fitting the Decision Tree. Hyperparameters were tuned using GridSearchCV to find the best parameters for the model.

```python
pipe = Pipeline(steps=[('std_slc', StandardScaler()), ('pca', decomposition.PCA()), ('dec_tree', tree.DecisionTreeClassifier())])
clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X, y)
```

## Results
The accuracy of the model and the classification report are displayed, along with the confusion matrix.

```python
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Conclusion
This project provides insights into predicting ICU admission for COVID-19 patients using machine learning. Future work could involve exploring other models and further optimizing hyperparameters to improve performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to the project by creating issues or submitting pull requests!
