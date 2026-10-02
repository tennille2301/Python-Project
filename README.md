# Machine Learning Based Network Traffic Classification

## 1. Description

This project focuses on the classification of network traffic data using Machine Learning techniques in Python. Multiple network traffic datasets are combined and preprocessed before applying machine learning algorithms.

The dataset is cleaned by removing unnecessary spaces, converting feature values into numeric format, handling infinite and missing values, and encoding the target labels. The dataset is then divided into training and testing samples.

The project evaluates the classification model using different performance metrics such as Confusion Matrix, Accuracy, Precision, Recall, F1 Score, and Classification Report.

The main algorithms assigned for this project are CatBoost, k-Nearest Neighbors, Support Vector Machine, Quadratic Discriminant Analysis, and Multilayer Perceptron.

---

## 2. Code

```python
import numpy as np
import pandas as pd

df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
    stratify=y
)

from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    random_state=0
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nPrecision : {:.4f}".format(
    precision_score(y_test, y_pred, average='weighted')
))
print("\nRecall : {:.4f}".format(
    recall_score(y_test, y_pred, average='weighted')
))
print("\nF1 Score : {:.4f}".format(
    f1_score(y_test, y_pred, average='weighted')
))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
```

---

## 3. Algorithm

### 11. CatBoost

```python
from catboost import CatBoostClassifier 
classifier = CatBoostClassifier() 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
```

### 13. k-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
```

### 15. Support Vector Machine

```python
# kernel can be swapped: 'rbf', 'linear', 'poly', etc.
from sklearn.svm import SVC 
classifier = SVC(kernel='rbf', probability=True, random_state=0) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
```

### 17. Quadratic Discriminant Analysis (QDA)

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
classifier = QuadraticDiscriminantAnalysis() 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
```

### 18. Multilayer Perceptron (Neural Network)

```python
from sklearn.neural_network import MLPClassifier 
classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    max_iter=500, 
    random_state=42
) 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
```
