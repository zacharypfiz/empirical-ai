import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # Alternative/simple model

# Load data from the specified root
DATASET_ROOT = os.environ.get('DATASET_ROOT', '/data')
train_df = pd.read_csv(os.path.join(DATASET_ROOT, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATASET_ROOT, 'test.csv'))

# Separate target variable
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Store PassengerId for submission
test_passenger_ids = test_df['PassengerId']

# Identify numerical and categorical features
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
    ],
    remainder='drop',
)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model),
])

full_pipeline.fit(X, y)

# Predict and write submission
test_predictions = full_pipeline.predict(test_df)
submission_df = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully: submission.csv")

