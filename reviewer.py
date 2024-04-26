# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Separate features and target variable
X = data.drop(columns=['net_debt'])
y = data['net_debt']

# Define categorical and numerical features
categorical_features = ['name', 'industry', 'sector']
numeric_features = ['last_12_months_revenue', 'EBITDA', 'enterprise_value','target_column']

# Define preprocessing steps
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print('training...')
# Train the model
model.fit(X_train, y_train)

print('predicting...')

print('Testing data:')
print(X_test)
# Make predictions
y_pred = model.predict(X_test)

print('prediction:')
print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred, normalize=False)
print("Accuracy:", accuracy)
