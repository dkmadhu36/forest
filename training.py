import pandas as pd
import numpy as np
import joblib
import mlflow as ml

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('/mnt/out/Fire_dataset_cleaned.csv')

# Split features / target
X = df.drop('FWI', axis=1)
y = df['FWI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# KNN pipeline (scaling + model)
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        metric='minkowski'
    ))
])

# Train
knn_pipeline.fit(X_train, y_train)

# Predict
y_pred = knn_pipeline.predict(X_test)
print(f"Predictions: {y_pred}")

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics to MLflow
ml.log_metrics({
    "mae": mae,
    "r2": r2
})

# Save model (pipeline)
joblib.dump(knn_pipeline, "/mnt/model/model.joblib")

print("KNN model trained and saved successfully!")
