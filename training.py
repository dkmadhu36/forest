import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib


df = pd.read_csv('/mnt/ou/Algerian_forest_fires_dataset_CLEANED_NEW.csv')
X = df.drop('FWI',axis=1)
y= df['FWI']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train.shape, X_test.shape
X_train.corr()
X_train.shape, X_test.shape
#model creation
lreg = LinearRegression()
lreg.fit(X_train, y_train)
lreg_pred = lreg.predict(X_test)
print(f"pred:{lreg_pred}")
test_data = pd.DataFrame([{
    "Temperature": 29,
    "RH": 57,
    "Ws": 18,
    "Rain": 0.0,
    "FFMC": 65.7,
    "DMC": 3.4,
    "DC": 7.6,
    "ISI": 1.3,
    "BUI": 3.4,
    "Classes": 0,   # not fire
    "Region": 1
}])

pred = lreg.predict(test_data)
print(f"prediction: {pred}")
mae = mean_absolute_error(y_test, lreg_pred)
r2 = r2_score(y_test, lreg_pred)
joblib.dump(lreg, "/mnt/model/model.joblib")
