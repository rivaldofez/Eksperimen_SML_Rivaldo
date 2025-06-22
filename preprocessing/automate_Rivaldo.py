import pandas as pd
from preprocessing_Rivaldo import *
 
data = pd.read_csv("co2emissions_raw.csv")
data.drop(["Make", "Model"], axis=1, inplace=True)
X_train, X_test, y_train, y_test = preprocess_data(data, 'CO2 Emissions(g/km)', 'scaler.joblib', 'co2emissions_preprocessing/columns.csv')

train_processed = pd.DataFrame(X_train)
train_processed['target'] = y_train

test_processed = pd.DataFrame(X_test)
test_processed['target'] = y_test

train_processed.to_csv("co2emissions_preprocessing/train_processed.csv", index=False)
test_processed.to_csv("co2emissions_preprocessing/test_processed.csv", index=False)
