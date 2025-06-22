import pandas as pd
from preprocessing_Rivaldo import *
import os
 
data = pd.read_csv("co2emissions_raw.csv")
data.drop(["Make", "Model"], axis=1, inplace=True)


print("Current working directory:", os.getcwd())
print("Files before saving:", os.listdir())

output_dir = os.path.join(os.getcwd(), "co2emissions_preprocessing")
os.makedirs(output_dir, exist_ok=True)
X_train, X_test, y_train, y_test = preprocess_data(data, 'CO2 Emissions(g/km)', os.path.join(output_dir, "scaler.pkl"), os.path.join(output_dir, "column.pkl"), os.path.join(output_dir, "numeric_column.pkl"), os.path.join(output_dir, "labeler.pkl"))

train_processed = pd.DataFrame(X_train)
train_processed['target'] = y_train

test_processed = pd.DataFrame(X_test)
test_processed['target'] = y_test

train_processed.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
test_processed.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
