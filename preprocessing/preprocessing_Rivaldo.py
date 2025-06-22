from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data, target_column, save_path, file_path):
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns
    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_column)

    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)

    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    Q1 = data.quantile(0.25, numeric_only=True)
    Q3 = data.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1

    left1, right1 = data.align(Q1 - 1.5 * IQR, axis=1, copy=False)
    left2, right2 = data.align(Q3 + 1.5 * IQR, axis=1, copy=False)
    left, right = (left1 < right1).align((left2 > right2), axis=1, copy=False)
    data = data[~(left | right).any(axis=1)]

    for column in categorical_features:
        data = pd.concat([data, pd.get_dummies(data[column], prefix='class')], axis=1)
    data.drop(categorical_features, axis=1, inplace=True)

    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    # Simpan
    dump(scaler, save_path)
    return X_train, X_test, y_train, y_test