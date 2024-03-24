import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def PreProcessData():
    # Get data from csv
    X = pd.read_csv('./ufc_data/ML_fighter_stats.csv')

    # Replace '--' with NaN
    X.replace('--', pd.NA, inplace=True)

    # Drop rows containing NaN
    X.dropna(inplace=True)

    # Get the dependent variable
    X.dropna(axis=0, subset=['WINNER'], inplace=True)
    y = X.WINNER
    X.drop(['WINNER'], axis=1, inplace=True)

    # Columns to Encode
    encode_cols = ['STANCE 1', 'STANCE 2', 'FIGHTER 1', 'FIGHTER 2']

    # Columns encoded
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[encode_cols]))

    # Remove categorical columns (will replace with one-hot encoding)
    num_X = X.drop(encode_cols, axis=1)
    cols = num_X.columns

    # Create StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    X_normalized = pd.DataFrame(scaler.fit_transform(num_X), columns=cols)

    # Add one-hot encoded columns to numerical features
    total_X = pd.concat([X_normalized, OH_cols], axis=1)

    # Ensure all columns have string type
    total_X.columns = total_X.columns.astype(str)

    # Break off  set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(total_X, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=False)

    return X_train, X_valid, y_train, y_valid