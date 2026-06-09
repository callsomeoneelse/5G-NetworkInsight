import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


#GPU SETUP
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {[g.name for g in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected — running on CPU.")




# SHARED PREPROCESSING STEPS
def load_and_preprocess_data(filepath="data/all_data.csv"):

    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    #Basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()

    # Filter out invalid GPS coordinates
    df = df[(df['latitude'] != 99.999) & (df['longitude'] != 99.999)]

    # Drop unneeded columns
    df = df.drop(columns=["time", "timezone"])

    # Rename column names to pass into to_datetime function in appropriate format
    df =df.rename(columns={'Date': 'day', 'min': 'minute', 'sec': 'second'})

    # Convert to datetime format
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'day', 'hour', 'minute', 'second']])
    df = df.set_index('datetime').sort_index()

    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


#CLUSTERING
CLUSTER_FEATURES = [
    'svr1','svr2', 'svr3', 'svr4',
    'send_data', 'Transfer size', 'Bitrate',
    'Transfer size-RX', 'Bitrate-RX'
]


def cluster_data(df):

    data = df.copy()

    scaler = StandardScaler()
    data[CLUSTER_FEATURES] = scaler.fit_transform(data[CLUSTER_FEATURES])

    # Elbow method for ideal k
    view_elbow = input("\nRun elbow method to find optimal k? (yes/no): ").strip().lower()
    if view_elbow == 'yes':
        print("Computing inertia for k = 2..14 (this may take a minute)...")
        inertia = []
        for k in range(2, 15):
            inertia.append(KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data[CLUSTER_FEATURES]).inertia_)
 
        plt.figure()
        plt.plot(range(2, 15), inertia, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.tight_layout()
        plt.savefig('outputs/elbow_plot.png', dpi=150)
        plt.show()
        print("Saved → elbow_plot.png")

    # Choose k
    while True:
        try:
            n_clusters = int(input("Enter the number of clusters (2-14): "))
            if 2 <= n_clusters <= 14:
                break
            else:
                print("Please enter a number between 2 and 14.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Fit and evaluate
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    data['cluster'] = kmeans.fit_predict(data[CLUSTER_FEATURES])
 
    db_index = davies_bouldin_score(data[CLUSTER_FEATURES], data['cluster'])
    print(f"\nDavies-Bouldin Index (lower is better): {db_index:.4f}")
    

    # Plot geographic clusters
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='viridis', s=0.1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clusters of Geographical Zones based on Network Performance')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('outputs/cluster_map.png', dpi=150)
    plt.show()
    print("Saved → cluster_map.png")

#Time-Series Data Prep
def build_sequences(series: pd.Series, window: int):
    
    #slide a window over a 'series' to build (X,y) arrays for LSTM training

    values = series.to_numpy()
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window].reshape(window, 1))
        y.append(values[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def split_data(X, y, train_ratio=0.80, val_ratio=0.10):
    
    n = len(X)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return (X[:train_end],  y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:],   y[val_end:])

#Forecasting

WINDOW_SIZE = 300  # 5 minutes of data (use 3600 for 1 hour on GPU)

def time_series_forecasting(df):
    #choose feature
    FORECAST_FEATURES = ['Bitrate', 'Bitrate-RX', 'Transfer size', 'Transfer size-RX', 'svr1', 'svr2', 'svr3', 'svr4']
    print("\nForecastable features:")
    for i, f in enumerate(FORECAST_FEATURES, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            choice = int(input("Enter the number of the feature to forecast: "))
            if 1 <= choice <= len(FORECAST_FEATURES):
                feature = FORECAST_FEATURES[choice - 1]
                break
            print(f"Please enter a number between 1 and {len(FORECAST_FEATURES)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    action = input("Train a new model or load an existing one? (train/load): ").strip().lower()

    if action == 'train':
        print(f"\nBuilding sequences (window={WINDOW_SIZE}) — this can take several minutes for large datasets...")

        X, y = build_sequences(df[feature], WINDOW_SIZE)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
 
        print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

        epochs = int(input("How many epochs to train?: "))
 
        model = Sequential([
            InputLayer((WINDOW_SIZE, 1)),
            LSTM(64),
            Dense(8, activation='relu'),
            Dense(1, activation='linear'),
        ])
        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.0001),
            metrics=[RootMeanSquaredError()]
        )
        model.summary()
 
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=epochs,
                  batch_size=64)
 
        test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest MSE:  {test_loss:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")
 
        model_name = input("Name for saved model (without extension): ").strip()
        save_path = f"{model_name}.keras"
        model.save(save_path)
        print(f"Saved → {save_path}")
 
        _plot_predictions(model, X_test, y_test, feature)
        
    elif action == 'load':
        model_name = input("Model filename (e.g. my_model.keras or my_model.h5): ").strip()
        if not os.path.exists(model_name):
            print("Model file not found.")
            return
 
        print("Loading model...")
        model = load_model(model_name)
 
        print(f"Building sequences (window={WINDOW_SIZE}) ...")
        X, y = build_sequences(df[feature], WINDOW_SIZE)
        _, _, _, _, X_test, y_test = split_data(X, y)
 
        test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest MSE:  {test_loss:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")
 
        _plot_predictions(model, X_test, y_test, feature)
 
    else:
        print("Invalid choice.")

def _plot_predictions(model, X_test, y_test, feature_name):
    """Predict on the first 2 hours of test data and plot actual vs predicted."""
    plot_len = min(WINDOW_SIZE * 2, len(X_test))
    predictions = model.predict(X_test[:plot_len], verbose=0).flatten()
    actual      = y_test[:plot_len]
 
    # Shift predictions by one window so they represent the *next* hour
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual[:WINDOW_SIZE], label='Actual')
    ax.plot(predictions[WINDOW_SIZE:plot_len], label='Predicted', color='red')
    ax.set_xlabel('Time steps')
    ax.set_ylabel(feature_name)
    ax.set_title(f'{feature_name} — Actual vs Predicted')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/forecast_plot.png', dpi=150)
    plt.show()
    print("Saved → forecast_plot.png")


def main():
    os.makedirs('outputs', exist_ok=True)

    df = load_and_preprocess_data()
 
    while True:
        print("\n--- Menu ---")
        choice = input("What would you like to do? (cluster / forecast / exit): ").strip().lower()
 
        if choice == 'cluster':
            cluster_data(df)
        elif choice == 'forecast':
            time_series_forecasting(df)
        elif choice == 'exit':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Enter 'cluster', 'forecast', or 'exit'.")
 
 
if __name__ == "__main__":
    main()
