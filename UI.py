import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os

# Load dataset
df = pd.read_csv("all_data.csv", low_memory=False)

# Drop missing/duplicate values
df = df.dropna()
df = df.drop_duplicates()

# Filter out invalid GPS coordinates
df = df[(df['latitude'] != 99.999) & (df['longitude'] != 99.999)]

# Drop unneeded columns
df = df.drop(columns=["time", "timezone"])

print(df.shape)

# Rename column names to pass into to_datetime function in appropriate format
df.rename(columns={'Date': 'day'}, inplace=True)
df.rename(columns={'min': 'minute'}, inplace=True)
df.rename(columns={'sec': 'second'}, inplace=True)

# Convert to datetime format
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'day', 'hour', 'minute', 'second']])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

def cluster_data():
    # Standardize features for clustering
    features = ['svr1', 'svr2', 'svr3', 'svr4', 'send_data', 'Transfer size', 'Bitrate', 
                'Transfer size-RX', 'Bitrate-RX']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Elbow method for ideal k
    inertia = []
    for k in range(2, 15):
        model = KMeans(n_clusters=k).fit(df[features])
        inertia.append(model.inertia_)

    # Ask user if they want to view elbow method graph
    view_elbow = input("Do you want to view the elbow method graph? (yes/no): ").strip().lower()
    if view_elbow == 'yes':
        # Plotting the inertia of the models    
        k_values = range(2, 15)
        plt.plot(k_values, inertia, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()

    # Ask user to choose the number of clusters
    while True:
        try:
            n_clusters = int(input("Enter the number of clusters (between 2 and 14): "))
            if 2 <= n_clusters <= 14:
                break
            else:
                print("Please enter a number between 2 and 14.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])
    
    # Plot clusters based on latitude and longitude
    scatter = plt.scatter(df['latitude'], df['longitude'], c=df['cluster'], cmap='viridis', s=0.1)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Clusters of Geographical Zones based on Network Performance')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster')
    plt.show()


def time_series_forecasting():
    # Ask which feature to train with, then process the data
    feature = input("Which feature would you like to train with? (Bitrate etc.): ").strip()
    bitrate = df[feature]

    # Data processing function
    def data_process(df, window):
        numpy_df = df.to_numpy()
        X = []
        y = []
        for i in range(len(numpy_df) - window):
            row = [[a] for a in numpy_df[i:i + 5]]  
            X.append(row)
            label = numpy_df[i + 5]
            y.append(label)
        return np.array(X), np.array(y)

    WINDOW_SIZE = 3600  # 1 hour
    X, y = data_process(bitrate, WINDOW_SIZE)

    # Split data into training, validation, and testing sets
    X_train, y_train = X[:1472000], y[:1472000]
    X_val, y_val = X[1472000:1659299], y[1472000:1659299]
    X_test, y_test = X[1659299:], y[1659299:]

    # Ask whether to train a new model or load an existing one
    action = input("Do you want to train a new model or load an existing model? (train/load): ").strip().lower()

    if action == 'train':
        # Build and train the model
        model = Sequential()
        model.add(InputLayer((WINDOW_SIZE, 1)))
        model.add(LSTM(64))
        model.add(Dense(8, 'relu'))
        model.add(Dense(1, 'linear'))

        
        epochs = int(input("How many rounds do you want to train (epochs): "))
        
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[Accuracy()])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

        
        # Save the model
        model_name = input("Enter a name for the saved model (without .h5): ").strip()
        model.save(f'{model_name}.h5')

    elif action == 'load':
        # Load the model
        model_name = input("Enter the name of the model you want to load (with .h5): ").strip()
        if os.path.exists(model_name):
            model = load_model(model_name)

            # Perform predictions
            predictions = model.predict(X_test).flatten()

            results = pd.DataFrame(data={'Predictions': predictions, 'Actual': y_test.flatten()})

            loss, rms = model.evaluate(X_test, y_test)
            
            shift_amount = 3600  
            results['Shifted_Predictions'] = np.nan
            results['Shifted_Predictions'][shift_amount:] = results['Predictions'][:-shift_amount]

            
            plt.plot(results['Actual'][:3600], label='Actual (Previous Hour)')
            plt.plot(results['Shifted_Predictions'][3600:7200], label='Predicted (Next Hour)')
            plt.xlabel('Time')
            plt.ylabel('Network Performance')
            plt.legend()
            plt.title('Time Series Forecasting Results')
            plt.show()
            
            print(loss,rms)
        else:
            print("Model file not found.")


while True:
    choice = input("Do you want to cluster or time series forecast? (cluster/forecast/exit): ").strip().lower()
    
    if choice == 'cluster':
        cluster_data()
    elif choice == 'forecast':
        time_series_forecasting()
    elif choice == 'exit':
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please select 'cluster', 'forecast', or 'exit'.")
