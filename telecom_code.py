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
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

#load_dataset
df = pd.read_csv("all_data.csv", low_memory=False)

#drop missing/duplicate values
df = df.dropna()
df = df.drop_duplicates()

#filter out invalid GPS coordinates
df = df[(df['latitude'] != 99.999) & (df['longitude'] != 99.999)]

#drop unneeded columns
df = df.drop(columns=["time", "timezone"])

#rename column names to pass into to_datetime function in appropriate format
df.rename(columns={'Date': 'day'}, inplace=True)
df.rename(columns={'min': 'minute'}, inplace=True)
df.rename(columns={'sec': 'second'}, inplace=True)

#convert to datetime format
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'day', 'hour', 'minute', 'second']])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)


#standardize features for clustering
features = ['svr1', 'svr2', 'svr3', 'svr4', 'send_data', 'Transfer size', 'Bitrate', 
            'Transfer size-RX', 'Bitrate-RX']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


# Elbow method for ideal k
inertia = []
for k in range(2,15):
    model = KMeans(n_clusters=k).fit(df[features])
    inertia.append(model.inertia_)

#Plotting the inertia of the models    
k_values = range(2,15)
plt.plot(k_values, inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(df[features])



# Plot clusters based on latitude and longitude
scatter = plt.scatter(df['latitude'], df['longitude'], c=df['cluster'], cmap='viridis', s=0.1)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Clusters of Geographical Zones based on Network Performance')
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')
plt.show()



clusters = df['cluster']  # This should be your cluster labels
features = df[features]

# Calculate evaluation metrics

db_index = davies_bouldin_score(features, clusters)


print(f'Davies-Bouldin Index: {db_index}')




bitrate = df["Bitrate"]

def data_process(df, window):
    numpy_df = df.to_numpy()
    
    X = []
    y = []
    
    for i in range(len(numpy_df) - window):
        row = [[a] for a in numpy_df[i:i+5]]
        X.append(row)
        label = numpy_df[i+5]
        y.append(label)
    
    return np.array(X), np.array(y)


WINDOW_SIZE = 3600 #1 hour

X,y = data_process(bitrate, WINDOW_SIZE)


X_train, y_train = X[:1472000], y[:1472000]
X_val, y_val = X[1472000:1659299], y[1472000:1659299]
X_test, y_test = X[1659299:], y[1659299:]


model = Sequential()
model.add(InputLayer((WINDOW_SIZE, 1)))
model.add(LSTM(64))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)

print(test_loss, test_rmse)

model.save('LSTM.h5')


model = load_model('LSTM.h5')

predictions = model.predict(X_test).flatten()

results = pd.DataFrame(data={'Predictions' : predictions, 'Actual' : y_test})


shift_amount = 3600  # Modify this according to the frequency of your data
results['Shifted_Predictions'] = np.nan
results['Shifted_Predictions'][shift_amount:] = results['Predictions'][:-shift_amount]

# Plotting
plt.plot(results['Actual'][:3600], label='Actual (Previous Hour)')
plt.plot(results['Shifted_Predictions'][3600:7200], label='Predicted (Next Hour)')

plt.xlabel('Time')
plt.ylabel('Network Performance')
plt.legend()
plt.show()












