import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read the data
df = pd.read_csv('all_data.csv', low_memory=False)

# Standard preprocessing
df = df.dropna()
df = df.drop_duplicates()


# Drop rows with invalid GPS coordinates
df = df[(df['latitude'] != 99.999) & (df['longitude'] != 99.999)]

# Drop unnecessary columns
df = df.drop(columns=['time', 'timezone'])

# Clustering
features = ['svr1', 'svr2', 'svr3', 'svr4', 'send_data', 'Transfer size', 'Bitrate']
kmeans = KMeans(n_clusters=5)
df['cluster'] = kmeans.fit_predict(df[features])

# Plot clusters
# plt.scatter(df['latitude'], df['longitude'], c=df['cluster'])
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.title('Clusters based on network performance')
# plt.show()

# Forecasting
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Date', 'hour', 'min', 'sec']])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# Example: Forecasting 'Bitrate'
model = ExponentialSmoothing(df['Bitrate'], trend='add', seasonal='add', seasonal_periods=24)
fit = model.fit()
forecast = fit.forecast(steps=24)

# Plot forecast
plt.plot(df['Bitrate'], label='Historical')
plt.plot(forecast, label='Forecast', color='red')
plt.xlabel('Time')
plt.ylabel('Bitrate')
plt.title('Bitrate Forecast')
plt.legend()
plt.show()

print(df.head())