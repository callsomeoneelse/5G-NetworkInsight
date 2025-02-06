import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


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
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'day', 'hour', 'minute', 'second']], format="%d.%m.%Y %H:%M:%S")
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)


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


# model = load_model('LSTM.h5')

# predictions = model.predict(X_test).flatten()

# results = pd.DataFrame(data={'Predictions' : predictions, 'Actual' : y_test})


# shift_amount = 3600  # Modify this according to the frequency of your data
# results['Shifted_Predictions'] = np.nan
# results['Shifted_Predictions'][shift_amount:] = results['Predictions'][:-shift_amount]

# # Plotting
# plt.plot(results['Actual'][:3600], label='Actual (Previous Hour)')
# plt.plot(results['Shifted_Predictions'][3600:7200], label='Predicted (Next Hour)')

# plt.xlabel('Time')
# plt.ylabel('Network Performance')
# plt.legend()
# plt.show()