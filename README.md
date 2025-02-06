5G Network Performance Analysis and Forecasting

Project Overview
This project applies machine learning techniques to analyze and predict 5G network performance across different geographical zones. The primary objectives include:

Clustering geographical zones based on network performance characteristics (e.g., throughput, latency).
Predicting future network performance using time-series forecasting to detect potential network bottlenecks.
By leveraging KMeans clustering and LSTM-based time-series forecasting, this project provides insights for network engineers, telecom providers, and city planners to optimize 5G infrastructure and enhance service quality.

Features
Geographical Clustering: Uses KMeans to categorize different network zones based on performance metrics.
Time-Series Forecasting: Implements LSTM (Long Short-Term Memory) neural networks to predict future network conditions.
Performance Evaluation: Analyzes network data for insights into bottlenecks, service quality, and coverage gaps.
Interactive Visualization: Generates clustering scatter plots and time-series prediction graphs.

Dataset
2.1 Data Source
The dataset consists of 5G network performance data collected from mobile testing trucks across various geographic zones. The key features include:

Time Data: UNIX timestamp, datetime attributes (day, month, hour, etc.).
Location Data: GPS coordinates (latitude, longitude).
Truck Information: Truck ID, speed.
Server Measurements: Data from four different servers (svr1, svr2, svr3, svr4).
Data Transfer Metrics: Transfer size, Bitrate, Retransmissions, CWnd (congestion window).
Receive Data: Incoming transfer rates (Transfer size-RX, Bitrate-RX).
This dataset enables both clustering analysis and time-series forecasting for enhanced network performance planning.

Data Processing
🔹 Preprocessing Steps:

Dropped missing and duplicate values for data integrity.
Filtered invalid GPS coordinates (99.999 values removed).
Feature selection and engineering to retain relevant columns.
Standardized numerical values using StandardScaler from sklearn.
Created a datetime column for easier time-series modeling.
Applied a sliding time window (1-hour) for forecasting.

Machine Learning Models
1. KMeans Clustering (Geographical Analysis)
Goal: Categorize different network zones based on performance metrics.
Method:
Used Elbow Method to determine the optimal number of clusters (k).
Applied KMeans clustering from scikit-learn.
Visualized clusters using Matplotlib scatter plots.
2. LSTM Time-Series Forecasting (Predictive Analysis)
Goal: Predict future network performance using past trends.
Model Architecture:
LSTM Layer for sequential learning.
Dense Layers with ReLU activation.
Adam Optimizer with learning_rate=0.0001.
Evaluation Metric: Root Mean Square Error (RMSE).
Training Setup:
Train/Validation/Test Split: 80% / 10% / 10%.
Sliding Window Approach (Uses 5 hours of past data to predict the next hour).

Evaluation Metrics
KMeans Clustering:

Evaluated using Inertia Score (sum of squared distances from each point to its cluster centroid).
The elbow point helps determine the best k value.
LSTM Forecasting:

Evaluated using Root Mean Square Error (RMSE) to measure prediction accuracy.
Lower RMSE indicates better forecasting performance.

AI Demonstrator
Clustering Visualization
Takes latitude, longitude, and network performance metrics as input.
Generates scatter plots where colors indicate cluster groups.
Time-Series Forecasting Demo
Takes a chosen time window (e.g., 3600 seconds) as input.
Uses the LSTM model to predict the next hour’s network performance.
Produces a time-series plot comparing predictions vs. actual values.
