# Forecasting Currency Volatility: A Deep Learning Study (GBP/IDR)

## 📌 Executive Summary
This project explores the application of Recurrent Neural Networks (RNN) to predict the exchange rate between the British Pound (GBP) and the Indonesian Rupiah (IDR). Utilizing real-world data from Yahoo Finance, the study implements a "Vanilla" RNN architecture to understand how sequential dependencies in financial markets can be modeled using deep learning.
**Key Objective:** To transition from static data analysis to temporal (time-based) forecasting, evaluating the strengths and limitations of basic RNNs in high-volatility environments.

## 🛠 Tech Stack & Tools
* **Data Acquisition:** `yfinance` (Yahoo Finance API)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Deep Learning Framework:** `TensorFlow` / `Keras`
* **Preprocessing:** `MinMaxScaler` (Scikit-Learn)

## 📈 Methodology
1. **Data Pipeline**
    - **Ticker:** GBPIDR=X
    - **Period:** Historical daily closing prices.
    - **Scaling:** Data was normalized to a range of $[0, 1]$ to ensure stable gradient descent during the training of the neural network.
2. **The Model Architecture**
For this introductory exploration, a Vanilla RNN was chosen. While more complex architectures like LSTMs exist, the Vanilla RNN serves as the fundamental building block for understanding:
    - Hidden State Persistence: How the model "remembers" previous days' prices.
    - Sequential Mapping: Mapping a look-back window (e.g., 30 days) to a single future prediction.
3. **Training Process**
    - **Loss Function:** Mean Squared Error (MSE)
    - **Optimizer:** Adam
    - **Validation:** A split-sample approach was used to test the model on unseen data, ensuring the model isn't simply memorizing the training set.

## 🔍 Key Findings & Analysis
**The "Lag" Effect**
In the resulting visualizations, the model shows a high degree of correlation with the actual price. However, a critical observation in this stage of deep learning is the "lag" effect—where the model often predicts the next day's price to be very close to today's price.
**Performance in Volatility**
The model effectively identifies general trends (Bullish/Bearish) but, like most baseline RNNs, struggles with "Black Swan" events or sudden spikes inherent in the Forex market.

## 🚀 Future Roadmap (Learning Progression)
As this represents an "Early Phase" project, the following improvements are planned to increase predictive accuracy:
1. Transition to LSTM/GRU: Implementing gated units to solve the vanishing gradient problem.
2. Multivariate Analysis: Incorporating external features like Interest Rates or Oil Prices to provide more context to the model.
3. Hyperparameter Optimization: Using Keras Tuner to find the optimal look-back window and neuron density.

## 📂 Project Structure
* `Foreign_Exchange_Temporal_Forecasting.ipynb`: Main development notebook.
* `data/`: (Optional) Cached CSVs of processed data.
* `assets/`: Visualizations and performance plots.

*Note: This project is for educational purposes and exposure to Deep Learning workflows. It is not intended for financial or algorithmic trading advice*.
