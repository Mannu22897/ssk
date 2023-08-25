wrong_prediction_made = False  # Flag to track if a wrong prediction was made
import optuna
from iqoptionapi.stable_api import IQ_Option
import logging
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score
from joblib import dump, load

logging.basicConfig(level=logging.INFO, format='%(message)s')

CANDLE_TIME = 2  # minutes
CANDLE_NUMBER = 1000
ASSETS = ["EURUSD-OTC"]
SHORT_SMA_PERIOD = 5
MEDIUM_SMA_PERIOD = 20
LONG_SMA_PERIOD = 50
ADDITIONAL_WAIT_TIME = 0  # seconds
investing = 20
s = "call"
def login_iq_option(email, password):
    api = IQ_Option(email, password)
    api.connect()
    api.change_balance('REAL')
    return api

def get_all_data(api, asset, interval, amount):
    candles = api.get_candles(asset, interval * 60, amount, time.time())
    with open('raw_data.txt', 'w') as file:
        for candle in candles:
            file.write(str(candle) + '\n')
    df = pd.DataFrame(candles)
    df = df.rename(columns={"from": "time", "min": "low", "max": "high"})
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "close", "high", "low"]]
    with open('converted_timestamps.txt', 'w') as file:
        for timestamp in df["time"]:
            file.write(str(timestamp) + '\n')
    return df

def compute_features(data):
    # Existing SMA calculations
    data["sma_short"] = data["close"].rolling(window=SHORT_SMA_PERIOD).mean()
    data["sma_medium"] = data["close"].rolling(window=MEDIUM_SMA_PERIOD).mean()
    data["sma_long"] = data["close"].rolling(window=LONG_SMA_PERIOD).mean()

    # Bollinger Bands
    sma_20 = data["close"].rolling(window=20).mean()
    std_dev_20 = data["close"].rolling(window=20).std()
    data["upper_band"] = sma_20 + (std_dev_20 * 2)
    data["lower_band"] = sma_20 - (std_dev_20 * 2)

    # Fibonacci Retracements
    peak = data["high"].max()
    trough = data["low"].min()
    diff = peak - trough
    data["fib_23"] = peak - diff * 0.236
    data["fib_38"] = peak - diff * 0.382
    data["fib_50"] = peak - diff * 0.5
    data["fib_61"] = peak - diff * 0.618
   
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Target variable
    data["target"] = data["close"].shift(-1) > data["close"]
    data["target"] = data["target"].astype(int)

    return data.dropna()

def determine_actual_outcome(data_point):
    if data_point['close'].iloc[0] > data_point['open'].iloc[0]:
        return "call"
    else:
        return "put"


# Hyperparameter tuning objective function
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Hyperparameter tuning objective function
def objective(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )

    # Using TimeSeriesSplit for time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    return cross_val_score(clf, X, y, cv=tscv).mean()

def train_decision_tree(data):
    if "time" in data.columns:
        X = data.drop(["time", "target"], axis=1)
    else:
        X = data.drop(["target"], axis=1)
    y = data["target"]

    # Define the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=10)

    # Getting the best hyperparameters
    best_params = study.best_params
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest Classifier with the best hyperparameters
    clf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Split data to calculate accuracy
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Random Forest Accuracy: {accuracy:.2f}")

    # Save the model
    dump(clf, 'trained_model.joblib')

    # Return the classifier and feature names
    return clf, X.columns.tolist()


def update_model_with_feedback(model, previous_prediction, actual_outcome, combined_data):
    predicted_outcome = previous_prediction['outcome']
    confidence = previous_prediction['confidence']
    if predicted_outcome != actual_outcome:
        confidence *= 0.9  # Example: reduce confidence by 10%
        updated_model, _ = train_decision_tree(combined_data)
    else:
        updated_model = model
    return updated_model, confidence

def update_model_with_feedback(model, previous_prediction, actual_outcome, combined_data):
    predicted_outcome = previous_prediction['outcome']
    confidence = previous_prediction['confidence']
    updated_model = model  # Default to the current model
    if predicted_outcome != actual_outcome:
        # Retrain the model if the prediction was wrong
        updated_model, _ = train_decision_tree(combined_data)
        confidence *= 0.9  # Example: reduce confidence by 10%
    return updated_model, confidence
def main(api_iq):
    wrong_prediction_made = False 
    logging.info("Fetching data for " + ASSETS[0])
    data = get_all_data(api_iq, ASSETS[0], CANDLE_TIME, CANDLE_NUMBER)
    logging.info("Data for " + ASSETS[0] + " fetched")

    logging.info("Computing features")
    data_with_features = compute_features(data)  # Define data_with_features here
    logging.info("Features computed")

    try:
        logging.info("Attempting to load existing model")
        model, combined_data = load('trained_model.joblib')
        # If loading an existing model, ensure feature_names is assigned appropriately
        feature_names = combined_data.columns.tolist()[:-1]  # You should define this part
        logging.info("Existing model loaded successfully")
    except FileNotFoundError:
        logging.info("No existing model found, training a new model")
        model, feature_names = train_decision_tree(data_with_features)  # feature_names is assigned here
        combined_data = data_with_features.copy()
        dump((model, combined_data), 'trained_model.joblib')
        logging.info("New model trained and saved")
    logging.info("Decision Tree model ready")


    # Storing the initial data for future use
    combined_data = data_with_features.copy()

   # Initialize previous_prediction before entering the loop
    previous_prediction = None

    while True:

        # Synchronize with the clock to place the trade at the exact 2-minute expiry time
        while int(time.time() % 60) != 0:
            time.sleep(0.1)

        logging.info("Fetching new data to make trading decisions")
        data = get_all_data(api_iq, ASSETS[0], CANDLE_TIME, CANDLE_NUMBER)
        logging.info("New data fetched")
    
        new_data_with_features = compute_features(data)
        logging.info("New features computed")

        # Combining old data with new data
        logging.info("Combining old and new data")
        combined_data = pd.concat([combined_data, new_data_with_features])
        logging.info("Data combined successfully")
    
        # Extract the most recent timestamp before dropping the 'time' column
        recent_timestamp = combined_data['time'].iloc[-1]

        # Exclude 'time' column from the training data
        training_data = combined_data.drop("time", axis=1)
        feature_names = training_data.columns.tolist()[:-1]  # Get feature names without 'time'

        # Retraining the model with combined data (without 'time' column)
        if wrong_prediction_made:
            logging.info("Retraining the model with combined data")
            model, _ = train_decision_tree(training_data.drop("target", axis=1))   # feature_names is updated here
            dump((model, combined_data), 'trained_model.joblib')
            logging.info("Model retrained successfully")
            print("Features used for training:", feature_names)
            wrong_prediction_made = False  # Reset the flag

        recent_data = training_data.drop("target", axis=1).iloc[-1]  # Exclude 'target' column for prediction

        if previous_prediction is not None:
            actual_data_point = combined_data[combined_data['time'] == previous_prediction['timestamp']]
            actual_outcome = determine_actual_outcome(actual_data_point)
            if previous_prediction['outcome'] == actual_outcome:
                logging.info("Won the previous trade!")
            else:
                logging.info("Lost the previous trade!")
            # Optional: Update model or confidence scoring based on feedback
            model, confidence = update_model_with_feedback(model, previous_prediction, actual_outcome, combined_data)

        # Get probability estimates for the two classes (Call and Put)
        probs = model.predict_proba(pd.DataFrame([recent_data], columns=feature_names))[0]
        ml_signal = 1 if probs[1] > 0.5 else 0
        confidence = max(probs) # Confidence in the prediction
        ml_signal_str = "call" if ml_signal == 1 else "put"
        logging.info(f"ML signal: {ml_signal_str}")
        logging.info(f"Confidence: {confidence:.2f}")

        # Define a base investment amount
        base_investment = 10
        investment_scaling_factor = 1 + (confidence - 0.5) * 2
        investing_amount = base_investment * investment_scaling_factor
        logging.info(f"Investment amount: {investing_amount:.2f}")

        # Store details of the previous prediction
        previous_prediction = {
           'timestamp': recent_timestamp,
           'outcome': ml_signal_str,
           'confidence': confidence,
           'investment': investing_amount
           }


        # Print the features used for prediction
        print("Features used for prediction:", feature_names)

        # Get probability estimates for the two classes (Call and Put)
        probs = model.predict_proba(pd.DataFrame([recent_data], columns=feature_names))[0]
        ml_signal = 1 if probs[1] > 0.5 else 0
        confidence = max(probs) # Confidence in the prediction
        ml_signal_str = "call" if ml_signal == 1 else "put"
        logging.info(f"ML signal: {ml_signal_str}")
        logging.info(f"Confidence: {confidence:.2f}")


        # Place the trade with the scaled investment amount
        ID = api_iq.buy(investing_amount, ASSETS[0], ml_signal_str, 2)
        logging.info(f"Placing a {ml_signal_str} trade")

        # Wait for 2 minutes plus the additional wait time
        logging.info(f"Waiting for {CANDLE_TIME * 60 + ADDITIONAL_WAIT_TIME} seconds before the next iteration")
        time.sleep(CANDLE_TIME * 60 + ADDITIONAL_WAIT_TIME)


if __name__ == "__main__":
    email = input("Please enter your login information...\nEmail: ")
    password = input("Password: ")
    api_iq = login_iq_option(email, password)
    logging.info("Successfully logged in!")
    logging.info(f"Balance mode: {api_iq.get_balance_mode().upper()}")
    logging.info(f"Balance: {api_iq.get_balance()}")
    main(api_iq)
