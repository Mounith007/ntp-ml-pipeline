# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def train_surrogate_model():
    # 1. Load Data
    df = pd.read_csv('ntp_synthetic_data.csv')

    # 2. Define Inputs (X) and Outputs (y)
    X = df[['Power (MW)', 'Flow Rate (kg/s)', 'Expansion Ratio']]
    y = df[['Thrust (N)', 'Isp (s)']]

    # 3. Split and Scale the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler().fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)


    # 4. Build the Neural Network Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'), # <-- NEW LAYER
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1]) # Output layer
    ])
    

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model Summary:")
    model.summary()

    # 5. Train the Model
    print("\nStarting model training...")
    model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=50, # Reduced for faster pipeline runs
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )

    # 6. Evaluate the Model
    loss = model.evaluate(X_test_scaled, y_test_scaled)
    print(f"\nFinal Test Loss (MSE): {loss}")

    # 7. Save the trained model
    model.save('ntp_surrogate_model.h5')
    print("\nModel saved as ntp_surrogate_model.h5")

if __name__ == "__main__":
    train_surrogate_model()