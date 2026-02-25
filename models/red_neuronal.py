from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def crear_red_simple(input_shape):
    modelo = Sequential([
        Input(shape=(input_shape,)),
        Dense(4, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo