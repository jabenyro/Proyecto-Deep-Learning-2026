from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam


def crear_modelo(input_shape):

    modelo = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.0005)

    modelo.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return modelo