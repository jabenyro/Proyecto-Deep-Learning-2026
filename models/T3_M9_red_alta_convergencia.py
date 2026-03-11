from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def crear_modelo(input_shape):

    modelo = Sequential([
        Input(shape=(input_shape,)),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        BatchNormalization(),

        Dense(16, activation='relu'),

        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.001)

    modelo.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return modelo