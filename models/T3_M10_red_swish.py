from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW

def crear_modelo(input_shape):

    modelo = Sequential([
        Input(shape=(input_shape,)),

        Dense(128, activation='swish'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='swish'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='swish'),
        BatchNormalization(),

        Dense(32, activation='swish'),

        Dense(1, activation='linear')
    ])

    optimizer = AdamW(learning_rate=0.001, weight_decay=0.004)

    modelo.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return modelo