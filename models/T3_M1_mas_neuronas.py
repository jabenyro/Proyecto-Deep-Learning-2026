from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def crear_modelo(input_shape):
    
    modelo = Sequential([
        Input(shape=(input_shape,)),
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