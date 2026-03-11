from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam

def crear_modelo(input_shape):
    entradas = Input(shape=(input_shape,))
    
    x1 = Dense(128, activation='relu')(entradas)
    x1 = BatchNormalization()(x1)
    
    x2 = Dense(128, activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    x3 = Add()([x1, x2])
    
    x4 = Dense(64, activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Dropout(0.2)(x4)
    
    salida = Dense(1, activation='linear')(x4)
    
    modelo = Model(inputs=entradas, outputs=salida)
    
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return modelo