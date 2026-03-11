from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def crear_modelo(input_shape):
    entradas = Input(shape=(input_shape,))
    
    deep = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(entradas)
    deep = BatchNormalization()(deep)
    deep = Dropout(0.3)(deep)
    
    deep = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(deep)
    deep = BatchNormalization()(deep)
    deep = Dropout(0.2)(deep)
    
    deep = Dense(32, activation='relu')(deep)
    
    concat = concatenate([entradas, deep])
    
    salida = Dense(1, activation='linear')(concat)
    
    modelo = Model(inputs=entradas, outputs=salida)
    
    modelo.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mae']
    )
    
    return modelo