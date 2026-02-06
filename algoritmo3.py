import pandas as pd
import numpy as np
import time
import platform # Librería para detectar hardware
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input

# ==========================================
# 1. PREPARACIÓN DE DATOS (Misma lógica anterior)
# ==========================================
def cargar_datos_mini(ruta):
    # Carga rápida y limpieza básica para el test de velocidad
    df = pd.read_excel(ruta)
    # Limpieza rápida de columnas numéricas
    cols = ['Radiación Movil W/m2', 'Temp. Ambiente °C', 'Temp. Panel °C', 'Veloc. Viento m/s', 'Potencia kW']
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace('.', '').str.replace(',', '.')
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = df.dropna()
    X = df[['Radiación Movil W/m2', 'Temp. Ambiente °C', 'Temp. Panel °C', 'Veloc. Viento m/s']]
    y = df['Potencia kW']
    return X, y

# ==========================================
# 2. FUNCIÓN DE MEDICIÓN DE TIEMPO
# ==========================================
def medir_inferencia(modelo, X_muestra, nombre, es_deep_learning=False):
    print(f"Midiendo velocidad para {nombre}...")
    
    # "Calentamiento" (Warm-up) para cargar librerías en memoria
    if es_deep_learning:
        modelo.predict(X_muestra, verbose=0)
    else:
        modelo.predict(X_muestra)
    
    # Medición
    N_ITERACIONES = 100
    tiempos = []
    
    for _ in range(N_ITERACIONES):
        start = time.perf_counter() # Reloj de alta precisión
        if es_deep_learning:
            modelo.predict(X_muestra, verbose=0)
        else:
            modelo.predict(X_muestra)
        end = time.perf_counter()
        tiempos.append(end - start)
    
    # Calcular promedio por muestra (en milisegundos)
    # X_muestra tiene N filas. Queremos tiempo por fila.
    tiempo_total_promedio = np.mean(tiempos)
    tiempo_por_muestra_ms = (tiempo_total_promedio / len(X_muestra)) * 1000
    
    print(f"--> {nombre}: {tiempo_por_muestra_ms:.4f} ms/muestra")
    return tiempo_por_muestra_ms

def obtener_info_sistema():
    print("\n=== ESPECIFICACIONES DEL HARDWARE (Copiar al Paper) ===")
    try:
        print(f"Sistema Operativo: {platform.system()} {platform.release()}")
        print(f"Versión SO: {platform.version()}")
        print(f"Arquitectura: {platform.machine()}")
        print(f"Procesador: {platform.processor()}")
        print(f"Python Versión: {platform.python_version()}")
        # Nota: Para RAM exacta se requeriría la librería externa 'psutil', 
        # pero con CPU y SO suele ser suficiente contexto.
    except Exception as e:
        print(f"No se pudo detectar info del sistema: {e}")

# ==========================================
# 3. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    try:
        X, y = cargar_datos_mini('dataset_solar.xlsx')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # --- 1. LightGBM ---
        print("\nEntrenando LightGBM (rápido)...")
        lgbm = lgb.LGBMRegressor(n_estimators=100, verbose=-1) # Pocos estimadores solo para el test
        lgbm.fit(X_train, y_train)
        
        t_lgbm = medir_inferencia(lgbm, X_test, "LightGBM", es_deep_learning=False)
        
        # --- 2. Bi-LSTM ---
        print("\nEntrenando Bi-LSTM (rápido)...")
        scaler = MinMaxScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        X_test_rs = X_test_sc.reshape((X_test_sc.shape[0], 1, X_test_sc.shape[1]))
        
        model = Sequential([
            Input(shape=(1, 4)),
            Bidirectional(LSTM(10, return_sequences=False)), # Red pequeña para test
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_sc.reshape(-1,1,4), y_train, epochs=1, verbose=0)
        
        t_lstm = medir_inferencia(model, X_test_rs, "Bi-LSTM", es_deep_learning=True)
        
        print("\n=== RESULTADOS PARA EL PAPER ===")
        print(f"LightGBM Inferencia: {t_lgbm:.5f} ms")
        print(f"Bi-LSTM Inferencia:  {t_lstm:.5f} ms")
        ratio = t_lstm/t_lgbm if t_lgbm > 0 else 0
        print(f"LightGBM es {ratio:.1f} veces más rápido.")
        
        obtener_info_sistema()
        
    except Exception as e:
        print(f"Error: {e}")