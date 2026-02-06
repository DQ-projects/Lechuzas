import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. CONFIGURACIÓN Y CARGA (Idéntico al anterior)
# ==========================================
COLS = {
    'fecha': 'Fecha',
    'target': 'Potencia kW',
    'features': ['Radiación Movil W/m2', 'Temp. Ambiente °C', 'Temp. Panel °C', 'Veloc. Viento m/s']
}

def limpiar_convertir(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    try:
        return float(x)
    except:
        return np.nan

def cargar_datos_paper(ruta):
    print("--- 1. Preparando Datos para Gráficos ---")
    df = pd.read_excel(ruta)
    df[COLS['fecha']] = pd.to_datetime(df[COLS['fecha']])
    df = df.sort_values(COLS['fecha']).set_index(COLS['fecha'])
    
    for col in COLS['features'] + [COLS['target']]:
        df[col] = df[col].apply(limpiar_convertir)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[COLS['target']] = df[COLS['target']].abs()
    
    # Truncamiento riguroso (Viento)
    idx = df[COLS['features'][-1]].first_valid_index()
    if idx: df = df.loc[idx:]
    
    df = df.interpolate(method='time', limit=4).dropna()
    df['Radiación Movil W/m2'] = df['Radiación Movil W/m2'].clip(lower=0)
    df.loc[df['Radiación Movil W/m2'] < 5, COLS['target']] = 0
    
    return df

def feature_engineering(df):
    d = df.copy()
    d['Hora'] = d.index.hour
    d['Mes'] = d.index.month
    d['Rad_Lag1'] = d['Radiación Movil W/m2'].shift(1)
    d['Temp_Lag1'] = d['Temp. Ambiente °C'].shift(1)
    return d.dropna()

# ==========================================
# 2. ENTRENAMIENTO RÁPIDO (Para obtener las predicciones)
# ==========================================
def obtener_predicciones(X_train, y_train, X_test, y_test):
    # --- LightGBM ---
    print("Entrenando LightGBM...")
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=40, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgbm = lgbm.predict(X_test)
    
    # --- Bi-LSTM ---
    print("Entrenando Bi-LSTM...")
    scaler = MinMaxScaler()
    Xt_sc = scaler.fit_transform(X_train)
    Xv_sc = scaler.transform(X_test)
    # Reshape
    Xt_rs = Xt_sc.reshape((Xt_sc.shape[0], 1, Xt_sc.shape[1]))
    Xv_rs = Xv_sc.reshape((Xv_sc.shape[0], 1, Xv_sc.shape[1]))
    
    model = Sequential([
        Input(shape=(1, Xt_sc.shape[1])),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xt_rs, y_train, validation_data=(Xv_rs, y_test), epochs=20, batch_size=64, verbose=0)
    p_lstm = model.predict(Xv_rs).flatten()
    
    return p_lgbm, p_lstm

# ==========================================
# 3. GENERACIÓN DE GRÁFICOS (Multilenguaje)
# ==========================================
def graficar_resultados(y_real, p_lgbm, p_lstm):
    # Configuración general de estilo
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    # Crear DataFrame de resultados para facilitar el ploteo
    res_df = pd.DataFrame({
        'Real': y_real,
        'LightGBM': p_lgbm,
        'Bi-LSTM': p_lstm
    }, index=y_real.index)

    # Diccionario de textos para traducción automática
    TEXTOS = {
        'es': {
            'fig1_title': 'Comparación de Modelos: Pronóstico de Potencia (Ventana 4 Días)',
            'fig1_ylabel': 'Potencia Activa (kW)',
            'fig1_xlabel': 'Fecha / Hora',
            'label_real': 'Datos Reales',
            'fig1_save': 'Fig1_Serie_Tiempo_Zoom_ES.png',
            'fig2_title': 'Correlación Real vs. Estimado',
            'fig2_xlabel': 'Potencia Real (kW)',
            'fig2_ylabel': 'Potencia Predicha (kW)',
            'label_perfect': 'Predicción Perfecta',
            'fig2_save': 'Fig2_Scatter_Plot_ES.png'
        },
        'en': {
            'fig1_title': 'Model Comparison: PV Power Forecasting (4-Day Window)',
            'fig1_ylabel': 'Active Power (kW)',
            'fig1_xlabel': 'Date / Time',
            'label_real': 'Actual Data',
            'fig1_save': 'Fig1_Time_Series_Zoom_EN.png',
            'fig2_title': 'Actual vs. Estimated Correlation',
            'fig2_xlabel': 'Actual Power (kW)',
            'fig2_ylabel': 'Predicted Power (kW)',
            'label_perfect': 'Perfect Prediction',
            'fig2_save': 'Fig2_Scatter_Plot_EN.png'
        }
    }

    # Bucle para generar gráficos en ambos idiomas
    for lang in ['es', 'en']:
        txt = TEXTOS[lang]
        print(f"Generando gráficos en idioma: {lang.upper()}...")

        # -------------------------------------------------------
        # GRÁFICO 1: ZOOM TEMPORAL (3-4 Días)
        # -------------------------------------------------------
        plt.figure(figsize=(12, 6))
        
        subset = res_df.iloc[:300] 
        
        plt.plot(subset.index, subset['Real'], label=txt['label_real'], color='black', linewidth=1.5, alpha=0.8)
        plt.plot(subset.index, subset['LightGBM'], label='LightGBM', color='#1f77b4', linestyle='--', linewidth=1.5)
        plt.plot(subset.index, subset['Bi-LSTM'], label='Bi-LSTM', color='#ff7f0e', linestyle='-.', linewidth=1.5)
        
        plt.title(txt['fig1_title'], fontsize=14, pad=15)
        plt.ylabel(txt['fig1_ylabel'])
        plt.xlabel(txt['fig1_xlabel'])
        plt.legend(frameon=True, loc='upper right')
        plt.tight_layout()
        plt.savefig(txt['fig1_save'], dpi=300)
        plt.close() # Cerrar para limpiar memoria

        # -------------------------------------------------------
        # GRÁFICO 2: SCATTER PLOT (Real vs Predicho)
        # -------------------------------------------------------
        plt.figure(figsize=(10, 8))
        
        sample = res_df.sample(n=min(5000, len(res_df)), random_state=42)
        
        # Calcular R2 para mostrarlos en la leyenda
        r2_lgbm = r2_score(y_real, p_lgbm)
        r2_lstm = r2_score(y_real, p_lstm)

        plt.scatter(sample['Real'], sample['LightGBM'], 
                    alpha=0.4, s=15, label=f'LightGBM ($R^2={r2_lgbm:.3f}$)', color='#1f77b4')
        
        plt.scatter(sample['Real'], sample['Bi-LSTM'], 
                    alpha=0.4, s=15, label=f'Bi-LSTM ($R^2={r2_lstm:.3f}$)', color='#ff7f0e', marker='x')

        max_val = sample['Real'].max()
        plt.plot([0, max_val], [0, max_val], 'k--', lw=2, label=txt['label_perfect'])
        
        plt.title(txt['fig2_title'], fontsize=14, pad=15)
        plt.xlabel(txt['fig2_xlabel'])
        plt.ylabel(txt['fig2_ylabel'])
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(txt['fig2_save'], dpi=300)
        plt.close()
    
    print("¡Proceso completado! Se han guardado 4 imágenes (2 ES, 2 EN) en alta resolución.")

if __name__ == "__main__":
    try:
        # Flujo completo
        df = cargar_datos_paper('dataset_solar.xlsx')
        df_proc = feature_engineering(df)
        
        features = COLS['features'] + ['Hora', 'Mes', 'Rad_Lag1', 'Temp_Lag1']
        X = df_proc[features]
        y = df_proc[COLS['target']]
        
        # Split (sin shuffle para mantener orden temporal)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Entrenar y Predecir
        pred_lgbm, pred_lstm = obtener_predicciones(X_train, y_train, X_test, y_test)
        
        # Graficar (ahora en ambos idiomas)
        graficar_resultados(y_test, pred_lgbm, pred_lstm)
        
    except FileNotFoundError:
        print("Error: No se encontró 'dataset_solar.xlsx'.")