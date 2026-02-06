import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Input
from scipy import stats

# ==========================================
# 1. CONFIGURACIÓN (Misma anterior)
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

def dm_test(actual, pred1, pred2, h = 1, critic_value = 0.05, power = 2):
    # Rutina simple para Diebold-Mariano
    d = np.abs(actual - pred1)**power - np.abs(actual - pred2)**power
    d_mean = np.mean(d)
    d_var = np.var(d)
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def cargar_y_procesar(ruta):
    print("--- Procesando Datos ---")
    df = pd.read_excel(ruta)
    df[COLS['fecha']] = pd.to_datetime(df[COLS['fecha']])
    df = df.sort_values(COLS['fecha']).set_index(COLS['fecha'])
    
    for col in COLS['features'] + [COLS['target']]:
        df[col] = df[col].apply(limpiar_convertir)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[COLS['target']] = df[COLS['target']].abs()
    idx = df[COLS['features'][-1]].first_valid_index()
    if idx: df = df.loc[idx:]
    df = df.interpolate(method='time', limit=4).dropna()
    df['Radiación Movil W/m2'] = df['Radiación Movil W/m2'].clip(lower=0)
    df.loc[df['Radiación Movil W/m2'] < 5, COLS['target']] = 0
    
    # Feature Engineering
    df['Hora'] = df.index.hour
    df['Mes'] = df.index.month
    df['Rad_Lag1'] = df['Radiación Movil W/m2'].shift(1)
    df['Temp_Lag1'] = df['Temp. Ambiente °C'].shift(1)
    
    return df.dropna()

if __name__ == "__main__":
    df = cargar_y_procesar('dataset_solar.xlsx')
    features = COLS['features'] + ['Hora', 'Mes', 'Rad_Lag1', 'Temp_Lag1']
    X = df[features]
    y = df[COLS['target']]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # --- MODELOS ---
    print("Entrenando LightGBM...")
    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=40, verbose=-1)
    lgbm.fit(X_train, y_train)
    p_lgbm = lgbm.predict(X_test)
    
    print("Entrenando Bi-LSTM (Rápido para demo)...")
    # Nota: Usamos configuración rápida. Para paper final usa la config completa.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    Xt_sc = scaler.fit_transform(X_train)
    Xv_sc = scaler.transform(X_test)
    Xt_rs = Xt_sc.reshape((Xt_sc.shape[0], 1, Xt_sc.shape[1]))
    Xv_rs = Xv_sc.reshape((Xv_sc.shape[0], 1, Xv_sc.shape[1]))
    
    model = Sequential([
        Input(shape=(1, len(features))),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xt_rs, y_train, epochs=5, verbose=0) # Pocas épocas para prueba
    p_lstm = model.predict(Xv_rs, verbose=0).flatten()
    
    # --- CÁLCULO DE MÉTRICAS NUEVAS ---
    # 1. Modelo de Persistencia (Naive) para Skill Score
    # Predicción = Valor de hace 15 min (shift 1 en el test set)
    p_persistencia = y_test.shift(1).fillna(method='bfill')
    rmse_pers = np.sqrt(mean_squared_error(y_test, p_persistencia))
    
    # Métricas LightGBM
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, p_lgbm))
    mae_lgbm = mean_absolute_error(y_test, p_lgbm)
    ss_lgbm = 1 - (rmse_lgbm / rmse_pers)
    
    # Métricas LSTM
    rmse_lstm = np.sqrt(mean_squared_error(y_test, p_lstm))
    mae_lstm = mean_absolute_error(y_test, p_lstm)
    ss_lstm = 1 - (rmse_lstm / rmse_pers)
    
    # Diebold-Mariano Test
    dm_stat, p_value = dm_test(y_test.values, p_lgbm, p_lstm)
    
    print("\n=== TABLA DE RESULTADOS FINAL ===")
    print(f"{'Modelo':<10} | {'RMSE':<10} | {'MAE':<10} | {'Skill Score':<12}")
    print("-" * 50)
    print(f"{'LightGBM':<10} | {rmse_lgbm:.2f} | {mae_lgbm:.2f} | {ss_lgbm:.2%}")
    print(f"{'Bi-LSTM':<10} | {rmse_lstm:.2f} | {mae_lstm:.2f} | {ss_lstm:.2%}")
    print("-" * 50)
    print(f"Diebold-Mariano p-value: {p_value:.6f}")
    if p_value < 0.05:
        print(">> Diferencia Estadísticamente Significativa CONFIRMADA")
    else:
        print(">> Diferencia NO significativa")

    # --- GENERACIÓN DE GRÁFICO 3 (Análisis de Falla) ---
    print("\nGenerando Figura 3...")
    res_df = pd.DataFrame({'Real': y_test, 'LightGBM': p_lgbm, 'Bi-LSTM': p_lstm})
    res_df['Error_LGBM'] = res_df['Real'] - res_df['LightGBM']
    res_df['Error_LSTM'] = res_df['Real'] - res_df['Bi-LSTM']
    
    # Buscar día nublado (alta varianza) y día despejado
    # Heurística: Día con mayor desviación estándar vs menor
    daily_std = res_df['Real'].resample('D').std()
    day_cloudy = daily_std.idxmax().strftime('%Y-%m-%d')
    day_clear = daily_std.idxmin().strftime('%Y-%m-%d')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # (a) Día Nublado
    subset_c = res_df.loc[day_cloudy]
    axes[0].plot(subset_c.index, subset_c['Real'], 'k-', label='Real', alpha=0.7)
    axes[0].plot(subset_c.index, subset_c['LightGBM'], 'b--', label='LightGBM')
    axes[0].plot(subset_c.index, subset_c['Bi-LSTM'], 'r:', label='Bi-LSTM')
    axes[0].set_title(f'(a) Día con Alta Variabilidad (Nublado) - {day_cloudy}')
    axes[0].set_ylabel('Potencia (kW)')
    axes[0].legend()
    
    # (b) Día Despejado
    subset_cl = res_df.loc[day_clear]
    axes[1].plot(subset_cl.index, subset_cl['Real'], 'k-', label='Real', alpha=0.7)
    axes[1].plot(subset_cl.index, subset_cl['LightGBM'], 'b--', label='LightGBM')
    axes[1].plot(subset_cl.index, subset_cl['Bi-LSTM'], 'r:', label='Bi-LSTM')
    axes[1].set_title(f'(b) Día Despejado (Perfil Suavizado) - {day_clear}')
    axes[1].set_ylabel('Potencia (kW)')
    
    # (c) Residuos (mismo día nublado para ver errores)
    axes[2].plot(subset_c.index, subset_c['Error_LGBM'], 'b', label='Error LightGBM')
    axes[2].plot(subset_c.index, subset_c['Error_LSTM'], 'r', alpha=0.6, label='Error Bi-LSTM')
    axes[2].axhline(0, color='k', linestyle='-', linewidth=1)
    axes[2].set_title(f'(c) Distribución de Errores Instantáneos - {day_cloudy}')
    axes[2].set_ylabel('Error (kW)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('Fig3_Analisis_Fallas.png', dpi=300)
    print("Figura 3 guardada.")