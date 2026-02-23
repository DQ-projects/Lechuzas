import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
COLS = {
    'fecha': 'Fecha',
    'target': 'Potencia kW',
    'features': ['Radiación Movil W/m2', 'Temp. Ambiente °C', 'Temp. Panel °C', 'Veloc. Viento m/s']
}

def limpiar_convertir(x):
    """Limpia strings numéricos con formato europeo/latino."""
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    try:
        return float(x)
    except:
        return np.nan

def cargar_datos_caracterizacion(ruta):
    print("--- Cargando datos para Figura de Caracterización ---")
    df = pd.read_excel(ruta)
    
    # Formateo de fechas
    df[COLS['fecha']] = pd.to_datetime(df[COLS['fecha']])
    df = df.sort_values(COLS['fecha']).set_index(COLS['fecha'])
    
    # Limpieza numérica
    for col in COLS['features'] + [COLS['target']]:
        df[col] = df[col].apply(limpiar_convertir)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Correcciones físicas (como en la metodología)
    df[COLS['target']] = df[COLS['target']].abs() # Corregir bidireccional
    df['Radiación Movil W/m2'] = df['Radiación Movil W/m2'].clip(lower=0)
    
    # Forzar a 0 la potencia nocturna para limpiar el ruido de los sensores
    df.loc[df['Radiación Movil W/m2'] < 5, COLS['target']] = 0
    
    # Eliminar nulos remanentes
    df = df.dropna()
    print(f"Datos listos: {len(df)} registros.")
    return df

# ==========================================
# 2. GENERACIÓN DE LA FIGURA COMPUESTA
# ==========================================
def graficar_caracterizacion_q1(df):
    print("Generando gráficos de alta calidad (Q1)...")
    
    # Estilo general científico
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 13,
        'axes.labelsize': 12
    })

    # Para que se vea bien, calculamos la media móvil diaria máxima
    df_diario_max = df[COLS['target']].resample('D').max()
    df_diario_mean = df_diario_max.rolling(window=7, min_periods=1).mean() # Suavizado semanal

    # Filtramos la noche para que el gráfico de dispersión no tenga una línea gigante en 0
    df_dia = df[df['Radiación Movil W/m2'] > 10]
    
    # Para no saturar el PDF/PNG, tomamos una muestra representativa (ej. 10,000 puntos) si el dataset es muy grande
    if len(df_dia) > 10000:
        df_dia = df_dia.sample(n=10000, random_state=42)

    # Diccionario de textos multilenguaje
    TEXTOS = {
        'es': {
            'title_a': '(a) Estacionalidad Anual de la Generación FV',
            'ylabel_a': 'Potencia Activa (kW)',
            'xlabel_a': 'Fecha',
            'label_raw': 'Potencia Activa 15-min',
            'label_roll': 'Max. Móvil 7 Días',
            'title_b': '(b) Potencia vs. Irradiancia (Dispersión Térmica)',
            'xlabel_b': 'Irradiancia Global Horizontal ($W/m^2$)',
            'ylabel_b': 'Potencia Activa (kW)',
            'cbar_label': 'Temperatura del Panel (°C)',
            'filename': 'Fig1_Caracterizacion_Datos_ES.png'
        },
        'en': {
            'title_a': '(a) Annual Seasonality of PV Generation',
            'ylabel_a': 'Active Power (kW)',
            'xlabel_a': 'Date',
            'label_raw': '15-min Active Power',
            'label_roll': '7-Day Rolling Max Power',
            'title_b': '(b) Power vs. Irradiance (Thermal Dispersion)',
            'xlabel_b': 'Global Horizontal Irradiance ($W/m^2$)',
            'ylabel_b': 'Active Power (kW)',
            'cbar_label': 'Panel Temperature (°C)',
            'filename': 'Fig1_Data_Characterization_EN.png'
        }
    }

    # Bucle para generar ambas versiones
    for lang in ['es', 'en']:
        txt = TEXTOS[lang]
        print(f"\nGenerando versión en {lang.upper()}...")

        # Crear figura con 2 subplots (1 fila, 2 columnas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.5, 1]})
        
        # -----------------------------------------------------------
        # PANEL A: Serie Temporal (Estacionalidad)
        # -----------------------------------------------------------
        # Graficar todos los puntos en gris/celeste muy tenue (background)
        ax1.plot(df.index, df[COLS['target']], color='#a6cee3', alpha=0.3, linewidth=0.5, label=txt['label_raw'])
        
        # Graficar la envolvente máxima (tendencia estacional) en azul oscuro
        ax1.plot(df_diario_mean.index, df_diario_mean, color='#1f78b4', linewidth=2, label=txt['label_roll'])
        
        ax1.set_title(txt['title_a'], loc='left', fontweight='bold')
        ax1.set_ylabel(txt['ylabel_a'])
        ax1.set_xlabel(txt['xlabel_a'])
        
        # Formatear el eje X para mostrar los meses bonitos
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.legend(loc='upper right', frameon=True)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # -----------------------------------------------------------
        # PANEL B: Scatter Plot (Potencia vs Irradiancia + Efecto Térmico)
        # -----------------------------------------------------------
        # Scatter coloreado por Temperatura del Panel
        scatter = ax2.scatter(
            df_dia['Radiación Movil W/m2'], 
            df_dia[COLS['target']], 
            c=df_dia['Temp. Panel °C'], 
            cmap='jet', # Escala de colores de frío (azul) a calor (rojo)
            alpha=0.6, 
            s=10, 
            edgecolors='none'
        )

        ax2.set_title(txt['title_b'], loc='left', fontweight='bold')
        ax2.set_xlabel(txt['xlabel_b'])
        ax2.set_ylabel(txt['ylabel_b'])
        
        # Agregar barra de colores para explicar la 3ra variable
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label(txt['cbar_label'], rotation=270, labelpad=15)
        
        ax2.grid(True, linestyle='--', alpha=0.5)

        # -----------------------------------------------------------
        # Ajustes finales y guardado
        # -----------------------------------------------------------
        plt.tight_layout()
        
        # Guardar en alta resolución (Q1 exige mínimo 300 dpi)
        plt.savefig(txt['filename'], dpi=300, bbox_inches='tight')
        print(f"¡Gráfico guardado exitosamente como '{txt['filename']}'!")
        
        # Cerrar la figura para liberar memoria y evitar sobreescritura visual
        plt.close()

if __name__ == "__main__":
    try:
        # Reemplaza con el nombre real de tu archivo de datos
        df_datos = cargar_datos_caracterizacion('dataset_solar.xlsx')
        graficar_caracterizacion_q1(df_datos)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'dataset_solar.xlsx'. Asegúrate de que esté en la misma carpeta.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")