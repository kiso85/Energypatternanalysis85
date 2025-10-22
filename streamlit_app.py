# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io
import locale

# --- Configuración de la Página y Localización ---
st.set_page_config(
    page_title="Dashboard Energético Avanzado",
    page_icon="⚡",
    layout="wide",
)
# Establecer localización en español para nombres de meses y días
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    st.warning("Localización en español no disponible, los nombres de los días/meses pueden aparecer en inglés.")
    locale.setlocale(locale.LC_TIME, '')


# --- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_csv_from_source(source):
    """Carga datos desde un archivo subido o una URL de GitHub."""
    if source is None:
        return pd.DataFrame()
    try:
        if isinstance(source, str) and source.startswith('http'):
            response = requests.get(source)
            response.raise_for_status() # Lanza un error si la URL falla (ej. 404)
            return pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
        return pd.read_csv(source, skipinitialspace=True)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

# ✨ FUNCIÓN RESTAURADA (del dashboard que sí funciona)
@st.cache_data
def get_github_csv_files(repo, path="data"):
    """Obtiene una lista de archivos CSV de un directorio en un repositorio de GitHub."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        response = requests.get(url=api_url)
        response.raise_for_status() # Lanza error si la API falla (ej. 404)
        files_json = response.json()
        if isinstance(files_json, dict) and 'message' in files_json:
            st.error(f"No se pudo encontrar el directorio '{path}'. Error de GitHub: {files_json['message']}")
            return []
        return [file['name'] for file in files_json if file['type'] == 'file' and file['name'].endswith('.csv')]
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de GitHub: {e}")
        return []

# --- Gestión de Agregación Temporal ---
def gestionar_agregacion_temporal(df, nivel):
    """
    Agrega (resume) el DataFrame a un nivel temporal específico (Diario, Mensual).
    """
    if df.empty:
        return pd.DataFrame()

    reglas = { "Diario": "D", "Mensual": "MS" }
    
    if nivel in reglas:
        df_agregado = df.resample(reglas[nivel]).agg({
            'Consumption_kWh': 'mean'
        }).dropna()
        return df_agregado
    
    return df


# --- Inicialización de DataFrames ---
df_energy = pd.DataFrame()
df_weather_raw = pd.DataFrame()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    
    st.header("1. Fuente de Datos")
    source_type = st.radio("Seleccionar origen", ["Cargar Archivos", "Desde GitHub"], key="source_type")

    df_energy_raw = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy_file = st.file_uploader("Archivo de Consumo (CSV)", type="csv")
        uploaded_weather_file = st.file_uploader("Archivo de Clima (CSV)", type="csv")
        df_energy_raw = load_csv_from_source(uploaded_energy_file)
        df_weather_raw = load_csv_from_source(uploaded_weather_file)
    else:
        # --- ✨ LÓGICA DE CARGA RESTAURADA (COMO EL DASHBOARD 2) ---
        
        # El repositorio correcto
        github_repo = st.text_input("Repositorio GitHub (usuario/repo)", "hardik5838/EnergyPatternAnalysis")
        
        if github_repo:
            # ✨ LA CORRECCIÓN CLAVE: path="data" (en minúscula)
            csv_files_list = get_github_csv_files(github_repo, path="data")
            
            if csv_files_list:
                # ✨ LA CORRECCIÓN CLAVE: base_url usa "data" (en minúscula)
                base_url = f"https://raw.githubusercontent.com/{github_repo}/main/data/"
                
                # Encontrar archivos por nombre (más robusto)
                energy_file_default = next((f for f in csv_files_list if 'energy' in f.lower()), None)
                weather_file_default = next((f for f in csv_files_list if 'weather' in f.lower()), None)
                
                # Índices para los selectbox
                index_energy = csv_files_list.index(energy_file_default) if energy_file_default else 0
                index_weather = csv_files_list.index(weather_file_default) if weather_file_default else 0
                
                selected_energy_file = st.selectbox(
                    "Selecciona archivo de consumo", 
                    csv_files_list, 
                    index=index_energy
                )
                selected_weather_file = st.selectbox(
                    "Selecciona archivo de clima", 
                    csv_files_list, 
                    index=index_weather
                )
                
                if selected_energy_file:
                    df_energy_raw = load_csv_from_source(base_url + selected_energy_file)
                if selected_weather_file:
                    df_weather_raw = load_csv_from_source(base_url + selected_weather_file)
            else:
                st.warning("No se encontraron archivos CSV en la carpeta 'data/' del repositorio.")

    # --- Procesamiento y Filtros ---
    if not df_energy_raw.empty:
        try:
            df_energy = df_energy_raw.copy()
            
            # Lógica de renombrado de columnas
            if 'Fecha' in df_energy.columns and 'Energía activa (kWh)' in df_energy.columns:
                 df_energy.rename(columns={'Fecha': 'datetime', 'Energía activa (kWh)': 'Consumption_kWh'}, inplace=True)
                 df_energy['datetime'] = pd.to_datetime(df_energy['datetime'], format='%d/%m/%Y %H:%M')
            elif 'Date & Time' in df_energy.columns and 'Consumption(kWh)' in df_energy.columns:
                 df_energy.rename(columns={'Date & Time': 'datetime', 'Consumption(kWh)': 'Consumption_kWh'}, inplace=True)
                 df_energy['datetime'] = pd.to_datetime(df_energy['datetime'])
            else:
                st.sidebar.error("Formato de CSV de energía no reconocido.")
                df_energy = pd.DataFrame()

            if not df_energy.empty:
                df_energy.set_index('datetime', inplace=True)
                st.sidebar.success("Datos de consumo cargados.")

                st.sidebar.markdown("---")
                st.sidebar.header("2. Filtros de Datos")
                
                dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
                selected_days = st.sidebar.multiselect("Días de la semana", options=list(dias_semana.keys()), format_func=lambda x: dias_semana[x], default=list(dias_semana.keys()))
                selected_hours = st.sidebar.slider("Horas del día", 0, 23, (0, 23))
                
                min_date, max_date = df_energy.index.min().date(), df_energy.index.max().date()
                date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

                st.sidebar.markdown("---")
                st.sidebar.header("3. Ajustes de Análisis")

                aggregation_level = st.sidebar.selectbox(
                    "Ver datos por",
                    options=["Horario", "Diario", "Mensual"],
                    index=0,
                    help="Cambia la resolución del gráfico de evolución temporal."
                )
                
                remove_baseline = st.sidebar.checkbox("Eliminar consumo base")
                baseline_threshold = st.sidebar.number_input("Umbral base (kWh)", value=float(df_energy['Consumption_kWh'].quantile(0.1)), disabled=not remove_baseline)
                
                remove_anomalies = st.sidebar.checkbox("Eliminar anomalías (picos)")
                anomaly_percentile = st.sidebar.number_input("Percentil para anomalías", value=99.0, min_value=90.0, max_value=100.0, disabled=not remove_anomalies)
                
                st.sidebar.markdown("---")
                st.sidebar.header("4. Constantes Matemáticas (HVAC)")
                base_temp_heating = st.sidebar.number_input("Temp. base calefacción (°C)", value=18.0, step=0.5)
                base_temp_cooling = st.sidebar.number_input("Temp. base refrigeración (°C)", value=21.0, step=0.5)

        except Exception as e:
            st.sidebar.error(f"Error procesando datos de energía: {e}")
            df_energy = pd.DataFrame()
    else:
         st.sidebar.info("Para comenzar, carga un archivo o selecciona 'Desde GitHub'.")


# --- Panel Principal ---
st.title("Dashboard de Análisis de Consumo Energético")

if not df_energy.empty:
    # --- Aplicación de Filtros ---
    df_filtered = df_energy.copy()
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df_filtered.loc[start_date:end_date]
    if selected_days:
        df_filtered = df_filtered[df_filtered.index.dayofweek.isin(selected_days)]
    df_filtered = df_filtered[(df_filtered.index.hour >= selected_hours[0]) & (df_filtered.index.hour <= selected_hours[1])]
    if remove_baseline:
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] > baseline_threshold]
    if remove_anomalies:
        upper_bound = df_filtered['Consumption_kWh'].quantile(anomaly_percentile / 100.0)
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] < upper_bound]

    # --- Aplicar agregación temporal ---
    df_display = gestionar_agregacion_temporal(df_filtered, aggregation_level)

    # --- Visualización de Datos ---
    st.markdown(f"Mostrando **{len(df_filtered):,}** registros horarios originales tras aplicar filtros.")
    if aggregation_level != "Horario":
        st.markdown(f"Agregados en **{len(df_display):,}** puntos de datos a nivel **{aggregation_level.lower()}**.")
    st.markdown("---")
    
    st.header("Análisis de Patrones Temporales")
    if not df_display.empty:
        st.subheader(f"Evolución del Consumo Energético ({aggregation_level})")
        
        df_plot = df_display.reset_index()
        
        if aggregation_level == "Mensual":
            hover_template = '<b>Mes</b>: %{x|%B %Y}<br><b>Consumo Medio</b>: %{y:.2f} kWh'
        elif aggregation_level == "Diario":
            hover_template = '<b>Día</b>: %{x|%A, %d %b %Y}<br><b>Consumo Medio</b>: %{y:.2f} kWh'
        else: # Horario
            hover_template = '<b>Fecha</b>: %{x|%d %b %Y - %H:%Mh}<br><b>Consumo</b>: %{y:.2f} kWh'
        
        fig_evolucion = px.line(
            df_plot, 
            x='datetime', 
            y='Consumption_kWh',
            labels={
                "datetime": f"Fecha ({aggregation_level})",
                "Consumption_kWh": "Consumo (kWh)"
            },
            hover_data={"datetime": False}
        )
        fig_evolucion.update_traces(hovertemplate=hover_template)
        st.plotly_chart(fig_evolucion, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patrón Diario Promedio")
            df_hourly = df_filtered.groupby(df_filtered.index.hour).mean(numeric_only=True).reset_index()
            df_hourly.rename(columns={'datetime': 'Hora'}, inplace=True)
            st.plotly_chart(px.bar(df_hourly, x='Hora', y='Consumption_kWh', title='Consumo Promedio por Hora'), use_container_width=True)
        with col2:
            st.subheader("Patrón Semanal Promedio")
            df_weekly = df_filtered.groupby(df_filtered.index.dayofweek).mean(numeric_only=True)
            if not df_weekly.empty:
                df_weekly.index = df_weekly.index.map(dias_semana)
                df_plot_weekly = df_weekly.reset_index()
                df_plot_weekly.rename(columns={'datetime': 'Día'}, inplace=True)
                st.plotly_chart(px.bar(df_plot_weekly, x='Día', y='Consumption_kWh', title='Consumo Promedio por Día'), use_container_width=True)
    else:
        st.warning("No hay datos de consumo para los filtros seleccionados.")

    # --- Análisis Climático (usa los datos horarios filtrados) ---
    if not df_weather_raw.empty:
        st.markdown("---")
        st.header("Correlación del Consumo con el Clima")
        try:
            df_weather = df_weather_raw.copy()
            
            if 'YEAR' in df_weather.columns:
                 df_weather['datetime'] = pd.to_datetime(df_weather[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
                 df_weather = df_weather[['datetime', 'T2M', 'RH2M']].set_index('datetime')
            elif 'Date & Time' in df_weather.columns:
                df_weather.rename(columns={'Date & Time': 'datetime', 'Temperature(°C)': 'T2M', 'Humidity(%)': 'RH2M'}, inplace=True)
                df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
                df_weather = df_weather[['datetime', 'T2M', 'RH2M']].set_index('datetime')
            else:
                st.error("Formato de CSV de clima no reconocido.")
                df_weather = pd.DataFrame()

            if not df_weather.empty:
                df_merged = df_filtered.join(df_weather, how='inner').dropna()

                if not df_merged.empty:
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("Consumo vs. Temperatura")
                        st.plotly_chart(px.scatter(df_merged, x='T2M', y='Consumption_kWh', labels={'T2M': 'Temperatura (°C)'}, trendline="ols", trendline_color_override="red"), use_container_width=True)
                    with col4:
                        st.subheader("Consumo vs. Humedad")
                        st.plotly_chart(px.scatter(df_merged, x='RH2M', y='Consumption_kWh', labels={'RH2M': 'Humedad (%)'}, trendline="ols", trendline_color_override="red"), use_container_width=True)
                else:
                    st.warning("No hay datos climáticos para el rango y filtros seleccionados.")
        except Exception as e:
            st.error(f"Error al procesar o fusionar los datos de clima: {e}")
else:
    st.info("Para comenzar, carga un archivo de consumo o selecciona 'Desde GitHub' en la barra lateral.")
