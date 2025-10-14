# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import requests  # Para conectar con GitHub
import base64    # Para decodificar nombres de archivo de GitHub

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard Energético Avanzado de Asepeyo",
    page_icon="⚡",
    layout="wide",
)

# --- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_csv_data(file_or_url):
    """Carga datos desde un archivo subido o una URL."""
    if file_or_url is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(file_or_url, skipinitialspace=True)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def get_github_csv_files(repo, path="Data"):
    """Obtiene una lista de archivos CSV de un directorio en un repositorio de GitHub."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza un error para respuestas 4xx/5xx
        files_json = response.json()
        
        # Manejo de error si el directorio no existe o está vacío
        if isinstance(files_json, dict) and 'message' in files_json:
            st.error(f"No se pudo encontrar el directorio '{path}' en el repositorio. Error: {files_json['message']}")
            return []
            
        csv_files = [file['name'] for file in files_json if file['type'] == 'file' and file['name'].endswith('.csv')]
        return csv_files
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de GitHub: {e}")
        return []

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    
    st.header("1. Fuente de Datos")
    source_type = st.radio("Seleccionar origen de datos", ["Cargar Archivos", "Desde GitHub"])

    df_energy = pd.DataFrame()
    df_weather = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy_file = st.file_uploader("Archivo de Consumo (CSV)", type="csv")
        uploaded_weather_file = st.file_uploader("Archivo de Clima (CSV)", type="csv")
        
        if uploaded_energy_file:
            df_energy = load_csv_data(uploaded_energy_file)
        if uploaded_weather_file:
            df_weather = load_csv_data(uploaded_weather_file)
    else:
        github_repo = st.text_input("Repositorio GitHub (usuario/repo)", "Hardik-Derashri/EnergyPatternAnalysis")
        csv_files_list = get_github_csv_files(github_repo)
        
        if csv_files_list:
            base_url = f"https://raw.githubusercontent.com/{github_repo}/main/Data/"
            selected_energy_file = st.selectbox("Selecciona el archivo de consumo", [""] + csv_files_list)
            selected_weather_file = st.selectbox("Selecciona el archivo de clima", [""] + csv_files_list)

            if selected_energy_file:
                df_energy = load_csv_data(base_url + selected_energy_file)
            if selected_weather_file:
                df_weather = load_csv_data(base_url + selected_weather_file)
        else:
            st.warning("No se encontraron archivos CSV en el directorio 'Data/' del repositorio.")

    # --- Procesamiento y Filtros (se activan al cargar al menos el de energía) ---
    if not df_energy.empty:
        # Procesar datos de energía
        try:
            df_energy.rename(columns={'Fecha': 'datetime', 'Energía activa (kWh)': 'Consumption_kWh'}, inplace=True)
            df_energy['datetime'] = pd.to_datetime(df_energy['datetime'], format='%d/%m/%Y %H:%M')
            df_energy.set_index('datetime', inplace=True)
        except Exception as e:
            st.error(f"Error al procesar el archivo de consumo. Verifique el formato de las columnas 'Fecha' y 'Energía activa (kWh)'. Error: {e}")
            st.stop()
            
        st.success("Datos de consumo cargados.")
        st.markdown("---")
        st.header("2. Filtros de Datos")

        # Filtros de días y horas
        dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        selected_days = st.multiselect("Filtrar por día de la semana", options=list(dias_semana.keys()), format_func=lambda x: dias_semana[x], default=list(dias_semana.keys()))
        selected_hours = st.slider("Filtrar por hora del día", 0, 23, (0, 23))
        
        # Filtro de rango de fechas
        min_date, max_date = df_energy.index.min().date(), df_energy.index.max().date()
        date_range = st.date_input("Filtrar por rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        st.markdown("---")
        st.header("3. Ajustes de Análisis")
        
        remove_baseline = st.checkbox("Eliminar consumo base (Outliers bajos)")
        baseline_threshold = st.number_input("Umbral de consumo base (kWh)", value=float(df_energy['Consumption_kWh'].quantile(0.1)), disabled=not remove_baseline)
        
        remove_anomalies = st.checkbox("Eliminar anomalías (Outliers altos)")
        anomaly_percentile = st.number_input("Percentil para anomalías", value=99.0, min_value=90.0, max_value=100.0, disabled=not remove_anomalies)
        
        st.markdown("---")
        st.header("4. Constantes Matemáticas (HVAC)")
        
        base_temp_heating = st.number_input("Temp. base de calefacción (°C)", value=18.0, step=0.5)
        base_temp_cooling = st.number_input("Temp. base de refrigeración (°C)", value=21.0, step=0.5)

# --- Panel Principal ---
st.title("Dashboard de Análisis de Consumo Energético")

if 'df_energy' in locals() and not df_energy.empty:
    
    # Aplicar filtros al dataframe de energía
    df_filtered = df_energy.copy()
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df_filtered.loc[start_date:end_date]
    df_filtered = df_filtered[df_filtered.index.dayofweek.isin(selected_days) & (df_filtered.index.hour >= selected_hours[0]) & (df_filtered.index.hour <= selected_hours[1])]
    if remove_baseline:
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] > baseline_threshold]
    if remove_anomalies:
        upper_bound = df_filtered['Consumption_kWh'].quantile(anomaly_percentile / 100.0)
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] < upper_bound]

    st.markdown(f"Mostrando **{len(df_filtered):,}** registros de datos después de aplicar filtros.")
    st.markdown("---")
    
    # --- Análisis Temporal ---
    st.header("Análisis de Patrones Temporales")
    st.plotly_chart(px.line(df_filtered.reset_index(), x='datetime', y='Consumption_kWh', title='Evolución del Consumo Energético (kWh)'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patrón Diario Promedio")
        df_hourly = df_filtered.groupby(df_filtered.index.hour).mean().reset_index()
        st.plotly_chart(px.bar(df_hourly, x='datetime', y='Consumption_kWh', title='Consumo Promedio por Hora'), use_container_width=True)
    with col2:
        st.subheader("Patrón Semanal Promedio")
        df_weekly = df_filtered.groupby(df_filtered.index.dayofweek).mean()
        df_weekly.index = df_weekly.index.map(dias_semana)
        st.plotly_chart(px.bar(df_weekly.reset_index(), x='index', y='Consumption_kWh', title='Consumo Promedio por Día'), use_container_width=True)

    # --- Análisis Climático ---
    if 'df_weather' in locals() and not df_weather.empty:
        try:
            df_weather.rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}, inplace=True, errors='ignore')
            if 'datetime' not in df_weather.columns:
                df_weather['datetime'] = pd.to_datetime(df_weather[['year', 'month', 'day', 'hour']])
            df_weather.set_index('datetime', inplace=True, drop=True)
        except Exception as e:
            st.error(f"Error al procesar el archivo de clima. Verifique las columnas de fecha. Error: {e}")
        
        st.success("Datos de clima cargados. Mostrando análisis de correlación.")
        st.markdown("---")
        st.header("Correlación del Consumo con el Clima")

        df_merged = df_filtered.join(df_weather, how='inner').dropna()

        if not df_merged.empty:
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Consumo vs. Temperatura")
                st.plotly_chart(px.scatter(df_merged, x='T2M', y='Consumption_kWh', labels={'T2M': 'Temperatura (°C)'}, trendline="ols", trendline_color_override="red"), use_container_width=True)
            with col4:
                st.subheader("Consumo vs. Humedad")
                st.plotly_chart(px.scatter(df_merged, x='RH2M', y='Consumption_kWh', labels={'RH2M': 'Humedad Relativa (%)'}, trendline="ols", trendline_color_override="red"), use_container_width=True)
            
            daily_df = df_merged.resample('D').agg({'Consumption_kWh': 'sum', 'T2M': 'mean'})
            daily_df['HDD'] = (base_temp_heating - daily_df['T2M']).clip(lower=0)
            daily_df['CDD'] = (daily_df['T2M'] - base_temp_cooling).clip(lower=0)

            col5, col6 = st.columns(2)
            with col5:
                st.subheader("Consumo Diario vs. HDD")
                st.plotly_chart(px.scatter(daily_df, x='HDD', y='Consumption_kWh', trendline="ols", trendline_color_override="blue"), use_container_width=True)
            with col6:
                st.subheader("Consumo Diario vs. CDD")
                st.plotly_chart(px.scatter(daily_df, x='CDD', y='Consumption_kWh', trendline="ols", trendline_color_override="red"), use_container_width=True)
        else:
            st.warning("No se encontraron datos climáticos para el rango y filtros seleccionados.")
else:
    st.info("Para comenzar, carga un archivo de consumo o selecciona uno desde GitHub en la barra lateral.")
