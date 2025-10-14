# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard Energético Avanzado de Asepeyo",
    page_icon="⚡",
    layout="wide",
)

# --- Funciones de Carga y Procesamiento ---
@st.cache_data
def load_csv_from_source(source):
    """Carga datos desde un archivo subido o una URL de GitHub."""
    if source is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(source, skipinitialspace=True)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def get_github_csv_files(repo, path="Data"):
    """Obtiene una lista de archivos CSV de un directorio en un repositorio de GitHub."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        files_json = response.json()
        if isinstance(files_json, dict) and 'message' in files_json:
            st.error(f"No se pudo encontrar el directorio '{path}'. Error de GitHub: {files_json['message']}")
            return []
        return [file['name'] for file in files_json if file['type'] == 'file' and file['name'].endswith('.csv')]
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API de GitHub: {e}")
        return []

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    
    # 1. Selección de Fuente de Datos
    st.header("1. Fuente de Datos")
    source_type = st.radio("Seleccionar origen", ["Cargar Archivos", "Desde GitHub"], key="source_type")

    df_energy_raw = pd.DataFrame()
    df_weather_raw = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy_file = st.file_uploader("Archivo de Consumo (CSV)", type="csv")
        uploaded_weather_file = st.file_uploader("Archivo de Clima (CSV)", type="csv")
        df_energy_raw = load_csv_from_source(uploaded_energy_file)
        df_weather_raw = load_csv_from_source(uploaded_weather_file)
    else:
        github_repo = st.text_input("Repositorio GitHub (usuario/repo)", "Hardik-Derashri/EnergyPatternAnalysis")
        if github_repo:
            csv_files_list = get_github_csv_files(github_repo)
            if csv_files_list:
                base_url = f"https://raw.githubusercontent.com/{github_repo}/main/Data/"
                selected_energy_file = st.selectbox("Selecciona archivo de consumo", [""] + csv_files_list)
                selected_weather_file = st.selectbox("Selecciona archivo de clima", [""] + csv_files_list)
                if selected_energy_file:
                    df_energy_raw = load_csv_from_source(base_url + selected_energy_file)
                if selected_weather_file:
                    df_weather_raw = load_csv_from_source(base_url + selected_weather_file)
            else:
                st.warning("No se encontraron archivos CSV en la carpeta 'Data/' del repositorio.")

    # --- Procesamiento y Filtros (si se carga al menos el de energía) ---
    df_energy = pd.DataFrame()
    if not df_energy_raw.empty:
        try:
            df_energy = df_energy_raw.copy()
            df_energy.rename(columns={'Fecha': 'datetime', 'Energía activa (kWh)': 'Consumption_kWh'}, inplace=True)
            df_energy['datetime'] = pd.to_datetime(df_energy['datetime'], format='%d/%m/%Y %H:%M')
            df_energy.set_index('datetime', inplace=True)
            st.sidebar.success("Datos de consumo cargados.")

            st.sidebar.markdown("---")
            st.sidebar.header("2. Filtros de Datos")
            
            # Filtros de días y horas
            dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
            selected_days = st.sidebar.multiselect("Días de la semana", options=list(dias_semana.keys()), format_func=lambda x: dias_semana[x], default=list(dias_semana.keys()))
            selected_hours = st.sidebar.slider("Horas del día", 0, 23, (0, 23))
            
            # Filtro de rango de fechas
            min_date, max_date = df_energy.index.min().date(), df_energy.index.max().date()
            date_range = st.sidebar.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

            st.sidebar.markdown("---")
            st.sidebar.header("3. Ajustes de Análisis")
            
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
            df_energy = pd.DataFrame() # Reset on error

# --- Panel Principal ---
st.title("Dashboard de Análisis de Consumo Energético")

if not df_energy.empty:
    # Aplicar filtros
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

    st.markdown(f"Mostrando **{len(df_filtered):,}** registros tras aplicar filtros.")
    st.markdown("---")
    
    st.header("Análisis de Patrones Temporales")
    if not df_filtered.empty:
        st.plotly_chart(px.line(df_filtered.reset_index(), x='datetime', y='Consumption_kWh', title='Evolución del Consumo Energético'), use_container_width=True)
        
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
    else:
        st.warning("No hay datos de consumo para los filtros seleccionados.")

    # --- Análisis Climático ---
    if not df_weather_raw.empty:
        st.success("Datos de clima detectados. Mostrando análisis de correlación.")
        st.markdown("---")
        st.header("Correlación del Consumo con el Clima")
        
        try:
            df_weather = df_weather_raw.copy()
            df_weather['datetime'] = pd.to_datetime(df_weather['YEAR'].astype(str) + '-' + df_weather['MO'].astype(str) + '-' + df_weather['DY'].astype(str) + ' ' + df_weather['HR'].astype(str) + ':00')
            df_weather = df_weather[['datetime', 'T2M', 'RH2M']].set_index('datetime')
            
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
                st.warning("No se encontraron datos climáticos para el rango de fechas y filtros seleccionados.")
        except Exception as e:
            st.error(f"Error al procesar o fusionar los datos de clima: {e}")
else:
    st.info("Para comenzar, carga un archivo de consumo o selecciona uno desde GitHub en la barra lateral.")
