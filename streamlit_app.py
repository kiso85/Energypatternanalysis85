# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard Energético Avanzado de Asepeyo",
    page_icon="⚡",
    layout="wide",
)

# --- Funciones de Carga y Procesamiento de Datos ---
@st.cache_data
def load_energy_data(file_path):
    """Carga y procesa únicamente los datos de consumo energético."""
    if file_path is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        df.rename(columns={'Fecha': 'datetime', 'Energía activa (kWh)': 'Consumption_kWh'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo de consumo: {e}")
        return pd.DataFrame()

@st.cache_data
def load_weather_data(file_path):
    """Carga y procesa únicamente los datos meteorológicos."""
    if file_path is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, skiprows=10)
        df['datetime'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
        df = df[['datetime', 'T2M', 'RH2M']]
        df.set_index('datetime', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo de clima: {e}")
        return pd.DataFrame()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title('⚡ Panel de Control')
    st.markdown("Configura el análisis de datos energéticos.")

    # --- Selección de Fuente de Datos ---
    st.header("1. Fuente de Datos")
    source_type = st.radio("Seleccionar origen de datos", ["Cargar Archivos", "Desde GitHub"], key="source_type")

    df_energy = pd.DataFrame()
    df_weather = pd.DataFrame()

    if source_type == "Cargar Archivos":
        uploaded_energy_file = st.file_uploader("Archivo de Consumo (CSV)", type="csv")
        uploaded_weather_file = st.file_uploader("Archivo de Clima (CSV)", type="csv")
        df_energy = load_energy_data(uploaded_energy_file)
        df_weather = load_weather_data(uploaded_weather_file)
    else:
        st.info("Asegúrate de que tus archivos están en la carpeta `Data/` de tu repositorio.")
        github_repo = st.text_input("Repositorio GitHub (usuario/repo)", "Hardik-Derashri/EnergyPatternAnalysis")
        base_url = f"https://raw.githubusercontent.com/{github_repo}/main/Data/"
        
        # Listar archivos (simulado, el usuario debe saber los nombres)
        energy_files = ["251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv"]
        weather_files = ["POWER_Point_Hourly_20230401_20250831_041d38N_002d18E_LST.csv"]
        
        selected_energy_file = st.selectbox("Selecciona el archivo de consumo", energy_files)
        selected_weather_file = st.selectbox("Selecciona el archivo de clima", weather_files)

        if st.button("Cargar desde GitHub"):
            df_energy = load_energy_data(base_url + selected_energy_file)
            df_weather = load_weather_data(base_url + selected_weather_file)

    # --- Filtros (se activan al cargar al menos el de energía) ---
    if not df_energy.empty:
        st.markdown("---")
        st.header("2. Filtros de Datos")

        # Filtro de Días de la Semana
        dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        selected_days = st.multiselect(
            "Filtrar por día de la semana",
            options=list(dias_semana.keys()),
            format_func=lambda x: dias_semana[x],
            default=list(dias_semana.keys())
        )

        # Filtro de Horas del Día
        selected_hours = st.slider(
            "Filtrar por hora del día",
            0, 23, (0, 23)
        )
        
        # Filtro de Rango de Fechas
        min_date, max_date = df_energy.index.min().date(), df_energy.index.max().date()
        date_range = st.date_input("Filtrar por rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        st.markdown("---")
        st.header("3. Ajustes de Análisis")
        
        # Opciones para remover outliers y baseline
        remove_baseline = st.checkbox("Eliminar consumo base")
        baseline_threshold = st.number_input("Umbral de consumo base (kWh)", value=df_energy['Consumption_kWh'].quantile(0.1), min_value=0.0, step=1.0, disabled=not remove_baseline)
        
        remove_anomalies = st.checkbox("Eliminar anomalías (picos)")
        anomaly_threshold = st.number_input("Umbral de anomalías (percentil)", value=99.0, min_value=90.0, max_value=100.0, step=0.5, disabled=not remove_anomalies)
        
        st.markdown("---")
        st.header("4. Constantes Matemáticas")
        
        # Variables para HDD/CDD
        st.markdown("**Análisis de Climatización (HVAC)**")
        base_temp_heating = st.number_input("Temp. base de calefacción (°C)", value=18.0, step=0.5)
        base_temp_cooling = st.number_input("Temp. base de refrigeración (°C)", value=21.0, step=0.5)

# --- Panel Principal ---
st.title("Dashboard de Análisis de Consumo Energético")

if 'df_energy' in locals() and not df_energy.empty:
    
    # Aplicar todos los filtros al dataframe de energía
    df_filtered = df_energy.copy()
    
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df_filtered = df_filtered.loc[start_date:end_date]

    df_filtered = df_filtered[df_filtered.index.dayofweek.isin(selected_days)]
    df_filtered = df_filtered[(df_filtered.index.hour >= selected_hours[0]) & (df_filtered.index.hour <= selected_hours[1])]

    if remove_baseline:
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] > baseline_threshold]
    
    if remove_anomalies:
        upper_bound = df_filtered['Consumption_kWh'].quantile(anomaly_threshold / 100.0)
        df_filtered = df_filtered[df_filtered['Consumption_kWh'] < upper_bound]

    st.markdown(f"Mostrando **{len(df_filtered):,}** registros de datos después de aplicar filtros.")
    st.markdown("---")
    
    # --- Análisis Temporal (siempre disponible) ---
    st.header("Análisis de Patrones Temporales")
    fig_time_series = px.line(df_filtered.reset_index(), x='datetime', y='Consumption_kWh', title='Consumo Energético a lo largo del Tiempo')
    st.plotly_chart(fig_time_series, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patrón Diario Promedio")
        df_hourly = df_filtered.groupby(df_filtered.index.hour).mean().reset_index()
        fig_hourly = px.bar(df_hourly, x='datetime', y='Consumption_kWh', title='Consumo Promedio por Hora')
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col2:
        st.subheader("Patrón Semanal Promedio")
        df_weekly = df_filtered.groupby(df_filtered.index.dayofweek).mean()
        df_weekly.index = df_weekly.index.map(dias_semana)
        fig_weekly = px.bar(df_weekly.reset_index(), x='index', y='Consumption_kWh', title='Consumo Promedio por Día')
        st.plotly_chart(fig_weekly, use_container_width=True)

    # --- Análisis Climático (solo si se cargan los datos de clima) ---
    if 'df_weather' in locals() and not df_weather.empty:
        st.markdown("---")
        st.header("Correlación del Consumo con el Clima")

        # Unir los dataframes ya filtrados
        df_merged = df_filtered.join(df_weather, how='inner').dropna()

        if not df_merged.empty:
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Consumo vs. Temperatura")
                fig_temp = px.scatter(df_merged, x='T2M', y='Consumption_kWh', labels={'T2M': 'Temperatura (°C)'}, trendline="ols", trendline_color_override="red")
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col4:
                st.subheader("Consumo vs. Humedad")
                fig_hum = px.scatter(df_merged, x='RH2M', y='Consumption_kWh', labels={'RH2M': 'Humedad Relativa (%)'}, trendline="ols", trendline_color_override="red")
                st.plotly_chart(fig_hum, use_container_width=True)
            
            # Análisis HDD/CDD
            daily_df = df_merged.resample('D').agg({'Consumption_kWh': 'sum', 'T2M': 'mean'})
            daily_df['HDD'] = [max(0, base_temp_heating - temp) for temp in daily_df['T2M']]
            daily_df['CDD'] = [max(0, temp - base_temp_cooling) for temp in daily_df['T2M']]

            col5, col6 = st.columns(2)
            with col5:
                st.subheader("Consumo Diario vs. Grados Día de Calefacción (HDD)")
                fig_hdd = px.scatter(daily_df, x='HDD', y='Consumption_kWh', trendline="ols", trendline_color_override="blue")
                st.plotly_chart(fig_hdd, use_container_width=True)

            with col6:
                st.subheader("Consumo Diario vs. Grados Día de Refrigeración (CDD)")
                fig_cdd = px.scatter(daily_df, x='CDD', y='Consumption_kWh', trendline="ols", trendline_color_override="red")
                st.plotly_chart(fig_cdd, use_container_width=True)

        else:
            st.warning("No hay datos superpuestos entre el consumo y el clima para el rango de fechas y filtros seleccionados.")
else:
    st.info("Para comenzar, carga un archivo de consumo o selecciona uno desde GitHub en la barra lateral.")
