# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard Energético Asepeyo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funciones de Carga y Procesamiento de Datos ---
@st.cache_data
def load_and_merge_data(energy_file, weather_file):
    """Carga, limpia y fusiona los datos de consumo y clima."""
    if energy_file is None or weather_file is None:
        return pd.DataFrame()

    try:
        # Cargar datos de energía
        df_energy = pd.read_csv(energy_file, skipinitialspace=True)
        df_energy.rename(columns={'Fecha': 'datetime', 'Energía activa (kWh)': 'Consumption_kWh'}, inplace=True)
        df_energy['datetime'] = pd.to_datetime(df_energy['datetime'], format='%d/%m/%Y %H:%M')

        # Cargar datos meteorológicos
        df_weather = pd.read_csv(weather_file, skiprows=10)
        df_weather['datetime'] = pd.to_datetime(df_weather['YEAR'].astype(str) + '-' + df_weather['MO'].astype(str) + '-' + df_weather['DY'].astype(str) + ' ' + df_weather['HR'].astype(str) + ':00')
        df_weather = df_weather[['datetime', 'T2M', 'RH2M']] # Seleccionar solo columnas relevantes

        # Fusionar los dataframes por 'datetime'
        df_merged = pd.merge(df_energy, df_weather, on='datetime', how='inner')
        df_merged.set_index('datetime', inplace=True)
        
        return df_merged

    except Exception as e:
        st.error(f"Ocurrió un error al procesar los archivos: {e}")
        return pd.DataFrame()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title('⚡ Dashboard de Análisis Energético')
    st.markdown("Sube tus archivos de consumo y clima para comenzar el análisis.")

    # Widgets para cargar archivos
    uploaded_energy_file = st.file_uploader("1. Carga tu archivo de consumo (CSV)", type="csv")
    uploaded_weather_file = st.file_uploader("2. Carga tu archivo de clima (CSV)", type="csv")

    # Cargar y procesar los datos
    df_original = load_and_merge_data(uploaded_energy_file, uploaded_weather_file)
    
    # Filtros (solo se muestran si los datos se cargan correctamente)
    if not df_original.empty:
        st.success("¡Archivos cargados y fusionados con éxito!")
        st.markdown("---")
        st.header("Filtros de Visualización")
        
        # Selector de Rango de Fechas
        min_date = df_original.index.min().date()
        max_date = df_original.index.max().date()
        
        date_range = st.date_input(
            "Selecciona el rango de fechas:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        
        # Selector de Tipo de Análisis
        analysis_type = st.radio(
            "Selecciona el tipo de análisis:",
            ('Resumen General', 'Análisis Temporal', 'Correlación con el Clima')
        )

# --- Panel Principal ---
st.title("Dashboard de Análisis de Consumo Energético")
st.markdown("Bienvenido al panel de control para el análisis de eficiencia energética en Asepeyo.")

if 'df_original' in locals() and not df_original.empty and 'date_range' in locals() and len(date_range) == 2:
    
    # Filtrar el dataframe por el rango de fechas
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    df_filtered = df_original.loc[start_date:end_date]

    st.markdown(f"Mostrando datos desde **{start_date.strftime('%d/%m/%Y')}** hasta **{end_date.strftime('%d/%m/%Y')}**.")
    st.markdown("---")

    # --- Resumen General ---
    if analysis_type == 'Resumen General':
        st.header("Resumen General del Consumo")
        
        total_consumption = df_filtered['Consumption_kWh'].sum()
        avg_hourly_consumption = df_filtered['Consumption_kWh'].mean()
        peak_consumption = df_filtered['Consumption_kWh'].max()
        baseline_consumption = df_filtered['Consumption_kWh'].min()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Consumo Total del Período", value=f"{total_consumption:,.0f} kWh")
        kpi2.metric(label="Consumo Horario Promedio", value=f"{avg_hourly_consumption:,.2f} kWh")
        kpi3.metric(label="Pico Máximo de Consumo", value=f"{peak_consumption:,.2f} kWh")
        kpi4.metric(label="Consumo Base (Mínimo)", value=f"{baseline_consumption:,.2f} kWh")
        
        st.markdown("---")
        st.subheader("Datos Completos Filtrados")
        st.dataframe(df_filtered)

    # --- Análisis Temporal ---
    elif analysis_type == 'Análisis Temporal':
        st.header("Análisis de Patrones Temporales")

        st.subheader("Evolución del Consumo Energético (kWh)")
        fig_time_series = px.line(df_filtered.reset_index(), x='datetime', y='Consumption_kWh', title='Consumo de Energía (kWh) por Hora')
        st.plotly_chart(fig_time_series, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patrón Diario Promedio")
            df_hourly = df_filtered.groupby(df_filtered.index.hour)['Consumption_kWh'].mean().reset_index()
            df_hourly.rename(columns={'index': 'Hora', 'Consumption_kWh': 'Consumo Promedio (kWh)'}, inplace=True)
            fig_hourly = px.bar(df_hourly, x='datetime', y='Consumo Promedio (kWh)', title='Consumo Promedio por Hora del Día')
            fig_hourly.update_xaxes(title="Hora del día")
            st.plotly_chart(fig_hourly, use_container_width=True)

        with col2:
            st.subheader("Patrón Semanal Promedio")
            df_weekly = df_filtered.groupby(df_filtered.index.dayofweek)['Consumption_kWh'].mean()
            df_weekly.index = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            fig_weekly = px.bar(df_weekly.reset_index(), x='index', y='Consumption_kWh', title='Consumo Promedio por Día de la Semana')
            fig_weekly.update_xaxes(title="Día de la semana")
            st.plotly_chart(fig_weekly, use_container_width=True)
            
        st.subheader("Consumo Total Mensual")
        df_monthly = df_filtered['Consumption_kWh'].resample('M').sum().reset_index()
        fig_monthly = px.bar(df_monthly, x='datetime', y='Consumption_kWh', title='Consumo Total por Mes')
        fig_monthly.update_xaxes(title="Mes")
        st.plotly_chart(fig_monthly, use_container_width=True)

    # --- Correlación con el Clima ---
    elif analysis_type == 'Correlación con el Clima':
        st.header("Correlación del Consumo con Variables Climáticas")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Consumo vs. Temperatura")
            fig_temp = px.scatter(df_filtered, x='T2M', y='Consumption_kWh', title='Consumo de Energía vs. Temperatura', labels={'T2M': 'Temperatura (°C)', 'Consumption_kWh': 'Consumo (kWh)'}, trendline="ols", trendline_color_override="red")
            st.plotly_chart(fig_temp, use_container_width=True)

        with col2:
            st.subheader("Consumo vs. Humedad")
            fig_hum = px.scatter(df_filtered, x='RH2M', y='Consumption_kWh', title='Consumo de Energía vs. Humedad', labels={'RH2M': 'Humedad Relativa (%)', 'Consumption_kWh': 'Consumo (kWh)'}, trendline="ols", trendline_color_override="red")
            st.plotly_chart(fig_hum, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Matriz de Correlación")
        correlation_matrix = df_filtered[['Consumption_kWh', 'T2M', 'RH2M']].corr()
        st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))
        st.info("""
        **Interpretación de la Matriz de Correlación:**
        - **Cercano a 1 (Rojo):** Fuerte correlación positiva. Al subir la temperatura/humedad, sube el consumo.
        - **Cercano a -1 (Azul):** Fuerte correlación negativa. Al subir la temperatura/humedad, baja el consumo.
        - **Cercano a 0 (Blanco):** Correlación débil o nula.
        """)

else:
    st.info("Por favor, carga ambos archivos CSV en la barra lateral para iniciar el análisis.")
