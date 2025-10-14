# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Consumo Energético de Asepeyo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Carga y Cacheo de Datos ---
@st.cache_data
def load_data(energy_file_path, weather_file_path):
    """Carga, limpia y fusiona los datos de consumo energético y meteorológicos."""
    try:
        # Cargar datos de energía
        df_energy = pd.read_csv(energy_file_path, skipinitialspace=True)
        df_energy.rename(columns={'Fecha': 'datetime', 'Energía activa (kWh)': 'Consumption_kWh'}, inplace=True)
        df_energy['datetime'] = pd.to_datetime(df_energy['datetime'], format='%d/%m/%Y %H:%M')

        # Cargar datos meteorológicos
        df_weather = pd.read_csv(weather_file_path, skiprows=10)
        df_weather['datetime'] = pd.to_datetime(df_weather['YEAR'].astype(str) + '-' + df_weather['MO'].astype(str) + '-' + df_weather['DY'].astype(str) + ' ' + df_weather['HR'].astype(str) + ':00')
        df_weather.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)

        # Fusionar los dataframes
        df_merged = pd.merge(df_energy, df_weather, on='datetime', how='inner')
        df_merged.set_index('datetime', inplace=True)
        
        return df_merged

    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo de datos. Verifique la ruta: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ocurrió un error al procesar los archivos: {e}")
        return pd.DataFrame()

# --- Barra Lateral y Lógica de Carga de Datos ---
with st.sidebar:
    st.title('⚡ Dashboard de Análisis Energético')

    # Asumiendo que los archivos están en el mismo directorio que el script
    energy_file = '251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv'
    weather_file = 'POWER_Point_Hourly_20230401_20250831_041d38N_002d18E_LST.csv'
    
    df_original = load_data(energy_file, weather_file)

    if not df_original.empty:
        st.success("Datos cargados correctamente!")
        
        # Filtros
        st.header("Filtros")
        
        # Selector de Rango de Fechas
        date_range = st.date_input(
            "Seleccione el rango de fechas",
            value=(df_original.index.min(), df_original.index.max()),
            min_value=df_original.index.min(),
            max_value=df_original.index.max(),
        )
        
        # Selector de Tipo de Análisis
        analysis_type = st.radio(
            "Seleccionar Tipo de Análisis",
            ('Resumen General', 'Análisis Temporal', 'Correlación con el Clima')
        )

# --- Lógica de la Aplicación Principal ---
if 'df_original' in locals() and not df_original.empty and 'date_range' in locals() and len(date_range) == 2:
    
    # Filtrar el dataframe según el rango de fechas seleccionado
    df_filtered = df_original.loc[date_range[0]:date_range[1]]

    st.title(f"Análisis de Consumo Energético")
    st.markdown(f"**Período seleccionado:** del `{date_range[0].strftime('%d/%m/%Y')}` al `{date_range[1].strftime('%d/%m/%Y')}`")
    st.markdown("---")

    # --- Resumen General ---
    if analysis_type == 'Resumen General':
        st.header("Resumen General del Consumo")
        
        # KPIs
        total_consumption = df_filtered['Consumption_kWh'].sum()
        avg_hourly_consumption = df_filtered['Consumption_kWh'].mean()
        peak_consumption = df_filtered['Consumption_kWh'].max()
        baseline_consumption = df_filtered['Consumption_kWh'].min()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Consumo Total", value=f"{total_consumption:,.2f} kWh")
        kpi2.metric(label="Consumo Horario Promedio", value=f"{avg_hourly_consumption:,.2f} kWh")
        kpi3.metric(label="Pico de Consumo", value=f"{peak_consumption:,.2f} kWh")
        kpi4.metric(label="Consumo Base (Mínimo)", value=f"{baseline_consumption:,.2f} kWh")
        
        st.markdown("---")
        st.subheader("Tabla de Datos")
        st.dataframe(df_filtered)

    # --- Análisis Temporal ---
    elif analysis_type == 'Análisis Temporal':
        st.header("Análisis de Patrones Temporales")

        # Gráfico de Consumo a lo largo del tiempo
        st.subheader("Consumo Energético a lo largo del Tiempo")
        fig_time_series = px.line(df_filtered, x=df_filtered.index, y='Consumption_kWh', title='Consumo de Energía (kWh) por Hora')
        st.plotly_chart(fig_time_series, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Patrón de consumo diario
            st.subheader("Patrón de Consumo Diario Promedio")
            df_hourly = df_filtered.groupby(df_filtered.index.hour).mean()
            fig_hourly = px.bar(df_hourly, x=df_hourly.index, y='Consumption_kWh', title='Consumo Promedio por Hora del Día')
            fig_hourly.update_xaxes(title="Hora del día")
            st.plotly_chart(fig_hourly, use_container_width=True)

        with col2:
            # Patrón de consumo semanal
            st.subheader("Patrón de Consumo Semanal Promedio")
            df_weekly = df_filtered.groupby(df_filtered.index.dayofweek).mean()
            df_weekly.index = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            fig_weekly = px.bar(df_weekly, x=df_weekly.index, y='Consumption_kWh', title='Consumo Promedio por Día de la Semana')
            fig_weekly.update_xaxes(title="Día de la semana")
            st.plotly_chart(fig_weekly, use_container_width=True)
            
        # Patrón de consumo mensual
        st.subheader("Consumo Total Mensual")
        df_monthly = df_filtered.resample('M').sum()
        fig_monthly = px.bar(df_monthly, x=df_monthly.index, y='Consumption_kWh', title='Consumo Total por Mes')
        fig_monthly.update_xaxes(title="Mes")
        st.plotly_chart(fig_monthly, use_container_width=True)

    # --- Correlación con el Clima ---
    elif analysis_type == 'Correlación con el Clima':
        st.header("Correlación del Consumo con Variables Climáticas")

        col1, col2 = st.columns(2)

        with col1:
            # Correlación con la Temperatura
            st.subheader("Consumo vs. Temperatura")
            fig_temp = px.scatter(df_filtered, x='T2M', y='Consumption_kWh', title='Consumo de Energía vs. Temperatura', labels={'T2M': 'Temperatura (°C)', 'Consumption_kWh': 'Consumo (kWh)'})
            st.plotly_chart(fig_temp, use_container_width=True)

        with col2:
            # Correlación con la Humedad
            st.subheader("Consumo vs. Humedad")
            fig_hum = px.scatter(df_filtered, x='RH2M', y='Consumption_kWh', title='Consumo de Energía vs. Humedad', labels={'RH2M': 'Humedad Relativa (%)', 'Consumption_kWh': 'Consumo (kWh)'})
            st.plotly_chart(fig_hum, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Matriz de Correlación")
        correlation_matrix = df_filtered[['Consumption_kWh', 'T2M', 'RH2M']].corr()
        st.write(correlation_matrix)
        st.info("""
        **Interpretación de la Matriz de Correlación:**
        - **Valores cercanos a 1:** Indican una fuerte correlación positiva (a medida que una variable aumenta, la otra también lo hace).
        - **Valores cercanos a -1:** Indican una fuerte correlación negativa (a medida que una variable aumenta, la otra disminuye).
        - **Valores cercanos a 0:** Indican una correlación débil o nula.
        """)

else:
    st.warning("Por favor, asegúrese de que los archivos de datos estén en el directorio correcto y seleccione un rango de fechas.")
