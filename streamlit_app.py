
# Importar librerías
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io


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

    # --- Visualización de Datos ---
    st.markdown(f"Mostrando **{len(df_filtered):,}** registros tras aplicar filtros.")
    st.markdown("---")
    
    st.header("Análisis de Patrones Temporales")
    if not df_filtered.empty:
        # Gráfico de evolución temporal
        st.plotly_chart(px.line(df_filtered.reset_index(), x='datetime', y='Consumption_kWh', title='Evolución del Consumo Energético'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patrón Diario Promedio")
            # Agrupamos por hora y reiniciamos el índice
            df_hourly = df_filtered.groupby(df_filtered.index.hour).mean().reset_index()
            # ✨ MEJORA: Renombramos la columna 'datetime' (que contiene horas) a 'Hora' para mayor claridad
            df_hourly.rename(columns={'datetime': 'Hora'}, inplace=True)
            st.plotly_chart(px.bar(df_hourly, x='Hora', y='Consumption_kWh', title='Consumo Promedio por Hora'), use_container_width=True)

        with col2:
            st.subheader("Patrón Semanal Promedio")
            # Agrupamos por día de la semana
            df_weekly = df_filtered.groupby(df_filtered.index.dayofweek).mean()

            if not df_weekly.empty:
                # Mapeamos los números del día (0-6) a los nombres ('Lunes', 'Martes', ...)
                df_weekly.index = df_weekly.index.map(dias_semana)
                
                # Convertimos el índice a una columna. La nueva columna se llamará 'datetime' (el nombre original del índice)
                df_plot_weekly = df_weekly.reset_index()
                
                # ✨ CORRECCIÓN Y MEJORA: Renombramos la columna 'datetime' a 'Día' para que sea más claro
                df_plot_weekly.rename(columns={'datetime': 'Día'}, inplace=True)
                
                # Usamos el nombre de columna correcto 'Día' en el eje x
                st.plotly_chart(px.bar(df_plot_weekly, x='Día', y='Consumption_kWh', title='Consumo Promedio por Día'), use_container_width=True)
    else:
        st.warning("No hay datos de consumo para los filtros seleccionados.")

    # --- Análisis Climático (sin cambios en esta parte) ---
    if not df_weather_raw.empty:
        st.markdown("---")
        st.header("Correlación del Consumo con el Clima")
        try:
            df_weather = df_weather_raw.copy()
            df_weather['datetime'] = pd.to_datetime(df_weather[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
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
                st.warning("No hay datos climáticos para el rango y filtros seleccionados.")
        except Exception as e:
            st.error(f"Error al procesar o fusionar los datos de clima: {e}")
else:
    st.info("Para comenzar, carga un archivo de consumo o selecciona uno desde GitHub en la barra lateral.")
