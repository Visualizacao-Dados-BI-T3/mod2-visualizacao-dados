import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(
    page_title="Dashboard de Suporte - Zenvia",
    page_icon="📊",
    layout="wide"
)

# 2. Carregamento dos Dados
@st.cache_data
def load_data():
    df = pd.read_csv("base_chamados_zenvia_looker.csv")
    # Converter colunas de data para datetime
    datetime_cols = ['createdDate', 'resolvedIn', 'reopenedIn', 'closedIn']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo CSV: {e}")
    st.stop()

# 3. Sidebar (Filtros Globais)
st.sidebar.header("🔍 Filtros de Análise")

# Filtro de Período do Mês
min_date = df['createdDate'].min().date()
max_date = df['createdDate'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Período de Criação do Chamado",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filtro de Atendente
owners = ["Todos"] + sorted([x for x in df['owner'].dropna().unique() if x != ""])
selected_owner = st.sidebar.selectbox("Atendente", owners)

# Aplicar Filtros aos Dados
filtered_df = df[
    (df['createdDate'].dt.date >= start_date) & 
    (df['createdDate'].dt.date <= end_date)
]

if selected_owner != "Todos":
    filtered_df = filtered_df[filtered_df['owner'] == selected_owner]

# 4. Cálculo dos KPIs Principais
# Chamados do mês atual vs acumulados anteriores
entradas_mes = filtered_df[filtered_df['createdDate'] >= '2026-09-01']
pendentes_anteriores = filtered_df[filtered_df['createdDate'] < '2026-09-01']

total_chamados = len(filtered_df)
total_entradas = len(entradas_mes)
total_pendentes_inicio = len(pendentes_anteriores)

resolved_df = filtered_df[filtered_df['baseStatus'].isin(['Resolved', 'Closed'])]
total_resolvidos = len(resolved_df)

taxa_resolucao = (total_resolvidos / total_chamados * 100) if total_chamados > 0 else 0

sla_met_count = len(resolved_df[resolved_df['sla_met'] == 'Sim'])
taxa_sla = (sla_met_count / total_resolvidos * 100) if total_resolvidos > 0 else 0

reopened_count = len(resolved_df[resolved_df['is_reopened_7d'] == 'Sim'])
taxa_reabertura = (reopened_count / total_resolvidos * 100) if total_resolvidos > 0 else 0

tma_ativo = resolved_df['lifetimeWorkingTime'].mean() if total_resolvidos > 0 else 0

# 5. Interface Principal
st.title("📊 Painel Executivo de Atendimento ao Cliente")
st.markdown("Análise de capacidade da equipe, SLA de solução e qualidade dos chamados.")
st.markdown("---")

# Linha 1: Cartões / Scorecards
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Entradas no Mês", f"{total_entradas}", help="Chamados criados a partir de 01/09/2026")

with col2:
    st.metric("Resolvidos", f"{total_resolvidos}", delta=f"{total_resolvidos - total_entradas} saldo", delta_color="inverse")

with col3:
    st.metric("Taxa Resolução", f"{taxa_resolucao:.1f}%")

with col4:
    st.metric("SLA Cumprido", f"{taxa_sla:.1f}%", help="Prazo acordado atingido")

with col5:
    st.metric("Taxa Reabertura", f"{taxa_reabertura:.1f}%", help="Reabertos em até 7 dias")

with col6:
    st.metric("TMA Ativo", f"{tma_ativo:.1f} min", help="Tempo médio ativo de solução")

st.markdown("---")

# Linha 2: Gráficos de Fluxo de Fila e Atendentes
col_chart1, col_chart2 = st.columns([6, 4])

with col_chart1:
    st.subheader("📈 Volume Diário: Criados vs Resolvidos")
    
    # Agrupamento diário
    df_created_daily = filtered_df.groupby(filtered_df['createdDate'].dt.date).size().reset_index(name='Criados')
    df_resolved_daily = resolved_df.groupby(resolved_df['resolvedIn'].dt.date).size().reset_index(name='Resolvidos')
    
    df_daily = pd.merge(df_created_daily, df_resolved_daily, left_on='createdDate', right_on='resolvedIn', how='outer').fillna(0)
    df_daily['Data'] = df_daily['createdDate'].combine_first(df_daily['resolvedIn'])
    df_daily = df_daily.sort_values('Data')
    
    fig_flow = px.line(
        df_daily, 
        x='Data', 
        y=['Criados', 'Resolvidos'], 
        labels={'value': 'Quantidade de Chamados', 'variable': 'Métrica'},
        color_discrete_map={'Criados': '#EF553B', 'Resolvidos': '#00CC96'}
    )
    st.plotly_chart(fig_flow, use_container_width=True)

with col_chart2:
    st.subheader("📌 Pendentes por Status Atual")
    unresolved_df = filtered_df[~filtered_df['baseStatus'].isin(['Resolved', 'Closed'])]
    
    if not unresolved_df.empty:
        fig_donut = px.pie(
            unresolved_df, 
            names='status', 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Nenhum chamado pendente no momento.")

# Linha 3: Tabela de Desempenho por Agente
st.markdown("---")
st.subheader("👥 Desempenho por Agente")

if not resolved_df.empty:
    agent_summary = resolved_df.groupby('owner').agg(
        Resolvidos=('id', 'count'),
        SLA_Met=('sla_met', lambda x: (x == 'Sim').sum()),
        Reabertos=('is_reopened_7d', lambda x: (x == 'Sim').sum()),
        TMA_Ativo_Medio=('lifetimeWorkingTime', 'mean')
    ).reset_index()

    agent_summary['% SLA Cumprido'] = (agent_summary['SLA_Met'] / agent_summary['Resolvidos'] * 100).round(1)
    agent_summary['% Reabertura'] = (agent_summary['Reabertos'] / agent_summary['Resolvidos'] * 100).round(1)
    agent_summary['TMA_Ativo_Medio'] = agent_summary['TMA_Ativo_Medio'].round(1)

    st.dataframe(
        agent_summary[['owner', 'Resolvidos', '% SLA Cumprido', '% Reabertura', 'TMA_Ativo_Medio']],
        column_config={
            "owner": "Atendente",
            "Resolvidos": "Resolvidos",
            "% SLA Cumprido": st.column_config.NumberColumn("% SLA Cumprido", format="%.1f%%"),
            "% Reabertura": st.column_config.NumberColumn("% Reabertura", format="%.1f%%"),
            "TMA_Ativo_Medio": st.column_config.NumberColumn("TMA Ativo (min)", format="%.1f min"),
        },
        hide_index=True,
        use_container_width=True
    )

# Detalhamento dos dados em tabela expansível
with st.expander("📋 Ver Base de Dados Filtrada"):
    st.dataframe(filtered_df)