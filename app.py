import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------
# Configuração inicial da página
# -----------------------------------------
st.set_page_config(
    page_title="Painel SVG - PMDF",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🛡️"
)

DEFAULT_HEIGHT = 600

DEFAULT_LAYOUT = dict(
    paper_bgcolor="#F8FAFC",
    plot_bgcolor="#F8FAFC",
    font=dict(color="#003366"),
    title_x=0.4
)

st.markdown("""
<style>
/* ============================== */
/* 🎨 Tema personalizado PMDF      */
/* ============================== */

/* Fundo geral e texto */
body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

/* Container das abas */
div[data-baseweb="tab-list"] {
    justify-content: center !important;      /* centraliza as abas */
    gap: 0.8rem !important;                  /* espaçamento entre abas */
    margin-top: 0.5rem !important;
}

/* Aba ativa */
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #003366 !important;    /* azul PMDF */
    color: #FFFFFF !important;
    padding: 5px 10px !important;
    border-radius: 5px 5px 0 0 !important;
    font-weight: 700 !important;
    box-shadow: 0px -3px 6px rgba(0, 0, 0, 0.1);
}

/* Abas inativas */
div[data-baseweb="tab-list"] button[aria-selected="false"] {
    background-color: #E2E8F0 !important;
    color: #003366 !important;
    padding: 5px 10px !important;
    border-radius: 5px 5px 0 0 !important;
    font-weight: 600 !important;
    border: 1px solid #CBD5E1 !important;
    transition: all 0.2s ease-in-out;
}

/* Hover nas abas inativas */
div[data-baseweb="tab-list"] button[aria-selected="false"]:hover {
    background-color: #C8A100 !important;  /* dourado PMDF */
    color: #FFFFFF !important;
    transform: scale(1.03);
}

/* Efeito de foco na aba ativa */
div[data-baseweb="tab-list"] button[aria-selected="true"]:hover {
    background-color: #00224D !important;
}

/* ============================== */
/* Layout e estética geral        */
/* ============================== */

/* Centralizar os títulos h1/h2 */
h1, h2 {
    text-align: center !important;
    color: #003366 !important;
    font-weight: 700 !important;
}

/* Links e botões */
a, .stButton>button {
    background-color: #003366 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
a:hover, .stButton>button:hover {
    background-color: #C8A100 !important;
    color: #003366 !important;
}

/* Linhas divisórias */
hr {
    border: 1px solid #003366 !important;
}

/* Cards e métricas */
[data-testid="stMetricValue"] {
    color: #003366 !important;
    font-weight: 700 !important;
}
/* ==========================
    METRICS (st.metric)
    ========================== */

[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 0.7rem 1rem !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.12);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 0.8rem !important;
}

[data-testid="stMetricValue"] {
    color: #003366 !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Painel de Análise do SVG (Serviço Voluntário Gratificado)")

# -----------------------------------------
# Função para carregar e padronizar a base
# -----------------------------------------
@st.cache_data
def load_svg_data():
    df_raw = pd.read_excel("analise_svg.xlsx")

    # Renomeia colunas
    rename_map = {
        "MÊS": "MES",
        "UNIDADE": "UNIDADE",
        "GRADUAÇÃO": "GRADUACAO",
        "NOME": "NOME",
        "MATRÍCULA": "MATRICULA",
        "QUANTIDADE": "QTD_SVG",
    }
    df = df_raw.rename(columns=rename_map)

    # Garante tipos adequados
    df["QTD_SVG"] = pd.to_numeric(df["QTD_SVG"], errors="coerce").fillna(0).astype(int)
    df["MES"] = df["MES"].astype(str).str.strip()
    df["UNIDADE"] = df["UNIDADE"].astype(str).str.strip()
    df["GRADUACAO"] = df["GRADUACAO"].astype(str).str.strip()
    df["NOME"] = df["NOME"].astype(str).str.strip()
    df["MATRICULA"] = df["MATRICULA"].astype(str).str.strip()

    # Mapa de meses em português -> número
    MAPA_MESES = {
        "JANEIRO": 1,
        "FEVEREIRO": 2,
        "MARCO": 3,
        "MARÇO": 3,
        "ABRIL": 4,
        "MAIO": 5,
        "JUNHO": 6,
        "JULHO": 7,
        "AGOSTO": 8,
        "SETEMBRO": 9,
        "OUTUBRO": 10,
        "NOVEMBRO": 11,
        "DEZEMBRO": 12,
    }

    # Converte os meses para número
    df["MES_NUM"] = df["MES"].map(MAPA_MESES)
    df["MES_NUM"] = df["MES_NUM"].astype(int)
    df = df.sort_values("MES_NUM")

    return df

def calcular_gini(valores):
    """
    Calcula o índice de Gini.
    valores: array/list com quantidades de SVG por policial.
    """
    array = np.array(valores, dtype=float)
    
    if array.size == 0:
        return 0
    
    array = array.flatten()
    array = array[array >= 0]  # remove negativos
    if np.sum(array) == 0:
        return 0
    
    array = np.sort(array)
    n = array.size
    indice = np.arange(1, n+1)
    
    gini = (
        (np.sum((2 * indice - n - 1) * array)) 
        / (n * np.sum(array))
    )
    return gini

# Função para classificar a faixa
def classificar_faixa(qtd):
    if 10 <= qtd <= 20:
        return "10-20 serviços"
    elif 20 <= qtd <= 40:
        return "20-40 serviços"
    elif 40 <= qtd <= 60:
        return "40-60 serviços"
    else:
        return "60+ serviços"

# Carrega a base
df = load_svg_data()

# -----------------------------------------
# Filtros da Sidebar
# -----------------------------------------
st.sidebar.header("Filtros")

meses = sorted(df["MES"].unique())
unidades = sorted(df["UNIDADE"].unique())
graduacoes = sorted(df["GRADUACAO"].unique())

f_mes = st.sidebar.multiselect("Mês", meses, default=meses)
f_unidade = st.sidebar.multiselect("Unidade", unidades, default=unidades)
f_grad = st.sidebar.multiselect("Graduação", graduacoes, default=graduacoes)
# -----------------------------------------
# Aplicação dos filtros no DataFrame
# -----------------------------------------
df_filtrado = df[
    (df["MES"].isin(f_mes)) &
    (df["UNIDADE"].isin(f_unidade)) &
    (df["GRADUACAO"].isin(f_grad))
]
if df_filtrado.empty:
    st.warning("Nenhum registro encontrado para os filtros selecionados.")
    st.stop()

st.markdown("---")

# ============== Seções em Abas ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "1) 📊 Indicadores Gerais",
    "2) 📈 Distribuição do SVG",
    "3) ⚖️ Equilíbrio Operacional",
    "4) 📄 Relatório"
])

# =========================================
# 1. INDICADORES GERAIS (DINÂMICOS)
# =========================================

total_svg = int(df_filtrado["QTD_SVG"].sum())
total_policiais = df_filtrado["MATRICULA"].nunique()

svg_medio_por_policial = (
    total_svg / total_policiais if total_policiais > 0 else 0
)

# Média mensal por policial
por_pol_mes = (
    df_filtrado.groupby(["MATRICULA", "MES"], as_index=False)["QTD_SVG"]
    .sum()
)

svg_medio_mensal_por_policial = por_pol_mes["QTD_SVG"].mean()

# Concentração no top 10%
svg_por_policial = (
    df_filtrado.groupby("MATRICULA", as_index=False)["QTD_SVG"]
    .sum()
    .rename(columns={"QTD_SVG": "TOTAL_SVG"})
)

svg_sorted = svg_por_policial.sort_values("TOTAL_SVG", ascending=False)
n_top10 = max(1, int(len(svg_sorted) * 0.10))

svg_top10 = svg_sorted.head(n_top10)["TOTAL_SVG"].sum()
concentracao_top10_pct = (svg_top10 / total_svg * 100) if total_svg > 0 else 0

with tab1:
    st.subheader("📊 Indicadores Gerais do SVG")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total de SVG realizados", f"{total_svg:,}".replace(",", "."))
    col2.metric("Policiais participantes", f"{total_policiais:,}".replace(",", "."))
    col3.metric("Média de SVG / policial", f"{svg_medio_por_policial:.1f}")
    col4.metric("Média mensal / policial", f"{svg_medio_mensal_por_policial:.1f}")
    col5.metric("SVG concentrado no TOP 10% (policiais)", f"{concentracao_top10_pct:.1f}%")

    #Barras agrupadas por Unidade e Mês
    svg_unid_mes = (
        df_filtrado
        .groupby(["MES_NUM", "UNIDADE", "MES"], as_index=False)["QTD_SVG"]
        .sum()
        .sort_values(["MES_NUM", "UNIDADE"])
    )
    unidades_selecionadas = df_filtrado["UNIDADE"].value_counts().sort_values(ascending=False).head(10).index
    svg_unid_mes = svg_unid_mes[svg_unid_mes["UNIDADE"].isin(unidades_selecionadas)]

    fig = px.bar(
        svg_unid_mes,
        x="MES",
        y="QTD_SVG",
        color="UNIDADE",
        barmode="group",
        title="SVG por Unidade e Mês (exibindo as 10 unidades com mais SVG)",
        labels={"MES": "", "QTD_SVG": "Quantidade de SVG", "UNIDADE": "Unidades"}
    )
    fig.update_layout(**DEFAULT_LAYOUT)
    st.plotly_chart(fig, width="stretch")

# =========================================
# 2. DISTRIBUIÇÃO DO SVG
# =========================================
with tab2:
    st.subheader("📊 Distribuição do SVG")

    col1, col2 = st.columns(2)

    with col1:
        svg_por_unidade = (
            df_filtrado
            .groupby("UNIDADE", as_index=False)["QTD_SVG"]
            .sum()
            .sort_values("QTD_SVG", ascending=False)  
        )
        total_svg = svg_por_unidade["QTD_SVG"].sum()
        svg_por_unidade["PORCENTAGEM"] = (svg_por_unidade["QTD_SVG"] / total_svg) * 100
        svg_por_unidade = svg_por_unidade.sort_values("PORCENTAGEM", ascending=False)
        fig = px.bar(svg_por_unidade,
            x="UNIDADE",
            y="QTD_SVG",
            orientation="v",
            title="SVG por Unidade",
            labels={"UNIDADE": "", "QTD_SVG": "Quantidade de SVG"}
        )
        fig.update_traces(
            hovertemplate="Unidade: %{x}<br>Total SVG: %{y}<br>Porcentagem: %{customdata[0]:.1f}%",
            customdata=svg_por_unidade[["PORCENTAGEM"]].to_numpy()
        )
        fig.update_layout(**DEFAULT_LAYOUT)
        st.plotly_chart(fig, width="stretch")

    with col2:
        svg_por_mes = (
            df_filtrado
            .groupby(["MES_NUM", "MES"], as_index=False)["QTD_SVG"]
            .sum()
            .sort_values("MES_NUM")  
        )
        
        svg_por_mes["MES"] = pd.Categorical(
            svg_por_mes["MES"],
            categories=svg_por_mes.sort_values("MES_NUM")["MES"],
            ordered=True
        )
        svg_por_mes["MM3"] = svg_por_mes["QTD_SVG"].rolling(window=3, min_periods=1).mean()
        fig_trend = go.Figure()

        # Linha real dos valores mensais
        fig_trend.add_trace(go.Scatter(
            x=svg_por_mes["MES"],
            y=svg_por_mes["QTD_SVG"],
            mode="lines+markers",
            name="SVG Mensal",
            line=dict(color="#003366", width=3),
            marker=dict(size=8),
            hovertemplate="Mês: %{x}<br>Quantidade de SVG: %{y}"
        ))

        # Linha da média móvel (suavização)
        fig_trend.add_trace(go.Scatter(
            x=svg_por_mes["MES"],
            y=svg_por_mes["MM3"],
            mode="lines",
            name="Média Móvel (3 meses)",
            line=dict(color="#C8A100", width=4, dash="dash")
        ))

        fig_trend.update_layout(
            title="Tendência Mensal do SVG com Média Móvel (MM3)",
            xaxis_title="Mês",
            yaxis_title="Quantidade de SVG",
            height=500,
            xaxis=dict(tickangle=-30)
        )
        fig_trend.update_layout(**DEFAULT_LAYOUT)

        st.plotly_chart(fig_trend, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        svg_por_grad = (
            df_filtrado
            .groupby("GRADUACAO", as_index=False)["QTD_SVG"]
            .sum()
            .sort_values("QTD_SVG", ascending=False)
        )
        fig = px.bar(svg_por_grad,
            x="GRADUACAO",
            y="QTD_SVG",
            orientation="v",
            title="SVG por Graduação",
            labels={"GRADUACAO": "", "QTD_SVG": "Quantidade de SVG"}
        )   
        fig.update_layout(**DEFAULT_LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with col4:
        # Top 10 policiais (por quantidade de SVG)
        top10_pol_porsvg = (
            df_filtrado
            .groupby(["MATRICULA", "NOME"], as_index=False)["QTD_SVG"]
            .sum()
            .sort_values("QTD_SVG", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top10_pol_porsvg.sort_values("QTD_SVG"),
            x="QTD_SVG",
            y="NOME",
            orientation="h",
            title="Top 10 Policiais por Quantidade de SVG",
            labels={"QTD_SVG": "Quantidade de SVG", "NOME": ""}
        )
        fig.update_layout(**DEFAULT_LAYOUT)
        fig.update_traces(
            hovertemplate="Nome: %{y}<br>SVG: %{x}<br>Matrícula: %{customdata}"
        )
        fig.update_traces(customdata=top10_pol_porsvg["MATRICULA"])
        st.plotly_chart(fig, width="stretch")

# =========================================
# 3. ANÁLISE OPERACIONAL
# =========================================
with tab3:
    st.subheader("📊 Análise Operacional")
    col1, col2 = st.columns(2)
    with col1:
        #SVG por Policial e faixa de distribuição
        svg_por_policial = (
            df_filtrado
            .groupby("MATRICULA", as_index=False)["QTD_SVG"]
            .sum()
            .rename(columns={"QTD_SVG": "TOTAL_SVG"})
        )
        svg_por_policial["FAIXA_SVG"] = svg_por_policial["TOTAL_SVG"].apply(classificar_faixa)
        # Agrupa por faixa
        dist_faixas = (
            svg_por_policial
            .groupby("FAIXA_SVG", as_index=False)
            .agg(
                POLICIAIS=("MATRICULA", "nunique"),
                TOTAL_SVG=("TOTAL_SVG", "sum")
            )
        )
        # Ordena faixas manualmente
        ordem_faixas = ["10-20 serviços", "20-40 serviços", "40-60 serviços", "60+ serviços"]
        dist_faixas["FAIXA_SVG"] = pd.Categorical(dist_faixas["FAIXA_SVG"],
                                                categories=ordem_faixas,
                                                ordered=True)
        dist_faixas = dist_faixas.sort_values("FAIXA_SVG")
        # Percentual de policiais em cada faixa
        total_policiais = dist_faixas["POLICIAIS"].sum()
        dist_faixas["PCT_POLICIAIS"] = dist_faixas["POLICIAIS"] / total_policiais * 100
        fig = px.bar(dist_faixas,
            x="FAIXA_SVG",
            y="POLICIAIS",
            text="PCT_POLICIAIS",
            title="Distribuição de policiais por faixa de SVG no período filtrado",
            labels={"FAIXA_SVG": "Faixa de serviços realizados", "POLICIAIS": "Quantidade de policiais"}
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",  
            textposition="auto"
        )
        fig.update_layout(**DEFAULT_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Ver tabela detalhada por faixa"):
            st.dataframe(dist_faixas.assign(PCT_POLICIAIS=lambda d: d["PCT_POLICIAIS"].round(1)))

    with col2:
        # SVG total por unidade
        svg_por_unidade = (
            df_filtrado
            .groupby("UNIDADE", as_index=False)["QTD_SVG"]
            .sum()
            .rename(columns={"QTD_SVG": "TOTAL_SVG"})
        )

        # Policiais distintos por unidade
        policiais_por_unidade = (
            df_filtrado
            .groupby("UNIDADE", as_index=False)["MATRICULA"]
            .nunique()
            .rename(columns={"MATRICULA": "POLICIAIS_UNICOS"})
        )

        # Quantidade de meses em que cada unidade aparece (meses ativos)
        meses_por_unidade = (
            df_filtrado
            .groupby("UNIDADE", as_index=False)["MES"]
            .nunique()
            .rename(columns={"MES": "MESES_ATIVOS"})
        )

        # Junta tudo
        media_svg_unidade = (
            svg_por_unidade
            .merge(policiais_por_unidade, on="UNIDADE")
            .merge(meses_por_unidade, on="UNIDADE")
        )

        # Calcula média MENSAL de SVG por policial em cada unidade
        # media_svg_unidade["MEDIA_MENSAL_SVG"] = (
        #     media_svg_unidade["TOTAL_SVG"] /
        #     (media_svg_unidade["POLICIAIS_UNICOS"] * media_svg_unidade["MESES_ATIVOS"])
        # ).round(2)
        
        # Calcula média MENSAL de SVG em cada unidade (sem policiais)
        media_svg_unidade["MEDIA_MENSAL_SVG"] = (
            media_svg_unidade["TOTAL_SVG"] / media_svg_unidade["MESES_ATIVOS"]
        )


        # Ordena decrescente pela média mensal
        media_svg_unidade = media_svg_unidade.sort_values("MEDIA_MENSAL_SVG", ascending=False)

        fig_media = px.bar(
            media_svg_unidade,
            x="UNIDADE",
            y="MEDIA_MENSAL_SVG",
            title="Média MENSAL em cada Unidade",
            labels={"UNIDADE": "", "MEDIA_MENSAL_SVG": "Média mensal de SVG / unidade"}
        )

        fig_media.update_layout(**DEFAULT_LAYOUT)

        st.plotly_chart(fig_media, width="stretch")

        with st.expander("Ver tabela detalhada"):
            st.dataframe(media_svg_unidade)


    # Mapa de calor - SVG por Unidade e Mês (ordenado)
    heatmap_svg = (
        df_filtrado
        .groupby(["UNIDADE", "MES_NUM", "MES"], as_index=False)["QTD_SVG"]
        .sum()
    )

    # Garantir ordem correta
    heatmap_svg = heatmap_svg.sort_values("MES_NUM")

    # Pivot: MES como índice, mas usando MES_NUM para ordenar
    pivot_heatmap = heatmap_svg.pivot_table(
        index="MES", 
        columns="UNIDADE", 
        values="QTD_SVG",
        fill_value=0
    )

    # Criamos um dataframe auxiliar apenas para pegar a ordem
    ordem_meses = (
        heatmap_svg[["MES_NUM", "MES"]]
        .drop_duplicates()
        .sort_values("MES_NUM")["MES"]
        .tolist()
    )

    # Reindexa seguindo MES_NUM
    pivot_heatmap = pivot_heatmap.reindex(ordem_meses)

    # Criar Heatmap Plotly
    fig = px.imshow(
        pivot_heatmap,
        x=pivot_heatmap.columns,
        y=pivot_heatmap.index,
        color_continuous_scale=["#E8EEF7", "#003366", "#C8A100"],
        aspect="auto",
        labels=dict(color="Total SVG"),
        title="Heatmap de SVG por Unidade e Mês",
    )

    fig.update_layout(**DEFAULT_LAYOUT)
    fig.update_traces(text=pivot_heatmap.values, texttemplate="%{text}")
    st.plotly_chart(fig, width="stretch")


# =========================================
# 4. RELATÓRIO
# =========================================
with tab4:
    st.subheader("📄 Relatório Automático de Insights – SVG")

    # ======================================================
    # 1) Estatísticas gerais
    # ======================================================
    total_svg = int(df_filtrado["QTD_SVG"].sum())
    total_policiais = df_filtrado["MATRICULA"].nunique()
    total_unidades = df_filtrado["UNIDADE"].nunique()
    total_meses = df_filtrado["MES"].nunique()

    # ======================================================
    # 2) SVG por policial
    # ======================================================
    svg_por_pol = (
        df_filtrado
        .groupby("MATRICULA", as_index=False)["QTD_SVG"]
        .sum()
        .rename(columns={"QTD_SVG": "TOTAL_SVG"})
    )

    svg_por_pol["FAIXA_SVG"] = svg_por_pol["TOTAL_SVG"].apply(classificar_faixa)

    dist_faixas = (
        svg_por_pol
        .groupby("FAIXA_SVG", as_index=False)
        .agg(POLICIAIS=("MATRICULA", "nunique"))
    )

    dist_faixas_dict = dict(
        zip(dist_faixas["FAIXA_SVG"], dist_faixas["POLICIAIS"])
    )

    # ======================================================
    # 3) Concentração TOP 10%
    # ======================================================
    svg_sorted = svg_por_pol.sort_values("TOTAL_SVG", ascending=False)
    n_top10 = max(1, int(len(svg_sorted) * 0.10))
    svg_top10 = svg_sorted.head(n_top10)["TOTAL_SVG"].sum()
    conc_top10 = (svg_top10 / total_svg) * 100 if total_svg > 0 else 0

    # ======================================================
    # 4) Gini
    # ======================================================
    def calcular_gini(valores):
        import numpy as np
        array = np.array(valores, dtype=float)
        if len(array) == 0 or array.sum() == 0:
            return 0
        array = np.sort(array)
        n = array.size
        indice = np.arange(1, n+1)
        gini = (np.sum((2 * indice - n - 1) * array)) / (n * array.sum())
        return gini

    gini_svg = calcular_gini(svg_por_pol["TOTAL_SVG"].values)

    # ======================================================
    # 5) SVG por unidade
    # ======================================================
    svg_unid = (
        df_filtrado
        .groupby("UNIDADE", as_index=False)["QTD_SVG"]
        .sum()
        .rename(columns={"QTD_SVG": "TOTAL_SVG"})
        .sort_values("TOTAL_SVG", ascending=False)
    )
    unidade_top = svg_unid.iloc[0] if not svg_unid.empty else None

    svg_unid_nonzero = svg_unid[svg_unid["TOTAL_SVG"] > 0]
    unidade_min = svg_unid_nonzero.iloc[-1] if not svg_unid_nonzero.empty else None

    # ======================================================
    # 6) SVG por mês – tendência
    # ======================================================
    svg_por_mes = (
        df_filtrado
        .groupby(["MES_NUM", "MES"], as_index=False)["QTD_SVG"]
        .sum()
        .sort_values("MES_NUM")
    )

    # Identifica o mês com maior quantidade de SVG
    mes_top = svg_por_mes.loc[svg_por_mes["QTD_SVG"].idxmax()] if not svg_por_mes.empty else None
    mm3 = svg_por_mes["QTD_SVG"].rolling(window=3, min_periods=1).mean()

    relatorio_texto = f"""

    ### 1) Visão Geral
    No período filtrado, observou-se:
    - **{total_svg:,}** serviços de SVG realizados;
    - Participação de **{total_policiais:,}** policiais;
    - Atuação distribuída em **{total_unidades}** unidades;
    - Total de **{total_meses}** mês(es) analisados.

    A unidade com maior utilização de SVG foi **{unidade_top['UNIDADE']}**, com **{int(unidade_top['TOTAL_SVG']):,}** serviços.  
    A de menor demanda (não zero) foi **{unidade_min['UNIDADE']}**, com **{int(unidade_min['TOTAL_SVG']):,}** serviços.

    ---

    ### 2) Perfil Individual de Engajamento
    A distribuição dos policiais por número de SVG indica:

    - **10-20 serviços:** {dist_faixas_dict.get("10-20 serviços", 0)} policiais  
    - **20-40 serviços:** {dist_faixas_dict.get("20-40 serviços", 0)} policiais  
    - **40-60 serviços:** {dist_faixas_dict.get("40-60 serviços", 0)} policiais  
    - **60+ serviços:** {dist_faixas_dict.get("60+ serviços", 0)} policiais  

    Esse perfil mostra quantos estão em engajamento pontual, moderado e intensivo.

    ---

    ### 3) Concentração da Carga
    - Os **10% mais atuantes** responderam por **{conc_top10:.1f}%** de todo o SVG.
    - O índice de **Gini** foi **{gini_svg:.2f}**, indicando:
    **{"baixa" if gini_svg < 0.2 else "moderada" if gini_svg < 0.4 else "alta" if gini_svg < 0.6 else "muito alta"} desigualdade** na distribuição da carga.

    ---

    ### 4) Tendência Mensal do SVG
    O mês de maior demanda foi: **{mes_top['MES']}**, com **{int(mes_top['QTD_SVG']):,}** SVG.
    
    A média móvel (MM3) indica que:
    - Há **{"+ aumento" if mm3.iloc[-1] > mm3.iloc[0] else "queda"}** na tendência geral do uso do SVG ao longo do período.
    - A série apresenta comportamento **{"estável" if abs(mm3.iloc[-1] - mm3.iloc[0]) < 10 else "volátil"}**.

    ---

    ### 5) Interpretação Operacional

    - A desigualdade detectada pelo Gini sugere que **parte do efetivo concentra grande volume de SVG**, podendo indicar risco de fadiga.
    - Unidades com alto uso de SVG podem estar **dependentes do serviço** para manter escala operacional.
    - Unidades com média baixa por policial indicam **baixa adesão** ou equilíbrio operacional.
    - Picos mensais podem estar associados a **eventos estratégicos**, devendo ser considerados no planejamento do efetivo.

    ---

    ### 6) Recomendações

    - Avaliar redistribuição interna do SVG em unidades com alta concentração.
    - Planejar reforço em meses historicamente mais críticos.
    - Incentivar participação mais homogênea para reduzir desigualdade.
    """

    st.markdown(relatorio_texto)

