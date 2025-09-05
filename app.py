# app.py
# -*- coding: utf-8 -*-
import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Relatório Agro IBGE (LSPA)", layout="wide")

# ========= Utilidades =========
def fmt_br(x, nd=0):
    """Formata número no padrão pt-BR (milhar com ponto e decimais com vírgula)."""
    if pd.isna(x):
        return ""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8").encode("utf-8")

def barh_sorted(df, x_col, y_col="produto_clean", topn=10, title="", x_title="", nd=0):
    """
    Barras horizontais MAIOR->MENOR com rótulos, ordem fixa e eixo Y invertido
    para exibir o maior no topo.
    """
    df2 = df.sort_values(x_col, ascending=False).head(topn).copy()
    fig = px.bar(
        df2,
        x=x_col,
        y=y_col,
        orientation="h",
        text=df2[x_col].map(lambda v: fmt_br(v, nd)),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="",
        yaxis=dict(
            categoryorder="array",
            categoryarray=df2[y_col].tolist(),
            autorange="reversed",  # maior no topo
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def barh_grouped_sorted(df, y_col, x_cols, labels, topn=10, title="", x_title=""):
    """
    Barras horizontais agrupadas (ex.: Ano A vs Ano B), ordenadas pela soma
    das séries (MAIOR->MENOR) e com eixo Y invertido.
    - x_cols: lista de colunas numéricas
    - labels: lista de tuplas (rótulo_legenda, casas_decimais)
    """
    df2 = df.assign(_sort=df[x_cols].sum(axis=1)).sort_values("_sort", ascending=False).head(topn).copy()
    fig = go.Figure()
    for x_col, (name, nd) in zip(x_cols, labels):
        fig.add_trace(
            go.Bar(
                y=df2[y_col],
                x=df2[x_col],
                orientation="h",
                name=name,
                text=[fmt_br(v, nd) for v in df2[x_col]],
                textposition="outside",
                cliponaxis=False,
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=x_title,
        yaxis_title="",
        yaxis=dict(
            categoryorder="array",
            categoryarray=df2[y_col].tolist(),
            autorange="reversed",  # maior no topo
        ),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

@st.cache_data(show_spinner=False)
def tidy_from_sheet(sheet_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Lê uma aba do SIDRA (LSPA) e transforma em formato tidy."""
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None)

    # detectar linha de cabeçalho com "Safra 20xx"
    header_row = None
    for i in range(0, 12):
        row = df.iloc[i].astype(str).tolist()
        if any(re.search(r"Safra\s*\d{4}", x) for x in row):
            header_row = i
            break
    if header_row is None:
        header_row = 4  # fallback

    # anos nas colunas a partir da 3ª coluna
    year_cells = df.iloc[header_row, 2:].tolist()
    years = [
        int(re.search(r"(\d{4})", str(y)).group(1))
        for y in year_cells
        if pd.notna(y) and re.search(r"(\d{4})", str(y))
    ]

    # dados começam abaixo
    data = df.iloc[header_row + 1 :, : 2 + len(years)].copy()
    data.columns = ["grupo_produto", "produto", *years]
    data = data.dropna(how="all")
    data["grupo_produto"] = data["grupo_produto"].ffill()

    long = data.melt(
        id_vars=["grupo_produto", "produto"],
        value_vars=years,
        var_name="ano_safra",
        value_name="valor",
    )
    long = long.dropna(subset=["valor"])
    long["grupo"] = long["grupo_produto"].astype(str).str.strip()
    long["produto"] = long["produto"].astype(str).str.strip()
    long = long.drop(columns=["grupo_produto"])
    long["variavel"] = sheet_name
    long["unidade"] = "Hectares" if "Área" in sheet_name else "Toneladas"
    long["ano_safra"] = long["ano_safra"].astype(int)
    long["valor"] = pd.to_numeric(long["valor"], errors="coerce")
    long = long.dropna(subset=["valor"])
    return long

@st.cache_data(show_spinner=False)
def build_all(file_bytes: bytes):
    """Lê as três abas, unifica, calcula indicadores e pendências."""
    ap = tidy_from_sheet("Área plantada (Hectares)", file_bytes)
    ac = tidy_from_sheet("Área colhida (Hectares)", file_bytes)
    pr = tidy_from_sheet("Produção (Toneladas)", file_bytes)

    tidy = pd.concat([ap, ac, pr], ignore_index=True)
    tidy["produto_clean"] = tidy["produto"].str.replace(
        r"^\d+(?:\.\d+)*\s+", "", regex=True
    )

    pivot = tidy.pivot_table(
        index=["grupo", "produto_clean", "ano_safra"],
        columns="variavel",
        values="valor",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(
        columns={
            "Área plantada (Hectares)": "area_plantada_ha",
            "Área colhida (Hectares)": "area_colhida_ha",
            "Produção (Toneladas)": "producao_t",
        }
    )
    pivot["produtividade_t_ha"] = (
        pivot["producao_t"] / pivot["area_colhida_ha"]
    ).round(3)

    # pendências
    df_prog = pivot.copy()
    df_prog["produtividade_t_ha"] = df_prog["produtividade_t_ha"].where(
        df_prog["produtividade_t_ha"].notna(),
        df_prog["producao_t"] / df_prog["area_colhida_ha"],
    )
    df_prog["area_pendente_ha"] = np.maximum(
        df_prog["area_plantada_ha"] - df_prog["area_colhida_ha"], 0
    )
    df_prog["producao_realizada_t_calc"] = (
        df_prog["produtividade_t_ha"].fillna(0) * df_prog["area_colhida_ha"].fillna(0)
    )
    df_prog["producao_pendente_t_est"] = (
        df_prog["produtividade_t_ha"].fillna(0) * df_prog["area_pendente_ha"].fillna(0)
    )
    df_prog["producao_total_est_t"] = (
        df_prog["producao_realizada_t_calc"] + df_prog["producao_pendente_t_est"]
    )

    return tidy, pivot, df_prog


# ========= Sidebar (Entrada) =========
st.sidebar.header("Entrada de dados")
uploaded = st.sidebar.file_uploader("Envie o Excel do IBGE (SIDRA)", type=["xlsx"])

default_path = "dados_sidra_ibge.xlsx"
if uploaded is None:
    try:
        with open(default_path, "rb") as f:
            file_bytes = f.read()
        st.sidebar.success(f"Carregado arquivo local: {default_path}")
    except FileNotFoundError:
        st.warning("Envie um arquivo .xlsx do SIDRA para continuar.")
        st.stop()
else:
    file_bytes = uploaded.read()

tidy, pivot, df_prog = build_all(file_bytes)

anos = sorted(df_prog["ano_safra"].unique().tolist())
ano_ref = st.sidebar.selectbox("Safra (para seções 1 e 3)", options=anos, index=len(anos) - 1)
topn = st.sidebar.slider("Top N nos gráficos", 5, 25, 10, 1)
min_area_produt = st.sidebar.slider(
    "Área mínima (ha) para ranking de produtividade",
    0,
    200_000,
    10_000,
    5_000,
)

# ========= Header =========
st.title(f"Relatório Agro IBGE (LSPA)")
st.caption(
    "Análises por métrica, pendências, comparativo entre anos e rankings. "
    "Gráficos sempre do maior para o menor (topo para baixo)."
)

# ========= Base do ano selecionado (para seções 1 e 3) =========
df_ano = df_prog[
    (df_prog["ano_safra"] == ano_ref)
    & (~df_prog["produto_clean"].str.lower().eq("total"))
].copy()

# ========= Seção 1) Progresso por cultura =========
st.subheader(f"Progresso por cultura — Safra {ano_ref}")
cols_show = [
    "grupo",
    "produto_clean",
    "area_plantada_ha",
    "area_colhida_ha",
    "area_pendente_ha",
    "producao_t",
    "produtividade_t_ha",
    "producao_realizada_t_calc",
    "producao_pendente_t_est",
    "producao_total_est_t",
]
tabela = (
    df_ano[cols_show]
    .sort_values("producao_pendente_t_est", ascending=False)
    .reset_index(drop=True)
)
st.dataframe(
    tabela.style.format(
        {
            "area_plantada_ha": "{:,.0f}",
            "area_colhida_ha": "{:,.0f}",
            "area_pendente_ha": "{:,.0f}",
            "producao_t": "{:,.0f}",
            "produtividade_t_ha": "{:,.3f}",
            "producao_realizada_t_calc": "{:,.0f}",
            "producao_pendente_t_est": "{:,.0f}",
            "producao_total_est_t": "{:,.0f}",
        }
    ),
    use_container_width=True,
)

st.divider()

# ========= Mapa de métricas =========
metric_map = {
    "Área plantada (ha)": ("area_plantada_ha", 0, "Área plantada (ha)"),
    "Área colhida (ha)": ("area_colhida_ha", 0, "Área colhida (ha)"),
    "Área pendente (ha)": ("area_pendente_ha", 0, "Área pendente (ha)"),
    "Produção IBGE (t)": ("producao_t", 0, "Produção (t) — IBGE (mês de ref.)"),
    "Produção realizada (t)": ("producao_realizada_t_calc", 0, "Produção realizada (t)"),
    "Produção projetada/pendente (t)": ("producao_pendente_t_est", 0, "Produção projetada/pendente (t)"),
    "Produção total estimada (t)": ("producao_total_est_t", 0, "Produção total estimada (t)"),
    "Produtividade (t/ha)": ("produtividade_t_ha", 3, "Produtividade (t/ha)"),
}

# ========= Seção 2) Comparativo entre anos =========
# ========= Seção 2) Comparativo entre anos =========
st.header("Comparativo entre anos")

c1, c2, c3 = st.columns([1,1,2])
with c1:
    metric_comp_label = st.selectbox("Métrica", list(metric_map.keys()), index=3)
with c2:
    idx_b = len(anos) - 1
    idx_a = max(0, idx_b - 1)
    ano_a = st.selectbox("Ano A", anos, index=idx_a)
    ano_b = st.selectbox("Ano B", anos, index=idx_b)

metric_col, metric_nd, metric_x_title = metric_map[metric_comp_label]

# agrega por produto nos dois anos
base_a = (
    df_prog[df_prog["ano_safra"] == ano_a]
    .groupby("produto_clean", as_index=False)[metric_col]
    .sum(numeric_only=True)
    .rename(columns={metric_col: f"{metric_col}_{ano_a}"})
)
base_b = (
    df_prog[df_prog["ano_safra"] == ano_b]
    .groupby("produto_clean", as_index=False)[metric_col]
    .sum(numeric_only=True)
    .rename(columns={metric_col: f"{metric_col}_{ano_b}"})
)

cmp = pd.merge(base_a, base_b, on="produto_clean", how="outer").fillna(0.0)
cmp = cmp[~cmp["produto_clean"].str.lower().eq("total")].copy()
cmp["delta"] = cmp[f"{metric_col}_{ano_b}"] - cmp[f"{metric_col}_{ano_a}"]
cmp["delta_pct"] = np.where(
    cmp[f"{metric_col}_{ano_a}"] != 0,
    cmp["delta"] / cmp[f"{metric_col}_{ano_a}"],
    np.nan,
)

# gráfico agrupado: Ano A vs Ano B (ordenado pela soma)
fig_group = barh_grouped_sorted(
    cmp,
    y_col="produto_clean",
    x_cols=[f"{metric_col}_{ano_a}", f"{metric_col}_{ano_b}"],
    labels=[(str(ano_a), metric_nd), (str(ano_b), metric_nd)],
    topn=topn,
    title=f"Top {topn} — {metric_comp_label}: {ano_a} × {ano_b} (ordenado pela soma)",
    x_title=metric_x_title,
)
st.plotly_chart(fig_group, use_container_width=True)

# tabela resumo do comparativo (sem gráfico de variação)
st.dataframe(
    cmp.sort_values(f"{metric_col}_{ano_b}", ascending=False)
      .head(topn)[["produto_clean", f"{metric_col}_{ano_a}", f"{metric_col}_{ano_b}", "delta", "delta_pct"]]
      .rename(columns={
          f"{metric_col}_{ano_a}": f"{metric_comp_label} {ano_a}",
          f"{metric_col}_{ano_b}": f"{metric_comp_label} {ano_b}",
          "delta": f"Δ {metric_comp_label} ({ano_b} − {ano_a})",
          "delta_pct": "% Δ",
      })
      .style.format({
          f"{metric_comp_label} {ano_a}": "{:,.0f}",
          f"{metric_comp_label} {ano_b}": "{:,.0f}",
          f"Δ {metric_comp_label} ({ano_b} − {ano_a})": "{:,.0f}",
          "% Δ": "{:.1%}",
      }),
    use_container_width=True,
)

st.divider()


# ========= Seção 3) Análise por métrica (um ano) =========
st.subheader(f"Análise por métrica — Safra {ano_ref} (ordenado do maior para o menor)")
metric_label = st.radio("Escolha a métrica", list(metric_map.keys()), horizontal=True, index=3)
col, nd, x_title = metric_map[metric_label]

fig_metric = barh_sorted(
    df_ano,
    x_col=col,
    topn=topn,
    title=f"Top {topn} — {x_title} • Safra {ano_ref}",
    x_title=x_title,
    nd=nd,
)
st.plotly_chart(fig_metric, use_container_width=True)

st.divider()

# ========= Seção 4) Ranking clássico (produção/produtividade) =========
st.subheader(f"Ranking por produção / produtividade — Safra {ano_ref}")
metric_opt = st.radio("Métrica do ranking", ["Produção (t)", "Produtividade (t/ha)"], horizontal=True)

if metric_opt == "Produção (t)":
    base = df_ano.copy()
    fig_rank = barh_sorted(
        base,
        x_col="producao_t",
        topn=topn,
        title=f"Top {topn} — Produção • Safra {ano_ref}",
        x_title="Produção (t)",
        nd=0,
    )
else:
    base = df_ano[df_ano["area_colhida_ha"] >= min_area_produt].copy()
    fig_rank = barh_sorted(
        base,
        x_col="produtividade_t_ha",
        topn=topn,
        title=f"Top {topn} — Produtividade • Safra {ano_ref}",
        x_title="Produtividade (t/ha)",
        nd=3,
    )

st.plotly_chart(fig_rank, use_container_width=True)

st.caption(
    "Notas: estimativas baseadas no LSPA (IBGE) do mês de referência. "
    "Produção/área pendente assumem produtividade estável até o fim da safra. "
    "Em culturas com área colhida muito baixa, a produtividade pode ficar instável."
)
