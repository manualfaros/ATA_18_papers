from __future__ import annotations
import io, re, sqlite3
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="EDA expresión diferencial (SQLite)",
    page_icon="🧬",
    layout="wide",
)

# ============================================================
# UTILIDADES
# ============================================================
def elegir_columna_gen(df: pd.DataFrame) -> Optional[str]:
    for c in ["gene", "gene_id", "GeneID", "Geneid", "locus_tag"]:
        if c in df.columns:
            return c
    return None


def cargar_lista_sasp(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        cols = pd.read_sql_query("PRAGMA table_info(SaSP_list);", conn)["name"].tolist()
        if "gene" in cols:
            q = "SELECT gene FROM SaSP_list;"
        elif "gene_id" in cols:
            q = "SELECT gene_id AS gene FROM SaSP_list;"
        else:
            return []
        return pd.read_sql_query(q, conn)["gene"].astype(str).tolist()
    finally:
        conn.close()


@st.cache_data
def cargar_long_desde_sqlite(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    cols_genes = pd.read_sql_query("PRAGMA table_info(Genes_SA);", conn)["name"].tolist()
    q_genes = (
        "SELECT DISTINCT gene AS gene FROM Genes_SA;"
        if "gene" in cols_genes
        else "SELECT DISTINCT gene_id AS gene FROM Genes_SA;"
    )
    genes = pd.read_sql_query(q_genes, conn)["gene"].astype(str)

    tablas = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'DEG_%';",
        conn,
    )["name"].tolist()

    registros = []

    for t in tablas:
        df = pd.read_sql_query(f"SELECT * FROM {t};", conn)
        if df.empty:
            continue

        gene_col = elegir_columna_gen(df)
        if gene_col is None or "logFC" not in df.columns:
            continue

        padj_col = next((c for c in ["adjP", "adj_P_Val", "adj.P.Val", "padj"] if c in df.columns), None)
        if padj_col is None:
            continue

        ave_col = "AveExpr" if "AveExpr" in df.columns else None

        keep = [gene_col, "logFC", padj_col] + ([ave_col] if ave_col else [])
        df = df[keep].rename(columns={gene_col: "gene", padj_col: "padj"})
        if ave_col:
            df = df.rename(columns={ave_col: "AveExpr"})

        base = pd.DataFrame({"gene": genes})
        base = base.merge(df, on="gene", how="left")

        base["logFC"] = pd.to_numeric(base["logFC"], errors="coerce").fillna(0.0)
        base["padj"] = pd.to_numeric(base["padj"], errors="coerce").fillna(1.0)
        base["AveExpr"] = pd.to_numeric(base.get("AveExpr", 0), errors="coerce").fillna(0.0)

        base["contraste"] = re.sub(r"^DEG_", "", t)
        registros.append(base)

    conn.close()
    return pd.concat(registros, ignore_index=True)


@st.cache_data
def construir_matriz(df_long: pd.DataFrame) -> pd.DataFrame:
    return (
        df_long
        .pivot_table(index="gene", columns="contraste", values="logFC", aggfunc="mean")
        .fillna(0.0)
    )


def aplicar_zscore(mat: pd.DataFrame) -> pd.DataFrame:
    return (mat.sub(mat.mean(axis=1), axis=0)).div(mat.std(axis=1).replace(0, 1), axis=0)


def limpiar_para_correlacion(mat: pd.DataFrame) -> pd.DataFrame:
    m = mat.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return m.loc[m.var(axis=1) > 0]


def plot_heatmap_scroll(fig, height=900):
    with st.container(height=height):
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# CARGA DATOS
# ============================================================
st.sidebar.title("⚙️ Fuente de datos")
db_path = st.sidebar.text_input("Ruta BD SQLite", "transcriptomica_analisis.db")

df_long = cargar_long_desde_sqlite(db_path)
mat_logfc = construir_matriz(df_long)

genes = mat_logfc.index.tolist()
contrastes = mat_logfc.columns.tolist()


# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "🌋 Volcano + DEGs",
    "🔍 Gen",
    "🔥 Heatmap global",
    "🧬 Heatmap SaSP"
])

# ============================================================
# VOLCANO + DEGS
# ============================================================
with tabs[0]:
    csel = st.selectbox("Contraste", contrastes)
    sub = df_long[df_long["contraste"] == csel].copy()

    col1, col2 = st.columns(2)
    thr_logfc = col1.slider("|logFC| mínimo", 0.0, 5.0, 1.0)
    thr_padj = col2.slider("adj.P.Val máximo", 0.0, 1.0, 0.05)

    sub["neglog10"] = -np.log10(np.clip(sub["padj"], 1e-300, 1.0))
    sub["cat"] = "No sig"
    sub.loc[(sub["padj"] <= thr_padj) & (sub["logFC"] >= thr_logfc), "cat"] = "Up"
    sub.loc[(sub["padj"] <= thr_padj) & (sub["logFC"] <= -thr_logfc), "cat"] = "Down"

    fig = go.Figure()
    fig.add_scatter(
        x=sub["logFC"], y=sub["neglog10"],
        mode="markers",
        marker=dict(size=4,
            color=sub["cat"].map({"Up": "red", "Down": "blue", "No sig": "lightgray"})),
        text=sub["gene"],
    )
    fig.add_hline(y=-np.log10(thr_padj), line_dash="dash")
    fig.add_vline(x=thr_logfc, line_dash="dash")
    fig.add_vline(x=-thr_logfc, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📑 Genes diferencialmente expresados")
    degs = sub[(sub["padj"] <= thr_padj) & (sub["logFC"].abs() >= thr_logfc)]
    st.write(f"Genes significativos: **{degs.shape[0]}**")
    st.dataframe(degs[["gene", "logFC", "AveExpr", "padj"]], use_container_width=True)
# ============================================================
# Gen
# ============================================================    
with tabs[1]:
    st.header("🔍 Explorador por gen")

    gene_sel = st.selectbox(
        "Selecciona un gen",
        sorted(df_long["gene"].unique()),
        key="gen_selector",
    )

    sub = df_long[df_long["gene"] == gene_sel].copy()
    sub = sub.sort_values("contraste")

    st.subheader("📑 Valores por contraste")
    st.dataframe(
        sub[["contraste", "logFC", "AveExpr", "padj"]],
        width=max(900, 120 * sub.shape[0])
    )

    st.subheader("📊 logFC por contraste")

    fig = go.Figure()
    fig.add_bar(
        x=sub["contraste"],
        y=sub["logFC"],
        marker_color=[
            "red" if (p <= 0.05 and abs(l) >= 1)
            else "gray"
            for p, l in zip(sub["padj"], sub["logFC"])
        ],
    )
    fig.update_layout(
    xaxis_title="Contraste",
    yaxis_title="log2 Fold Change",
    height=450,
    width=max(300, 20 * sub.shape[0]),  # 👈 ancho REAL
    margin=dict(b=220),
    xaxis=dict(
        tickangle=-45,
        tickmode="array",
        tickvals=sub["contraste"],
        ticktext=sub["contraste"],
    ),
    transition_duration=0,
    uirevision=True,
    )

    st.plotly_chart(fig, use_container_width=False)
# ============================================================
# HEATMAP GLOBAL (AGRUPACIÓN SELECCIONABLE)
# ============================================================
with tabs[2]:
    st.header("🔥 Heatmap global de expresión diferencial")

    tipo = st.radio(
        "Modo de agrupación de genes",
        [
            "Sin agrupar",
            "Coexpresión (clusters)",
            "Clustering jerárquico (dendrograma)",
        ],
    )

    mat_base = mat_logfc.copy()

    # --------------------------------------------------------
    # SIN AGRUPAR
    # --------------------------------------------------------
    if tipo == "Sin agrupar":
        modo = st.selectbox("Valores a mostrar", ["logFC", "Z-score por gen"])

        mat_plot = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base

        st.caption(
            "Genes mostrados sin agrupamiento. "
            "Cada fila corresponde a un gen y cada columna a un contraste."
        )

    # --------------------------------------------------------
    # COEXPRESIÓN (clusters planos)
    # --------------------------------------------------------
    elif tipo == "Coexpresión (clusters)":
        n_clusters = st.slider("Número de clusters", 4, 80, 20)

        mat_z = limpiar_para_correlacion(aplicar_zscore(mat_base))

        Z = linkage(pdist(mat_z.values, metric="correlation"), method="average")
        cl = fcluster(Z, t=n_clusters, criterion="maxclust")

        mat_z["cluster"] = cl
        mat_z = mat_z.sort_values("cluster")

        mat_plot = mat_z.drop(columns="cluster")

        st.caption(
            "Genes agrupados por **coexpresión** usando:\n"
            "- Z-score por gen\n"
            "- Distancia: 1 − correlación de Pearson\n"
            "- Clustering jerárquico (average)\n"
            "- División en K clusters"
        )

        st.markdown("### 📑 Genes por cluster")
        cluster_sel = st.selectbox(
            "Seleccionar cluster",
            sorted(mat_z["cluster"].unique()),
        )
        genes_cluster = mat_z.index[mat_z["cluster"] == cluster_sel]
        st.dataframe(
            pd.DataFrame({"gene": genes_cluster}),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CLUSTERING JERÁRQUICO (orden por dendrograma)
    # --------------------------------------------------------
    else:
        modo = st.selectbox("Valores a mostrar", ["logFC", "Z-score por gen"])
        metric = st.selectbox("Métrica de distancia", ["euclidean", "correlation"])
        linkage_method = st.selectbox(
            "Método de linkage", ["average", "complete", "ward"]
        )

        mat_z = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base
        mat_z = limpiar_para_correlacion(mat_z)

        Z = linkage(pdist(mat_z.values, metric=metric), method=linkage_method)
        order = leaves_list(Z)

        mat_plot = mat_z.iloc[order]

        st.caption(
            "Clustering jerárquico completo:\n"
            f"- Valores: {modo}\n"
            f"- Distancia: {metric}\n"
            f"- Linkage: {linkage_method}\n"
            "El orden de los genes sigue el dendrograma."
        )

    # --------------------------------------------------------
    # HEATMAP (común)
    # --------------------------------------------------------
    h = min(3500, max(1400, int(25 * mat_plot.shape[0])))

    fig = go.Figure(
        go.Heatmap(
            z=mat_plot.values,
            x=mat_plot.columns,
            y=mat_plot.index,
            colorscale="RdBu_r",
            zmid=0,   # 👈 ESTO HACE QUE 0 SEA BLANCO
        )
    )

    fig.update_layout(height=h, margin=dict(l=320, r=20))
    plot_heatmap_scroll(fig, h)


# ============================================================
# HEATMAP SaSP (AGRUPACIÓN SELECCIONABLE)
# ============================================================
with tabs[3]:
    st.header("🧬 Heatmap SaSP de expresión diferencial")

    sasp = cargar_lista_sasp(db_path)
    mat_base = mat_logfc.loc[mat_logfc.index.isin(sasp)]

    if mat_base.shape[0] < 2:
        st.warning("No hay suficientes genes SaSP para mostrar el heatmap.")
        st.stop()

    tipo = st.radio(
        "Modo de agrupación de genes (SaSP)",
        [
            "Sin agrupar",
            "Coexpresión (clusters)",
            "Clustering jerárquico (dendrograma)",
        ],
    )

    # --------------------------------------------------------
    # SIN AGRUPAR
    # --------------------------------------------------------
    if tipo == "Sin agrupar":
        modo = st.selectbox(
            "Valores a mostrar",
            ["logFC", "Z-score por gen"],
            key="sasp_modo",
        )

        mat_plot = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base

        st.caption(
            "Heatmap de genes SaSP sin agrupamiento. "
            "Cada fila corresponde a un gen SaSP y cada columna a un contraste."
        )

    # --------------------------------------------------------
    # COEXPRESIÓN (clusters planos)
    # --------------------------------------------------------
    elif tipo == "Coexpresión (clusters)":
        n_clusters = st.slider(
            "Número de clusters",
            2,
            min(40, mat_base.shape[0]),
            min(10, mat_base.shape[0]),
            key="sasp_clusters",
        )

        mat_z = limpiar_para_correlacion(aplicar_zscore(mat_base))

        Z = linkage(pdist(mat_z.values, metric="correlation"), method="average")
        cl = fcluster(Z, t=n_clusters, criterion="maxclust")

        mat_z["cluster"] = cl
        mat_z = mat_z.sort_values("cluster")

        mat_plot = mat_z.drop(columns="cluster")

        st.caption(
            "Genes SaSP agrupados por **coexpresión** usando:\n"
            "- Z-score por gen\n"
            "- Distancia: 1 − correlación de Pearson\n"
            "- Clustering jerárquico (average)\n"
            "- División en K clusters"
        )

        st.markdown("### 📑 Genes SaSP por cluster")
        cluster_sel = st.selectbox(
            "Seleccionar cluster",
            sorted(mat_z["cluster"].unique()),
            key="sasp_cluster_sel",
        )
        genes_cluster = mat_z.index[mat_z["cluster"] == cluster_sel]
        st.dataframe(
            pd.DataFrame({"gene": genes_cluster}),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # CLUSTERING JERÁRQUICO (orden por dendrograma)
    # --------------------------------------------------------
    else:
        modo = st.selectbox(
            "Valores a mostrar",
            ["logFC", "Z-score por gen"],
            key="sasp_dendro_modo",
        )
        metric = st.selectbox(
            "Métrica de distancia",
            ["euclidean", "correlation"],
            key="sasp_metric",
        )
        linkage_method = st.selectbox(
            "Método de linkage",
            ["average", "complete", "ward"],
            key="sasp_linkage",
        )

        mat_z = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base
        mat_z = limpiar_para_correlacion(mat_z)

        Z = linkage(pdist(mat_z.values, metric=metric), method=linkage_method)
        order = leaves_list(Z)

        mat_plot = mat_z.iloc[order]

        st.caption(
            "Clustering jerárquico completo aplicado solo a genes SaSP:\n"
            f"- Valores: {modo}\n"
            f"- Distancia: {metric}\n"
            f"- Linkage: {linkage_method}\n"
            "El orden de los genes sigue el dendrograma."
        )

    # --------------------------------------------------------
    # HEATMAP (común)
    # --------------------------------------------------------
    h = min(1800, max(600, int(45 * mat_plot.shape[0])))

    fig = go.Figure(
        go.Heatmap(
            z=mat_plot.values,
            x=mat_plot.columns,
            y=mat_plot.index,
            colorscale="RdBu_r",
            zmid=0,   # 👈 ESTO HACE QUE 0 SEA BLANCO
        )
    )

    fig.update_layout(
        height=h,
        margin=dict(l=320, r=20, t=40, b=80),
    )

    plot_heatmap_scroll(fig, h)
