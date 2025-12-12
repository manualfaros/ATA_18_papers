from __future__ import annotations
import io, re, sqlite3
from typing import Optional, List

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="EDA expresión diferencial (SQLite)",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------
# ALTair Theme (API nueva, sin warnings)
# ---------------------------------------------------------------------
def make_alt_theme():
    return {
        "config": {
            "axis": {"labelFontSize": 12, "titleFontSize": 13},
            "legend": {"labelFontSize": 12, "titleFontSize": 13},
            "view": {"strokeWidth": 0},
            "title": {"fontSize": 16},
        }
    }

# Para Altair >=5.5: alt.theme
try:
    @alt.theme.register("clean", enable=True)
    def _clean_theme():
        return alt.theme.ThemeConfig(make_alt_theme())
except Exception:
    # Fallback por si la versión de Altair es más antigua
    alt.themes.register("clean", lambda: make_alt_theme())
    alt.themes.enable("clean")

# ---------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------
def elegir_columna_gen(df: pd.DataFrame) -> Optional[str]:
    """Detecta la columna que contiene el identificador de gen."""
    for c in ["gene", "gene_id", "GeneID", "Geneid", "locus_tag"]:
        if c in df.columns:
            return c
    return None


def cargar_lista_sasp(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        cols = pd.read_sql_query("PRAGMA table_info(SaSP_list);", conn)["name"].tolist()
        if "gene" in cols:
            q = "SELECT gene AS gene FROM SaSP_list;"
        elif "gene_id" in cols:
            q = "SELECT gene_id AS gene FROM SaSP_list;"
        else:
            return []
        sasp = pd.read_sql_query(q, conn)["gene"].tolist()
        return sasp
    finally:
        conn.close()


@st.cache_data(show_spinner=True)
def cargar_long_desde_sqlite(db_path: str) -> pd.DataFrame:
    """
    Carga todas las tablas DEG_* y construye un dataframe largo con:
    gene, logFC, AveExpr, padj, neg_log10_padj, contraste
    """
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        st.error(f"No se pudo abrir la base de datos '{db_path}': {e}")
        return pd.DataFrame()

    # Obtener lista de genes de Genes_SA
    try:
        cols_genes = pd.read_sql_query("PRAGMA table_info(Genes_SA);", conn)["name"].tolist()
        if "gene" in cols_genes:
            q_genes = "SELECT DISTINCT gene AS gene FROM Genes_SA;"
        elif "gene_id" in cols_genes:
            q_genes = "SELECT DISTINCT gene_id AS gene FROM Genes_SA;"
        else:
            st.error("La tabla Genes_SA no contiene columna 'gene' ni 'gene_id'.")
            conn.close()
            return pd.DataFrame()

        genes_sa = pd.read_sql_query(q_genes, conn)
    except Exception as e:
        st.error(f"No se pudo leer Genes_SA: {e}")
        conn.close()
        return pd.DataFrame()

    all_genes = sorted(genes_sa["gene"].astype(str).tolist())

    # Tablas DEG_
    tablas = pd.read_sql_query(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name LIKE 'DEG_%';",
        conn,
    )["name"].tolist()

    if not tablas:
        st.error("No se encontraron tablas que empiecen por 'DEG_' en la base de datos.")
        conn.close()
        return pd.DataFrame()

    registros: List[pd.DataFrame] = []

    for t in tablas:
        try:
            df_deg = pd.read_sql_query(f"SELECT * FROM {t};", conn)
        except Exception as e:
            st.warning(f"No se pudo leer la tabla {t}: {e}")
            continue

        if df_deg.empty:
            continue

        gene_col = elegir_columna_gen(df_deg)
        if gene_col is None:
            st.warning(f"No se encontró columna de gen en la tabla {t}. Se ignora.")
            continue
        df_deg = df_deg.rename(columns={gene_col: "gene"})

        if "logFC" not in df_deg.columns:
            st.warning(f"No hay columna 'logFC' en la tabla {t}. Se ignora.")
            continue

        ave_col = "AveExpr" if "AveExpr" in df_deg.columns else None

        padj_col = None
        for cand in ["adjP", "adj_P_Val", "adj.P.Val", "padj"]:
            if cand in df_deg.columns:
                padj_col = cand
                break
        if padj_col is None:
            st.warning(f"No hay columna de p-valor ajustado en la tabla {t}. Se ignora.")
            continue

        cols_keep = ["gene", "logFC", padj_col]
        if ave_col:
            cols_keep.append(ave_col)
        df_deg = df_deg[cols_keep].copy()

        if ave_col:
            df_deg = df_deg.rename(columns={ave_col: "AveExpr"})
        df_deg = df_deg.rename(columns={padj_col: "padj"})

        # Expandir a todos los genes
        df_full = pd.DataFrame({"gene": all_genes})
        df_full = df_full.merge(df_deg, on="gene", how="left")

        # Limpieza numérica
        df_full["logFC"] = pd.to_numeric(df_full["logFC"], errors="coerce").fillna(0.0)

        if "AveExpr" in df_full.columns:
            df_full["AveExpr"] = pd.to_numeric(df_full["AveExpr"], errors="coerce").fillna(0.0)
        else:
            df_full["AveExpr"] = 0.0

        df_full["padj"] = pd.to_numeric(df_full["padj"], errors="coerce").fillna(1.0)
        df_full["neg_log10_padj"] = -np.log10(np.clip(df_full["padj"], 1e-300, 1.0))

        contraste_name = re.sub(r"^DEG_", "", t)
        df_full["contraste"] = contraste_name

        registros.append(df_full)

    conn.close()

    if not registros:
        st.error("No se pudieron construir registros de DE a partir de las tablas DEG_.")
        return pd.DataFrame()

    df_long = pd.concat(registros, ignore_index=True)
    df_long["gene"] = df_long["gene"].astype(str)
    df_long["contraste"] = df_long["contraste"].astype(str)

    return df_long


@st.cache_data(show_spinner=False)
def cargar_lista_sasp_cached(db_path: str) -> List[str]:
    """Versión cacheada de la lista SaSP."""
    return cargar_lista_sasp(db_path)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("⚙️ Fuente de datos (SQLite)")
db_path = st.sidebar.text_input("Ruta BD SQLite", "transcriptomica_analisis.db")

df_long = cargar_long_desde_sqlite(db_path)

if df_long.empty:
    st.warning("No hay datos de expresión diferencial cargados desde la BD.")
    st.stop()

contrastes_disponibles = sorted(df_long["contraste"].dropna().unique().tolist())
genes_disponibles = sorted(df_long["gene"].dropna().unique().tolist())

st.sidebar.info(f"{len(contrastes_disponibles)} contrastes detectados")
st.sidebar.info(f"{len(genes_disponibles)} genes detectados")

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Explorador de Gen")
gsel = st.sidebar.selectbox(
    "Seleccionar Gen",
    genes_disponibles,
    key="gen_explorer_sidebar"
)

# ---------------------------------------------------------------------
# Helper z-score
# ---------------------------------------------------------------------
def aplicar_zscore(mat: pd.DataFrame, modo: str) -> pd.DataFrame:
    """Aplica z-score por gene (filas) o por contraste (columnas)."""
    m = mat.copy()
    if modo == "gene":
        mean = m.mean(axis=1)
        std = m.std(axis=1).replace(0, 1)
        m = (m.sub(mean, axis=0)).div(std, axis=0)
    elif modo == "contraste":
        mean = m.mean(axis=0)
        std = m.std(axis=0).replace(0, 1)
        m = (m - mean) / std
    return m

# ---------------------------------------------------------------------
# Helper Heatmap (con dendrograma) + versión cacheada
# ---------------------------------------------------------------------
def _generar_heatmap_interno(
    mat_base: pd.DataFrame,
    metodo: str,
    metric: str,
    zscore_mode: str,
    title: str,
    height: int = 1000,
) -> Optional[go.Figure]:

    # 1. Aplicar Z-score
    if zscore_mode == "Z-score por gene":
        mat = aplicar_zscore(mat_base, "gene")
    elif zscore_mode == "Z-score por contraste":
        mat = aplicar_zscore(mat_base, "contraste")
    else:
        mat = mat_base.copy()

    # 2. Limpieza para evitar NaN/Inf y filas constantes
    mat = mat.replace([np.inf, -np.inf], np.nan)
    mat = mat.dropna(axis=0, how="any")  # quitar genes con cualquier NaN

    # eliminar genes con varianza 0 (filas constantes → correlación indefinida)
    var = mat.var(axis=1)
    mat = mat.loc[var > 0]

    # Si tras limpiar no queda suficiente
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        st.warning(
            f"No hay suficientes genes/contrastes válidos tras limpiar NaN/Inf y filas "
            f"constantes para generar el clustering en: {title}"
        )
        return None

    mat_values = mat.values

    # 3. Clustering (filas)
    try:
        D = pdist(mat_values, metric=metric)
        # Por si acaso, comprobamos que D no tenga NaN/Inf
        if not np.all(np.isfinite(D)):
            raise ValueError("La matriz de distancias contiene valores no finitos.")
        Z = linkage(D, method=metodo)
    except Exception as e:
        st.error(f"Error al calcular el clustering ({title}): {e}")
        return None

    # 4. Dendrograma (filas)
    dendro = ff.create_dendrogram(
        mat_values,
        orientation="right",
        labels=mat.index.tolist(),
        linkagefun=lambda x: Z,
        color_threshold=float("inf"),
    )

    # Orden de genes según dendrograma
    dendro_leaves = dendro["layout"]["yaxis"]["ticktext"]
    mat_ord = mat.reindex(dendro_leaves)
    gene_labels_ord = mat_ord.index.tolist()

    # 5. Subplots: dendrograma + heatmap
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.2, 0.8],
        shared_yaxes=True,
        horizontal_spacing=0.01,
    )

    # Dendrograma en la columna 1
    for data in dendro["data"]:
        fig.add_trace(data, row=1, col=1)

    # Heatmap columna 2
    heatmap_trace = go.Heatmap(
        z=mat_ord.values,
        x=mat_ord.columns.tolist(),
        y=gene_labels_ord,
        colorscale="RdBu_r",
        colorbar=dict(title="logFC"),
        zmin=mat_ord.values.min(),
        zmax=mat_ord.values.max(),
    )
    fig.add_trace(heatmap_trace, row=1, col=2)

    # Ajustes de ejes
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(
        tickvals=[i for i in range(len(gene_labels_ord))],
        ticktext=gene_labels_ord,
        row=1,
        col=1,
    )

    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1, ticks="")
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8), row=1, col=2)

    # Layout
    fig.update_layout(
        height=height,
        width=1200,
        title=title,
        margin=dict(l=100, r=20, t=50, b=150),
        showlegend=False,
    )

    return fig


@st.cache_data(show_spinner=True)
def generar_heatmap(
    mat_base: pd.DataFrame,
    metodo: str,
    metric: str,
    zscore_mode: str,
    title: str,
    height: int = 1000,
) -> Optional[go.Figure]:
    """
    Versión cacheada del generador de heatmaps con dendrograma.
    Evita recalcular clustering y dendrograma en cada interacción.
    """
    # Para que la cache funcione bien, convertimos el índice/columnas a tipos hashables simples
    mat_base = mat_base.copy()
    mat_base.index = mat_base.index.astype(str)
    mat_base.columns = mat_base.columns.astype(str)
    return _generar_heatmap_interno(mat_base, metodo, metric, zscore_mode, title, height)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tabs = st.tabs([
    "📊 Exploración",
    "🌋 Volcano / MA",
    "📑 DEGs",
    "🔍 Gen",
    "🔥 Heatmap global",
    "🧬 Heatmap SaSP",
])

# ---------------------------------------------------------------------
# TAB 1 — Exploración general
# ---------------------------------------------------------------------
with tabs[0]:
    st.header("📊 Exploración general")

    st.subheader("Muestra de la tabla larga (todas las tablas DEG_ combinadas)")
    st.dataframe(df_long.head(200), use_container_width=True)

    st.subheader("Resumen por contraste")
    resumen = (
        df_long
        .groupby("contraste")
        .agg(
            n_genes=("gene", "nunique"),
            n_valid_padj=("padj", lambda x: x.notna().sum()),
        )
        .reset_index()
    )
    st.dataframe(resumen, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2 — Volcano + MA
# ---------------------------------------------------------------------
with tabs[1]:
    st.header("🌋 Volcano + MA plots")

    csel = st.selectbox("Contraste", contrastes_disponibles, key="contraste_volcano")
    sub = df_long[df_long["contraste"] == csel].copy()

    col1, col2 = st.columns(2)
    with col1:
        thr_logfc = st.number_input(
            "|logFC| mínimo", 0.0, 10.0, 1.0, key="logfc_volcano"
        )
    with col2:
        thr_padj = st.number_input(
            "adj.P.Val máximo", 0.0, 1.0, 0.05, key="padj_volcano"
        )

    thr_neg_log10_padj = -np.log10(thr_padj)

    def cat(row):
        if np.isnan(row["logFC"]) or np.isnan(row["padj"]):
            return "No eval"
        if abs(row["logFC"]) >= thr_logfc and row["padj"] <= thr_padj:
            return "Up" if row["logFC"] > 0 else "Down"
        return "No sig"

    sub["cat"] = sub.apply(cat, axis=1)

    base_volcano = alt.Chart(sub).encode(
        x=alt.X("logFC", title="log2 Fold Change"),
        y=alt.Y("neg_log10_padj", title="-log10(adj.P.Val)"),
        color=alt.Color("cat:N", title="Categoría"),
        tooltip=["gene", "logFC", "padj"],
    ).properties(height=400, title=f"Volcano – {csel}")

    volcano = base_volcano.mark_circle(size=50, opacity=0.7)

    h_rule = alt.Chart(
        pd.DataFrame({"y": [thr_neg_log10_padj]})
    ).mark_rule(color="gray", strokeDash=[3, 3]).encode(y="y")

    v_rule_pos = alt.Chart(
        pd.DataFrame({"x": [thr_logfc]})
    ).mark_rule(color="gray", strokeDash=[3, 3]).encode(x="x")

    v_rule_neg = alt.Chart(
        pd.DataFrame({"x": [-thr_logfc]})
    ).mark_rule(color="gray", strokeDash=[3, 3]).encode(x="x")

    st.altair_chart(volcano + h_rule + v_rule_pos + v_rule_neg, use_container_width=True)

    ma = (
        alt.Chart(sub)
        .mark_circle(size=50, opacity=0.7)
        .encode(
            x=alt.X("AveExpr", title="AveExpr"),
            y=alt.Y("logFC", title="log2 Fold Change"),
            color=alt.Color("cat:N", title="Categoría"),
            tooltip=["gene", "AveExpr", "logFC", "padj"],
        )
        .properties(height=400, title=f"MA plot – {csel}")
    )
    st.altair_chart(ma, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3 — Tabla de DEGs
# ---------------------------------------------------------------------
with tabs[2]:
    st.header("📑 Tabla de genes diferencialmente expresados")

    csel = st.selectbox("Contraste", contrastes_disponibles, key="contraste_tabla")
    sub = df_long[df_long["contraste"] == csel].copy()

    thr_logfc = st.number_input(
        "|logFC| mínimo", 0.0, 10.0, 1.0, key="logfc_degtable"
    )
    thr_padj = st.number_input(
        "adj.P.Val máximo", 0.0, 1.0, 0.05, key="padj_degtable"
    )

    mask = (sub["logFC"].abs() >= thr_logfc) & (sub["padj"] <= thr_padj)
    deg_filt = (
        sub.loc[mask, ["gene", "logFC", "AveExpr", "padj"]]
        .sort_values("padj")
    )

    st.write(f"Genes significativos: **{deg_filt.shape[0]}**")
    st.dataframe(deg_filt, use_container_width=True)

    if not deg_filt.empty:
        out = io.StringIO()
        deg_filt.to_csv(out, index=False)
        st.download_button(
            label="⬇️ Descargar CSV",
            data=out.getvalue().encode("utf-8"),
            file_name=f"DEGs_{csel}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------
# TAB 4 — Explorador por gen
# ---------------------------------------------------------------------
with tabs[3]:
    st.header("🔍 Explorador por gen")

    if gsel:
        sub = df_long[df_long["gene"] == gsel].copy()
        sub = sub[["contraste", "logFC", "AveExpr", "padj"]].sort_values("contraste")

        st.subheader(f"Resumen de {gsel} por contraste")
        st.dataframe(sub, use_container_width=True)

        chart = (
            alt.Chart(sub)
            .mark_bar()
            .encode(
                x=alt.X("contraste:N", sort="-y", title="Contraste"),
                y=alt.Y("logFC:Q", title="logFC"),
                tooltip=["contraste", "logFC", "padj"],
                color=alt.Color("logFC:Q", legend=None),
            )
            .properties(height=400, title=f"logFC por contraste – {gsel}")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info(
            "Selecciona un gen desde la barra lateral (Sidebar) "
            "para explorar sus valores en todos los contrastes."
        )

# ---------------------------------------------------------------------
# TAB 5 — Heatmap global
# ---------------------------------------------------------------------
with tabs[4]:
    st.header("🔥 Heatmap global (todos los genes) con Dendrograma")

    col1, col2, col3 = st.columns(3)
    with col1:
        metodo_global = st.selectbox(
            "Método de linkage (global)",
            ["average", "complete", "single", "ward"],
            key="metodo_global",
        )
    with col2:
        metric_global = st.selectbox(
            "Métrica de distancia (global)",
            ["euclidean", "correlation"],
            key="metric_global",
        )
    with col3:
        zscore_global = st.selectbox(
            "Normalización (global)",
            ["Sin normalizar", "Z-score por gene", "Z-score por contraste"],
            key="zscore_global",
        )

    mat_base = df_long.pivot_table(
        index="gene",
        columns="contraste",
        values="logFC",
        aggfunc="mean",
    ).fillna(0.0)

    fig = generar_heatmap(
        mat_base,
        metodo_global,
        metric_global,
        zscore_global,
        "Heatmap Global de Expresión Diferencial (logFC)",
        height=1500,
    )

    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay suficientes datos válidos para generar el heatmap global.")

# ---------------------------------------------------------------------
# TAB 6 — Heatmap SaSP
# ---------------------------------------------------------------------
with tabs[5]:
    st.header("🧬 Heatmap SOLO genes SaSP con Dendrograma")

    col1, col2, col3 = st.columns(3)
    with col1:
        metodo_sasp = st.selectbox(
            "Método de linkage (SaSP)",
            ["average", "complete", "single", "ward"],
            key="metodo_sasp",
        )
    with col2:
        metric_sasp = st.selectbox(
            "Métrica de distancia (SaSP)",
            ["euclidean", "correlation"],
            key="metric_sasp",
        )
    with col3:
        zscore_sasp = st.selectbox(
            "Normalización (SaSP)",
            ["Sin normalizar", "Z-score por gene", "Z-score por contraste"],
            key="zscore_sasp",
        )

    sasp_list = cargar_lista_sasp_cached(db_path)
    sasp_set = set(sasp_list)

    mat_base = df_long.pivot_table(
        index="gene",
        columns="contraste",
        values="logFC",
        aggfunc="mean",
    ).fillna(0.0)

    mat_sasp = mat_base.loc[mat_base.index.isin(sasp_set)]

    st.write(f"SaSP en lista de BD: {len(sasp_list)}")
    st.write(f"SaSP encontrados en el DataFrame: {mat_sasp.shape[0]}")

    # Altura dinámica basada en el número de genes SaSP
    sasp_height = max(400, min(1000, 30 * mat_sasp.shape[0]))

    fig_sasp = generar_heatmap(
        mat_sasp,
        metodo_sasp,
        metric_sasp,
        zscore_sasp,
        "Heatmap SaSP de Expresión Diferencial (logFC)",
        height=sasp_height,
    )

    if fig_sasp:
        st.plotly_chart(fig_sasp, use_container_width=True)
    else:
        st.warning(
            "No hay suficientes genes SaSP o contrastes válidos para generar el heatmap. "
            "Verifica la tabla 'SaSP_list' en tu BD."
        )
