from __future__ import annotations
import io, re, sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, fisher_exact

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Análisis Transcriptómico S. aureus",
    page_icon="🧬",
    layout="wide",
)

# ============================================================
# MEGA-GRUPO: agrupa las 17 clases originales en 9 coherentes
# ============================================================
MEGA_MAP = {
    "Transport":              "Transport",
    "Translation":            "Translation",
    "Energy metabolism":      "Metabolism",
    "Central metabolism":     "Metabolism",
    "Amino acid metabolism":  "Metabolism",
    "Nucleotide metabolism":  "Metabolism",
    "Lipid metabolism":       "Metabolism",
    "Cofactor biosynthesis":  "Metabolism",
    "DNA metabolism":         "DNA / Replication",
    "Cellular processes":     "Cellular processes",
    "Protein fate":           "Cellular processes",
    "Cell envelope":          "Cell envelope",
    "Virulence":              "Virulence",
    "Regulatory":             "Regulatory",
    "Transcription":          "Regulatory",
    "Signal transduction":    "Regulatory",
    "Mobile elements":        "Mobile elements",
}

# ============================================================
# UTILIDADES
# ============================================================
def elegir_columna_gen(df: pd.DataFrame) -> Optional[str]:
    for c in ("gene", "gene_id", "GeneID", "Geneid", "locus_tag"):
        if c in df.columns:
            return c
    return None


@st.cache_data
def cargar_long(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'DEG_%';", conn
    )["name"].tolist()
    registros = []
    for t in tables:
        df = pd.read_sql_query(f"SELECT * FROM [{t}];", conn)
        if df.empty:
            continue
        gc = elegir_columna_gen(df)
        if gc is None or "logFC" not in df.columns:
            continue
        pc = next((c for c in ("adjP","adj_P_Val","adj.P.Val","padj") if c in df.columns), None)
        if pc is None:
            continue
        ave = "AveExpr" if "AveExpr" in df.columns else None
        keep = [gc, "logFC", pc] + ([ave] if ave else [])
        df = df[keep].rename(columns={gc:"gene", pc:"padj"})
        if ave:
            df = df.rename(columns={ave:"AveExpr"})
        df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce").fillna(0.0)
        df["padj"]  = pd.to_numeric(df["padj"],  errors="coerce").fillna(1.0)
        if "AveExpr" not in df.columns:
            df["AveExpr"] = 0.0
        else:
            df["AveExpr"] = pd.to_numeric(df["AveExpr"], errors="coerce").fillna(0.0)
        df["contraste"] = re.sub(r"^DEG_", "", t)
        registros.append(df)
    conn.close()
    return pd.concat(registros, ignore_index=True)


@st.cache_data
def construir_matriz(df_long: pd.DataFrame) -> pd.DataFrame:
    return (
        df_long.pivot_table(index="gene", columns="contraste", values="logFC", aggfunc="mean")
        .fillna(0.0)
    )


def limpiar(mat: pd.DataFrame) -> pd.DataFrame:
    m = mat.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return m.loc[m.var(axis=1) > 0]


def zscore_mat(mat: pd.DataFrame) -> pd.DataFrame:
    return mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)


def scroll_heatmap(fig, height=900):
    with st.container(height=height):
        st.plotly_chart(fig, width="stretch")


# ============================================================
# CARGA DATOS
# ============================================================
st.sidebar.title("⚙️ Fuente de datos")
db_path = st.sidebar.text_input("Ruta BD SQLite", "transcriptomica_analisis.db")

try:
    df_long   = cargar_long(db_path)
    mat_logfc = construir_matriz(df_long)

    conn = sqlite3.connect(db_path)
    genes_sasp = pd.read_sql_query("SELECT gene FROM SaSP_list;", conn)["gene"].astype(str).tolist()
    conn.close()

    contrastes_all = mat_logfc.columns.tolist()

    # ============================================================
    # CONDICIONES (metadatos de contraste)
    # ============================================================
    CONDITION_GROUPS = {
        "Infection_like": ["Goldman", "costa", "Yousuf", "Ibberson", "Szafranska", "Bastakoti", "Garcia"],
        "Stress_like": ["Peyrusson", "Vlaemink", "Feng", "Bastock", "Im", "Chaves"],
        "Regulon": ["Rapun", "Das", "Kim", "Podkowik", "Sharkey", "Bezrukov"],
    }

    _PATTERNS: list[tuple[str,str]] = []
    for grp, lst in CONDITION_GROUPS.items():
        for c in lst:
            _PATTERNS.append((str(c).lower(), grp))
    _PATTERNS.sort(key=lambda x: len(x[0]), reverse=True)

    def cond_de_contraste(nombre: str) -> str:
        n = str(nombre).lower()
        for pat, grp in _PATTERNS:
            if pat and pat in n:
                return grp
        return "Sin_asignar"

    # ── sidebar: selección de condiciones ──
    st.sidebar.subheader("Condiciones")
    feature_mode = st.sidebar.radio(
        "Usar como features",
        ["Contrastes", "Agregado por condición"],
        index=0,
        key="feat_mode",
    )

    conds_presentes = sorted({cond_de_contraste(c) for c in contrastes_all})
    for _c in ["Infection_like", "Stress_like", "Regulon", "Sin_asignar"]:
        if _c not in conds_presentes:
            conds_presentes.append(_c)

    default_conds = [c for c in ["Infection_like", "Stress_like", "Regulon"] if c in conds_presentes]
    sel_conditions = st.sidebar.multiselect(
        "Filtrar contrastes por condición",
        options=conds_presentes,
        default=default_conds if default_conds else conds_presentes,
        key="cond_sel",
    )
    if not sel_conditions:
        sel_conditions = conds_presentes

    label_cols = st.sidebar.checkbox("Etiquetar columnas con condición", value=True, key="cond_label_cols")

    contrastes = [c for c in contrastes_all if cond_de_contraste(c) in sel_conditions]
    if not contrastes:
        contrastes = contrastes_all.copy()

    mat_contrastes_sel = mat_logfc.loc[:, contrastes].copy()

    if feature_mode == "Agregado por condición":
        cols = {}
        for cond in sel_conditions:
            cs = [c for c in contrastes_all if (cond_de_contraste(c) == cond) and (c in mat_logfc.columns)]
            if cs:
                cols[cond] = mat_logfc[cs].mean(axis=1)
        if not cols:
            mat_features = mat_contrastes_sel
        else:
            mat_features = pd.DataFrame(cols)
    else:
        mat_features = mat_contrastes_sel

    if label_cols and feature_mode == "Contrastes":
        mat_features = mat_features.rename(columns=lambda c: f"{c} ({cond_de_contraste(c)})")

except Exception as exc:
    st.error(f"Error cargando datos: {exc}")
    st.stop()

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "🌋 Volcano + DEGs",
    "🔍 Gen",
    "🔥 Heatmap global",
    "🧬 Heatmap SaSP",
    "🎯 Análisis funcional",
])

# ============================================================
# TAB 0 – VOLCANO
# ============================================================
with tabs[0]:
    st.header("🌋 Volcano Plot + DEGs")
    
    with st.expander("📐 ¿Qué muestra este gráfico?", expanded=False):
        st.markdown(r"""
        **Objetivo:** Identificar genes diferencialmente expresados en un contraste específico.
        
        **Cada punto = un gen.** Se evalúa si su expresión cambia significativamente entre dos condiciones.
        
        | Eje | Qué mide | Fórmula |
        |-----|----------|---------|
        | **X** | Magnitud del cambio | $\log_2(\text{Fold Change}) = \log_2\left(\frac{\text{expresión}_{\text{tratamiento}}}{\text{expresión}_{\text{control}}}\right)$ |
        | **Y** | Significancia estadística | $-\log_{10}(p_{\text{ajustado}})$ |
        
        **Interpretación:**
        - logFC > 0 → gen **sobreexpresado** en tratamiento
        - logFC < 0 → gen **subexpresado** en tratamiento
        - Mayor altura → más significativo (menor p-valor)
        
        **Umbrales típicos:** |logFC| ≥ 1 (cambio ≥2x) y p-adj ≤ 0.05 (5% falsos positivos).
        """)
    
    csel = st.selectbox("Contraste", contrastes, key="volcano_contraste")
    sub = df_long[df_long["contraste"] == csel].copy()

    c1, c2 = st.columns(2)
    thr_lfc = c1.slider("|logFC| mínimo", 0.0, 5.0, 1.0, key="v_lfc")
    thr_pad = c2.slider("adj.P.Val máximo", 0.0, 1.0, 0.05, key="v_pad")

    sub["neglog"] = -np.log10(np.clip(sub["padj"], 1e-300, 1.0))
    sub["cat"] = "No sig"
    sub.loc[(sub["padj"] <= thr_pad) & (sub["logFC"] >=  thr_lfc), "cat"] = "Up"
    sub.loc[(sub["padj"] <= thr_pad) & (sub["logFC"] <= -thr_lfc), "cat"] = "Down"

    fig = go.Figure()
    fig.add_scatter(
        x=sub["logFC"], y=sub["neglog"], mode="markers",
        marker=dict(size=4, color=sub["cat"].map({"Up":"red","Down":"blue","No sig":"lightgray"})),
        text=sub["gene"],
    )
    fig.add_hline(y=-np.log10(thr_pad), line_dash="dash")
    fig.add_vline(x= thr_lfc, line_dash="dash")
    fig.add_vline(x=-thr_lfc, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

    degs = sub[(sub["padj"] <= thr_pad) & (sub["logFC"].abs() >= thr_lfc)]
    st.write(f"Genes significativos: **{degs.shape[0]}**")
    st.dataframe(degs[["gene","logFC","AveExpr","padj"]], width="stretch")

# ============================================================
# TAB 1 – GEN
# ============================================================
with tabs[1]:
    st.header("🔍 Explorador por gen")
    
    with st.expander("📐 ¿Qué representa un gen aquí?", expanded=False):
        st.markdown(r"""
        **Objetivo:** Visualizar cómo responde un gen específico a todas las condiciones experimentales.
        
        **Cada gen es un vector numérico** con tantas dimensiones como contrastes:
        
        $$\vec{g} = (\text{logFC}_1, \text{logFC}_2, \ldots, \text{logFC}_n)$$
        
        Este vector es el **perfil de expresión** del gen: describe cómo responde a cada condición.
        
        **Ejemplo:** si un gen tiene logFC = +3 en infección y logFC = -1 en estrés térmico, 
        su perfil indica que se activa durante infección pero se reprime con calor.
        
        **Utilidad:** Esta representación vectorial es la base de todos los análisis posteriores 
        (clustering, correlación, predicción funcional).
        """)
    
    gene_sel = st.selectbox("Selecciona un gen", sorted(df_long["gene"].unique()), key="gen_sel")
    sub = df_long[(df_long["gene"] == gene_sel) & (df_long["contraste"].isin(contrastes))].sort_values("contraste")

    st.dataframe(sub[["contraste","logFC","AveExpr","padj"]], width="stretch")

    fig = go.Figure()
    fig.add_bar(
        x=sub["contraste"], y=sub["logFC"],
        marker_color=["red" if (p<=0.05 and abs(l)>=1) else "gray" for p,l in zip(sub["padj"],sub["logFC"])],
    )
    fig.update_layout(xaxis_title="Contraste", yaxis_title="logFC", height=450,
                      xaxis=dict(tickangle=-45), margin=dict(b=220))
    st.plotly_chart(fig, width="content")

# ============================================================
# TAB 2 – HEATMAP GLOBAL
# ============================================================
with tabs[2]:
    st.header("🔥 Heatmap global")
    
    with st.expander("📐 ¿Cómo funciona el clustering?", expanded=False):
        st.markdown(r"""
        **Objetivo:** Visualizar patrones de expresión de todos los genes y agrupar los que se comportan de forma similar.
        
        **1. Representación matricial**
        
        Cada fila = un gen, cada columna = un contraste. El color indica logFC (rojo = up, azul = down).
        
        **2. Distancia entre genes**
        
        Cada gen es un vector $\vec{g}$. La similitud entre dos genes se mide con:
        
        | Métrica | Fórmula | Captura |
        |---------|---------|---------|
        | Euclidiana | $d = \sqrt{\sum_i (g_{1i} - g_{2i})^2}$ | Diferencia absoluta |
        | Correlación | $d = 1 - r_{\text{Pearson}}$ | Similitud de forma (ignora magnitud) |
        
        **3. Clustering jerárquico**
        
        Agrupa genes iterativamente: une los más cercanos, recalcula distancias, repite.
        El **dendrograma** muestra este árbol de fusiones.
        
        **4. Z-score (opcional)**
        
        Normaliza cada gen: $z_i = \frac{x_i - \bar{x}}{\sigma}$ → centra en media=0, permite comparar **patrones** 
        independientemente de la magnitud absoluta del cambio.
        """)
    
    tipo = st.radio("Modo", ["Sin agrupar","Clusters","Dendrograma"], key="hg_tipo")
    mat_base = mat_features.copy()

    if tipo == "Sin agrupar":
        modo = st.selectbox("Valores", ["logFC","Z-score"], key="hg_modo")
        mat_plot = zscore_mat(mat_base) if modo == "Z-score" else mat_base
    elif tipo == "Clusters":
        n_cl = st.slider("Clusters", 4, 80, 20, key="hg_ncl")
        mz = limpiar(zscore_mat(mat_base))
        Z  = linkage(pdist(mz.values, metric="correlation"), method="average")
        cl = fcluster(Z, t=n_cl, criterion="maxclust")
        mz["_cl"] = cl
        mz = mz.sort_values("_cl")
        mat_plot = mz.drop(columns="_cl")

        with st.expander("📑 Genes por cluster"):
            csel2 = st.selectbox("Cluster", sorted(mz["_cl"].unique()), key="hg_csel")
            st.dataframe(pd.DataFrame({"gene": mz.index[mz["_cl"]==csel2]}), width="stretch")
    else:
        modo2  = st.selectbox("Valores", ["logFC","Z-score"], key="hg_modo2")
        metric = st.selectbox("Métrica", ["euclidean","correlation"], key="hg_metric")
        link   = st.selectbox("Linkage", ["average","complete","ward"], key="hg_link")
        mz = limpiar(zscore_mat(mat_base) if modo2=="Z-score" else mat_base)
        Z  = linkage(pdist(mz.values, metric=metric), method=link)
        mat_plot = mz.iloc[leaves_list(Z)]

    h = min(3500, max(1400, 25*mat_plot.shape[0]))
    fig = go.Figure(go.Heatmap(z=mat_plot.values, x=mat_plot.columns, y=mat_plot.index,
                               colorscale="RdBu_r", zmid=0))
    fig.update_layout(height=h, margin=dict(l=320, r=20))
    scroll_heatmap(fig, h)

# ============================================================
# TAB 3 – HEATMAP SaSP
# ============================================================
with tabs[3]:
    st.header("🧬 Heatmap SaSP")
    
    with st.expander("📐 ¿En qué se diferencia del Heatmap global?", expanded=False):
        st.markdown(r"""
        **Objetivo:** Visualizar patrones de expresión solo de los ~90 genes SaSP (small proteins de *S. aureus*).
        
        **Misma matemática, diferente subconjunto.**
        
        **¿Por qué es útil separar los SaSP?**
        
        - **Reduce ruido:** elimina ~2400 genes no relevantes para el análisis
        - **Mejor resolución:** permite ver agrupaciones entre SaSP que se perderían en el heatmap global
        - **Identifica módulos:** si varios SaSP clusterizan juntos → posible función compartida o co-regulación
        - **Foco en lo desconocido:** la mayoría de SaSP no tienen función asignada, este análisis ayuda a generar hipótesis
        
        Los métodos (distancia, clustering, Z-score) son idénticos al heatmap global.
        """)
    
    mat_base = mat_features.loc[mat_features.index.isin(genes_sasp)]
    if mat_base.shape[0] < 2:
        st.warning("Insuficientes genes SaSP.")
    else:
        tipo = st.radio("Modo", ["Sin agrupar","Clusters","Dendrograma"], key="hs_tipo")
        if tipo == "Sin agrupar":
            modo = st.selectbox("Valores", ["logFC","Z-score"], key="hs_modo")
            mat_plot = zscore_mat(mat_base) if modo=="Z-score" else mat_base
        elif tipo == "Clusters":
            n_cl = st.slider("Clusters", 2, min(40, mat_base.shape[0]), min(10, mat_base.shape[0]), key="hs_ncl")
            mz = limpiar(zscore_mat(mat_base))
            Z  = linkage(pdist(mz.values, metric="correlation"), method="average")
            cl = fcluster(Z, t=n_cl, criterion="maxclust")
            mz["_cl"] = cl; mz = mz.sort_values("_cl")
            mat_plot = mz.drop(columns="_cl")
            with st.expander("📑 SaSP por cluster"):
                cs = st.selectbox("Cluster", sorted(mz["_cl"].unique()), key="hs_csel")
                st.dataframe(pd.DataFrame({"gene": mz.index[mz["_cl"]==cs]}), width="stretch")
        else:
            modo2  = st.selectbox("Valores", ["logFC","Z-score"], key="hs_modo2")
            metric = st.selectbox("Métrica", ["euclidean","correlation"], key="hs_metric")
            link   = st.selectbox("Linkage", ["average","complete","ward"], key="hs_link")
            mz = limpiar(zscore_mat(mat_base) if modo2=="Z-score" else mat_base)
            Z  = linkage(pdist(mz.values, metric=metric), method=link)
            mat_plot = mz.iloc[leaves_list(Z)]

        h = min(1800, max(600, 45*mat_plot.shape[0]))
        fig = go.Figure(go.Heatmap(z=mat_plot.values, x=mat_plot.columns, y=mat_plot.index,
                                   colorscale="RdBu_r", zmid=0))
        fig.update_layout(height=h, margin=dict(l=320, r=20, t=40, b=80))
        scroll_heatmap(fig, h)

# ============================================================
# TAB 4 – ANÁLISIS FUNCIONAL (Simplificado)
# ============================================================
with tabs[4]:
    st.header("🎯 Análisis funcional de SaSP")
    
    with st.expander("📐 ¿Qué hace esta pestaña?", expanded=False):
        st.markdown(r"""
        **Objetivo:** Predecir la función de cada SaSP basándose en los genes con los que más correlaciona.
        
        **Método:**
        1. Para cada SaSP, calcular correlación con todos los genes SAOUHSC
        2. Seleccionar los *k* vecinos más correlacionados
        3. Entre los vecinos con función conocida, evaluar si alguna función está sobrerrepresentada (Test de Fisher)
        4. Asignar función si el enriquecimiento es significativo; si no → "Indeterminado"
        
        **Los vecinos sin función conocida también se muestran:** si en el futuro se caracteriza alguno, 
        el SaSP probablemente tenga función relacionada.
        """)
    
    # ── Cargar anotaciones ──
    st.subheader("📁 1. Anotaciones funcionales")
    
    usar_default = st.checkbox("Usar anotaciones por defecto (SAOUHSC)", value=True, key="usar_default")
    
    if usar_default:
        # Cargar archivo por defecto
        try:
            annot = pd.read_csv("anotaciones_SAOUHSC.csv")
            st.success(f"✅ Anotaciones por defecto cargadas: {len(annot)} genes")
        except FileNotFoundError:
            st.error("No se encontró el archivo de anotaciones por defecto. Sube uno manualmente.")
            st.stop()
    else:
        uploaded = st.file_uploader(
            "CSV con columnas: gene, functional_group",
            type=["csv", "tsv", "txt"], 
            key="func_upload",
        )
        
        if uploaded is None:
            st.info("Sube un archivo de anotaciones.")
            st.stop()
        
        # Parsear
        raw = uploaded.getvalue().decode("utf-8-sig")
        lines = [l.strip().rstrip(";").strip() for l in raw.splitlines() if l.strip()]
        annot = pd.read_csv(io.StringIO("\n".join(lines)), on_bad_lines="skip")
        st.success(f"✅ Anotaciones cargadas: {len(annot)} genes")
    
    if "gene" not in annot.columns or "functional_group" not in annot.columns:
        st.error("Columnas requeridas: **gene** y **functional_group**")
        st.stop()
    
    # Aplicar mega-map
    annot["mega_group"] = annot["functional_group"].map(MEGA_MAP).fillna("Unknown/Hypothetical")
    
    # Crear diccionario gen -> función
    gene_to_func = dict(zip(annot["gene"], annot["mega_group"]))
    
    # Preparar datos
    mat_clean = limpiar(mat_features)
    
    # Genes con función conocida
    genes_anotados = annot[
        (annot["mega_group"] != "Unknown/Hypothetical") &
        (annot["gene"].isin(mat_clean.index))
    ].drop_duplicates("gene")
    genes_anotados_set = set(genes_anotados["gene"].tolist())
    
    # Todos los genes SAOUHSC
    todos_saouhsc = [g for g in mat_clean.index if g.startswith("SAOUHSC")]
    
    # SaSP presentes
    sasp_en = [g for g in genes_sasp if g in mat_clean.index]
    
    st.success(f"✅ {len(genes_anotados)} genes anotados · {len(todos_saouhsc)} SAOUHSC totales · {len(sasp_en)} SaSP")
    
    # ── Parámetros ──
    st.subheader("⚙️ 2. Parámetros")
    
    c1, c2, c3 = st.columns(3)
    k_vecinos = c1.slider("Número de vecinos (k)", 5, 50, 20, key="func_k")
    min_corr = c2.slider("Correlación mínima", 0.3, 0.9, 0.5, key="func_min_r")
    p_threshold = c3.slider("p-valor máximo (Fisher)", 0.01, 0.1, 0.05, key="func_p")
    
    # ── Ejecutar análisis ──
    if st.button("🚀 Ejecutar análisis", type="primary", key="btn_func"):
        with st.spinner("Calculando correlaciones y predicciones..."):
            
            resultados = []
            todos_vecinos = []
            
            for sasp in sasp_en:
                perfil_sasp = mat_clean.loc[sasp].values
                
                # Calcular correlación con TODOS los SAOUHSC
                corr_lista = []
                for gen in todos_saouhsc:
                    if gen == sasp:
                        continue
                    perfil_gen = mat_clean.loc[gen].values
                    r, p = pearsonr(perfil_sasp, perfil_gen)
                    if not np.isnan(r) and r >= min_corr:
                        funcion = gene_to_func.get(gen, "Sin anotar")
                        if funcion == "Unknown/Hypothetical":
                            funcion = "Sin anotar"
                        corr_lista.append({
                            "gene": gen,
                            "r": r,
                            "funcion": funcion
                        })
                
                if not corr_lista:
                    resultados.append({
                        "SaSP": sasp,
                        "funcion_predicha": "Indeterminado",
                        "p_valor": 1.0,
                        "confianza": 0.0,
                        "n_vecinos": 0,
                        "r_medio": 0.0
                    })
                    continue
                
                # Ordenar y tomar top-k
                df_corr = pd.DataFrame(corr_lista).sort_values("r", ascending=False).head(k_vecinos)
                
                # Guardar vecinos
                for _, row in df_corr.iterrows():
                    todos_vecinos.append({
                        "SaSP": sasp,
                        "vecino": row["gene"],
                        "correlacion": round(row["r"], 4),
                        "funcion": row["funcion"]
                    })
                
                # Test de Fisher (solo con anotados)
                df_anotados = df_corr[df_corr["funcion"] != "Sin anotar"]
                
                if len(df_anotados) < 3:
                    resultados.append({
                        "SaSP": sasp,
                        "funcion_predicha": "Indeterminado",
                        "p_valor": 1.0,
                        "confianza": 0.0,
                        "n_vecinos": len(df_corr),
                        "r_medio": round(df_corr["r"].mean(), 3)
                    })
                    continue
                
                funciones_vecinos = df_anotados["funcion"].value_counts()
                n_anotados = len(df_anotados)
                
                mejor_funcion = None
                mejor_p = 1.0
                
                for func, n_func in funciones_vecinos.items():
                    total_con_func = (genes_anotados["mega_group"] == func).sum()
                    total_sin_func = len(genes_anotados) - total_con_func
                    
                    tabla = [
                        [n_func, n_anotados - n_func],
                        [total_con_func - n_func, total_sin_func - (n_anotados - n_func)]
                    ]
                    
                    if all(all(x >= 0 for x in row) for row in tabla):
                        _, p_fisher = fisher_exact(tabla, alternative="greater")
                        
                        if p_fisher < mejor_p:
                            mejor_p = p_fisher
                            mejor_funcion = func
                
                # Asignar resultado
                if mejor_p <= p_threshold and mejor_funcion:
                    confianza = min(-np.log10(max(mejor_p, 1e-300)) / 10, 1.0)
                    resultados.append({
                        "SaSP": sasp,
                        "funcion_predicha": mejor_funcion,
                        "p_valor": mejor_p,
                        "confianza": round(confianza, 3),
                        "n_vecinos": len(df_corr),
                        "r_medio": round(df_corr["r"].mean(), 3)
                    })
                else:
                    resultados.append({
                        "SaSP": sasp,
                        "funcion_predicha": "Indeterminado",
                        "p_valor": mejor_p,
                        "confianza": 0.0,
                        "n_vecinos": len(df_corr),
                        "r_medio": round(df_corr["r"].mean(), 3)
                    })
            
            # Guardar
            st.session_state["func_results"] = pd.DataFrame(resultados)
            st.session_state["func_vecinos"] = pd.DataFrame(todos_vecinos)
            st.session_state["func_done"] = True
        
        st.success("✅ Análisis completado")
    
    # ── Mostrar resultados ──
    if st.session_state.get("func_done"):
        results = st.session_state["func_results"]
        vecinos_df = st.session_state["func_vecinos"]
        
        # Resumen
        predichos = results[results["funcion_predicha"] != "Indeterminado"]
        
        st.subheader("📊 Resumen")
        c1, c2, c3 = st.columns(3)
        c1.metric("SaSP con predicción", f"{len(predichos)} / {len(results)}")
        c2.metric("Indeterminados", f"{len(results) - len(predichos)}")
        c3.metric("Tasa de asignación", f"{len(predichos)/len(results):.0%}")
        
        # Distribución
        if len(predichos) > 0:
            func_counts = predichos["funcion_predicha"].value_counts()
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.bar(x=func_counts.values, y=func_counts.index, orientation="h",
                             labels={"x": "Número de SaSP", "y": "Función"}, title="Funciones predichas")
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.dataframe(func_counts.to_frame("n_SaSP"), width="stretch")
        
        # Tabla
        st.subheader("📋 Predicciones")
        results_sorted = results.sort_values("confianza", ascending=False)
        st.dataframe(results_sorted, width="stretch")
        
        # ── Explorador ──
        st.markdown("---")
        st.subheader("🔍 Explorar un SaSP")
        
        sasp_sel = st.selectbox("Seleccionar SaSP", results_sorted["SaSP"].tolist(), key="func_sasp_sel")
        
        info = results[results["SaSP"] == sasp_sel].iloc[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicción", info["funcion_predicha"])
        c2.metric("Confianza", f"{info['confianza']:.3f}")
        c3.metric("p-valor", f"{info['p_valor']:.2e}")
        
        # Vecinos (todos juntos)
        vec_sasp = vecinos_df[vecinos_df["SaSP"] == sasp_sel].sort_values("correlacion", ascending=False)
        
        if len(vec_sasp) > 0:
            st.markdown("#### Vecinos")
            
            # Gráfico coloreado por función
            fig_vec = px.bar(
                vec_sasp,
                x="correlacion", y="vecino", color="funcion",
                orientation="h",
                labels={"correlacion": "Correlación (r)", "vecino": "Gen", "funcion": "Función"},
            )
            fig_vec.update_layout(
                height=max(350, 22 * len(vec_sasp)),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_vec, width="stretch")
            
            # Tabla
            with st.expander("Ver tabla"):
                st.dataframe(vec_sasp, width="stretch")
            
            # Comparar perfiles
            st.markdown("#### Comparar perfiles")
            
            vecino_sel = st.selectbox(
                "Vecino",
                vec_sasp["vecino"].tolist(),
                format_func=lambda x: f"{x} ({vec_sasp[vec_sasp['vecino']==x]['funcion'].values[0]})",
                key="func_vecino_sel"
            )
            
            fig_perfil = go.Figure()
            fig_perfil.add_scatter(
                x=mat_clean.columns, y=mat_clean.loc[sasp_sel].values,
                name=sasp_sel, mode="lines+markers", line=dict(color="red", width=2)
            )
            fig_perfil.add_scatter(
                x=mat_clean.columns, y=mat_clean.loc[vecino_sel].values,
                name=vecino_sel, mode="lines+markers", line=dict(color="blue", width=2)
            )
            fig_perfil.update_layout(
                xaxis_title="Contraste", yaxis_title="logFC",
                height=400, xaxis=dict(tickangle=-45), hovermode="x unified"
            )
            st.plotly_chart(fig_perfil, width="stretch")
            
            r_sel = vec_sasp[vec_sasp["vecino"] == vecino_sel]["correlacion"].values[0]
            st.metric("Correlación", f"{r_sel:.4f}")
        
        # Descargas
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Predicciones", results_sorted.to_csv(index=False),
                               "predicciones_SaSP.csv", "text/csv", key="dl_pred")
        with c2:
            st.download_button("📥 Vecinos", vecinos_df.to_csv(index=False),
                               "vecinos_SaSP.csv", "text/csv", key="dl_vec")

# ── sidebar info ──
st.sidebar.markdown("---")
st.sidebar.info("""
**S. aureus Transcriptómica**

5 pestañas:
- 🌋 Volcano: DEGs por contraste
- 🔍 Gen: perfil de un gen
- 🔥 Heatmap global: todos los genes
- 🧬 Heatmap SaSP: solo SaSP
- 🎯 Análisis funcional: predicción + exploración
""")
