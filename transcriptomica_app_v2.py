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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Análisis Transcriptómico S. aureus – ML Suite",
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
    # CONDICIONES (metadatos de contraste) + features agregadas
    # ============================================================
    CONDITION_GROUPS = {
        "Infection_like": ["Goldman", "costa", "Yousuf", "Ibberson", "Szafranska", "Bastakoti",
                           "human_infected", "pig_infected"],
        "Stress_like": ["Peyrusson", "Vlaemink", "Feng", "Bastock", "Im", "Chaves"],
        "Regulon": ["Rapun", "Das", "Kim", "Podkowik", "Sharkey", "Bezrukov"],
    }

    # normaliza a minúsculas para que 'costa' y 'Costa' no fallen
    # y permite coincidencias por *subcadena* (p.ej. "Goldman_vs_ctrl" -> Goldman)
    _PATTERNS: list[tuple[str,str]] = []
    for grp, lst in CONDITION_GROUPS.items():
        for c in lst:
            _PATTERNS.append((str(c).lower(), grp))
    # patrones más largos primero para reducir colisiones
    _PATTERNS.sort(key=lambda x: len(x[0]), reverse=True)

    def cond_de_contraste(nombre: str) -> str:
        n = str(nombre).lower()
        for pat, grp in _PATTERNS:
            if pat and pat in n:
                return grp
        return "Sin_asignar"

    # ── sidebar: selección de condiciones / modo de features ──
    st.sidebar.subheader("Condiciones")
    feature_mode = st.sidebar.radio(
        "Usar como features",
        ["Contrastes", "Agregado por condición"],
        index=0,
        key="feat_mode",
    )

    # condiciones realmente presentes en la BD (según nombres de contrastes)
    conds_presentes = sorted({cond_de_contraste(c) for c in contrastes_all})
    # asegura que las 3 principales aparezcan aunque falte alguna en la BD
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

    # ── contrastes disponibles en el UI (para Volcano y filtros generales) ──
    contrastes = [c for c in contrastes_all if cond_de_contraste(c) in sel_conditions]
    if not contrastes:
        contrastes = contrastes_all.copy()

    # ── matriz de features global para Tabs 2–5 ──
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
    "🤖 ML Suite",
    "🔗 Coexpresión",
])

# ============================================================
# TAB 0 – VOLCANO
# ============================================================
with tabs[0]:
    st.header("🌋 Volcano Plot + DEGs")
    
    with st.expander("📐 ¿Qué muestra este gráfico?", expanded=False):
        st.markdown(r"""
        **Cada punto = un gen.** Se evalúa si su expresión cambia significativamente entre dos condiciones.
        
        | Eje | Qué mide | Fórmula |
        |-----|----------|---------|
        | **X** | Magnitud del cambio | $\log_2(\text{Fold Change}) = \log_2\left(\frac{\text{expresión}_{\text{tratamiento}}}{\text{expresión}_{\text{control}}}\right)$ |
        | **Y** | Significancia estadística | $-\log_{10}(p_{\text{ajustado}})$ |
        
        **Interpretación:**
        - logFC > 0 → gen **sobreexpresado** en tratamiento
        - logFC < 0 → gen **subexpresado** en tratamiento
        - Mayor altura → más significativo (menor p-valor)
        
        Los umbrales (líneas discontinuas) definen qué consideramos "diferencial": típicamente |logFC| ≥ 1 y p-adj ≤ 0.05.
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
        **Cada gen es un vector numérico** con tantas dimensiones como contrastes:
        
        $$\vec{g} = (\text{logFC}_1, \text{logFC}_2, \ldots, \text{logFC}_n)$$
        
        Este vector es el **perfil de expresión** del gen: describe cómo responde a cada condición experimental.
        
        **Ejemplo:** si un gen tiene logFC = +3 en infección y logFC = -1 en estrés térmico, 
        su perfil indica que se activa durante infección pero se reprime con calor.
        
        Esta representación vectorial es la base de todos los análisis posteriores (clustering, ML, correlación).
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
        **Objetivo:** agrupar genes con perfiles de expresión similares.
        
        **1. Distancia entre genes**
        
        Cada gen es un vector $\vec{g}$. La similitud entre dos genes se mide con:
        
        | Métrica | Fórmula | Captura |
        |---------|---------|---------|
        | Euclidiana | $d = \sqrt{\sum_i (g_{1i} - g_{2i})^2}$ | Diferencia absoluta |
        | Correlación | $d = 1 - r_{\text{Pearson}}$ | Similitud de forma (ignora escala) |
        
        **2. Clustering jerárquico**
        
        Agrupa genes iterativamente: une los más cercanos, recalcula distancias, repite.
        El **dendrograma** muestra este árbol de fusiones.
        
        **3. Z-score (opcional)**
        
        Normaliza cada gen: $z_i = \frac{x_i - \bar{x}}{\sigma}$
        
        Esto centra todos los genes en media=0 y permite comparar **patrones** independientemente de la magnitud.
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
    
    with st.expander("📐 ¿En qué se diferencia de Heatmap global?", expanded=False):
        st.markdown(r"""
        **Misma matemática, diferente subconjunto.**
        
        Aquí filtramos solo los **~90 genes SaSP** (Small proteins de *S. aureus*).
        
        **¿Por qué es útil?**
        - Reduce ruido: elimina ~2400 genes no relevantes
        - Permite ver si los SaSP forman **módulos funcionales** (clusters coherentes)
        - Identifica SaSP con perfiles similares → posible función compartida
        
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
# TAB 4 – ML SUITE
# ============================================================
with tabs[4]:
    st.header("🤖 ML Suite – Predicción Funcional")

    with st.expander("📐 ¿Cómo funcionan los métodos de ML?", expanded=False):
        st.markdown(r"""
        **Objetivo:** predecir la función de genes SaSP desconocidos usando su perfil de expresión.
        
        ---
        
        **1️⃣ Random Forest (Supervisado)**
        
        Aprende de genes con función **conocida**:
        - Entrada: vector de expresión $\vec{g} = (\text{logFC}_1, \ldots, \text{logFC}_n)$
        - Salida: categoría funcional (ej. "Virulence", "Transport")
        
        Entrena múltiples árboles de decisión que "votan" la clase. La **confianza** es la proporción de árboles que coinciden.
        
        ---
        
        **2️⃣ K-means + Enriquecimiento (No supervisado)**
        
        No usa etiquetas funcionales para agrupar:
        
        1. **K-means:** minimiza $\sum_{i=1}^{k} \sum_{\vec{g} \in C_i} \|\vec{g} - \vec{\mu}_i\|^2$
           - Agrupa genes por cercanía a **centroides** $\vec{\mu}_i$
        
        2. **Test de Fisher:** para cada cluster, evalúa si una función está **sobrerrepresentada**:
           $$p = P(\text{ver } \geq k \text{ genes con función } F \text{ por azar})$$
        
        La **confianza** deriva del p-valor del enriquecimiento.
        
        ---
        
        **3️⃣ Ensemble**
        
        Combina ambos métodos. Si RF y K-means **coinciden** → predicción más fiable.
        """)
    
    st.markdown("""
    | Método | Tipo | Qué hace |
    |--------|------|----------|
    | **Random Forest** | Supervisado | Clasifica función basándose en perfil de expresión |
    | **K-means + Fisher** | No supervisado | Agrupa genes y busca funciones enriquecidas por cluster |
    | **Ensemble** | Combinado | Cruza ambas predicciones; las que coinciden son más fiables |

    > **Nota técnica:** las 17 categorías originales se agrupan en **9 mega-categorías**
    > para que cada clase tenga suficientes ejemplos de entrenamiento.
    """)

    # ── upload anotaciones ──
    st.subheader("📁 1. Anotaciones SAOUHSC")
    uploaded = st.file_uploader(
        "CSV con columnas: gene, functional_group  (usar anotaciones_SAOUHSC_simple.csv)",
        type=["csv","tsv","txt"], key="ml_upload",
    )

    if uploaded is None:
        st.info("Sube el archivo `anotaciones_SAOUHSC_simple.csv` para continuar.")
        st.stop()

    # ── parsear archivo (robusto) ──
    raw = uploaded.getvalue().decode("utf-8-sig")
    lines = [l.strip().rstrip(";").strip() for l in raw.splitlines() if l.strip()]
    annotations = pd.read_csv(io.StringIO("\n".join(lines)), on_bad_lines="skip")

    if "gene" not in annotations.columns or "functional_group" not in annotations.columns:
        st.error("Columnas requeridas: **gene** y **functional_group**")
        st.stop()

    st.success(f"✅ {len(annotations)} anotaciones cargadas")

    # ── preparar datos compartidos por los 3 métodos ──
    mat_clean = limpiar(mat_features)
    mat_z     = zscore_mat(mat_clean)          # Z-score: normaliza cada gen a media=0 sd=1

    # Aplicar mega-map
    annot = annotations.copy()
    annot["mega_group"] = annot["functional_group"].map(MEGA_MAP).fillna("Unknown/Hypothetical")

    # Solo genes conocidos presentes en la matriz
    train_df = annot[
        (annot["mega_group"] != "Unknown/Hypothetical") &
         annot["gene"].isin(mat_clean.index)
    ].drop_duplicates("gene").copy()

    # Fusionar clases con < 15 ejemplos en "Other"
    MIN_CLASE = 15
    cc = train_df["mega_group"].value_counts()
    small = cc[cc < MIN_CLASE].index.tolist()
    if small:
        train_df.loc[train_df["mega_group"].isin(small), "mega_group"] = "Other"
        st.info(f"ℹ️ Clases pequeñas fusionadas en *Other*: {small}")

    X_train_genes = train_df["gene"].values
    X_train       = mat_z.loc[X_train_genes].values
    y_train       = train_df["mega_group"].values

    sasp_en   = [g for g in genes_sasp if g in mat_z.index]
    X_sasp    = mat_z.loc[sasp_en].values

    # ── mostrar distribución de clases ──
    class_dist = pd.Series(y_train).value_counts()
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_d = px.bar(x=class_dist.values, y=class_dist.index, orientation="h",
                       labels={"x":"Genes","y":"Función"}, title="Clases de entrenamiento")
        st.plotly_chart(fig_d, width="stretch")
    with c2:
        st.dataframe(class_dist.to_frame("n_genes"), width="stretch")

    st.write(f"**Genes para entrenar:** {len(X_train)} · **Clases:** {class_dist.shape[0]} · **SaSP para predecir:** {len(sasp_en)}")

    # ── selector de método ──
    st.subheader("🎯 2. Método de ML")
    metodo = st.radio("Método", [
        "📊 Clasificación Supervisada (Random Forest)",
        "🔬 Clustering + Enriquecimiento (K-means)",
        "🎯 Ensemble (los dos métodos)",
    ], key="ml_method")

    # ============================================================
    # MÉTODO 1 – RANDOM FOREST
    # ============================================================
    if "Clasificación" in metodo or "Ensemble" in metodo:
        st.markdown("---")
        st.subheader("📊 Random Forest – Clasificación Supervisada")

        c1, c2, c3 = st.columns(3)
        n_est   = c1.slider("Árboles",    50,  500, 300, key="rf_est")
        max_dep = c2.slider("Profundidad", 5,   30,  20,  key="rf_dep")
        t_size  = c3.slider("% Test",     10,   30,  20,  key="rf_tst") / 100

        if st.button("🚀 Entrenar Random Forest", type="primary", key="btn_rf"):
            with st.spinner("Entrenando…"):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train, y_train, test_size=t_size, random_state=42, stratify=y_train
                )
                rf = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_dep,
                    class_weight="balanced", random_state=42, n_jobs=-1, min_samples_leaf=3
                )
                rf.fit(X_tr, y_tr)

                cv   = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1_weighted")
                y_pred = rf.predict(X_te)

                sasp_pred  = rf.predict(X_sasp)
                sasp_proba = rf.predict_proba(X_sasp)
                sasp_conf  = sasp_proba.max(axis=1)

                st.session_state["rf_model"] = rf
                st.session_state["rf_preds"] = pd.DataFrame({
                    "gene": sasp_en,
                    "predicted_function": sasp_pred,
                    "confidence": sasp_conf.round(3),
                })
                st.session_state["rf_done"] = True
                st.session_state["rf_cv"]   = cv
                st.session_state["rf_cm"]   = (confusion_matrix(y_te, y_pred, labels=rf.classes_), rf.classes_)
                st.session_state["rf_imp"]  = pd.DataFrame({
                    "Contraste":   mat_clean.columns,
                    "Importancia": rf.feature_importances_,
                }).sort_values("Importancia", ascending=False)
                st.session_state["rf_f1test"] = f1_score(y_te, y_pred, average="weighted")
                st.session_state["rf_acc"]    = (y_pred == y_te).mean()

            st.success("✅ Modelo entrenado")

        # ── resultados RF ──
        if st.session_state.get("rf_done"):
            cv = st.session_state["rf_cv"]
            c1, c2, c3 = st.columns(3)
            c1.metric("F1 (CV 5-fold)", f"{cv.mean():.3f} ± {cv.std():.3f}")
            c2.metric("F1 (Test)",      f"{st.session_state['rf_f1test']:.3f}")
            c3.metric("Accuracy (Test)",f"{st.session_state['rf_acc']:.3f}")

            # Matriz de confusión
            st.subheader("📊 Matriz de Confusión")
            cm, classes = st.session_state["rf_cm"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=classes, y=classes,
                text=cm, texttemplate="%{text}",
                colorscale="Blues", showscale=False,
            ))
            fig_cm.update_layout(xaxis_title="Predicción", yaxis_title="Real",
                                 height=500, xaxis=dict(tickangle=-45))
            st.plotly_chart(fig_cm, width="stretch")

            # Feature importance
            st.subheader("🎯 Importancia de contrastes (top 20)")
            imp = st.session_state["rf_imp"].head(20)
            fig_imp = go.Figure(go.Bar(
                x=imp["Importancia"], y=imp["Contraste"],
                orientation="h", marker_color="#667eea",
            ))
            fig_imp.update_layout(xaxis_title="Importancia", height=550,
                                  yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, width="stretch")

            # Predicciones SaSP
            st.subheader("🔮 Predicciones en SaSP")
            preds = st.session_state["rf_preds"].sort_values("confidence", ascending=False)
            min_c = st.slider("Confianza mínima", 0.0, 1.0, 0.15, key="rf_minc")
            pf = preds[preds["confidence"] >= min_c]

            c1, c2 = st.columns([2, 1])
            with c1:
                pc = pf["predicted_function"].value_counts()
                fig_pc = px.bar(x=pc.values, y=pc.index, orientation="h",
                                title=f"Funciones predichas (conf ≥ {min_c})",
                                labels={"x":"SaSP","y":"Función"})
                st.plotly_chart(fig_pc, width="stretch")
            with c2:
                st.dataframe(pf, width="stretch")

            st.download_button("📥 Descargar predicciones RF",
                               pf.to_csv(index=False),
                               f"pred_RF_conf{min_c}.csv", "text/csv", key="dl_rf")

    # ============================================================
    # MÉTODO 2 – KMEANS + ENRIQUECIMIENTO
    # ============================================================
    if "Clustering" in metodo or "Ensemble" in metodo:
        st.markdown("---")
        st.subheader("🔬 K-means + Enriquecimiento Funcional")

        n_km = st.slider("Número de clusters", 5, 30, 15, key="km_k")

        if st.button("🧮 Ejecutar Clustering", type="primary", key="btn_km"):
            with st.spinner("Ejecutando K-means…"):
                # todos los genes anotados (conocidos) + SaSP
                todos = list(set(X_train_genes) | set(sasp_en))
                X_todos = mat_z.loc[todos].values

                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(X_todos)

                km = KMeans(n_clusters=n_km, random_state=42, n_init=10)
                clusters = km.fit_predict(X_scaled)

                cdf = pd.DataFrame({"gene": todos, "cluster": clusters})
                cdf = cdf.merge(annot[["gene","mega_group"]], on="gene", how="left")
                cdf["is_sasp"] = cdf["gene"].isin(sasp_en)

                # Fisher por cluster × función
                enrich = []
                for cid in range(n_km):
                    in_c = cdf["cluster"] == cid
                    for func in cdf.loc[in_c, "mega_group"].dropna().unique():
                        if func in ("Unknown/Hypothetical",):
                            continue
                        a = ( in_c &  (cdf["mega_group"]==func)).sum()
                        b = ( in_c & ~(cdf["mega_group"]==func)).sum()
                        c = (~in_c &  (cdf["mega_group"]==func)).sum()
                        d = (~in_c & ~(cdf["mega_group"]==func)).sum()
                        if a > 0:
                            _, p = fisher_exact([[a,b],[c,d]], alternative="greater")
                            enrich.append({"cluster":cid, "function":func,
                                           "n_genes":int(a), "p_value":p,
                                           "significant": p < 0.01})

                edf = pd.DataFrame(enrich).sort_values("p_value")

                # Asignar función a SaSP
                max_nl = float(np.clip(-np.log10(edf["p_value"].clip(lower=1e-300)),0,None).max()) or 1.0
                km_preds = []
                for sg in sasp_en:
                    cid = cdf[cdf["gene"]==sg]["cluster"].values[0]
                    sig = edf[(edf["cluster"]==cid) & edf["significant"]]
                    if len(sig):
                        pf   = sig.iloc[0]["function"]
                        conf = min(-np.log10(max(sig.iloc[0]["p_value"],1e-300))/max_nl, 1.0)
                    else:
                        pf, conf = "Unknown", 0.0
                    km_preds.append({"gene":sg,"cluster":int(cid),
                                     "predicted_function":pf,"confidence":round(conf,3)})

                st.session_state["km_preds"]  = pd.DataFrame(km_preds)
                st.session_state["km_enrich"] = edf
                st.session_state["km_cdf"]    = cdf
                st.session_state["km_scaled"] = X_scaled
                st.session_state["km_todos"]  = todos
                st.session_state["km_clusters"] = clusters
                st.session_state["km_done"]   = True

            st.success("✅ Clustering completado")

        # ── resultados clustering ──
        if st.session_state.get("km_done"):
            edf = st.session_state["km_enrich"]
            top = edf[edf["significant"]].head(20)

            st.subheader("📊 Enriquecimiento funcional significativo")
            if len(top):
                labels = [f"Cluster {r['cluster']}: {r['function']}" for _,r in top.iterrows()]
                fig_e = px.bar(
                    x=top["n_genes"].values, y=labels,
                    orientation="h",
                    color=(-np.log10(top["p_value"])).values,
                    labels={"x":"Genes","color":"-log10(p)"},
                    title="Top enriquecamientos (p < 0.01)",
                )
                fig_e.update_layout(height=600, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_e, width="stretch")
            else:
                st.warning("Ningún enriquecimiento significativo. Prueba otro valor de K.")

            # PCA
            st.subheader("🗺️ PCA")
            X_scaled  = st.session_state["km_scaled"]
            todos     = st.session_state["km_todos"]
            clusters  = st.session_state["km_clusters"]
            pca2 = PCA(n_components=2)
            Xpca = pca2.fit_transform(X_scaled)
            pca_df = pd.DataFrame({
                "PC1": Xpca[:,0], "PC2": Xpca[:,1],
                "cluster": clusters, "gene": todos,
                "Es SaSP": ["✓ SaSP" if g in sasp_en else "SAOUHSC" for g in todos],
            })
            fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="cluster",
                                 symbol="Es SaSP", hover_data=["gene"],
                                 title=f"PCA  PC1={pca2.explained_variance_ratio_[0]:.1%}  PC2={pca2.explained_variance_ratio_[1]:.1%}")
            st.plotly_chart(fig_pca, width="stretch")

            # Predicciones
            st.subheader("🔮 Predicciones por Clustering")
            kp = st.session_state["km_preds"].copy()
            kp_known = kp[kp["predicted_function"] != "Unknown"].sort_values("confidence", ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(kp_known, width="stretch")
            with c2:
                pc = kp_known["predicted_function"].value_counts()
                st.plotly_chart(px.pie(values=pc.values, names=pc.index,
                                       title="Funciones predichas"), width="stretch")

            st.download_button("📥 Descargar predicciones Clustering",
                               kp_known.to_csv(index=False),
                               "pred_clustering.csv", "text/csv", key="dl_km")

    # ============================================================
    # MÉTODO 3 – ENSEMBLE
    # ============================================================
    if "Ensemble" in metodo:
        st.markdown("---")
        st.subheader("🎯 Ensemble – Consenso")

        if not (st.session_state.get("rf_done") and st.session_state.get("km_done")):
            st.info("⚠️ Entrena primero el **Random Forest** y luego ejecuta el **Clustering** para ver el consenso.")
        else:
            rf_p = st.session_state["rf_preds"].copy()
            km_p = st.session_state["km_preds"].copy()

            ens = rf_p.merge(km_p[["gene","predicted_function","confidence"]],
                             on="gene", suffixes=("_rf","_km"))
            ens["agree"] = ens["predicted_function_rf"] == ens["predicted_function_km"]
            ens["ensemble_conf"] = ((ens["confidence_rf"] + ens["confidence_km"])/2).round(3)

            # Predicción final: si coinciden → esa; si no → la de mayor confianza, o Uncertain si muy cercanas
            def final(row):
                if row["agree"]:
                    return row["predicted_function_rf"]
                diff = abs(row["confidence_rf"] - row["confidence_km"])
                if diff < 0.05:
                    return "Uncertain"
                return row["predicted_function_rf"] if row["confidence_rf"] > row["confidence_km"] else row["predicted_function_km"]

            ens["final_prediction"] = ens.apply(final, axis=1)
            ens = ens.sort_values("ensemble_conf", ascending=False)

            # KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("Acuerdo RF ↔ Clustering", f"{ens['agree'].sum()} / {len(ens)}  ({ens['agree'].mean():.0%})")
            high = ens[ens["ensemble_conf"] >= 0.3]
            c2.metric("Alta confianza (≥ 0.3)", f"{len(high)} genes")
            unc  = ens[ens["final_prediction"] == "Uncertain"]
            c3.metric("Inciertos", f"{len(unc)} genes")

            # Filtro
            mc = st.slider("Confianza mínima (ensemble)", 0.0, 1.0, 0.1, key="ens_mc")
            ef = ens[ens["ensemble_conf"] >= mc].copy()

            # Tabla completa
            st.subheader("📋 Predicciones Ensemble")
            show_cols = ["gene","final_prediction","ensemble_conf","agree",
                         "predicted_function_rf","confidence_rf",
                         "predicted_function_km","confidence_km"]
            st.dataframe(ef[show_cols], width="stretch")

            # Distribución
            fc = ef["final_prediction"].value_counts()
            fig_fc = px.bar(x=fc.values, y=fc.index, orientation="h",
                            title=f"Funciones predichas – Ensemble (conf ≥ {mc})",
                            labels={"x":"SaSP","y":"Función"})
            st.plotly_chart(fig_fc, width="stretch")

            st.download_button("📥 Descargar predicciones Ensemble",
                               ef.to_csv(index=False),
                               f"pred_ensemble_conf{mc}.csv", "text/csv", key="dl_ens")

# ============================================================
# TAB 5 – COEXPRESIÓN
# ============================================================
with tabs[5]:
    st.header("🔗 Coexpresión SaSP – SAOUHSC")
    
    with st.expander("📐 ¿Qué mide la coexpresión?", expanded=False):
        st.markdown(r"""
        **Idea:** genes que se expresan juntos probablemente funcionan juntos.
        
        **Correlación de Pearson** entre dos genes $A$ y $B$:
        
        $$r = \frac{\sum_i (A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum_i (A_i - \bar{A})^2} \cdot \sqrt{\sum_i (B_i - \bar{B})^2}}$$
        
        | Valor de $r$ | Interpretación |
        |--------------|----------------|
        | $r \approx +1$ | Coexpresión: suben/bajan juntos |
        | $r \approx -1$ | Anticorrelación: uno sube cuando el otro baja |
        | $r \approx 0$ | Sin relación lineal |
        
        **Filtro de housekeeping:** genes que correlacionan con *muchos* SaSP probablemente son 
        genes de expresión constitutiva (ribosomales, metabólicos básicos), no co-regulados específicamente.
        
        **Aplicación:** si un SaSP desconocido correlaciona fuertemente con genes de virulencia conocidos, 
        sugiere que también participa en virulencia.
        """)

    if st.button("🧮 Calcular correlaciones", type="primary", key="btn_corr"):
        with st.spinner("Calculando…"):
            mat_c = limpiar(mat_features)
            sasp_ok  = [g for g in genes_sasp if g in mat_c.index]
            saouhsc  = [g for g in mat_c.index if g.startswith("SAOUHSC")]

            rows = []
            for s in sasp_ok:
                sp = mat_c.loc[s].values
                for h in saouhsc:
                    hp = mat_c.loc[h].values
                    r, p = pearsonr(sp, hp)
                    rows.append({"SaSP":s,"SAOUHSC":h,"correlacion":round(r,4),
                                 "p_valor":p,"abs_corr":abs(r)})
            st.session_state["corr_df"] = pd.DataFrame(rows)
        st.success(f"✅ {len(st.session_state['corr_df']):,} correlaciones calculadas")

    if "corr_df" in st.session_state:
        df_corr = st.session_state["corr_df"]

        c1, c2, c3 = st.columns(3)
        min_r = c1.slider("|r| mínimo", 0.0, 1.0, 0.7, key="cx_r")
        max_p = c2.slider("p-valor máx", 0.0, 0.1, 0.05, key="cx_p")
        max_prom = c3.slider("Max SaSP por SAOUHSC", 1, 50, 20, key="cx_prom",
                             help="Filtra housekeeping (genes que correlacionan con demasiados SaSP)")

        filt = df_corr[(df_corr["abs_corr"] >= min_r) & (df_corr["p_valor"] <= max_p)].copy()

        # Detectar housekeeping
        prom = filt.groupby("SAOUHSC").size().reset_index(name="n_sasp")
        hk   = prom[prom["n_sasp"] > max_prom]
        filt = filt[~filt["SAOUHSC"].isin(hk["SAOUHSC"])].sort_values("abs_corr", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Pares totales",      f"{len(df_corr):,}")
        c2.metric("Pares significativos",f"{len(filt)}")
        c3.metric("Housekeeping filtrados",f"{len(hk)}")

        if len(hk):
            with st.expander(f"🏠 {len(hk)} genes housekeeping removidos"):
                st.dataframe(hk.sort_values("n_sasp", ascending=False), width="stretch")

        if len(filt) == 0:
            st.warning("Sin correlaciones significativas con estos filtros.")
        else:
            st.subheader("🏆 Top correlaciones")
            st.dataframe(filt[["SaSP","SAOUHSC","correlacion","p_valor"]].head(50), width="stretch")

            # Por SaSP
            sasp_opts = sorted(filt["SaSP"].unique())
            sasp_sel  = st.selectbox("SaSP", sasp_opts, key="cx_sasp")
            sub = filt[filt["SaSP"]==sasp_sel].head(20)

            if len(sub):
                fig = go.Figure(go.Bar(
                    x=sub["correlacion"], y=sub["SAOUHSC"], orientation="h",
                    marker_color=np.where(sub["correlacion"]>0,"red","blue"),
                ))
                fig.update_layout(xaxis_title="r (Pearson)", height=550,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")

                # Comparar perfiles
                saouhsc_sel = st.selectbox("SAOUHSC para comparar", sub["SAOUHSC"].tolist(), key="cx_sao")
                fig2 = go.Figure()
                fig2.add_scatter(x=mat_features.columns, y=mat_features.loc[sasp_sel].values,
                                 name=sasp_sel, mode="lines+markers", line=dict(color="red"))
                fig2.add_scatter(x=mat_features.columns, y=mat_features.loc[saouhsc_sel].values,
                                 name=saouhsc_sel, mode="lines+markers", line=dict(color="blue"))
                fig2.update_layout(xaxis_title="Contraste", yaxis_title="logFC",
                                   height=450, xaxis=dict(tickangle=-45), hovermode="x unified")
                st.plotly_chart(fig2, width="stretch")

                row = sub[sub["SAOUHSC"]==saouhsc_sel]
                if len(row):
                    c1, c2 = st.columns(2)
                    c1.metric("r", f"{row['correlacion'].values[0]:.4f}")
                    c2.metric("p", f"{row['p_valor'].values[0]:.2e}")

            # Red
            st.subheader("🕸️ Hubs SAOUHSC (conectados a + SaSP)")
            hub = filt.groupby("SAOUHSC").agg(
                n_conexiones=("SaSP","count"),
                r_media=("correlacion","mean"),
            ).sort_values("n_conexiones", ascending=False).head(20).reset_index()

            fig_h = go.Figure(go.Bar(
                x=hub["n_conexiones"], y=hub["SAOUHSC"], orientation="h",
                marker_color=hub["r_media"], marker_colorscale="RdBu_r", marker_cmid=0,
                text=hub["r_media"].round(2), textposition="auto",
            ))
            fig_h.update_layout(xaxis_title="SaSP conectados", height=550,
                                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_h, width="stretch")

            st.download_button("📥 Descargar correlaciones",
                               filt.to_csv(index=False),
                               f"coexp_r{min_r}.csv","text/csv", key="dl_corr")

# ── sidebar info ──
st.sidebar.markdown("---")
st.sidebar.info("**S. aureus Transcriptómica**\nVolcano · Gen · Heatmaps · ML Suite · Coexpresión")
