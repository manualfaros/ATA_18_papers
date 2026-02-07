from __future__ import annotations
import io, re, sqlite3
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, fisher_exact

# ML imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Análisis Transcriptómico S. aureus + ML Suite",
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
        st.plotly_chart(fig, width="stretch")


def calcular_correlaciones_sasp(mat_logfc: pd.DataFrame, genes_sasp: List[str]) -> pd.DataFrame:
    """
    Calcula correlaciones de Pearson entre cada gen SaSP y todos los genes SAOUHSC
    """
    mat_clean = limpiar_para_correlacion(mat_logfc)
    
    sasp_en_matriz = [g for g in genes_sasp if g in mat_clean.index]
    saouhsc_genes = [g for g in mat_clean.index if g.startswith("SAOUHSC")]
    
    resultados = []
    
    for sasp in sasp_en_matriz:
        sasp_perfil = mat_clean.loc[sasp].values
        
        for saouhsc in saouhsc_genes:
            saouhsc_perfil = mat_clean.loc[saouhsc].values
            
            if len(sasp_perfil) > 2:
                corr, pval = pearsonr(sasp_perfil, saouhsc_perfil)
                
                resultados.append({
                    "SaSP": sasp,
                    "SAOUHSC": saouhsc,
                    "correlacion": corr,
                    "p_valor": pval,
                    "abs_correlacion": abs(corr)
                })
    
    return pd.DataFrame(resultados)


# ============================================================
# CARGA DATOS
# ============================================================
st.sidebar.title("⚙️ Fuente de datos")
db_path = st.sidebar.text_input("Ruta BD SQLite", "transcriptomica_analisis.db")

try:
    df_long = cargar_long_desde_sqlite(db_path)
    mat_logfc = construir_matriz(df_long)
    genes_sasp = cargar_lista_sasp(db_path)
    
    genes = mat_logfc.index.tolist()
    contrastes = mat_logfc.columns.tolist()
    
except Exception as e:
    st.error(f"Error cargando datos: {e}")
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
    "🔗 Coexpresión"
])

# ============================================================
# VOLCANO + DEGS
# ============================================================
with tabs[0]:
    csel = st.selectbox("Contraste", contrastes, key="volcano_contraste")
    sub = df_long[df_long["contraste"] == csel].copy()

    col1, col2 = st.columns(2)
    thr_logfc = col1.slider("|logFC| mínimo", 0.0, 5.0, 1.0, key="volcano_logfc")
    thr_padj = col2.slider("adj.P.Val máximo", 0.0, 1.0, 0.05, key="volcano_padj")

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

    st.plotly_chart(fig, width="stretch")

    st.markdown("### 📑 Genes diferencialmente expresados")
    degs = sub[(sub["padj"] <= thr_padj) & (sub["logFC"].abs() >= thr_logfc)]
    st.write(f"Genes significativos: **{degs.shape[0]}**")
    st.dataframe(degs[["gene", "logFC", "AveExpr", "padj"]], width="stretch")

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
        width="stretch"
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
        width=max(300, 20 * sub.shape[0]),
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

    st.plotly_chart(fig, width="content")

# ============================================================
# HEATMAP GLOBAL
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
        key="heatmap_global_tipo"
    )

    mat_base = mat_logfc.copy()

    if tipo == "Sin agrupar":
        modo = st.selectbox("Valores a mostrar", ["logFC", "Z-score por gen"], key="heatmap_global_modo")
        mat_plot = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base
        st.caption("Genes mostrados sin agrupamiento.")

    elif tipo == "Coexpresión (clusters)":
        n_clusters = st.slider("Número de clusters", 4, 80, 20, key="heatmap_global_n_clusters")
        mat_z = limpiar_para_correlacion(aplicar_zscore(mat_base))
        Z = linkage(pdist(mat_z.values, metric="correlation"), method="average")
        cl = fcluster(Z, t=n_clusters, criterion="maxclust")
        mat_z["cluster"] = cl
        mat_z = mat_z.sort_values("cluster")
        mat_plot = mat_z.drop(columns="cluster")
        st.caption("Genes agrupados por coexpresión.")

        st.markdown("### 📑 Genes por cluster")
        cluster_sel = st.selectbox("Seleccionar cluster", sorted(mat_z["cluster"].unique()), key="heatmap_global_cluster")
        genes_cluster = mat_z.index[mat_z["cluster"] == cluster_sel]
        st.dataframe(pd.DataFrame({"gene": genes_cluster}), width="stretch")

    else:
        modo = st.selectbox("Valores a mostrar", ["logFC", "Z-score por gen"], key="heatmap_global_dendro_modo")
        metric = st.selectbox("Métrica de distancia", ["euclidean", "correlation"], key="heatmap_global_metric")
        linkage_method = st.selectbox("Método de linkage", ["average", "complete", "ward"], key="heatmap_global_linkage")
        
        mat_z = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base
        mat_z = limpiar_para_correlacion(mat_z)
        Z = linkage(pdist(mat_z.values, metric=metric), method=linkage_method)
        order = leaves_list(Z)
        mat_plot = mat_z.iloc[order]
        st.caption(f"Clustering jerárquico: {modo}, distancia={metric}, linkage={linkage_method}")

    h = min(3500, max(1400, int(25 * mat_plot.shape[0])))
    fig = go.Figure(go.Heatmap(
        z=mat_plot.values, x=mat_plot.columns, y=mat_plot.index,
        colorscale="RdBu_r", zmid=0,
    ))
    fig.update_layout(height=h, margin=dict(l=320, r=20))
    plot_heatmap_scroll(fig, h)


# ============================================================
# HEATMAP SaSP
# ============================================================
with tabs[3]:
    st.header("🧬 Heatmap SaSP de expresión diferencial")

    sasp = cargar_lista_sasp(db_path)
    mat_base = mat_logfc.loc[mat_logfc.index.isin(sasp)]

    if mat_base.shape[0] < 2:
        st.warning("No hay suficientes genes SaSP para mostrar el heatmap.")
    else:
        tipo = st.radio(
            "Modo de agrupación de genes (SaSP)",
            ["Sin agrupar", "Coexpresión (clusters)", "Clustering jerárquico (dendrograma)"],
            key="heatmap_sasp_tipo"
        )

        if tipo == "Sin agrupar":
            modo = st.selectbox("Valores a mostrar", ["logFC", "Z-score por gen"], key="sasp_modo")
            mat_plot = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base
            st.caption("Heatmap de genes SaSP sin agrupamiento.")

        elif tipo == "Coexpresión (clusters)":
            n_clusters = st.slider("Número de clusters", 2, min(40, mat_base.shape[0]), 
                                   min(10, mat_base.shape[0]), key="sasp_clusters")
            mat_z = limpiar_para_correlacion(aplicar_zscore(mat_base))
            Z = linkage(pdist(mat_z.values, metric="correlation"), method="average")
            cl = fcluster(Z, t=n_clusters, criterion="maxclust")
            mat_z["cluster"] = cl
            mat_z = mat_z.sort_values("cluster")
            mat_plot = mat_z.drop(columns="cluster")
            st.caption("Genes SaSP agrupados por coexpresión.")

            st.markdown("### 📑 Genes SaSP por cluster")
            cluster_sel = st.selectbox("Seleccionar cluster", sorted(mat_z["cluster"].unique()), 
                                       key="sasp_cluster_sel")
            genes_cluster = mat_z.index[mat_z["cluster"] == cluster_sel]
            st.dataframe(pd.DataFrame({"gene": genes_cluster}), width="stretch")

        else:
            modo = st.selectbox("Valores a mostrar", ["logFC", "Z-score por gen"], key="sasp_dendro_modo")
            metric = st.selectbox("Métrica de distancia", ["euclidean", "correlation"], key="sasp_metric")
            linkage_method = st.selectbox("Método de linkage", ["average", "complete", "ward"], 
                                          key="sasp_linkage")
            
            mat_z = aplicar_zscore(mat_base) if modo.startswith("Z") else mat_base
            mat_z = limpiar_para_correlacion(mat_z)
            Z = linkage(pdist(mat_z.values, metric=metric), method=linkage_method)
            order = leaves_list(Z)
            mat_plot = mat_z.iloc[order]
            st.caption(f"Clustering jerárquico SaSP: {modo}, {metric}, {linkage_method}")

        h = min(1800, max(600, int(45 * mat_plot.shape[0])))
        fig = go.Figure(go.Heatmap(
            z=mat_plot.values, x=mat_plot.columns, y=mat_plot.index,
            colorscale="RdBu_r", zmid=0,
        ))
        fig.update_layout(height=h, margin=dict(l=320, r=20, t=40, b=80))
        plot_heatmap_scroll(fig, h)


# ============================================================
# ML SUITE
# ============================================================
with tabs[4]:
    st.header("🤖 Machine Learning Suite: Predicción Funcional Avanzada")
    
    st.markdown("""
    Esta suite utiliza los perfiles de expresión (logFC) de todos los contrastes 
    para entrenar un modelo que prediga el **functional_group** de genes no caracterizados.
    """)
    
    st.subheader("📁 1. Cargar anotaciones funcionales SAOUHSC")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo 'anotaciones_SAOUHSC_simple.csv'",
        type=['csv', 'tsv', 'txt'],
        key="ml_upload"
    )
    
    if uploaded_file is not None:
        try:
            # LECTURA ROBUSTA: Forzamos coma, saltamos errores de comillas y líneas malas
            annotations = pd.read_csv(
                uploaded_file, 
                sep=',', 
                engine='python', 
                on_bad_lines='skip', 
                quoting=3
            )
            
            # LIMPIEZA DE COLUMNAS: Quitamos ";", comillas y espacios de los nombres
            # Esto corrige el error de 'functional_group;;'
            annotations.columns = [c.strip(' ";') for c in annotations.columns]
            
            # Verificación de columnas necesarias
            if 'gene' in annotations.columns and 'functional_group' in annotations.columns:
                # Limpiar el contenido de las celdas
                annotations['gene'] = annotations['gene'].astype(str).str.strip(' ";')
                annotations['functional_group'] = annotations['functional_group'].astype(str).str.strip(' ";')
                
                st.success(f"✅ Cargadas {len(annotations)} anotaciones.")
                st.dataframe(annotations[['gene', 'functional_group']].head(5))
                
                # --- PREPARACIÓN DE DATOS PARA ML ---
                mat_clean = limpiar_para_correlacion(mat_logfc)
                
                # Unir matriz de expresión con anotaciones
                df_ml = mat_clean.reset_index().merge(
                    annotations[['gene', 'functional_group']], 
                    on='gene', 
                    how='inner'
                )
                
                # Filtrar genes con función conocida
                df_train = df_ml[~df_ml['functional_group'].isin(['Unknown/Hypothetical', 'Unknown', ''])]
                
                if len(df_train) < 10:
                    st.warning("Pocos genes con función conocida para entrenar (mínimo 10).")
                else:
                    st.write(f"Genes disponibles para entrenamiento: {len(df_train)}")
                    
                    # Botón para entrenar
                    if st.button("🚀 Entrenar Modelo de Predicción"):
                        # Aquí iría tu lógica de RandomForest o SVC
                        st.info("Entrenando modelo... por favor espera.")
                        # (Añade aquí tu lógica de entrenamiento si la tienes)
            else:
                st.error(f"❌ No encontré 'gene' o 'functional_group'. Columnas detectadas: {list(annotations.columns)}")
        
        except Exception as e:
            st.error(f"Error crítico al leer el archivo: {e}")
            
            # Verificar columnas
            required_cols = ['gene', 'functional_group']
            if not all(col in annotations.columns for col in required_cols):
                st.error("El archivo debe tener columnas: 'gene' y 'functional_group'")
                st.stop()
            
            # Preparar datos
            mat_clean = limpiar_para_correlacion(mat_logfc)
            
            # Merge anotaciones con matriz de expresión
            genes_con_anotacion = annotations[annotations['functional_group'] != 'Unknown/Hypothetical'].copy()
            genes_con_anotacion = genes_con_anotacion[genes_con_anotacion['gene'].isin(mat_clean.index)]
            
            if len(genes_con_anotacion) < 50:
                st.warning(f"Solo {len(genes_con_anotacion)} genes tienen anotación funcional conocida. Se necesitan al menos 50 para entrenar.")
                st.stop()
            
            st.write(f"**Genes SAOUHSC con función conocida:** {len(genes_con_anotacion)}")
            
            # Distribución de clases
            class_dist = genes_con_anotacion['functional_group'].value_counts()
            st.write("**Distribución de funciones:**")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_dist = px.bar(
                    x=class_dist.values,
                    y=class_dist.index,
                    orientation='h',
                    labels={'x': 'Número de genes', 'y': 'Función'},
                    title="Genes por categoría funcional"
                )
                st.plotly_chart(fig_dist, width="stretch")
            with col2:
                st.dataframe(class_dist.to_frame('count'), width="stretch")
            
            # Preparar features y labels
            X_train_genes = genes_con_anotacion['gene'].values
            X_train = mat_clean.loc[X_train_genes].values
            y_train = genes_con_anotacion['functional_group'].values
            
            # Genes SaSP para predicción
            sasp_en_matriz = [g for g in genes_sasp if g in mat_clean.index]
            X_sasp = mat_clean.loc[sasp_en_matriz].values
            
            st.write(f"**Genes SaSP para predicción:** {len(sasp_en_matriz)}")
            
            # ============================================================
            # MÉTODO SELECTOR
            # ============================================================
            st.subheader("🎯 2. Seleccionar método de ML")
            
            metodo = st.radio(
                "Método",
                [
                    "📊 Clasificación Supervisada (Random Forest)",
                    "🔬 Clustering + Enriquecimiento",
                    "🎯 Ensemble (Ambos métodos)"
                ],
                key="ml_method"
            )
            
            # ============================================================
            # MÉTODO 1: CLASIFICACIÓN SUPERVISADA
            # ============================================================
            if "Clasificación" in metodo or "Ensemble" in metodo:
                st.markdown("---")
                st.subheader("📊 Clasificación Supervisada con Random Forest")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("Número de árboles", 50, 500, 200, key="rf_trees")
                with col2:
                    max_depth = st.slider("Profundidad máxima", 5, 30, 15, key="rf_depth")
                with col3:
                    test_size = st.slider("% Test", 10, 30, 20, key="rf_test") / 100
                
                if st.button("🚀 Entrenar Clasificador", type="primary", key="train_classifier"):
                    with st.spinner("Entrenando Random Forest..."):
                        # Split train/test
                        X_tr, X_te, y_tr, y_te, genes_tr, genes_te = train_test_split(
                            X_train, y_train, X_train_genes, 
                            test_size=test_size, random_state=42, stratify=y_train
                        )
                        
                        # Entrenar
                        rf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            class_weight='balanced',
                            n_jobs=-1
                        )
                        rf.fit(X_tr, y_tr)
                        
                        # Validación cruzada
                        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_weighted')
                        
                        # Predicciones en test
                        y_pred = rf.predict(X_te)
                        
                        # Predicciones en SaSP
                        sasp_pred = rf.predict(X_sasp)
                        sasp_proba = rf.predict_proba(X_sasp)
                        sasp_conf = sasp_proba.max(axis=1)
                        
                        # Guardar en session state
                        st.session_state['rf_model'] = rf
                        st.session_state['rf_predictions'] = pd.DataFrame({
                            'gene': sasp_en_matriz,
                            'predicted_function': sasp_pred,
                            'confidence': sasp_conf
                        })
                        st.session_state['rf_trained'] = True
                        
                        # Métricas
                        st.success("✅ Modelo entrenado exitosamente!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("F1-Score (CV)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                        col2.metric("F1-Score (Test)", f"{f1_score(y_te, y_pred, average='weighted'):.3f}")
                        col3.metric("Accuracy (Test)", f"{(y_pred == y_te).mean():.3f}")
                        
                        # Matriz de confusión
                        st.subheader("📊 Matriz de Confusión (Test Set)")
                        cm = confusion_matrix(y_te, y_pred, labels=rf.classes_)
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=rf.classes_,
                            y=rf.classes_,
                            text=cm,
                            texttemplate="%{text}",
                            colorscale="Blues",
                            showscale=False
                        ))
                        fig_cm.update_layout(
                            xaxis_title="Predicción",
                            yaxis_title="Real",
                            height=500,
                            xaxis=dict(tickangle=-45)
                        )
                        st.plotly_chart(fig_cm, width="stretch")
                        
                        # Feature importance
                        st.subheader("🎯 Importancia de Contrastes")
                        feature_imp = pd.DataFrame({
                            "Contraste": mat_clean.columns,
                            "Importancia": rf.feature_importances_
                        }).sort_values("Importancia", ascending=False)
                        
                        fig_imp = go.Figure(go.Bar(
                            x=feature_imp["Importancia"][:20],
                            y=feature_imp["Contraste"][:20],
                            orientation="h",
                            marker_color='#667eea'
                        ))
                        fig_imp.update_layout(
                            xaxis_title="Importancia",
                            yaxis_title="Contraste",
                            height=600,
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig_imp, width="stretch")
                
                # Mostrar predicciones si ya está entrenado
                if st.session_state.get('rf_trained', False):
                    st.subheader("🔮 Predicciones en genes SaSP")
                    
                    pred_df = st.session_state['rf_predictions'].copy()
                    pred_df = pred_df.sort_values('confidence', ascending=False)
                    
                    # Filtro de confianza
                    min_conf = st.slider("Confianza mínima", 0.0, 1.0, 0.5, key="rf_min_conf")
                    pred_filt = pred_df[pred_df['confidence'] >= min_conf]
                    
                    st.write(f"**Predicciones con confianza ≥ {min_conf}:** {len(pred_filt)}")
                    
                    # Distribución de predicciones
                    pred_counts = pred_filt['predicted_function'].value_counts()
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig_pred = px.bar(
                            x=pred_counts.values,
                            y=pred_counts.index,
                            orientation='h',
                            title=f"Funciones predichas (conf ≥ {min_conf})",
                            labels={'x': 'Número de SaSP', 'y': 'Función'}
                        )
                        st.plotly_chart(fig_pred, width="stretch")
                    
                    with col2:
                        st.dataframe(pred_filt, width="stretch")
                    
                    # Descargar
                    csv = pred_filt.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar predicciones",
                        csv,
                        f"predicciones_RF_conf{min_conf}.csv",
                        "text/csv",
                        key="download_rf"
                    )
            
            # ============================================================
            # MÉTODO 2: CLUSTERING + ENRIQUECIMIENTO
            # ============================================================
            if "Clustering" in metodo or "Ensemble" in metodo:
                st.markdown("---")
                st.subheader("🔬 Clustering No Supervisado + Enriquecimiento Funcional")
                
                n_clusters_km = st.slider("Número de clusters (K-means)", 5, 30, 15, key="kmeans_k")
                
                if st.button("🧮 Ejecutar Clustering", type="primary", key="run_clustering"):
                    with st.spinner("Ejecutando K-means..."):
                        # Clustering en todos los genes con anotación + SaSP
                        todos_genes = list(set(X_train_genes) | set(sasp_en_matriz))
                        X_todos = mat_clean.loc[todos_genes].values
                        
                        # Normalizar
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_todos)
                        
                        # K-means
                        kmeans = KMeans(n_clusters=n_clusters_km, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Crear DataFrame de resultados
                        cluster_df = pd.DataFrame({
                            'gene': todos_genes,
                            'cluster': clusters
                        })
                        
                        # Merge con anotaciones
                        cluster_df = cluster_df.merge(
                            annotations[['gene', 'functional_group']], 
                            on='gene', 
                            how='left'
                        )
                        cluster_df['is_sasp'] = cluster_df['gene'].isin(sasp_en_matriz)
                        
                        # Enriquecimiento por cluster
                        enrichment_results = []
                        
                        for cluster_id in range(n_clusters_km):
                            cluster_genes = cluster_df[cluster_df['cluster'] == cluster_id]
                            sasp_in_cluster = cluster_genes[cluster_genes['is_sasp']]
                            
                            # Contar funciones en el cluster
                            func_counts = cluster_genes['functional_group'].value_counts()
                            
                            # Test de Fisher para cada función
                            for func in func_counts.index:
                                if pd.isna(func):
                                    continue
                                
                                # Tabla de contingencia
                                in_cluster_with_func = ((cluster_df['cluster'] == cluster_id) & 
                                                       (cluster_df['functional_group'] == func)).sum()
                                in_cluster_without_func = ((cluster_df['cluster'] == cluster_id) & 
                                                          (cluster_df['functional_group'] != func)).sum()
                                out_cluster_with_func = ((cluster_df['cluster'] != cluster_id) & 
                                                        (cluster_df['functional_group'] == func)).sum()
                                out_cluster_without_func = ((cluster_df['cluster'] != cluster_id) & 
                                                           (cluster_df['functional_group'] != func)).sum()
                                
                                # Fisher exact test
                                if in_cluster_with_func > 0:
                                    _, p_value = fisher_exact([
                                        [in_cluster_with_func, in_cluster_without_func],
                                        [out_cluster_with_func, out_cluster_without_func]
                                    ], alternative='greater')
                                    
                                    enrichment_results.append({
                                        'cluster': cluster_id,
                                        'function': func,
                                        'n_genes': in_cluster_with_func,
                                        'n_sasp': len(sasp_in_cluster),
                                        'p_value': p_value,
                                        'significant': p_value < 0.01
                                    })
                        
                        enrich_df = pd.DataFrame(enrichment_results)
                        enrich_df = enrich_df.sort_values('p_value')
                        
                        # Asignar función a SaSP por cluster
                        sasp_cluster_pred = []
                        for sasp_gene in sasp_en_matriz:
                            cluster_id = cluster_df[cluster_df['gene'] == sasp_gene]['cluster'].values[0]
                            
                            # Función más enriquecida en ese cluster
                            cluster_enrichment = enrich_df[
                                (enrich_df['cluster'] == cluster_id) & 
                                (enrich_df['significant'])
                            ]
                            
                            if len(cluster_enrichment) > 0:
                                pred_func = cluster_enrichment.iloc[0]['function']
                                conf = 1 - cluster_enrichment.iloc[0]['p_value']
                            else:
                                pred_func = "Unknown"
                                conf = 0.0
                            
                            sasp_cluster_pred.append({
                                'gene': sasp_gene,
                                'cluster': cluster_id,
                                'predicted_function': pred_func,
                                'confidence': conf
                            })
                        
                        sasp_cluster_df = pd.DataFrame(sasp_cluster_pred)
                        
                        # Guardar en session state
                        st.session_state['kmeans_model'] = kmeans
                        st.session_state['cluster_df'] = cluster_df
                        st.session_state['enrichment_df'] = enrich_df
                        st.session_state['kmeans_predictions'] = sasp_cluster_df
                        st.session_state['kmeans_trained'] = True
                        
                        st.success("✅ Clustering completado!")
                        
                        # Mostrar resultados
                        st.subheader("📊 Enriquecimiento Funcional por Cluster")
                        
                        # Top enrichments
                        top_enrich = enrich_df[enrich_df['significant']].head(20)
                        
                        fig_enrich = px.bar(
                            top_enrich,
                            x='n_genes',
                            y=[f"Cluster {row['cluster']}: {row['function']}" 
                               for _, row in top_enrich.iterrows()],
                            orientation='h',
                            color=-np.log10(top_enrich['p_value']),
                            labels={'x': 'Genes en cluster', 'color': '-log10(p)'},
                            title="Top 20 enriquecimientos funcionales"
                        )
                        fig_enrich.update_layout(height=600, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_enrich, width="stretch")
                        
                        # PCA visualization
                        st.subheader("🗺️ Visualización PCA de Clusters")
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        pca_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'cluster': clusters,
                            'gene': todos_genes,
                            'is_sasp': [g in sasp_en_matriz for g in todos_genes]
                        })
                        
                        fig_pca = px.scatter(
                            pca_df,
                            x='PC1',
                            y='PC2',
                            color='cluster',
                            symbol='is_sasp',
                            hover_data=['gene'],
                            title=f"PCA de genes (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})",
                            labels={'is_sasp': 'Es SaSP'}
                        )
                        st.plotly_chart(fig_pca, width="stretch")
                
                # Mostrar predicciones si ya está entrenado
                if st.session_state.get('kmeans_trained', False):
                    st.subheader("🔮 Predicciones por Clustering")
                    
                    pred_cluster = st.session_state['kmeans_predictions'].copy()
                    pred_cluster = pred_cluster[pred_cluster['predicted_function'] != 'Unknown']
                    pred_cluster = pred_cluster.sort_values('confidence', ascending=False)
                    
                    st.write(f"**SaSP con predicción funcional:** {len(pred_cluster)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(pred_cluster, width="stretch")
                    
                    with col2:
                        pred_counts = pred_cluster['predicted_function'].value_counts()
                        fig = px.pie(
                            values=pred_counts.values,
                            names=pred_counts.index,
                            title="Distribución de funciones predichas"
                        )
                        st.plotly_chart(fig, width="stretch")
                    
                    # Descargar
                    csv = pred_cluster.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar predicciones (clustering)",
                        csv,
                        "predicciones_clustering.csv",
                        "text/csv",
                        key="download_cluster"
                    )
            
            # ============================================================
            # MÉTODO 3: ENSEMBLE
            # ============================================================
            if "Ensemble" in metodo:
                st.markdown("---")
                st.subheader("🎯 Predicción Ensemble (Consenso)")
                
                if (st.session_state.get('rf_trained', False) and 
                    st.session_state.get('kmeans_trained', False)):
                    
                    # Combinar predicciones
                    pred_rf = st.session_state['rf_predictions'].copy()
                    pred_km = st.session_state['kmeans_predictions'].copy()
                    
                    # Merge
                    ensemble = pred_rf.merge(
                        pred_km[['gene', 'predicted_function', 'confidence']], 
                        on='gene', 
                        suffixes=('_rf', '_km')
                    )
                    
                    # Consenso
                    ensemble['agree'] = ensemble['predicted_function_rf'] == ensemble['predicted_function_km']
                    ensemble['ensemble_confidence'] = (ensemble['confidence_rf'] + ensemble['confidence_km']) / 2
                    ensemble['final_prediction'] = ensemble.apply(
                        lambda row: row['predicted_function_rf'] if row['agree'] 
                        else ('Uncertain' if abs(row['confidence_rf'] - row['confidence_km']) < 0.1 
                              else row['predicted_function_rf'] if row['confidence_rf'] > row['confidence_km'] 
                              else row['predicted_function_km']),
                        axis=1
                    )
                    
                    st.write(f"**Consenso entre métodos:** {ensemble['agree'].sum()} / {len(ensemble)} ({ensemble['agree'].mean():.1%})")
                    
                    # Filtrar por confianza
                    min_conf_ens = st.slider("Confianza mínima (ensemble)", 0.0, 1.0, 0.6, key="ens_conf")
                    ensemble_filt = ensemble[ensemble['ensemble_confidence'] >= min_conf_ens]
                    
                    st.write(f"**Predicciones con alta confianza:** {len(ensemble_filt)}")
                    
                    # Mostrar resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Acuerdo total", f"{ensemble['agree'].sum()} genes")
                        st.metric("% Acuerdo", f"{ensemble['agree'].mean():.1%}")
                    
                    with col2:
                        high_conf = ensemble[ensemble['ensemble_confidence'] >= 0.7]
                        st.metric("Alta confianza (≥0.7)", f"{len(high_conf)} genes")
                        st.metric("% Alta conf", f"{len(high_conf)/len(ensemble):.1%}")
                    
                    with col3:
                        uncertain = ensemble[ensemble['final_prediction'] == 'Uncertain']
                        st.metric("Inciertos", f"{len(uncertain)} genes")
                        st.metric("% Inciertos", f"{len(uncertain)/len(ensemble):.1%}")
                    
                    # Tabla de resultados
                    st.subheader("📋 Tabla de Predicciones Ensemble")
                    display_cols = ['gene', 'final_prediction', 'ensemble_confidence', 
                                   'predicted_function_rf', 'confidence_rf',
                                   'predicted_function_km', 'confidence_km', 'agree']
                    st.dataframe(
                        ensemble_filt[display_cols].sort_values('ensemble_confidence', ascending=False),
                        width="stretch"
                    )
                    
                    # Gráfico de distribución
                    final_counts = ensemble_filt['final_prediction'].value_counts()
                    fig_final = px.bar(
                        x=final_counts.values,
                        y=final_counts.index,
                        orientation='h',
                        title=f"Funciones predichas (ensemble, conf ≥ {min_conf_ens})",
                        labels={'x': 'Número de SaSP', 'y': 'Función'}
                    )
                    st.plotly_chart(fig_final, width="stretch")
                    
                    # Descargar
                    csv = ensemble_filt.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar predicciones ensemble",
                        csv,
                        f"predicciones_ensemble_conf{min_conf_ens}.csv",
                        "text/csv",
                        key="download_ensemble"
                    )
                else:
                    st.info("⚠️ Debes entrenar ambos métodos (Clasificación y Clustering) para usar el Ensemble.")
        
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("""
        **💡 Para usar la ML Suite:**
        
        1. Sube el archivo de anotaciones SAOUHSC (CSV con columnas: gene, functional_group)
        2. Selecciona el método de ML (o usa los tres)
        3. Entrena el modelo
        4. Explora predicciones para tus genes SaSP
        5. Descarga resultados
        
        **Recomendación:** Usa el método Ensemble para máxima confianza en predicciones.
        """)


# ============================================================
# COEXPRESIÓN SaSP-SAOUHSC
# ============================================================
with tabs[5]:
    st.header("🔗 Análisis de Coexpresión SaSP - SAOUHSC")
    
    st.markdown("""
    Análisis complementario basado en correlaciones de Pearson.
    Útil para validar predicciones de ML y explorar genes específicos.
    """)
    
    if st.button("🧮 Calcular Correlaciones", type="primary", key="calc_corr"):
        with st.spinner("Calculando correlaciones de Pearson..."):
            df_corr = calcular_correlaciones_sasp(mat_logfc, genes_sasp)
            
            if df_corr.empty:
                st.warning("No se pudieron calcular correlaciones.")
            else:
                st.success(f"✅ Calculadas {len(df_corr):,} correlaciones!")
                st.session_state['df_correlaciones'] = df_corr
    
    # Si ya hay correlaciones calculadas
    if 'df_correlaciones' in st.session_state:
        df_corr = st.session_state['df_correlaciones']
        
        # Filtros
        st.subheader("⚙️ Filtros")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_corr = st.slider("Correlación mínima (|r|)", 0.0, 1.0, 0.7, key="coexp_min_corr")
        with col2:
            max_pval = st.slider("p-valor máximo", 0.0, 0.1, 0.05, key="coexp_max_pval")
        with col3:
            max_promiscuity = st.slider(
                "Max. SaSP correlacionados por SAOUHSC", 
                1, 50, 20,
                key="coexp_max_promiscuity",
                help="Filtra genes SAOUHSC que correlacionan con demasiados SaSP (posibles housekeeping)"
            )
        
        # Filtrar
        df_filt = df_corr[
            (df_corr["abs_correlacion"] >= min_corr) & 
            (df_corr["p_valor"] <= max_pval)
        ].copy()
        
        # Contar cuántos SaSP correlaciona cada SAOUHSC
        promiscuity = df_filt.groupby('SAOUHSC').size().reset_index(name='n_sasp_correlacionados')
        
        # Identificar genes "promiscuos" (housekeeping candidates)
        housekeeping_candidates = promiscuity[
            promiscuity['n_sasp_correlacionados'] > max_promiscuity
        ].sort_values('n_sasp_correlacionados', ascending=False)
        
        # Filtrar genes promiscuos
        df_filt = df_filt[
            ~df_filt['SAOUHSC'].isin(housekeeping_candidates['SAOUHSC'])
        ].sort_values("abs_correlacion", ascending=False)
        
        # Mostrar estadísticas
        col1, col2, col3 = st.columns(3)
        col1.metric("Pares totales", f"{len(df_corr):,}")
        col2.metric("Pares significativos", f"{len(df_filt)}")
        col3.metric("Genes housekeeping filtrados", f"{len(housekeeping_candidates)}")
        
        # Mostrar genes housekeeping detectados
        if len(housekeeping_candidates) > 0:
            with st.expander(f"🔍 Ver {len(housekeeping_candidates)} genes SAOUHSC removidos (posibles housekeeping)"):
                st.dataframe(housekeeping_candidates, width="stretch")
                st.caption(
                    "Estos genes correlacionan con muchos SaSP diferentes, lo que sugiere que son "
                    "housekeeping (metabolismo básico, transcripción, traducción) o reguladores globales."
                )
        
        # Mostrar top correlaciones
        st.subheader("🏆 Top Correlaciones SaSP - SAOUHSC (filtradas)")
        st.dataframe(
            df_filt[["SaSP", "SAOUHSC", "correlacion", "p_valor"]].head(100),
            width="stretch"
        )
        
        if len(df_filt) == 0:
            st.warning("No hay correlaciones significativas con los filtros actuales. Ajusta los umbrales.")
        else:
            # Análisis por SaSP
            st.subheader("📊 Genes SAOUHSC más correlacionados por SaSP")
            
            sasp_sel = st.selectbox(
                "Seleccionar gen SaSP",
                sorted(df_filt["SaSP"].unique()),
                key="coexp_sasp_select"
            )
            
            df_sasp = df_filt[df_filt["SaSP"] == sasp_sel].sort_values(
                "abs_correlacion", ascending=False
            ).head(20)
            
            if len(df_sasp) == 0:
                st.info(f"No hay genes SAOUHSC correlacionados con {sasp_sel} bajo estos filtros.")
            else:
                fig = go.Figure(go.Bar(
                    x=df_sasp["correlacion"],
                    y=df_sasp["SAOUHSC"],
                    orientation="h",
                    marker_color=np.where(df_sasp["correlacion"] > 0, "red", "blue")
                ))
                fig.update_layout(
                    xaxis_title="Correlación de Pearson",
                    yaxis_title="Gen SAOUHSC",
                    height=600,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, width="stretch")
                
                # Comparación de perfiles
                st.subheader("📈 Comparación de perfiles de expresión")
                
                saouhsc_sel = st.selectbox(
                    "Seleccionar gen SAOUHSC para comparar",
                    df_sasp["SAOUHSC"].tolist(),
                    key="coexp_saouhsc_comparar"
                )
                
                # Obtener perfiles
                perfil_sasp = mat_logfc.loc[sasp_sel]
                perfil_saouhsc = mat_logfc.loc[saouhsc_sel]
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=perfil_sasp.index,
                    y=perfil_sasp.values,
                    name=f"{sasp_sel} (SaSP)",
                    mode="lines+markers",
                    line=dict(color="red", width=2)
                ))
                fig_comp.add_trace(go.Scatter(
                    x=perfil_saouhsc.index,
                    y=perfil_saouhsc.values,
                    name=f"{saouhsc_sel} (SAOUHSC)",
                    mode="lines+markers",
                    line=dict(color="blue", width=2)
                ))
                fig_comp.update_layout(
                    xaxis_title="Contraste",
                    yaxis_title="logFC",
                    height=500,
                    xaxis=dict(tickangle=-45),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_comp, width="stretch")
                
                # Mostrar correlación específica
                corr_info = df_filt[
                    (df_filt["SaSP"] == sasp_sel) & 
                    (df_filt["SAOUHSC"] == saouhsc_sel)
                ]
                if not corr_info.empty:
                    col1, col2 = st.columns(2)
                    col1.metric("Correlación", f"{corr_info['correlacion'].values[0]:.4f}")
                    col2.metric("p-valor", f"{corr_info['p_valor'].values[0]:.2e}")
                
                # Análisis de red
                st.subheader("🕸️ Red de coexpresión")
                st.markdown("""
                Muestra qué genes SAOUHSC están conectados a múltiples SaSP (posibles hubs funcionales).
                """)
                
                # Contar conexiones por SAOUHSC
                network_stats = df_filt.groupby('SAOUHSC').agg({
                    'SaSP': 'count',
                    'correlacion': 'mean'
                }).rename(columns={
                    'SaSP': 'n_conexiones',
                    'correlacion': 'correlacion_media'
                }).reset_index()
                
                network_stats = network_stats.sort_values('n_conexiones', ascending=False).head(20)
                
                fig_network = go.Figure()
                fig_network.add_trace(go.Bar(
                    x=network_stats['n_conexiones'],
                    y=network_stats['SAOUHSC'],
                    orientation='h',
                    marker_color=network_stats['correlacion_media'],
                    marker_colorscale='RdBu_r',
                    marker_cmid=0,
                    text=network_stats['correlacion_media'].round(2),
                    textposition='auto',
                ))
                fig_network.update_layout(
                    xaxis_title="Número de SaSP conectados",
                    yaxis_title="Gen SAOUHSC",
                    height=600,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_network, width="stretch")
                
                st.caption(
                    "Genes SAOUHSC conectados a múltiples SaSP podrían ser reguladores de "
                    "vías compartidas o participar en procesos biológicos relacionados."
                )
                
                # Exportar
                csv = df_filt.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar todas las correlaciones (CSV)",
                    data=csv,
                    file_name=f"coexpresion_SaSP_SAOUHSC_minCorr{min_corr}.csv",
                    mime="text/csv",
                    key="download_corr"
                )


st.sidebar.markdown("---")
st.sidebar.info("""
**Análisis Transcriptómico S. aureus**

- Volcano plots y DEGs
- Exploración por gen
- Heatmaps con clustering
- **ML Suite completa**
- Análisis de coexpresión
""")
