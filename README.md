
---

# 🗃️ Estructura de la base de datos SQLite (`expresion.db`)

La base de datos contiene **75 tablas** de expresión diferencial y **una tabla maestra con la anotación GTF**.

## **1. `Sa_gtf` — Tabla de anotación génica**

Esta tabla contiene la información del GTF base de *S. aureus*.

Columnas principales:

| columna       | descripción |
|---------------|-------------|
| gene_id       | Identificador único del gen |
| locus_tag     | Tag del locus |
| gene_name     | Nombre del gen (si existe) |
| feature       | Tipo (gene, CDS, mRNA, etc.) |
| start         | Inicio |
| end           | Fin |
| strand        | Hebra +/− |
| product       | Descripción funcional |

👉 Esta tabla ofrece anotación estable y sirve como **tabla padre** para todas las tablas DEG.

---

## **2. Tablas de expresión diferencial (`DEG_*`)**

Cada archivo .tsv importado desde limma-voom se convierte en una tabla en SQLite.

Ejemplos:

- `DEG_AC_EX`
- `DEG_D16TCS_braR`
- `DEG_ST239_BF_48h_ST239_PL_48h`
- `DEG_wt_graR`
- `DEG_X25_PC1_X25_TSB1`
- (hasta 74 tablas…)

Todas las tablas siguen el **mismo formato:**

| columna     | descripción |
|-------------|-------------|
| gene        | Nombre del gen o locus_tag |
| logFC       | log2 Fold Change |
| AveExpr     | Expresión media |
| adjP        | p-value ajustado (FDR) |
| neg_log10_padj | Transformación para volcano plot |
| contrast_name | Nombre limpio del contraste |

---

# 🔗 Relaciones entre tablas

Aunque SQLite no requiere claves externas explícitas, la app asume estas relaciones:

- Cada fila de las tablas DEG representa el mismo gen descrito en `Sa_gtf`.
- El `gene` en cada tabla DEG coincide con `gene_id` o `locus_tag` del GTF.
- Esto permite a la app:
  - buscar anotación del gen
  - mostrar su descripción
  - comparar logFC del mismo gen en diferentes contrastes
  - construir matrices genes × contrastes

---

# 🎨 Funcionalidades de la app

✔ Carga automática de todas las tablas DEG de la base de datos  
✔ Explorador general de contrastes  
✔ Volcano plot interactivo (Altair)  
✔ MA-plot interactivo  
✔ Tabla filtrable de DEGs  
✔ Exploración por gen (todos los contrastes)  
✔ 🔥 Clustermap con dendrograma (Seaborn + SciPy)  
✔ 🔍 Vista especial para genes que empiezan por `SaSP*`
🗂️ Datos de entrada

La app trabaja con una base de datos SQLite que contiene:

Tablas DEG_* con resultados de RNA-seq

columnas mínimas: gene / gene_id, logFC, padj

Tabla Genes_SA con el universo de genes

Tabla SaSP_list con genes SaSP (opcional, pero recomendado)

Para la ML Suite, además se requiere:

Un archivo CSV con anotaciones funcionales (gene, functional_group)

🧭 Flujo general de análisis

Identificar genes diferencialmente expresados

Visualizar perfiles de expresión

Agrupar genes por similitud

Explorar relaciones de coexpresión

Inferir función mediante ML

Las pestañas están ordenadas siguiendo este flujo lógico.

📑 Pestañas de la app
🌋 1. Volcano + DEGs

Qué hace
Muestra genes diferencialmente expresados en un contraste concreto.

Matemáticamente
Cada gen se compara contra la hipótesis de no cambio:

eje X → log2 Fold Change

eje Y → −log10(p-valor)

👉 No compara genes entre sí.

Pregunta clave

¿Qué genes cambian más en este experimento?

🔍 2. Explorador por gen

Qué hace
Permite inspeccionar el perfil de un gen a través de todos los contrastes.

Matemáticamente
Es una visualización directa de un vector (logFC por contraste).
No hay inferencia ni clustering.

Pregunta clave

¿Cómo se comporta este gen en todos los experimentos?

🔥 3. Heatmap global

Qué hace
Visualiza patrones globales de expresión y agrupa genes por similitud.

Matemáticamente
Cada gen es un vector.
Se calculan distancias entre vectores para:

ordenar genes

o agruparlos en clusters

Pregunta clave

¿Qué genes tienen perfiles de expresión parecidos?

🧬 4. Heatmap SaSP

Qué hace
Aplica el mismo análisis del heatmap global, pero solo sobre genes SaSP.

Matemáticamente
La operación es la misma (distancias entre perfiles),
pero restringida a un subconjunto funcional.

Pregunta clave

¿Los SaSP forman módulos coherentes o subgrupos?

🔗 5. Coexpresión

Qué hace
Explora relaciones entre genes SaSP y genes SAOUHSC.

Matemáticamente
Calcula correlación de Pearson entre pares de genes:

comparación uno a uno

no clustering global

Permite identificar:

genes vecinos

hubs de coexpresión

posibles reguladores compartidos

Pregunta clave

¿Qué genes se regulan de forma coordinada?

🤖 6. ML Suite (última pestaña)

Qué hace
Predice funciones biológicas a partir de perfiles de expresión.

Incluye tres enfoques:

📊 Clasificación supervisada (Random Forest)

Aprende reglas que conectan perfiles → funciones

Produce predicciones con confianza

Muestra qué contrastes son más informativos

🔬 Clustering + enriquecimiento (K-means)

Agrupa genes por patrón promedio

Detecta funciones sobre-representadas en cada cluster

Asigna funciones a genes no caracterizados

🎯 Ensemble

Combina ambos métodos

Prioriza predicciones consistentes y robustas

Matemáticamente
Aquí no se comparan genes entre sí,
sino perfiles de expresión contra etiquetas funcionales.

Pregunta clave

¿Qué función biológica sugiere este patrón de expresión?

🧮 Resumen matemático rápido
Análisis	Operación principal	Tipo de comparación
Volcano	Contraste vs cero	Gen individual
Heatmap	Distancia	Global (muchos genes)
Coexpresión	Correlación	Par a par
K-means	Distancia a centroides	Módulos
Random Forest	Reglas predictivas	Perfil → función



Asegúrate de:

tener la base de datos SQLite accesible

usar Python ≥ 3.9

tener instaladas las dependencias habituales (streamlit, pandas, scikit-learn, plotly, scipy)

🎯 Objetivo final

Esta app no busca solo listas de genes, sino:

estructuras

módulos

relaciones

y predicciones funcionales

a partir de datos transcriptómicos complejos, de forma interpretable y guiada.
---

# ▶️ Cómo ejecutar la app localmente

1. Instalar dependencias:

```bash
pip install -r requirements.txt

2. 🚀 
streamlit run transcriptomica_ML_SUITE_COMPLETA_con_ayuda_matematica.py
