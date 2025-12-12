
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

---

# ▶️ Cómo ejecutar la app localmente

1. Instalar dependencias:

```bash
pip install -r requirements.txt


