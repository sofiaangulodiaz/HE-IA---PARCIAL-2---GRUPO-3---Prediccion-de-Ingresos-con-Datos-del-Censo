# 📊 Predicción de Ingresos con Datos del Censo

Proyecto de IA aplicada con **PyTorch** para predecir si una persona gana más de **$50K anuales** usando el dataset **Adult Census (UCI, 1994)**.

Incluye:
- Exploratory Data Analysis (EDA)
- Preprocesamiento de datos (imputación, escalado, one-hot)
- Regresión Logística como modelo baseline
- Redes Neuronales (MLP)
- Técnicas de regularización: **Dropout**, **BatchNorm**, **EarlyStopping**, **weight decay** y **gradient clipping**

Se comparan modelos y métricas en **entrenamiento**, **validación** y **prueba**, destacando la importancia del preprocesamiento y la regularización en el desempeño final.

## 👩‍💻 Proyecto desarrollado como parte del curso HE2 - Inteligencia Artificial Aplicada

## 👩‍💻 Autores
- Sofía Angulo
- Juan Felipe Benites

---

## 📂 Estructura del Repositorio
├── parcial2.ipynb                # Notebook principal (pipeline completo)    
├── Reporte                      # Informe (secciones 1–4) y anexos  
└── README.md                     # Este archivo

---

## 🧭 Resumen ejecutivo
- **Splits:** TRAIN **32,561**, VALIDACIÓN **8,140**, PRUEBA **8,141** (estratificados)
- **Clases:** ~**76%** (≤$50K) / **24%** (>$50K) estables en los 3 conjuntos
- **Features finales:** **100** columnas tras One-Hot Encoding (OHE)

### Resultados clave 
**Regresión Logística** (umbral optimizado en validación **t\*=0.53**):  
- TRAIN → Loss **0.5789**, Acc **0.8190**, Prec **0.5886**, Rec **0.8249**, F1 **0.6870**, AUC **0.9070**  
- VALIDACIÓN → Loss **0.5987**, Acc **0.8117**, Prec **0.5711**, Rec **0.8144**, F1 **0.6714**, AUC **0.8984**  
- PRUEBA → Loss **0.5654**, Acc **0.8216**, Prec **0.5876**, Rec **0.8216**, F1 **0.6852**, AUC **0.9106**

**MLP con regularización (mejor experimento)**  
Arquitectura **(128, 64)** + **BatchNorm** + **Dropout=0.2**, **AdamW lr=1e-3**, **weight_decay=1e-4**, **BCEWithLogitsLoss**, **EarlyStopping=6**, **batch=128**  
- VALIDACIÓN → **ValLoss≈0.576** (rango 0.576–0.582), **ValAcc pico≈0.809**  


**Conclusión rápida:** el baseline lineal ya logra **AUC≈0.91** y **F1≈0.685** en PRUEBA. El MLP **sin** regularización sobreajusta; **con** regularización se estabiliza y **se acerca** al baseline, sin superarlo con la evidencia actual.

---

## 🔬 Datos y preprocesamiento
- **Variables:** 14 columnas originales (numéricas y categóricas)
- **Imputación:** mediana (numéricas) y moda (categóricas), **fit solo en TRAIN**
- **Escalado:** **StandardScaler** en numéricas (**fit en TRAIN**)
- **Codificación:** **One-Hot** con `drop='first'` y `handle_unknown='ignore'`
- **Dimensión final:** **100** features
- **Loaders:** `batch=128`, `shuffle=True` solo en TRAIN
- **Evitar fuga de información:** todas las estadísticas (imputación, escalado, OHE) se ajustan **exclusivamente con TRAIN** y se aplican a VAL/TEST

**Figuras sugeridas (títulos exactos del notebook/PDF):**
- `income (TRAIN)`, `income (VAL)`, `income (TEST)`
- `Histogramas numéricas (TRAIN)`
- `Distribución de occupation (TOP 15, TRAIN)` (u otra categórica clave)
- `hours-per-week según income (TRAIN)`

---

## 🧠 Modelos

### Baseline — Regresión Logística
- Entrenada sobre 100 features (post-OHE)
- Selección de **umbral t\*** en VALIDACIÓN (max **F1**, t\*≈**0.53**)
- Métricas completas en TRAIN/VALIDACIÓN/PRUEBA (ver “Resultados clave”)

### MLP — sin vs. con regularización
- **Sin regularización:** sobreajuste (TrainAcc >0.90; ValAcc ~0.80–0.82; ValLoss creciente)
- **Con regularización (mejor experimento):**
  - Arquitectura **(128, 64)**, **BatchNorm**, **Dropout=0.2**
  - **AdamW** (lr=1e-3, wd=1e-4), **BCEWithLogitsLoss**
  - **Gradient clipping** y **EarlyStopping=6**
  - VALIDACIÓN: **ValLoss≈0.576**, **ValAcc pico≈0.809**

**Figuras de entrenamiento (títulos exactos):**  
`NoReg Run <ID> - Pérdida vs Épocas` · `NoReg Run <ID> - Accuracy vs Épocas` · `Run <ID> - Pérdida vs Épocas (con EarlyStopping)`



---

## ▶️ Cómo reproducir
1. **Instalar**  
   ```bash
   # Python >= 3.10
   pip install torch scikit-learn pandas numpy matplotlib


