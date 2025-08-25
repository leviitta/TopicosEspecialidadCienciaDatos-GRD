Predicción de Grupos Relacionados por el Diagnóstico (GRD) mediante Machine Learning

Este proyecto tiene como objetivo desarrollar y evaluar modelos de machine learning para predecir la clasificación de un paciente en un Grupo Relacionado por el Diagnóstico (GRD). La predicción se basa en datos clínicos como la edad, el sexo, los diagnósticos y los procedimientos médicos asociados a su hospitalización.

El GRD es un sistema de clasificación de pacientes que agrupa casos clínicamente similares y que se espera que consuman una cantidad parecida de recursos hospitalarios. Una predicción precisa del GRD puede ayudar a los centros de salud en la gestión de recursos, la planificación y la optimización de la atención al paciente.

📜 Tabla de Contenidos
Metodología del Proyecto

  - Librerías y Tecnologías

  - Estructura del Código

  - Cómo Ejecutar el Código

  - Resultados

  - Posibles Mejoras

🛠️ Metodología del Proyecto
El flujo de trabajo de este proyecto se puede dividir en las siguientes etapas clave:

1. Carga y Limpieza de Datos:

    - Se carga el dataset dataset_elpino.csv desde Google Drive.
    - Se crea la variable objetivo Label extrayendo y limpiando la información de la columna original GRD.
    - Se eliminan las categorías (clases de Label) con una frecuencia muy baja (inferior al 0.4%) para reducir el ruido y el desbalance extremo.
    - Se descartan filas y columnas que superan un umbral del 75% de valores nulos para asegurar la calidad de los datos.

2. Análisis Exploratorio de Datos (EDA):

    - Se visualiza la distribución de la variable objetivo (Label) para entender el desbalance de clases.
    - Se analizan las distribuciones de características demográficas clave como la edad y el sexo de los pacientes.

3. Preprocesamiento e Ingeniería de Características:

    - Imputación de Nulos: Los valores faltantes en columnas numéricas se rellenan con la mediana, y en las categóricas, con la moda.
    - Extracción de Códigos: Para las columnas de diagnósticos y procedimientos, se extrae únicamente el código (ej. C189), descartando la descripción textual para simplificar el procesamiento.

    - Pipeline de Preprocesamiento: Se utiliza ColumnTransformer de Scikit-learn para aplicar transformaciones específicas a cada tipo de columna:

      - Variables Numéricas (Edad en años): Imputación con la mediana y escalado con StandardScaler.

      - Variables Categóricas (Sexo (Desc)): Imputación con la moda y codificación con OneHotEncoder.

      - Códigos de Diagnóstico y Procedimientos: Se utiliza TargetEncoder, una técnica de codificación avanzada ideal para variables categóricas de alta cardinalidad. Este codificador reemplaza cada código por la media de la variable objetivo, capturando así poder predictivo.

4. Modelado y Optimización:

    - Manejo del Desbalance de Clases: Se integra SMOTETomek en el pipeline. Esta técnica de remuestreo combina el sobremuestreo de las clases minoritarias (SMOTE) con la limpieza de enlaces entre clases (Tomek Links), generando un conjunto de datos de entrenamiento más balanceado.

    - Reducción de Dimensionalidad: Se aplica PCA (Análisis de Componentes Principales) para reducir la dimensionalidad del conjunto de datos tras la codificación, conservando un porcentaje de la varianza (95%, 97% o 99%).

    - Comparación de Modelos: Se entrenan y comparan dos potentes algoritmos de clasificación: RandomForestClassifier y LightGBM.

    - Búsqueda de Hiperparámetros: Se utiliza GridSearchCV con validación cruzada estratificada de 5 folds (StratifiedKFold) para encontrar la mejor combinación de hiperparámetros para cada modelo. La métrica de evaluación optimizada es el F1-score ponderado, adecuada para problemas multiclase con desbalance.

    - Evaluación Final: El mejor modelo encontrado durante la búsqueda se evalúa sobre un conjunto de prueba (test set) previamente separado para medir su rendimiento en datos no vistos.

💻 Librerías y Tecnologías
El proyecto se desarrolla en Python 3 y se ejecuta en un entorno de Google Colab. Las principales librerías utilizadas son:

    - Análisis y Manipulación de Datos: pandas, numpy

    - Visualización: matplotlib, seaborn

    - Machine Learning y Preprocesamiento: scikit-learn

    - Codificación Avanzada: category-encoders

    - Manejo de Desbalance: imbalanced-learn

    - Modelos de Clasificación: lightgbm

📂 Estructura del Código:

El script Topicos_de_especialidad_Ciencia_de_Datos_GRD.ipynb contiene todo el flujo de trabajo, incluyendo varias funciones personalizadas para facilitar la reproducibilidad:

    - construir_columna_label(): Procesa la columna GRD original para crear la variable objetivo.

    - eliminar_label_por_porcentaje(): Filtra las clases minoritarias del target.

    - eliminar_filas_nulas_x_porcentaje() y eliminar_columnas_nulas_x_porcentaje(): Funciones para la limpieza de datos nulos.

    - imputar_numericas_con_mediana() y imputar_categoricas_con_moda(): Rellenan valores faltantes.

    - Funciones de visualización (plot_*): Generan gráficos para el EDA.

🚀 Cómo Ejecutar el Código:

Este proyecto está diseñado para ejecutarse en Google Colab. Para reproducir los resultados, sigue estos pasos:

    - Clona o descarga el repositorio.

    - Sube el notebook a Google Colab.

    - Prepara el dataset:

      - Asegúrate de tener el archivo dataset_elpino.csv.

      - Súbelo a tu Google Drive.

    - Modifica la siguiente línea en el código para que apunte a la ruta correcta de tu archivo:

      - ruta_data_set_GRD = 'gdrive/My Drive/ruta/a/tu/dataset/dataset_elpino.csv'

    - Instala las dependencias: La primera celda del notebook se encarga de instalar las librerías necesarias:

      - !pip install category_encoders

Ejecuta todas las celdas: Haz clic en Entorno de ejecución > Ejecutar todo en Google Colab. El script te pedirá autorización para montar tu Google Drive.

📊 Resultados:

El script realiza una búsqueda exhaustiva de hiperparámetros para los modelos RandomForest y LightGBM. La siguiente tabla resume los resultados obtenidos durante la validación cruzada:

    - Modelo	Mejor F1-score Ponderado (CV)	Mejores Hiperparámetros
      - LightGBM	0.490532	{'model__learning_rate': 0.1, 'model__n_estimators': 200, 'model__num_leaves': 40, 'pca__n_components': 0.99}
      - Random Forest	0.490744	{'model__max_depth': None, 'model__min_samples_leaf': 1, 'model__n_estimators': 200, 'pca__n_components': 0.99}

Exportar a Hojas de cálculo
El mejor modelo, LightGBM, fue evaluado en el conjunto de prueba, obteniendo un rendimiento final de:

    - F1-score ponderado en el conjunto de prueba: 0.4714

Este resultado indica que el modelo final es no capaz de predecir correctamente el GRD de un paciente con una alta precisión y robustez.

💡 Posibles Mejoras:

    - Explorar otros modelos: Probar otros algoritmos como XGBoost, CatBoost o incluso redes neuronales.

    - Ingeniería de Características Avanzada: Crear nuevas características a partir de las existentes, como el número de diagnósticos secundarios o el tipo de procedimiento principal.

    - Optimización de TargetEncoder: Ajustar los parámetros de suavizado (smoothing) del TargetEncoder para evitar el sobreajuste.

    - Análisis de Errores: Investigar en qué clases de GRD el modelo comete más errores para entender sus debilidades y proponer mejoras específicas.
