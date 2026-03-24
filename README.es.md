# Proyecto Naive Bayes

Este repositorio ya quedó preparado para arrancar con un flujo base de clasificación usando `GaussianNB` de `scikit-learn`.

La primera versión entrena un modelo sobre el dataset de cáncer de mama incluido en `scikit-learn`, evalúa el resultado y guarda el modelo entrenado dentro de `models/`.

## Estructura

- **`src/app.py`** → Ejecuta el pipeline completo de entrenamiento y evaluación.
- **`src/utils.py`** → Carga el dataset base y guarda el modelo entrenado.
- **`src/explore.ipynb`** → Espacio para exploración y pruebas rápidas.
- **`models/`** → Carpeta donde se guardan los modelos entrenados.
- **`data/`** → Reservada para futuros datasets propios del proyecto.

## Cómo ejecutar

Instala dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta el proyecto:

```bash
python src/app.py
```

## Qué hace hoy

Al correr el script:

- divide el dataset en entrenamiento y prueba,
- entrena un modelo `GaussianNB`,
- imprime `accuracy`, matriz de confusión y `classification report`,
- guarda el modelo en `models/gaussian_nb_breast_cancer.joblib`.

## Próximo paso natural

Cuando quieras pasar del ejemplo al proyecto real, podemos hacer cualquiera de estos pasos:

- conectar un CSV propio dentro de `data/raw/`,
- agregar preprocesamiento,
- comparar varias variantes de Naive Bayes (`GaussianNB`, `MultinomialNB`, `BernoulliNB`),
- guardar métricas y predicciones como archivos reproducibles.
