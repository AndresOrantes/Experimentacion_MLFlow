from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def training_pipeline(preprocessor, pca: bool=True, n_components: float=0.90, max_iter:int=1000, C:float = 1.0, solver: str = "lbfgs") -> Pipeline:
    """Pipeline de entrenamiento. Se definen los pasos de entrenamiento del pipeline que incluyen el pipeline de preprocesamiento, PCA y entrenamiento del modelo"""

    steps = [("preprocessor", preprocessor)]

    if pca:
        steps.append(("pca", PCA(n_components=n_components)))

    steps.append(
        ("classifier", LogisticRegression(max_iter=max_iter, C=C, solver=solver))
    )

    return Pipeline(steps)