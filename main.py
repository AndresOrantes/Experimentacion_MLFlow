# Importación de librerías externas
import argparse
import logging
import mlflow
import mlflow.sklearn

# Importación de funciones internas
from src.dataset import clean_target, load_data, remove_duplicates, split_data
from src.features import preprocessing
from src.modeling.predict import evaluate_model
from src.modeling.train import training_pipeline


# Configuración de loggings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuracion inicial de parámetros y características ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

PCA_COMPONENTS = 15
EXPERIMENT_NAME = "Adult_Income_Classification"

# parse_args para pasar argumentos en CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Logistic Regression con argumentos de CLI para MLflow."
    )
    
    # Hiperparámetros de Logistic Regression
    parser.add_argument(
        "--C",
        type=float,
        default=0.5,
        help="Inverso de la fuerza de regularización (default: 0.5)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Número máximo de iteraciones (default: 100)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "saga", "sag", "newton-cg"],
        help="Algoritmo de optimización (default: lbfgs)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Mapeo de argumentos recibidos por CLI
    logistic_reg_params = {
        "C": args.C,
        "max_iter": args.max_iter,
        "solver": args.solver,
    }

    # Nombre dinámico para diferenciar las corridas en el UI de MLflow
    run_name = f"LR_C{args.C}_iter{args.max_iter}_{args.solver}"

    logging.info("Cargando datos desde librería UCI Repository...")
    X, y_raw = load_data()
    y = clean_target(y_raw)

    logging.info("Eliminando filas duplicadas...")
    X, y = remove_duplicates(X, y)

    logging.info("Ejecutando train_test_split...")
    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logging.info("Construyendo preprocesador...")
    preprocessor = preprocessing(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    logging.info("Creando pipeline del modelo...")
    pipeline = training_pipeline(
        preprocessor=preprocessor,
        n_components=PCA_COMPONENTS,
        **logistic_reg_params,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        logging.info(f"Entrenando modelo con parámetros: {logistic_reg_params}...")
        pipeline.fit(X_train, y_train)

        logging.info("Registrando parámetros en MLflow...")
        mlflow.log_params(logistic_reg_params)
        mlflow.log_param("pca_components", PCA_COMPONENTS)

        logging.info("Evaluando modelo...")
        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        logging.info("Guardando pipeline en MLflow...")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

    logging.info("Ejecución completada. Métricas:")
    for metric, value in metrics.items():
        logging.info(f"  - {metric}: {value:.4f}")


if __name__ == "__main__":
    main()