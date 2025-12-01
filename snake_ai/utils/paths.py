import os

# Definição de caminhos padrão
ROOT_DIR = os.getcwd()

MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
SNAPSHOTS_DIR = os.path.join(ROOT_DIR, "snapshots")

def create_directories():
    for d in [MODELS_DIR, LOGS_DIR, PLOTS_DIR, SNAPSHOTS_DIR]:
        os.makedirs(d, exist_ok=True)

