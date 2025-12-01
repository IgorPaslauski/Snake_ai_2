import numpy as np
import os
import argparse
from snake_ai.agents.neural_net import NeuralNetwork
from snake_ai.visualization.live_view import play_episode

def main():
    parser = argparse.ArgumentParser(description="Assistir ao melhor agente jogando Snake.")
    parser.add_argument("--model", type=str, default="models/best_overall.npy", help="Caminho para o arquivo .npy do genoma.")
    parser.add_argument("--speed", type=int, default=10, help="Velocidade do jogo (FPS).")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Erro: Modelo não encontrado em {args.model}")
        print("Rode 'python main_train.py' primeiro para gerar um modelo.")
        return

    print(f"Carregando modelo de {args.model}...")
    genome = np.load(args.model)
    
    # Deve corresponder à config usada no treinamento
    # Idealmente salvar config junto com o modelo, mas vamos usar hardcoded por enquanto conforme o exercício
    ENV_CONFIG = {
        "width": 10,
        "height": 10,
        "initial_energy": 100
    }
    
    LAYER_SIZES = [6, 16, 12, 3]
    nn = NeuralNetwork(LAYER_SIZES)
    
    print("Iniciando visualização... (Pressione ESC ou feche a janela para sair)")
    play_episode(genome, ENV_CONFIG, nn, speed=args.speed)

if __name__ == "__main__":
    main()

