import numpy as np
import matplotlib.pyplot as plt
from ..env.snake_env import SnakeEnv
from ..env.state_encoding import encode_state
from ..agents.neural_net import NeuralNetwork

def save_generation_snapshot(
    genome: np.ndarray,
    env_config: dict,
    nn: NeuralNetwork,
    output_path: str,
    max_steps: int = 50,
    frames: int = 16
) -> None:
    """
    Roda um episódio curto e salva um grid de imagens do jogo.
    """
    nn.set_weights_flat(genome)
    env = SnakeEnv(**env_config)
    
    # Rodar episódio e capturar estados
    states_to_plot = []
    
    steps = 0
    done = False
    
    # Captura inicial
    # Vamos capturar frames distribuídos ou os primeiros N?
    # Vamos capturar os primeiros 'frames' passos
    
    while not done and steps < frames:
        # Renderizar estado atual em matriz numérica
        # 0: Vazio, 1: Corpo, 2: Cabeça, 3: Maçã
        grid = np.zeros((env.height, env.width))
        
        # Corpo
        for (x, y) in env.snake:
            if 0 <= x < env.width and 0 <= y < env.height:
                grid[y, x] = 1
        
        # Cabeça (sobrescreve corpo)
        hx, hy = env.snake[0]
        if 0 <= hx < env.width and 0 <= hy < env.height:
            grid[hy, hx] = 2
            
        # Maçã
        ax, ay = env.apple
        grid[ay, ax] = 3
        
        states_to_plot.append(grid.copy())
        
        # Step
        state_vec = encode_state(env)
        output = nn.forward(state_vec)
        action = np.argmax(output)
        _, _, done, _ = env.step(action)
        steps += 1
        
    # Plotar
    # Definir tamanho do grid (ex: 4x4 para 16 frames)
    side = int(np.ceil(np.sqrt(len(states_to_plot))))
    if side == 0: return

    fig, axes = plt.subplots(side, side, figsize=(10, 10))
    
    # Cores: 0=Preto (Fundo), 1=VerdeEscuro (Corpo), 2=VerdeClaro (Cabeça), 3=Vermelho (Maçã)
    # Custom cmap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['black', 'green', '#00FF00', 'red'])
    
    for i, ax in enumerate(axes.flat):
        if i < len(states_to_plot):
            ax.imshow(states_to_plot[i], cmap=cmap, vmin=0, vmax=3)
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

