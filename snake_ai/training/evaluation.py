import numpy as np
from ..env.snake_env import SnakeEnv
from ..env.state_encoding import encode_state
from ..agents.neural_net import NeuralNetwork

def evaluate_genome(
    genome: np.ndarray,
    nn: NeuralNetwork,
    env_config: dict,
    num_episodes: int = 3
) -> float:
    """
    Avalia o fitness de um genoma jogando múltiplos episódios.
    
    Args:
        genome (np.ndarray): Vetor de pesos/biases.
        nn (NeuralNetwork): Instância da rede neural (reutilizada para eficiência).
        env_config (dict): Configuração do ambiente (width, height, etc).
        num_episodes (int): Quantidade de jogos para média.
        
    Returns:
        float: Fitness médio.
    """
    
    nn.set_weights_flat(genome)
    
    total_fitness = 0.0
    total_apples = 0
    
    # Criar ambiente
    env = SnakeEnv(**env_config)
    
    for _ in range(num_episodes):
        env.reset()
        done = False
        steps = 0
        
        # Proteção contra loops infinitos (embora a energia cuide disso)
        max_steps = 2000 # Limite hard
        
        while not done and steps < max_steps:
            state_vec = encode_state(env)
            output = nn.forward(state_vec)
            action = np.argmax(output) # 0, 1, 2
            
            _, _, done, info = env.step(action)
            steps += 1
        
        # Fitness calculation
        score = env.score
        total_apples += score
        
        # Fórmula de fitness
        # Prioridade máxima: maçãs. 
        # Prioridade secundária: sobreviver (steps).
        # Se score alto, steps altos é bom. Se score 0, steps altos é bom (sobreviveu).
        
        # Ex: fitness = (score * 100) + (steps * 0.1)
        # Se morrer cedo com 0 maçãs -> baixo.
        # Se ficar rodando sem comer -> steps aumenta, mas energia mata.
        # Energia limita steps a width*height se não comer.
        
        fitness = (score * 100) + (steps * 0.1)
        
        # Penalidade se bateu na parede muito rápido? Não precisa, fitness será baixo.
        
        total_fitness += fitness
        
    return total_fitness / num_episodes

