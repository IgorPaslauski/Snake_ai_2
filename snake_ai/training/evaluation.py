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
    Avalia o fitness com heurística dinâmica.
    
    Fase 1 (Pequena): Foco total em comer (reward alto por maçã).
    Fase 2 (Grande): Foco em sobreviver (reward alto por passos) e penalidade alta por colisão.
    """
    
    nn.set_weights_flat(genome)
    
    total_fitness = 0.0
    
    # Criar ambiente
    env = SnakeEnv(**env_config)
    
    # Threshold para considerar "Grande" (ex: 10% do grid)
    # Se grid 10x10 = 100. Grande > 10.
    size_threshold = (env.width * env.height) * 0.1
    
    for _ in range(num_episodes):
        env.reset()
        done = False
        steps = 0
        episode_fitness = 0.0
        
        max_steps = 2000 
        
        while not done and steps < max_steps:
            state_vec = encode_state(env)
            output = nn.forward(state_vec)
            action = np.argmax(output)
            
            _, _, done, info = env.step(action)
            steps += 1
            
        score = env.score
        final_len = len(env.snake)
        
        # --- HEURÍSTICA DINÂMICA ---
        
        if final_len < size_threshold:
            # FASE DE CRESCIMENTO
            # Prioridade: Comer.
            # Maçã vale muito (100). Passo vale pouco (0.1).
            # Penalidade por morrer na parede (colisão simples) deve ser alta para aprender limites.
            
            episode_fitness = (score * 100) + (steps * 0.1)
            
            # Penalidade extra se morreu cedo sem comer nada
            if score == 0:
                episode_fitness -= 50 # Punição forte por incompetência inicial
                
        else:
            # FASE DE SOBREVIVÊNCIA
            # Prioridade: Manter-se vivo (evitar auto-colisão).
            # Maçã vale menos relativo ao passo (ainda boa, mas passos somam muito).
            # Passo vale muito mais (1.0 ou mais), pois cada passo vivo é vitória.
            
            episode_fitness = (score * 200) + (steps * 2.0) # Scaling up rewards
            
            # Aqui a morte é natural, mas queremos maximizar steps.
            # O score * 200 garante que comer ainda é melhor que só rodar,
            # mas steps * 2.0 faz com que 100 passos valham tanto quanto 1 maçã.
        
        total_fitness += max(0, episode_fitness) # Fitness não negativo
        
    return total_fitness / num_episodes
