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
        
        collision_reason = None
        while not done and steps < max_steps:
            state_vec = encode_state(env)
            output = nn.forward(state_vec)
            action = np.argmax(output)
            
            _, _, done, info = env.step(action)
            steps += 1
            
            # Capturar motivo da colisão se o jogo terminou
            if done and "reason" in info:
                collision_reason = info["reason"]
            
        score = env.score
        final_len = len(env.snake)
        
        # --- HEURÍSTICA DINÂMICA ---
        
        # Recompensa base por passos (incentiva movimento e sobrevivência)
        episode_fitness = (score * 100) + (steps * 0.5)
        
        if final_len < size_threshold:
            # FASE DE CRESCIMENTO
            # Prioridade: Comer.
            # Maçã vale muito (100). Passo vale pouco (0.5 já adicionado).
            
            # Penalidade extra se morreu cedo sem comer nada
            if score == 0:
                episode_fitness -= 50 # Punição forte por incompetência inicial
            
            # Penalidade aumentada por bater na parede
            if collision_reason == "wall_collision":
                episode_fitness -= 500 # Penalidade alta por colisão com parede
                
        else:
            # FASE DE SOBREVIVÊNCIA
            # Prioridade: Manter-se vivo (evitar auto-colisão).
            # Passo vale mais na fase de sobrevivência
            episode_fitness += steps * 1.5 # Total steps reward = 2.0
            
            # Bônus extra por tamanho grande
            episode_fitness += score * 200 
            
            # Penalidade aumentada por bater na parede
            if collision_reason == "wall_collision":
                episode_fitness -= 800 # Penalidade extremamente alta por colisão com parede na fase de sobrevivência
            
            # Penalidade por colisão com corpo (muito alta para evitar esse comportamento)
            if collision_reason == "body_collision":
                episode_fitness -= 1000 # Penalidade extremamente severa por auto-colisão
            
            # Aqui a morte é natural, mas queremos maximizar steps.
            # O score * 200 garante que comer ainda é melhor que só rodar,
            # mas steps * 2.0 faz com que 100 passos valham tanto quanto 1 maçã.
        
        total_fitness += max(0, episode_fitness) # Fitness não negativo
        
    return total_fitness / num_episodes
