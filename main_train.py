import numpy as np
import os
import sys
from tqdm import tqdm
import time

from snake_ai.env.snake_env import SnakeEnv
from snake_ai.agents.neural_net import NeuralNetwork
from snake_ai.agents.genetic_algorithm import GeneticAlgorithm
from snake_ai.training.evaluation import evaluate_genome
from snake_ai.utils.logger import TrainingLogger
from snake_ai.utils.paths import create_directories, MODELS_DIR, LOGS_DIR, PLOTS_DIR, SNAPSHOTS_DIR
from snake_ai.visualization.plots import plot_training_curves
from snake_ai.visualization.board_snapshots import save_generation_snapshot
from snake_ai.visualization.dashboard import DashboardRenderer
from snake_ai.utils.launcher import ConfigScreen

def main():
    # --- 1. Tela de Configuração ---
    print("Abrindo tela de configuração...")
    launcher = ConfigScreen()
    user_config = launcher.show()
    
    if user_config is None:
        print("Configuração cancelada ou janela fechada. Saindo.")
        sys.exit(0)
        
    # --- 2. Aplicar Configurações ---
    ENV_CONFIG = {
        "width": user_config["width"],
        "height": user_config["height"],
        "initial_energy": user_config["initial_energy"],
        "grow_on_eat": user_config["grow_on_eat"]
    }
    
    POPULATION_SIZE = user_config["population_size"]
    GENERATIONS = user_config["generations"]
    MUTATION_RATE = user_config["mutation_rate"]
    
    # Fixos ou derivados
    ELITISM = max(2, int(POPULATION_SIZE * 0.05)) # 5% de elitismo
    MUTATION_STD = 0.2
    
    # Arquitetura da MLP: Input=4 (Danger=3, Angle=1), Hidden=[16, 12], Output=3
    LAYER_SIZES = [4, 16, 12, 3]
    
    EPISODES_PER_EVAL = 3
    SNAPSHOT_INTERVAL = 50
    
    # Visualização
    LIVE_DASHBOARD = user_config["live_dashboard"]
    VIEW_SPEED = user_config["fps"]
    
    print("\n--- Configuração Iniciada ---")
    print(f"Gerações: {GENERATIONS}")
    print(f"População: {POPULATION_SIZE}")
    print(f"Crescer corpo: {user_config['grow_on_eat']}")
    print(f"Dashboard: {LIVE_DASHBOARD}")
    print("-----------------------------\n")
    
    # --- Inicialização ---
    create_directories()
    
    # Instância de Rede Neural usada para avaliação (pesos serão injetados)
    nn = NeuralNetwork(LAYER_SIZES)
    genome_size = len(nn.get_weights_flat())
    print(f"Tamanho do Genoma (Weights + Biases): {genome_size}")
    
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        genome_size=genome_size,
        elitism=ELITISM,
        mutation_rate=MUTATION_RATE,
        mutation_std=MUTATION_STD,
        crossover_type="uniform"
    )
    
    # Logger
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"training_{timestamp}.csv")
    logger = TrainingLogger(log_path, ["generation", "best_fitness", "mean_fitness", "min_fitness"])
    
    best_overall_fitness = -float('inf')
    best_overall_genome = None
    
    # Inicializar Dashboard
    dashboard = None
    if LIVE_DASHBOARD:
        print("Inicializando Dashboard Interativo...")
        dashboard = DashboardRenderer(ENV_CONFIG, LAYER_SIZES, caption="Treinamento Snake AI - Monitoramento em Tempo Real")
    
    print(f"Iniciando treinamento por {GENERATIONS} gerações...")
    
    try:
        for gen in tqdm(range(GENERATIONS), desc="Generations"):
            population = ga.get_population()
            fitness_scores = []
            
            # 1. Avaliação (Loop de treino)
            for genome in population:
                fit = evaluate_genome(genome, nn, ENV_CONFIG, num_episodes=EPISODES_PER_EVAL)
                fitness_scores.append(fit)
                
            # Estatísticas
            fitness_scores = np.array(fitness_scores)
            best_fit = np.max(fitness_scores)
            mean_fit = np.mean(fitness_scores)
            min_fit = np.min(fitness_scores)
            
            # Ordenar para pegar os melhores
            sorted_indices = np.argsort(fitness_scores)[::-1]
            best_idx = sorted_indices[0]
            best_gen_genome = population[best_idx]
            
            # Salvar melhor global
            if best_fit > best_overall_fitness:
                best_overall_fitness = best_fit
                best_overall_genome = best_gen_genome.copy()
                np.save(os.path.join(MODELS_DIR, "best_overall.npy"), best_overall_genome)
                
            np.save(os.path.join(MODELS_DIR, f"best_gen_{gen:04d}.npy"), best_gen_genome)
            
            logger.log({
                "generation": gen,
                "best_fitness": best_fit,
                "mean_fitness": mean_fit,
                "min_fitness": min_fit
            })
            
            # 2. Visualização Dashboard
            if dashboard:
                # Atualizar dados do gráfico
                dashboard.update_graph_data(gen, best_fit, mean_fit)
                
                # Selecionar top 9 para exibir
                top_9_indices = sorted_indices[:9]
                top_9_genomes = [population[i] for i in top_9_indices]
                
                # Renderizar visualização paralela
                nn.set_weights_flat(best_gen_genome) 
                should_quit = dashboard.render_generation(top_9_genomes, nn, speed=VIEW_SPEED)
                
                if should_quit:
                    print("\nVisualização fechada pelo usuário. Encerrando treinamento...")
                    break
            
            # Snapshot Estático
            if gen % SNAPSHOT_INTERVAL == 0:
                snap_path = os.path.join(SNAPSHOTS_DIR, f"gen_{gen:04d}.png")
                save_generation_snapshot(best_gen_genome, ENV_CONFIG, nn, snap_path)
                
            # 3. Evolução
            ga.evolve(fitness_scores)
            
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
        
    finally:
        if dashboard:
            dashboard.close()
            
        print("\nTreinamento concluído (ou encerrado)!")
        print(f"Melhor Fitness Global: {best_overall_fitness:.2f}")
        
        plot_path = os.path.join(PLOTS_DIR, f"fitness_curve_{timestamp}.png")
        plot_training_curves(log_path, plot_path)
        print(f"Gráfico final salvo em {plot_path}")

if __name__ == "__main__":
    main()
