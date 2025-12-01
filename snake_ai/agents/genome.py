import numpy as np

def create_random_genome(size: int, scale: float = 0.1) -> np.ndarray:
    """Cria um genoma aleatório com distribuição normal."""
    return np.random.randn(size) * scale

def mutate_genome(genome: np.ndarray, mutation_rate: float, mutation_std: float) -> np.ndarray:
    """
    Aplica mutação gaussiana ao genoma.
    Cada gene tem 'mutation_rate' chance de ser alterado.
    """
    # Máscara booleana para decidir quais genes mutar
    mask = np.random.rand(len(genome)) < mutation_rate
    
    # Ruído gaussiano
    noise = np.random.randn(len(genome)) * mutation_std
    
    # Aplicar ruído apenas onde mask é True
    mutated_genome = genome.copy()
    mutated_genome[mask] += noise[mask]
    
    return mutated_genome

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crossover Uniforme: cada gene do filho vem do pai1 ou pai2 com 50% de chance.
    Gera 2 filhos (inversos).
    """
    mask = np.random.rand(len(parent1)) < 0.5
    
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    
    return child1, child2

def crossover_single_point(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crossover de ponto único."""
    point = np.random.randint(1, len(parent1) - 1)
    
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    
    return child1, child2

