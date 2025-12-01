import numpy as np
from .genome import create_random_genome, mutate_genome, crossover_single_point, crossover_uniform

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        genome_size: int,
        elitism: int,
        mutation_rate: float,
        mutation_std: float,
        crossover_type: str = "uniform"
    ):
        self.population_size = population_size
        self.genome_size = genome_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.crossover_type = crossover_type
        
        # População atual: lista de np.arrays
        self.population = [create_random_genome(genome_size) for _ in range(population_size)]
        self.generation = 0
        self.best_genome = None
        self.best_fitness_history = []

    def get_population(self) -> list[np.ndarray]:
        return self.population

    def evolve(self, fitness_scores: list[float]) -> None:
        """
        Evolui a população para a próxima geração baseada nos scores de fitness.
        """
        # 1. Ordenar índices por fitness (decrescente)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        
        # Salvar melhor da geração
        self.best_genome = sorted_population[0].copy()
        self.best_fitness_history.append(fitness_scores[sorted_indices[0]])
        
        new_population = []
        
        # 2. Elitismo: manter os k melhores
        for i in range(self.elitism):
            if i < len(sorted_population):
                new_population.append(sorted_population[i].copy())
                
        # 3. Preencher o resto da população
        # Usaremos seleção por torneio simples para escolher pais
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(sorted_population, fitness_scores, sorted_indices)
            parent2 = self._tournament_selection(sorted_population, fitness_scores, sorted_indices)
            
            # Crossover
            if self.crossover_type == "uniform":
                child1, child2 = crossover_uniform(parent1, parent2)
            else:
                child1, child2 = crossover_single_point(parent1, parent2)
            
            # Mutação
            child1 = mutate_genome(child1, self.mutation_rate, self.mutation_std)
            child2 = mutate_genome(child2, self.mutation_rate, self.mutation_std)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population[:self.population_size]
        self.generation += 1

    def _tournament_selection(self, sorted_pop, fitness_scores, sorted_indices, k=3):
        """Seleciona um indivíduo via torneio."""
        # Escolher k índices aleatórios da população
        contestants_indices = np.random.choice(len(sorted_pop), k, replace=False)
        
        # Achar o melhor entre eles (que tem maior fitness)
        # Precisamos mapear de volta para o fitness original
        # Mas sorted_pop já está ordenado, então o menor índice em sorted_pop vence?
        # contestants_indices são índices de sorted_pop (0 = melhor global).
        # Então basta pegar o menor valor numérico de índice.
        best_idx = np.min(contestants_indices)
        return sorted_pop[best_idx]

