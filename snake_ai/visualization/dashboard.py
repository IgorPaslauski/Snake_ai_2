import pygame
import numpy as np
from ..env.snake_env import SnakeEnv
from ..env.state_encoding import encode_state
from ..agents.neural_net import NeuralNetwork

class DashboardRenderer:
    def __init__(self, env_config: dict, layer_sizes: list[int], caption: str = "Snake AI Training Dashboard"):
        pygame.init()
        
        # Configurações de Layout
        self.grid_rows = 3
        self.grid_cols = 3
        self.num_games = self.grid_rows * self.grid_cols
        
        self.env_config = env_config
        self.base_cell_size = 15 # Menor que o normal para caber 9
        self.game_w = env_config["width"] * self.base_cell_size
        self.game_h = env_config["height"] * self.base_cell_size
        self.margin = 5
        
        # Area de Jogos (Esquerda)
        self.games_area_w = (self.game_w + self.margin) * self.grid_cols + self.margin
        self.games_area_h = (self.game_h + self.margin) * self.grid_rows + self.margin
        
        # Area de Info (Direita)
        self.info_area_w = 400
        self.info_area_h = self.games_area_h
        
        self.total_w = self.games_area_w + self.info_area_w
        self.total_h = self.games_area_h
        
        self.screen = pygame.display.set_mode((self.total_w, self.total_h))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 18, bold=True)
        
        self.layer_sizes = layer_sizes
        
        # Dados do gráfico
        self.gen_history = []
        self.best_history = []
        self.mean_history = []
        
        # Pre-calcular posições dos neurônios para visualização
        self.node_positions = self._calculate_node_positions()

    def _calculate_node_positions(self):
        """Calcula as coordenadas (x, y) de cada neurônio na área de info."""
        positions = []
        
        area_w = self.info_area_w
        area_h = self.info_area_h // 2 # Metade superior para NN
        
        offset_x = self.games_area_w
        offset_y = 40 # Título
        
        num_layers = len(self.layer_sizes)
        layer_spacing = (area_w - 40) / (num_layers - 1)
        
        for l_idx, size in enumerate(self.layer_sizes):
            layer_nodes = []
            x = offset_x + 20 + l_idx * layer_spacing
            
            # Espaçamento vertical
            total_h_needed = size * 15 # 15px por node
            start_y = offset_y + (area_h - total_h_needed) / 2
            
            for n_idx in range(size):
                y = start_y + n_idx * 15
                layer_nodes.append((int(x), int(y)))
            positions.append(layer_nodes)
            
        return positions

    def update_graph_data(self, generation, best_score, mean_score):
        self.gen_history.append(generation)
        self.best_history.append(best_score)
        self.mean_history.append(mean_score)

    def render_generation(self, genomes: list[np.ndarray], nn_template: NeuralNetwork, speed: int = 30):
        """
        Executa N jogos em paralelo e visualiza.
        """
        num_agents = min(len(genomes), self.num_games)
        
        # Inicializar ambientes e estados
        envs = []
        nns = []
        dones = [False] * num_agents
        active_genomes = genomes[:num_agents]
        
        for genome in active_genomes:
            e = SnakeEnv(**self.env_config)
            n = NeuralNetwork(self.layer_sizes) # Nova instância para pesos diferentes
            n.set_weights_flat(genome)
            envs.append(e)
            nns.append(n)
            
        running = True
        while running and not all(dones):
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            self.screen.fill((20, 20, 20)) # Dark background
            
            # 1. Desenhar Jogos (Esquerda)
            first_agent_activations = None
            
            for i in range(num_agents):
                if not dones[i]:
                    env = envs[i]
                    nn = nns[i]
                    
                    # Step
                    state_vec = encode_state(env)
                    
                    if i == 0:
                        # Capturar ativações do primeiro agente para visualizar
                        output, activations = nn.forward_debug(state_vec)
                        first_agent_activations = activations
                    else:
                        output = nn.forward(state_vec)
                        
                    action = np.argmax(output)
                    _, _, done, _ = env.step(action)
                    dones[i] = done
                    
                # Desenhar Grid
                row = i // self.grid_cols
                col = i % self.grid_cols
                
                x_base = self.margin + col * (self.game_w + self.margin)
                y_base = self.margin + row * (self.game_h + self.margin)
                
                # Fundo do jogo
                pygame.draw.rect(self.screen, (0, 0, 0), (x_base, y_base, self.game_w, self.game_h))
                pygame.draw.rect(self.screen, (50, 50, 50), (x_base, y_base, self.game_w, self.game_h), 1)
                
                # Cobra
                for idx, (sx, sy) in enumerate(envs[i].snake):
                    color = (0, 255, 0) if idx > 0 else (0, 200, 0)
                    # Se morreu, fica cinza
                    if dones[i]: color = (100, 100, 100)
                        
                    rect = (x_base + sx * self.base_cell_size, y_base + sy * self.base_cell_size, self.base_cell_size, self.base_cell_size)
                    pygame.draw.rect(self.screen, color, rect)
                
                # Maçã
                ax, ay = envs[i].apple
                rect = (x_base + ax * self.base_cell_size, y_base + ay * self.base_cell_size, self.base_cell_size, self.base_cell_size)
                pygame.draw.rect(self.screen, (255, 0, 0), rect)
                
                # Score Overlay
                score_surf = self.font.render(f"{envs[i].score}", True, (255, 255, 255))
                self.screen.blit(score_surf, (x_base + 2, y_base + 2))

            # 2. Visualizar Rede Neural (Direita Top)
            if first_agent_activations:
                self._draw_nn(first_agent_activations, nn_template)
            
            # 3. Visualizar Gráfico (Direita Bottom)
            self._draw_graph()
            
            pygame.display.flip()
            self.clock.tick(speed)

    def _draw_nn(self, activations: list[np.ndarray], nn_template: NeuralNetwork):
        """Desenha conexões e nós."""
        # Título
        title = self.title_font.render("Neural Network (Best Agent)", True, (200, 200, 200))
        self.screen.blit(title, (self.games_area_w + 10, 10))
        
        weights = nn_template.weights # Pesos da rede (usando a template, que deveria ser a mesma do agente 0 se atualizada corretamente)
        # Nota: nn_template pode não ter os pesos exatos do agente 0 se não for passado explicitamente.
        # Mas como os weights são fixos na topologia, usamos apenas para desenhar linhas.
        # A intensidade das linhas pode ser fixa ou baseada no peso médio. Vamos simplificar e desenhar conexões fixas.
        
        # Desenhar Conexões
        for l in range(len(self.node_positions) - 1):
            layer_a = self.node_positions[l]
            layer_b = self.node_positions[l+1]
            
            # Otimização: desenhar apenas conexões fortes ou todas muito finas
            for i, start_pos in enumerate(layer_a):
                val_a = activations[l][i] if l < len(activations) else 0
                if val_a <= 0: continue # Se neurônio inativo, não destaca conexões saindo dele
                
                for j, end_pos in enumerate(layer_b):
                    # Cor baseada na ativação de origem
                    alpha = int(min(val_a, 1.0) * 100)
                    color = (100, 100, 100)
                    pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        # Desenhar Nós
        for l, layer_pos in enumerate(self.node_positions):
            vals = activations[l] if l < len(activations) else [0]*len(layer_pos)
            
            for i, pos in enumerate(layer_pos):
                val = vals[i]
                # Cor: Azul (negativo/baixo) -> Branco (zero) -> Vermelho (positivo/alto)
                # ReLU: 0 -> Positivo. Tanh: -1 -> 1.
                # Simplificação: Brilho verde baseado na magnitude
                intensity = min(max(val, 0), 1.0) * 255
                color = (0, int(intensity), 0)
                if l == len(self.node_positions) - 1: # Output
                    # Tanh outputs (-1 a 1)
                    # Azul=Negativo, Vermelho=Positivo
                    norm = (val + 1) / 2 # 0 a 1
                    r = int(norm * 255)
                    b = int((1-norm) * 255)
                    color = (r, 0, b)
                
                pygame.draw.circle(self.screen, color, pos, 4)
                pygame.draw.circle(self.screen, (150, 150, 150), pos, 5, 1)

    def _draw_graph(self):
        """Desenha gráfico de fitness."""
        area_x = self.games_area_w + 20
        area_y = self.total_h // 2 + 20
        area_w = self.info_area_w - 40
        area_h = self.total_h // 2 - 40
        
        # Fundo e borda
        pygame.draw.rect(self.screen, (30, 30, 30), (area_x, area_y, area_w, area_h))
        pygame.draw.rect(self.screen, (100, 100, 100), (area_x, area_y, area_w, area_h), 1)
        
        if len(self.gen_history) < 2:
            return
            
        # Escalas
        max_gen = max(self.gen_history[-1], 1)
        max_fit = max(max(self.best_history), 1)
        
        # Helper para converter coords
        def to_screen(gen, fit):
            px = area_x + (gen / max_gen) * area_w
            # py invertido (topo é max_fit)
            py = (area_y + area_h) - (fit / max_fit) * area_h
            return (px, py)
            
        # Desenhar Best Fitness (Verde)
        points_best = [to_screen(g, f) for g, f in zip(self.gen_history, self.best_history)]
        if len(points_best) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, points_best, 2)
            
        # Desenhar Mean Fitness (Azul)
        points_mean = [to_screen(g, f) for g, f in zip(self.gen_history, self.mean_history)]
        if len(points_mean) > 1:
            pygame.draw.lines(self.screen, (0, 100, 255), False, points_mean, 1)
            
        # Labels
        lbl = self.font.render(f"Gen: {self.gen_history[-1]}", True, (200, 200, 200))
        self.screen.blit(lbl, (area_x + 5, area_y + 5))
        
        lbl_best = self.font.render(f"Best: {self.best_history[-1]:.1f}", True, (0, 255, 0))
        self.screen.blit(lbl_best, (area_x + 5, area_y + 20))

    def close(self):
        pygame.quit()

