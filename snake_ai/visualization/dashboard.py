import pygame
import numpy as np
from ..env.snake_env import SnakeEnv
from ..env.state_encoding import encode_state
from ..agents.neural_net import NeuralNetwork

class DashboardRenderer:
    def __init__(self, env_config: dict, layer_sizes: list[int], caption: str = "Snake AI Training Dashboard", num_games: int = 9):
        pygame.init()
        
        self.env_config = env_config
        self.layer_sizes = layer_sizes
        
        # Configurações de Grid de Jogos
        # num_games pode ser 1 ou 9
        self.num_games = max(1, min(9, num_games))
        if self.num_games == 1:
            self.grid_rows = 1
            self.grid_cols = 1
        else:
            self.grid_rows = 3
            self.grid_cols = 3
        self.margin = 10 # Margem entre jogos
        
        # Tamanho inicial da janela
        self.total_w = 1200
        self.total_h = 800
        
        # Janela Redimensionável
        self.screen = pygame.display.set_mode((self.total_w, self.total_h), pygame.RESIZABLE)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("Arial", 12, bold=True)
        self.title_font = pygame.font.SysFont("Arial", 16, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 9)  # Fonte menor para nomes dos neurônios
        
        # Nomes dos neurônios de entrada (baseado em state_encoding.py)
        self.input_names = ["Perigo F", "Perigo D", "Perigo E", "Ângulo", "Tamanho", "Cauda F", "Cauda D", "Cauda E"]
        # Nomes dos neurônios de saída
        self.output_names = ["Esquerda", "Frente", "Direita"]
        
        # Dados do gráfico
        self.gen_history = []
        self.best_history = []
        self.mean_history = []
        
        # Inicializar Layout
        self.recalculate_layout(self.total_w, self.total_h)

    def recalculate_layout(self, w, h):
        """Recalcula dimensões baseada no tamanho da janela."""
        self.total_w = w
        self.total_h = h
        
        # Divisão: Jogos (Esquerda) vs Info (Direita)
        # Info ocupa 30% ou min 350px
        self.info_area_w = max(int(w * 0.30), 350)
        self.games_area_w = w - self.info_area_w
        
        # Calcular tamanho da célula do jogo para caber no grid 3x3 na área esquerda
        # Largura disponível para um jogo (descontando margens)
        avail_w_per_game = (self.games_area_w - (self.grid_cols + 1) * self.margin) / self.grid_cols
        avail_h_per_game = (h - (self.grid_rows + 1) * self.margin) / self.grid_rows
        
        env_w = self.env_config["width"]
        env_h = self.env_config["height"]
        
        # Escala baseada no menor fator limitante (largura ou altura)
        scale_w = avail_w_per_game / env_w
        scale_h = avail_h_per_game / env_h
        
        self.cell_size = int(min(scale_w, scale_h))
        self.cell_size = max(2, self.cell_size) # Mínimo de 2px
        
        self.game_pixel_w = self.cell_size * env_w
        self.game_pixel_h = self.cell_size * env_h
        
        # Recalcular posições dos neurônios (área da direita, metade superior)
        self.node_positions = self._calculate_node_positions()

    def _calculate_node_positions(self):
        """Calcula as coordenadas (x, y) de cada neurônio na área de info."""
        positions = []
        
        start_x = self.games_area_w + 20
        width = self.info_area_w - 40
        
        # Metade superior para NN
        height = (self.total_h // 2) - 40 
        start_y = 40
        
        num_layers = len(self.layer_sizes)
        layer_spacing = width / (num_layers - 1) if num_layers > 1 else 0
        
        for l_idx, size in enumerate(self.layer_sizes):
            layer_nodes = []
            x = start_x + l_idx * layer_spacing
            
            # Espaçamento vertical entre nós
            # Tentar usar todo o espaço vertical, mas limitar espaçamento máximo
            max_node_spacing = 25
            node_spacing = min(height / max(size, 1), max_node_spacing)
            total_nodes_h = size * node_spacing
            
            layer_start_y = start_y + (height - total_nodes_h) / 2
            
            for n_idx in range(size):
                y = layer_start_y + n_idx * node_spacing
                layer_nodes.append((int(x), int(y)))
            positions.append(layer_nodes)
            
        return positions

    def update_graph_data(self, generation, best_score, mean_score):
        self.gen_history.append(generation)
        self.best_history.append(best_score)
        self.mean_history.append(mean_score)

    def render_generation(self, genomes: list[np.ndarray], nn_template: NeuralNetwork, speed: int = 30) -> bool:
        """
        Executa N jogos em paralelo e visualiza.
        Retorna True se o usuário solicitou o fechamento (QUIT).
        """
        num_agents = min(len(genomes), self.num_games)
        
        envs = []
        nns = []
        dones = [False] * num_agents
        active_genomes = genomes[:num_agents]
        
        for genome in active_genomes:
            # Passar grow_on_eat do config se existir, senão True
            grow = self.env_config.get("grow_on_eat", True)
            # Recriar env para garantir config atualizada
            e = SnakeEnv(
                width=self.env_config["width"],
                height=self.env_config["height"],
                initial_energy=self.env_config["initial_energy"],
                grow_on_eat=grow
            )
            n = NeuralNetwork(self.layer_sizes)
            n.set_weights_flat(genome)
            envs.append(e)
            nns.append(n)
            
        running = True
        while running and not all(dones):
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                elif event.type == pygame.VIDEORESIZE:
                    self.recalculate_layout(event.w, event.h)
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    # Teclas 1 e 9 para mudar número de jogos
                    if event.key == pygame.K_1:
                        self.num_games = 1
                        self.grid_rows = 1
                        self.grid_cols = 1
                        self.recalculate_layout(self.total_w, self.total_h)
                    elif event.key == pygame.K_9:
                        self.num_games = 9
                        self.grid_rows = 3
                        self.grid_cols = 3
                        self.recalculate_layout(self.total_w, self.total_h)
            
            self.screen.fill((20, 20, 25)) # Fundo levemente azulado escuro
            
            # 1. Desenhar Jogos (Esquerda)
            first_agent_activations = None
            
            for i in range(num_agents):
                if not dones[i]:
                    env = envs[i]
                    nn = nns[i]
                    state_vec = encode_state(env)
                    
                    if i == 0:
                        output, activations = nn.forward_debug(state_vec)
                        first_agent_activations = activations
                    else:
                        output = nn.forward(state_vec)
                        
                    action = np.argmax(output)
                    _, _, done, _ = env.step(action)
                    dones[i] = done
                
                # --- Renderização do Jogo ---
                row = i // self.grid_cols
                col = i % self.grid_cols
                
                # Calcular posição centralizada na célula do grid
                cell_w = (self.games_area_w - (self.grid_cols+1)*self.margin) / self.grid_cols
                cell_h = (self.total_h - (self.grid_rows+1)*self.margin) / self.grid_rows
                
                center_x = self.margin + col * (cell_w + self.margin) + cell_w/2
                center_y = self.margin + row * (cell_h + self.margin) + cell_h/2
                
                game_x = int(center_x - self.game_pixel_w / 2)
                game_y = int(center_y - self.game_pixel_h / 2)
                
                # Moldura/Background do Jogo
                pygame.draw.rect(self.screen, (0, 0, 0), (game_x, game_y, self.game_pixel_w, self.game_pixel_h))
                border_color = (100, 100, 100) if not dones[i] else (50, 0, 0)
                pygame.draw.rect(self.screen, border_color, (game_x-2, game_y-2, self.game_pixel_w+4, self.game_pixel_h+4), 2)
                
                # Cobra
                for idx, (sx, sy) in enumerate(envs[i].snake):
                    color = (0, 255, 0) if idx > 0 else (255, 255, 0) # Verde corpo, Amarelo cabeça
                    if dones[i]: color = (80, 80, 80) # Cinza se morto
                        
                    rect = (game_x + sx * self.cell_size, game_y + sy * self.cell_size, self.cell_size - 1, self.cell_size - 1)
                    pygame.draw.rect(self.screen, color, rect)
                
                # Maçã
                ax, ay = envs[i].apple
                # Maçã levemente arredondada/menor
                margin_apple = max(1, self.cell_size // 4)
                rect_apple = (
                    game_x + ax * self.cell_size + margin_apple, 
                    game_y + ay * self.cell_size + margin_apple, 
                    self.cell_size - 2*margin_apple, 
                    self.cell_size - 2*margin_apple
                )
                pygame.draw.rect(self.screen, (255, 50, 50), rect_apple)
                
                # --- Infos (Food & Energy) ---
                # Desenhar na parte superior do quadrado do jogo ou em baixo
                # Vamos desenhar em cima (Food) e em baixo (Energy)
                
                # Food
                score_text = f"Food: {envs[i].score}"
                score_surf = self.font.render(score_text, True, (255, 255, 255))
                self.screen.blit(score_surf, (game_x, game_y - 15))
                
                # Energy
                energy_pct = max(0, envs[i].energy / envs[i].initial_energy)
                energy_color = (0, 255, 255) if energy_pct > 0.3 else (255, 100, 0) # Ciano -> Laranja
                energy_text = f"Energy: {envs[i].energy}"
                energy_surf = self.font.render(energy_text, True, energy_color)
                self.screen.blit(energy_surf, (game_x, game_y + self.game_pixel_h + 2))
                
                # Barra de Energia simples abaixo
                bar_w = self.game_pixel_w
                bar_h = 4
                pygame.draw.rect(self.screen, (50, 50, 50), (game_x, game_y + self.game_pixel_h + 16, bar_w, bar_h))
                pygame.draw.rect(self.screen, energy_color, (game_x, game_y + self.game_pixel_h + 16, int(bar_w * energy_pct), bar_h))

            # 2. Visualizar Rede Neural
            if first_agent_activations:
                self._draw_nn(first_agent_activations, nn_template)
            
            # 3. Visualizar Gráfico
            self._draw_graph()
            
            pygame.display.flip()
            self.clock.tick(speed)

        return False

    def _draw_nn(self, activations, nn_template):
        # Fundo da área de NN
        area_x = self.games_area_w
        area_y = 0
        area_w = self.info_area_w
        area_h = self.total_h // 2
        
        # Título
        title = self.title_font.render("Neural Network (Best Agent)", True, (220, 220, 220))
        self.screen.blit(title, (area_x + 20, 10))
        
        # Instruções para mudar número de jogos
        instrucoes = self.font.render("Pressione 1 ou 9 para mudar número de jogos", True, (150, 150, 150))
        self.screen.blit(instrucoes, (area_x + 20, area_h - 20))
        
        if len(activations) != len(self.node_positions): return

        # Conexões
        for l in range(len(self.node_positions) - 1):
            layer_a = self.node_positions[l]
            layer_b = self.node_positions[l+1]
            
            for i, start_pos in enumerate(layer_a):
                val_a = activations[l][i] if i < len(activations[l]) else 0
                if val_a <= 0.01: continue 
                
                for j, end_pos in enumerate(layer_b):
                    alpha = int(min(val_a, 1.0) * 60) # Mais transparência
                    if alpha < 5: continue
                    
                    # Cor fixa ou variável? Vamos usar cinza claro
                    color = (150, 150, 150)
                    
                    # Hack para desenhar linha com alpha no Pygame (precisa de Surface)
                    # Simplificado: desenha linha solida se for forte o suficiente
                    if alpha > 20:
                        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        # Nós
        for l, layer_pos in enumerate(self.node_positions):
            vals = activations[l] if l < len(activations) else []
            
            for i, pos in enumerate(layer_pos):
                val = vals[i] if i < len(vals) else 0
                
                # Cor do nó
                # Input/Hidden (ReLU): 0 -> Preto/VerdeEscuro, 1+ -> Verde Vivo
                # Output (Tanh): -1 -> Azul, 0 -> Branco, 1 -> Vermelho
                
                radius = 5
                if l == len(self.node_positions) - 1: # Output
                    norm = (val + 1) / 2 # 0..1
                    norm = max(0, min(1, norm))
                    r = int(norm * 255)
                    b = int((1-norm) * 255)
                    color = (r, 0, b)
                    radius = 7 # Output maior
                else:
                    intensity = min(max(val, 0), 1.0) * 255
                    color = (0, int(intensity), 0)
                
                pygame.draw.circle(self.screen, color, pos, radius)
                pygame.draw.circle(self.screen, (200, 200, 200), pos, radius, 1) # Borda branca
                
                # Adicionar nome do neurônio
                nome = ""
                if l == 0:  # Camada de entrada
                    if i < len(self.input_names):
                        nome = self.input_names[i]
                elif l == len(self.node_positions) - 1:  # Camada de saída
                    if i < len(self.output_names):
                        nome = self.output_names[i]
                else:  # Camadas ocultas
                    nome = f"H{i+1}"
                
                if nome:
                    # Desenhar nome ao lado do neurônio
                    # Posicionar à direita do neurônio para inputs/hidden, abaixo para outputs
                    if l == len(self.node_positions) - 1:  # Output: abaixo
                        text_surf = self.small_font.render(nome, True, (200, 200, 200))
                        text_rect = text_surf.get_rect(center=(pos[0], pos[1] + radius + 10))
                    else:  # Input/Hidden: à direita
                        text_surf = self.small_font.render(nome, True, (200, 200, 200))
                        text_rect = text_surf.get_rect(midleft=(pos[0] + radius + 5, pos[1]))
                    
                    self.screen.blit(text_surf, text_rect)

    def _draw_graph(self):
        area_x = self.games_area_w + 20
        area_y = self.total_h // 2 + 20
        area_w = self.info_area_w - 40
        area_h = (self.total_h // 2) - 40
        
        # Fundo Gráfico
        pygame.draw.rect(self.screen, (10, 10, 10), (area_x, area_y, area_w, area_h))
        pygame.draw.rect(self.screen, (100, 100, 100), (area_x, area_y, area_w, area_h), 1)
        
        if len(self.gen_history) < 2: return
            
        max_gen = max(self.gen_history[-1], 1)
        max_fit = max(max(self.best_history), 1)
        
        def to_screen(gen, fit):
            px = area_x + (gen / max_gen) * area_w
            py = (area_y + area_h) - (fit / max_fit) * area_h
            return (px, py)
            
        # Best
        points_best = [to_screen(g, f) for g, f in zip(self.gen_history, self.best_history)]
        if len(points_best) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, points_best, 2)
            
        # Mean
        points_mean = [to_screen(g, f) for g, f in zip(self.gen_history, self.mean_history)]
        if len(points_mean) > 1:
            pygame.draw.lines(self.screen, (0, 100, 255), False, points_mean, 1)
            
        # Labels
        lbl = self.font.render(f"Generation: {self.gen_history[-1]}", True, (200, 200, 200))
        self.screen.blit(lbl, (area_x, area_y - 15))
        
        lbl_best = self.font.render(f"Best Fitness: {self.best_history[-1]:.1f}", True, (0, 255, 0))
        self.screen.blit(lbl_best, (area_x, area_y + 5))

    def close(self):
        pygame.quit()
