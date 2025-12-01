import pygame
import numpy as np
import time
from ..env.snake_env import SnakeEnv
from ..env.state_encoding import encode_state
from ..agents.neural_net import NeuralNetwork

class PygameRenderer:
    """
    Gerenciador de visualização Pygame persistente.
    Mantém a janela aberta entre episódios.
    """
    def __init__(self, env_config: dict, caption: str = "Snake AI Training"):
        pygame.init()
        self.env_config = env_config
        self.cell_size = 20
        self.width_px = env_config["width"] * self.cell_size
        self.height_px = env_config["height"] * self.cell_size
        
        # Área extra para texto (opcional)
        self.hud_height = 40
        
        self.screen = pygame.display.set_mode((self.width_px, self.height_px + self.hud_height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
    def render_episode(self, genome: np.ndarray, nn: NeuralNetwork, speed: int = 20) -> None:
        """
        Roda e renderiza um episódio completo com o genoma fornecido.
        """
        # Configurar rede e ambiente
        nn.set_weights_flat(genome)
        env = SnakeEnv(**self.env_config)
        
        done = False
        
        while not done:
            # Processar eventos (permite fechar a janela ou mover)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # IA decide ação
            state_vec = encode_state(env)
            output = nn.forward(state_vec)
            action = np.argmax(output)
            
            _, _, done, info = env.step(action)
            
            # Desenhar
            self._draw_frame(env)
            
            # Controle de FPS
            self.clock.tick(speed)
            
    def _draw_frame(self, env: SnakeEnv):
        self.screen.fill((0, 0, 0)) # Fundo preto
        
        # Offset do HUD
        y_offset = self.hud_height
        
        # Desenhar Cobra
        for i, (x, y) in enumerate(env.snake):
            color = (0, 255, 0) # Verde
            if i == 0:
                color = (0, 200, 0) # Cabeça
            
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size + y_offset, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 50, 0), rect, 1) # Borda
            
        # Desenhar Maçã
        ax, ay = env.apple
        rect = pygame.Rect(ax * self.cell_size, ay * self.cell_size + y_offset, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), rect)
        
        # HUD
        # Limpar área do HUD (já foi limpa pelo fill, mas podemos separar se quiser cor diferente)
        # pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.width_px, self.hud_height))
        
        score_text = self.font.render(f"Score: {env.score}", True, (255, 255, 255))
        energy_text = self.font.render(f"Energy: {env.energy}", True, (255, 255, 255))
        
        self.screen.blit(score_text, (5, 5))
        self.screen.blit(energy_text, (100, 5))
        
        pygame.display.flip()

    def close(self):
        pygame.quit()

def play_episode(genome: np.ndarray, env_config: dict, nn: NeuralNetwork, speed: int = 10):
    """
    Função standalone para manter compatibilidade com play_best.py existente,
    mas agora usando a classe PygameRenderer.
    """
    renderer = PygameRenderer(env_config, caption="Snake AI - Replay")
    renderer.render_episode(genome, nn, speed)
    # Não fechamos aqui imediatamente para não matar o script se tiver mais coisas, 
    # mas o PygameRenderer não tem loop infinito.
    # Se play_best espera loop único, ok.
    renderer.close()
