import numpy as np
import random
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeEnv:
    """
    Ambiente do jogo Snake (Cobrinha) para treinamento de IA.
    
    Atributos:
        width (int): Largura do tabuleiro.
        height (int): Altura do tabuleiro.
        initial_energy (int): Energia inicial da cobra.
        snake (list[tuple]): Lista de coordenadas do corpo da cobra (cabeça é o índice 0).
        apple (tuple): Coordenada da maçã (x, y).
        direction (Direction): Direção atual da cobra.
        score (int): Pontuação atual (número de maçãs comidas).
        energy (int): Energia restante.
        done (bool): Se o jogo terminou.
        steps (int): Contador de passos no episódio.
    """

    def __init__(self, width: int = 10, height: int = 10, initial_energy: int | None = None):
        self.width = width
        self.height = height
        # Se energia inicial não for definida, usa largura * altura como padrão
        self.initial_energy = initial_energy if initial_energy is not None else width * height
        self.reset()

    def reset(self) -> dict:
        """
        Reseta o ambiente para o estado inicial.
        
        Retorna:
            dict: Dicionário contendo informações do estado (cabeça, corpo, maçã, direção, energia).
        """
        # Inicia a cobra no centro
        self.direction = Direction.RIGHT
        head_x = self.width // 2
        head_y = self.height // 2
        self.snake = [(head_x, head_y), (head_x - 1, head_y), (head_x - 2, head_y)]
        
        self.score = 0
        self.energy = self.initial_energy
        self.steps = 0
        self.done = False
        
        self._place_apple()
        
        return self._get_state_info()

    def _place_apple(self):
        """Posiciona a maçã em um local aleatório livre (não ocupado pela cobra)."""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                self.apple = (x, y)
                break

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """
        Executa um passo no ambiente.
        
        Args:
            action (int): 0 = virar esquerda, 1 = seguir reto, 2 = virar direita.
            
        Retorna:
            state_info (dict): Informações do estado atual.
            reward (float): Recompensa recebida.
            done (bool): Se o episódio terminou.
            info (dict): Informações extras (score, steps).
        """
        if self.done:
            return self._get_state_info(), 0, True, {"score": self.score}

        self.steps += 1
        self.energy -= 1
        
        # Atualizar direção baseada na ação
        # action: 0 -> esquerda, 1 -> frente, 2 -> direita
        # Ordem Enum: UP(0), RIGHT(1), DOWN(2), LEFT(3)
        # Virar esquerda: (dir - 1) % 4
        # Virar direita: (dir + 1) % 4
        
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)
        
        if action == 0: # Esquerda (relativo)
            new_idx = (idx - 1) % 4
            self.direction = clock_wise[new_idx]
        elif action == 2: # Direita (relativo)
            new_idx = (idx + 1) % 4
            self.direction = clock_wise[new_idx]
        # Se action == 1, mantém direção
        
        # Calcular nova posição da cabeça
        x, y = self.snake[0]
        if self.direction == Direction.UP:
            y -= 1
        elif self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.LEFT:
            x -= 1
            
        new_head = (x, y)
        reward = 0
        
        # Verificar colisões (Parede ou Corpo)
        if (x < 0 or x >= self.width or y < 0 or y >= self.height or new_head in self.snake[:-1]):
            # Colisão! Nota: new_head in snake[:-1] verifica colisão com corpo exceto a cauda que vai mover
            # Mas tecnicamente se não comeu maçã, a cauda sai. Se comeu, a cauda fica.
            # Simplificação: colisão com qualquer parte atual conta como morte, exceto se for movimento válido onde a cauda sai.
            # Vamos verificar colisão estrita.
            self.done = True
            reward = -1.0
            return self._get_state_info(), reward, self.done, {"score": self.score, "reason": "collision"}
        
        # Verificar energia
        if self.energy <= 0:
            self.done = True
            reward = -1.0 # Punir por morrer de fome? Ou 0? O user pediu -1 ao morrer (colisão), mas energia 0 é game over também.
            # Vou colocar punição também para incentivar comer.
            return self._get_state_info(), reward, self.done, {"score": self.score, "reason": "starvation"}

        # Mover cobra
        self.snake.insert(0, new_head)
        
        # Verificar se comeu maçã
        if new_head == self.apple:
            self.score += 1
            reward = 10.0
            self.energy = self.initial_energy # Resetar energia
            self._place_apple()
        else:
            # Se não comeu, remove a cauda
            self.snake.pop()
            reward = 0 # Passo neutro
            
        return self._get_state_info(), reward, self.done, {"score": self.score}

    def _get_state_info(self) -> dict:
        """Retorna dados crus do estado para serem processados pelo encoder."""
        return {
            "snake": self.snake,
            "apple": self.apple,
            "direction": self.direction,
            "energy": self.energy,
            "width": self.width,
            "height": self.height,
            "initial_energy": self.initial_energy
        }

