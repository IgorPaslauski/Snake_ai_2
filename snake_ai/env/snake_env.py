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
    """

    def __init__(self, width: int = 10, height: int = 10, initial_energy: int | None = None, grow_on_eat: bool = True):
        self.width = width
        self.height = height
        self.initial_energy = initial_energy if initial_energy is not None else width * height
        self.grow_on_eat = grow_on_eat
        self.reset()

    def reset(self) -> dict:
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
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                self.apple = (x, y)
                break

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        if self.done:
            return self._get_state_info(), 0, True, {"score": self.score}

        self.steps += 1
        self.energy -= 1
        
        # Atualizar direção (0=Esquerda, 1=Reto, 2=Direita)
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)
        
        if action == 0: # Esquerda
            new_idx = (idx - 1) % 4
            self.direction = clock_wise[new_idx]
        elif action == 2: # Direita
            new_idx = (idx + 1) % 4
            self.direction = clock_wise[new_idx]
        
        # Calcular nova posição
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
        
        # Colisão (Parede ou Corpo)
        # Verificar colisão com parede primeiro
        if (x < 0 or x >= self.width or y < 0 or y >= self.height):
            self.done = True
            reward = -10.0
            return self._get_state_info(), reward, self.done, {"score": self.score, "reason": "wall_collision"}
        # Verificar colisão com corpo
        elif new_head in self.snake[:-1]:
            self.done = True
            reward = -30.0
            return self._get_state_info(), reward, self.done, {"score": self.score, "reason": "body_collision"}
        
        # Energia
        if self.energy <= 0:
            self.done = True
            reward = -1.0
            return self._get_state_info(), reward, self.done, {"score": self.score, "reason": "starvation"}

        # Mover
        self.snake.insert(0, new_head)
        
        # Comer Maçã
        if new_head == self.apple:
            self.score += 1
            reward = 10.0
            self.energy = self.initial_energy
            self._place_apple()
            
            # Se NÃO deve crescer, remove a cauda mesmo comendo
            if not self.grow_on_eat:
                self.snake.pop()
        else:
            # Se não comeu, remove cauda
            self.snake.pop()
            reward = 0
            
        return self._get_state_info(), reward, self.done, {"score": self.score}

    def _get_state_info(self) -> dict:
        return {
            "snake": self.snake,
            "apple": self.apple,
            "direction": self.direction,
            "energy": self.energy,
            "width": self.width,
            "height": self.height,
            "initial_energy": self.initial_energy
        }
