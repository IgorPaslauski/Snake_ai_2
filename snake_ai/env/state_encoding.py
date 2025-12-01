import numpy as np
import math
from .snake_env import SnakeEnv, Direction

def encode_state(env: SnakeEnv) -> np.ndarray:
    """
    Converte o estado atual do ambiente em um vetor de entrada para a rede neural.
    Usando coordenadas polares (distância e ângulo) para a maçã e visão egocêntrica.
    
    Inputs:
    1. Perigo (3 valores): Frente, Direita, Esquerda.
    2. Maçã (2 valores): Distância (normalizada), Ângulo (relativo à cabeça, normalizado).
    3. Energia (1 valor): Normalizada.
    
    Total: 3 + 2 + 1 = 6 inputs.
    """
    
    state_info = env._get_state_info()
    snake = state_info["snake"]
    head = snake[0]
    apple = state_info["apple"]
    direction = state_info["direction"]
    width = state_info["width"]
    height = state_info["height"]
    energy = state_info["energy"]
    initial_energy = state_info["initial_energy"]

    # Helper para verificar colisão em um ponto arbitrário
    def is_collision(pt):
        x, y = pt
        if x < 0 or x >= width or y < 0 or y >= height:
            return True
        if pt in snake: # Colisão com corpo
            return True
        return False

    # Coordenadas ao redor da cabeça
    point_l = (head[0] - 1, head[1])
    point_r = (head[0] + 1, head[1])
    point_u = (head[0], head[1] - 1)
    point_d = (head[0], head[1] + 1)

    # Vetores de direção atual
    # UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)
    dir_vector = (0, 0)
    
    if direction == Direction.UP:
        dir_vector = (0, -1)
        pt_fwd = point_u
        pt_right = point_r
        pt_left = point_l
    elif direction == Direction.RIGHT:
        dir_vector = (1, 0)
        pt_fwd = point_r
        pt_right = point_d
        pt_left = point_u
    elif direction == Direction.DOWN:
        dir_vector = (0, 1)
        pt_fwd = point_d
        pt_right = point_l
        pt_left = point_r
    elif direction == Direction.LEFT:
        dir_vector = (-1, 0)
        pt_fwd = point_l
        pt_right = point_u
        pt_left = point_d

    # 1. Perigo (3 valores)
    danger = [
        1.0 if is_collision(pt_fwd) else 0.0,
        1.0 if is_collision(pt_right) else 0.0,
        1.0 if is_collision(pt_left) else 0.0
    ]

    # 2. Maçã Polar (Distância e Ângulo)
    # Vetor Cabeça -> Maçã
    apple_vec_x = apple[0] - head[0]
    apple_vec_y = apple[1] - head[1]
    
    # Distância Euclidiana
    dist = math.sqrt(apple_vec_x**2 + apple_vec_y**2)
    max_dist = math.sqrt(width**2 + height**2)
    norm_dist = dist / max_dist # 0 a 1
    
    # Ângulo
    # Ângulo da maçã em relação ao eixo X global
    angle_apple = math.atan2(apple_vec_y, apple_vec_x)
    # Ângulo da direção atual em relação ao eixo X global
    angle_head = math.atan2(dir_vector[1], dir_vector[0])
    
    # Diferença (ângulo relativo)
    angle_diff = angle_apple - angle_head
    
    # Normalizar para [-PI, PI]
    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff <= -math.pi:
        angle_diff += 2 * math.pi
        
    # Normalizar para [-1, 1]
    norm_angle = angle_diff / math.pi
    
    # 3. Energia
    energy_norm = [energy / initial_energy]

    # Concatenar: [Danger(3), Dist(1), Angle(1), Energy(1)]
    state_vector = np.array(danger + [norm_dist, norm_angle] + energy_norm, dtype=np.float32)
    
    return state_vector
