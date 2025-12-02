import numpy as np
import math
import heapq
from .snake_env import SnakeEnv, Direction

def get_dijkstra_distance(start: tuple, goal: tuple, obstacles: set, width: int, height: int) -> float:
    """
    Calcula a distância do caminho mais curto até o objetivo usando Dijkstra.
    Retorna:
    - 1.0: Se existe caminho (inverso da distância normalizada)
    - 0.0: Se não existe caminho
    """
    if start == goal:
        return 1.0
        
    # Verificar limites e colisão inicial
    if start[0] < 0 or start[0] >= width or start[1] < 0 or start[1] >= height:
        return 0.0
    if start in obstacles:
        # Se o objetivo for a própria cauda, e 'start' for a cauda, é válido.
        # Mas obstacles geralmente contém a cauda.
        # Vamos assumir que obstacles NÃO deve conter o 'goal' se ele for acessível.
        if start != goal:
            return 0.0
        
    queue = [(0, start)]
    visited = {start: 0}
    
    while queue:
        cost, current = heapq.heappop(queue)
        
        if cost > visited[current]:
            continue
            
        if current == goal:
            # Retorna inverso da distância normalizada
            return 1.0 / max(1.0, cost)
            
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            new_cost = cost + 1
            
            if 0 <= nx < width and 0 <= ny < height:
                if neighbor not in obstacles or neighbor == goal:
                    if neighbor not in visited or new_cost < visited[neighbor]:
                        visited[neighbor] = new_cost
                        heapq.heappush(queue, (new_cost, neighbor))
                    
    return 0.0

def encode_state(env: SnakeEnv) -> np.ndarray:
    """
    Converte o estado atual do ambiente em um vetor de entrada para a rede neural.
    Versão com Perigo, Ângulo, Tamanho e Dijkstra para Cauda (8 Inputs):
    
    1. Perigo (3 valores): Frente, Direita, Esquerda.
    2. Maçã (1 valor): Ângulo relativo à cabeça.
    3. Tamanho (1 valor): Comprimento atual normalizado.
    4. Cauda (3 valores): Dijkstra para a cauda em cada direção (sobrevivência).
    
    Total: 3 + 1 + 1 + 3 = 8 inputs.
    """
    
    state_info = env._get_state_info()
    snake = state_info["snake"]
    head = snake[0]
    apple = state_info["apple"]
    direction = state_info["direction"]
    width = state_info["width"]
    height = state_info["height"]

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

    # 2. Maçã (Apenas Ângulo)
    apple_vec_x = apple[0] - head[0]
    apple_vec_y = apple[1] - head[1]
    
    # Ângulo
    angle_apple = math.atan2(apple_vec_y, apple_vec_x)
    angle_head = math.atan2(dir_vector[1], dir_vector[0])
    angle_diff = angle_apple - angle_head
    
    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff <= -math.pi:
        angle_diff += 2 * math.pi
        
    norm_angle = angle_diff / math.pi
    
    # 3. Tamanho da Cobra (Normalizado)
    max_len = width * height
    norm_len = len(snake) / max_len
    
    # 4. Caminho para Cauda (Dijkstra) - Sobrevivência
    # A cauda é um alvo móvel, mas alcançar a posição atual da cauda é uma boa heurística de segurança
    tail = snake[-1]
    
    # Obstáculos: corpo da cobra, exceto a cauda (que se move)
    obstacles = set(snake[:-1]) 
    
    # Calcular caminho para a cauda a partir de cada direção
    tail_path_fwd = get_dijkstra_distance(pt_fwd, tail, obstacles, width, height)
    tail_path_right = get_dijkstra_distance(pt_right, tail, obstacles, width, height)
    tail_path_left = get_dijkstra_distance(pt_left, tail, obstacles, width, height)
    
    # Atualizar PERIGO para incluir "sem saída" (se não alcança a cauda)
    # Se já é colisão (1.0) ou se não tem caminho para cauda (0.0), vira Perigo=1.0
    if tail_path_fwd == 0.0: danger[0] = 1.0
    if tail_path_right == 0.0: danger[1] = 1.0
    if tail_path_left == 0.0: danger[2] = 1.0
    
    tail_paths = [tail_path_fwd, tail_path_right, tail_path_left]
    
    # Concatenar: [Danger(3), Angle(1), Size(1), TailPath(3)]
    state_vector = np.array(danger + [norm_angle, norm_len] + tail_paths, dtype=np.float32)
    
    return state_vector
