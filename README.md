# Snake AI Training System

Sistema completo para treinar uma Inteligência Artificial para jogar Snake (Jogo da Cobrinha) usando **Redes Neurais Artificiais (MLP)** e **Algoritmos Genéticos (GA)**. O código foi desenvolvido em Python puro com NumPy, sem frameworks de Deep Learning (como PyTorch ou TensorFlow).

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Arquitetura e Funcionamento](#arquitetura-e-funcionamento)
- [Configurações e Parâmetros](#configurações-e-parâmetros)
- [Arquivos e Módulos](#arquivos-e-módulos)

---

## 🎯 Visão Geral

Este projeto implementa um sistema de aprendizado por reforço evolutivo onde:

1. **Agentes** (cobras controladas por IA) jogam Snake em um ambiente simulado
2. Cada agente possui uma **Rede Neural (MLP)** que decide as ações baseadas no estado do jogo
3. Um **Algoritmo Genético** evolui os pesos das redes neurais através de gerações
4. Os melhores agentes são selecionados, cruzados e mutados para criar a próxima geração
5. O processo se repete até que os agentes aprendam a jogar eficientemente

### Características Principais

- ✅ **Implementação do zero**: Sem dependências de frameworks de ML
- ✅ **Visualização em tempo real**: Dashboard interativo com 9 jogos simultâneos
- ✅ **Heurística dinâmica**: Sistema de recompensas que se adapta conforme a cobra cresce
- ✅ **Interface gráfica**: Tela de configuração para ajustar hiperparâmetros facilmente
- ✅ **Logging completo**: Estatísticas salvas em CSV e gráficos de evolução

---

## 📁 Estrutura do Projeto

```
Snake/
├── snake_ai/                    # Pacote principal
│   ├── agents/                  # Agentes e algoritmos genéticos
│   │   ├── neural_net.py       # Implementação da MLP
│   │   ├── genome.py           # Funções de manipulação de genomas
│   │   └── genetic_algorithm.py # Algoritmo genético
│   ├── env/                    # Ambiente do jogo
│   │   ├── snake_env.py        # Lógica do jogo Snake
│   │   └── state_encoding.py   # Codificação do estado para a rede
│   ├── training/               # Lógica de treinamento
│   │   └── evaluation.py       # Função de fitness e avaliação
│   ├── utils/                  # Utilitários
│   │   ├── launcher.py         # Tela de configuração (Tkinter)
│   │   ├── logger.py           # Logger CSV
│   │   └── paths.py            # Definição de diretórios
│   └── visualization/         # Visualização
│       ├── dashboard.py        # Dashboard unificado (Pygame)
│       ├── plots.py            # Gráficos estáticos e dinâmicos
│       ├── board_snapshots.py  # Snapshots do tabuleiro
│       └── live_view.py        # Visualizador simples (legado)
├── main_train.py               # Script principal de treinamento
├── play_best.py                # Script para assistir melhor agente
├── requirements.txt            # Dependências Python
└── README.md                   # Este arquivo
```

**Diretórios gerados durante o treinamento:**
- `models/` - Genomas salvos (melhor de cada geração + melhor global)
- `logs/` - Arquivos CSV com estatísticas de treinamento
- `plots/` - Gráficos de evolução do fitness
- `snapshots/` - Imagens estáticas do tabuleiro em gerações específicas

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passos

1. **Clone o repositório** (ou baixe os arquivos):
   ```bash
   git clone <url-do-repositorio>
   cd Snake
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

   As dependências incluem:
   - `numpy` - Operações matemáticas e arrays
   - `matplotlib` - Gráficos e visualizações
   - `pygame` - Renderização gráfica e dashboard
   - `tqdm` - Barras de progresso no terminal

---

## 🎮 Como Usar

### 1. Treinamento

Para iniciar o treinamento, execute:

```bash
python main_train.py
```

**O que acontece:**

1. **Tela de Configuração**: Uma janela Tkinter será aberta onde você pode configurar:
   - Número de gerações
   - Tamanho da população
   - Dimensões do tabuleiro (largura x altura)
   - Energia inicial (atualiza automaticamente baseado no tamanho do grid)
   - Taxa de mutação
   - Se a cobra cresce ao comer
   - Se deseja visualizar o dashboard ao vivo
   - Velocidade de visualização (FPS)

2. **Inicialização**: O sistema cria os diretórios necessários (`models/`, `logs/`, `plots/`, `snapshots/`)

3. **Loop de Treinamento**: Para cada geração:
   - Avalia todos os agentes da população
   - Calcula fitness (baseado em maçãs comidas e sobrevivência)
   - Salva o melhor genoma da geração e o melhor global
   - Atualiza logs CSV
   - Renderiza dashboard (se habilitado) mostrando os top 9 agentes
   - Gera snapshots estáticos a cada 50 gerações
   - Evolui a população (seleção, crossover, mutação)

4. **Finalização**: Ao terminar (ou ao fechar o dashboard), gera um gráfico final de evolução do fitness

**Interrupção:**
- Pressione `Ctrl+C` no terminal para interromper o treinamento
- Feche a janela do dashboard para parar o treinamento graciosamente

### 2. Visualização do Melhor Agente

Para assistir o melhor agente treinado jogando:

```bash
python play_best.py
```

**Opções:**
- `--model`: Caminho para o arquivo de modelo (padrão: `models/best_overall.npy`)
- `--speed`: Velocidade do jogo em FPS (padrão: 10)

**Exemplo:**
```bash
python play_best.py --model models/best_gen_0100.npy --speed 15
```

---

## 🧠 Arquitetura e Funcionamento

### 1. Ambiente do Jogo (`SnakeEnv`)

O ambiente simula o jogo Snake em um grid 2D:

- **Estado**: Posições da cobra (cabeça + corpo), posição da maçã, direção atual, energia restante
- **Ações**: 0 (virar esquerda), 1 (seguir reto), 2 (virar direita)
- **Regras**:
  - A cobra se move 1 célula por passo
  - Cresce ao comer uma maçã (se `grow_on_eat=True`)
  - Morre ao colidir com parede ou com seu próprio corpo
  - Sistema de energia: começa com `width * height`, reseta ao comer, morre se chegar a 0
- **Recompensas**:
  - +10 por comer maçã
  - -1 por morrer
  - 0 para passos neutros

### 2. Codificação de Estado (`encode_state`)

O estado do jogo é convertido em um vetor numérico de **6 entradas**:

1. **Perigo (3 valores)**: Flags binárias indicando colisão iminente à frente, direita e esquerda
2. **Maçã (2 valores)**:
   - Ângulo relativo à direção atual da cabeça (normalizado entre -1 e 1)
   - Distância normalizada (0 a 1)
3. **Tamanho (1 valor)**: Comprimento atual da cobra normalizado pelo tamanho máximo do grid

### 3. Rede Neural (`NeuralNetwork`)

**Arquitetura MLP:**
- **Entrada**: 6 neurônios (estado codificado)
- **Camadas Ocultas**: [16, 12] neurônios com ativação **ReLU**
- **Saída**: 3 neurônios (scores para cada ação) com ativação **Tanh**
- **Decisão**: Ação com maior score (`argmax`)

**Métodos principais:**
- `forward(state)`: Propagação para frente, retorna scores das ações
- `forward_debug(state)`: Versão que retorna ativações de todas as camadas (para visualização)
- `get_weights_flat()`: Retorna todos os pesos e biases como vetor 1D (genoma)
- `set_weights_flat(genome)`: Define pesos e biases a partir de um genoma

### 4. Algoritmo Genético (`GeneticAlgorithm`)

**Ciclo evolutivo:**

1. **Inicialização**: População de genomas aleatórios (pesos da rede)
2. **Avaliação**: Cada genoma é testado no jogo e recebe um fitness
3. **Seleção**: Os melhores são selecionados (elitismo + torneio)
4. **Crossover**: Genomas são cruzados (uniforme ou single-point)
5. **Mutação**: Pesos são modificados com ruído gaussiano
6. **Nova Geração**: Processo se repete

**Hiperparâmetros:**
- **Elitismo**: 5% da população (melhores são preservados)
- **Taxa de Mutação**: Configurável (padrão: 0.1)
- **Desvio Padrão da Mutação**: 0.2
- **Tipo de Crossover**: Uniforme (cada peso vem de um dos pais aleatoriamente)

### 5. Função de Fitness (`evaluate_genome`)

**Heurística Dinâmica:**

A função de fitness se adapta conforme a cobra cresce:

- **Fase 1 (Cobra Pequena)**: Foco em comer
  - Recompensa alta por maçãs (100 pontos por maçã)
  - Recompensa baixa por passos (0.1 por passo)
  - Penalidade forte se morrer sem comer nada (-50)

- **Fase 2 (Cobra Grande)**: Foco em sobrevivência
  - Recompensa alta por maçãs (200 pontos por maçã)
  - Recompensa alta por passos (2.0 por passo)
  - Penalidade severa por colisão com o corpo (-500)
  - Penalidade por colisão com parede (-100)
  - Bônus contínuo se a cauda for alcançável (+0.5 por passo)
  - Penalidade se a cauda não for alcançável (-0.5 por passo)

**Avaliação:**
- Cada genoma é testado em 3 episódios
- Fitness final = média dos fitness dos episódios

---

## ⚙️ Configurações e Parâmetros

### Configurações do Ambiente

Definidas na tela de configuração ou em `main_train.py`:

```python
ENV_CONFIG = {
    "width": 10,              # Largura do tabuleiro
    "height": 10,             # Altura do tabuleiro
    "initial_energy": 100,    # Energia inicial (padrão: width * height)
    "grow_on_eat": True       # Se a cobra cresce ao comer
}
```

### Configurações do Algoritmo Genético

```python
POPULATION_SIZE = 150        # Tamanho da população
GENERATIONS = 1000           # Número de gerações
MUTATION_RATE = 0.1         # Taxa de mutação (0-1)
MUTATION_STD = 0.2           # Desvio padrão do ruído gaussiano
ELITISM = 5% da população    # Quantos melhores preservar
```

### Configurações da Rede Neural

```python
LAYER_SIZES = [6, 16, 12, 3]  # [Input, Hidden1, Hidden2, Output]
```

### Configurações de Treinamento

```python
EPISODES_PER_EVAL = 3        # Episódios por avaliação de genoma
SNAPSHOT_INTERVAL = 50       # Intervalo para salvar snapshots
```

---

## 📄 Arquivos e Módulos

### Scripts Principais

#### `main_train.py`
Script principal que orquestra todo o processo de treinamento:
- Abre tela de configuração
- Inicializa ambiente, rede neural e algoritmo genético
- Loop principal: avaliação → evolução → logging → visualização
- Salva modelos e gera gráficos finais

#### `play_best.py`
Script para visualizar o melhor agente jogando:
- Carrega genoma salvo
- Cria ambiente e rede neural
- Renderiza jogo em tempo real com Pygame

### Módulos do Pacote `snake_ai`

#### `env/snake_env.py`
**Classe `SnakeEnv`**: Implementa o ambiente do jogo
- `reset()`: Reinicia o jogo para estado inicial
- `step(action)`: Executa uma ação e retorna (estado, recompensa, done, info)
- `_get_state_info()`: Retorna dicionário com informações do estado atual
- `is_tail_reachable()`: Verifica se a cabeça pode alcançar a cauda (BFS)

#### `env/state_encoding.py`
**Função `encode_state(env)`**: Converte estado do jogo em vetor numérico
- Calcula perigos (colisões iminentes)
- Calcula posição relativa da maçã (ângulo e distância)
- Normaliza valores para o intervalo adequado

#### `agents/neural_net.py`
**Classe `NeuralNetwork`**: Implementação manual de MLP
- Inicialização de pesos (He initialization)
- Forward pass com ReLU (ocultas) e Tanh (saída)
- Métodos para serializar/deserializar pesos (genoma)

#### `agents/genome.py`
Funções auxiliares para manipulação de genomas:
- `create_random_genome(size)`: Cria genoma aleatório
- `mutate_genome(genome, rate, std)`: Aplica mutação gaussiana
- `crossover_uniform(parent1, parent2)`: Crossover uniforme
- `crossover_single_point(parent1, parent2)`: Crossover single-point

#### `agents/genetic_algorithm.py`
**Classe `GeneticAlgorithm`**: Implementa o algoritmo genético
- `__init__()`: Inicializa população aleatória
- `get_population()`: Retorna população atual
- `evolve(fitness_scores)`: Executa um ciclo evolutivo completo

#### `training/evaluation.py`
**Função `evaluate_genome()`**: Avalia fitness de um genoma
- Cria ambiente e rede neural
- Executa múltiplos episódios
- Calcula fitness com heurística dinâmica
- Retorna fitness médio

#### `utils/launcher.py`
**Classe `ConfigScreen`**: Interface gráfica de configuração (Tkinter)
- Campos para todos os hiperparâmetros
- Validação de entradas
- Atualização automática de energia baseada no tamanho do grid
- Retorna dicionário com configurações

#### `utils/logger.py`
**Classe `TrainingLogger`**: Logger CSV simples
- `log(dict)`: Adiciona linha ao CSV
- Cabeçalhos automáticos

#### `utils/paths.py`
Define diretórios padrão:
- `MODELS_DIR = "models/"`
- `LOGS_DIR = "logs/"`
- `PLOTS_DIR = "plots/"`
- `SNAPSHOTS_DIR = "snapshots/"`
- `create_directories()`: Cria todos os diretórios se não existirem

#### `visualization/dashboard.py`
**Classe `DashboardRenderer`**: Dashboard unificado em Pygame
- **Área Esquerda**: Grid 3x3 mostrando 9 jogos simultâneos
- **Área Direita Superior**: Visualização da rede neural (nós e conexões)
- **Área Direita Inferior**: Gráfico de fitness em tempo real
- Janela redimensionável
- Atualização em tempo real durante treinamento

#### `visualization/plots.py`
Funções para geração de gráficos:
- `plot_training_curves(csv_path, output_path)`: Gera gráfico estático de fitness
- `LivePlotter`: Classe para gráficos interativos (não usada no dashboard atual)

#### `visualization/board_snapshots.py`
**Função `save_generation_snapshot()`**: Salva imagem estática do tabuleiro
- Renderiza tabuleiro com cobra e maçã
- Salva em `snapshots/gen_XXXX.png`

#### `visualization/live_view.py`
**Função `play_episode()`**: Visualizador simples de um episódio
- Renderiza jogo em tempo real
- Controles: ESC para sair
- Usado por `play_best.py`

---

## 📊 Resultados e Logs

### Arquivos Gerados

Durante o treinamento, os seguintes arquivos são criados:

1. **`models/best_gen_XXXX.npy`**: Melhor genoma de cada geração
2. **`models/best_overall.npy`**: Melhor genoma de todas as gerações
3. **`logs/training_YYYYMMDD-HHMMSS.csv`**: Estatísticas de treinamento
   - Colunas: `generation`, `best_fitness`, `mean_fitness`, `min_fitness`
4. **`plots/fitness_curve_YYYYMMDD-HHMMSS.png`**: Gráfico de evolução do fitness
5. **`snapshots/gen_XXXX.png`**: Imagens do tabuleiro em gerações específicas

### Interpretando os Resultados

- **Fitness Crescente**: Indica que os agentes estão melhorando
- **Fitness Estagnado**: Pode indicar convergência ou necessidade de ajustar hiperparâmetros
- **Dashboard**: Permite observar comportamento em tempo real e identificar padrões

---

## 🔧 Troubleshooting

### Erro: "ModuleNotFoundError"
- Certifique-se de que todas as dependências estão instaladas: `pip install -r requirements.txt`

### Erro: "FileNotFoundError" ao carregar modelo
- Execute `main_train.py` primeiro para gerar modelos
- Verifique se o caminho do modelo está correto

### Dashboard não abre ou trava
- Reduza a velocidade de visualização (FPS)
- Desabilite o dashboard e use apenas logs/gráficos
- Verifique se o Pygame está instalado corretamente

### Performance lenta
- Reduza o tamanho da população
- Reduza o número de gerações para testes
- Desabilite o dashboard ao vivo
- Reduza o número de episódios por avaliação

---

## 📝 Notas Adicionais

- O código foi desenvolvido para fins educacionais e demonstração de conceitos de IA
- A implementação é intencionalmente simples e didática (sem frameworks de ML)
- Para melhor performance, considere usar frameworks como PyTorch ou TensorFlow
- O sistema de heurística dinâmica pode ser ajustado conforme necessário

---

## 📄 Licença

Este projeto é fornecido como está, para fins educacionais.

---

**Desenvolvido com ❤️ usando Python, NumPy e Algoritmos Genéticos**
