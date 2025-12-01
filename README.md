# Snake AI Training System

Este projeto implementa um sistema completo para treinar uma Inteligência Artificial para jogar Snake (Jogo da Cobrinha) usando Redes Neurais (MLP) e Algoritmos Genéticos (GA). O código foi desenvolvido em Python puro com NumPy, sem frameworks de Deep Learning (como PyTorch ou TensorFlow).

## Estrutura do Projeto

```
snake_ai/
  agents/          # Implementação da Rede Neural e Algoritmo Genético
  env/             # Lógica do jogo Snake e codificação de estado
  training/        # Funções de avaliação e loop de treino
  utils/           # Loggers e helpers de caminhos
  visualization/   # Gráficos e visualização Pygame
main_train.py      # Script principal de treinamento
play_best.py       # Script para assistir o melhor agente jogar
requirements.txt   # Dependências
```

## Instalação

1. Clone o repositório.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como Usar

### 1. Treinamento

Para iniciar o treinamento, execute:

```bash
python main_train.py
```

Isso irá:
- Criar pastas `models/`, `logs/`, `plots/`, `snapshots/`.
- Rodar o Algoritmo Genético por N gerações (configurável no script).
- Salvar o melhor genoma de cada geração e o melhor global em `models/`.
- Gerar logs CSV em `logs/` e gráficos de evolução em `plots/`.
- Salvar imagens (snapshots) dos tabuleiros jogados em `snapshots/`.

### 2. Visualização

Para assistir o melhor agente treinado jogando em tempo real:

```bash
python play_best.py
```

Opções:
- `--model`: Caminho para o arquivo de modelo (padrão: `models/best_overall.npy`).
- `--speed`: Velocidade do jogo em FPS (padrão: 10).

## Detalhes Técnicos

- **Ambiente**: Grid 10x10 (configurável), inputs normalizados.
- **Rede Neural**: MLP (Input -> ReLU -> Hidden -> ReLU -> Hidden -> Tanh -> Output).
- **Algoritmo Genético**:
  - Seleção por Torneio.
  - Crossover Uniforme.
  - Mutação Gaussiana.
  - Elitismo.
- **Fitness**: Baseado principalmente em maçãs comidas, com bônus para sobrevivência.

## Configuração

Você pode alterar hiperparâmetros (tamanho da população, taxa de mutação, camadas da rede) editando as variáveis no início de `main_train.py`.

