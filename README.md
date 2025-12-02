# 🐍 Snake AI - Evolutionary Reinforcement Learning

> Um sistema completo de IA que aprende a jogar Snake do zero, utilizando **Redes Neurais (MLP)** construídas manualmente com NumPy e otimizadas via **Algoritmos Genéticos**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Pygame](https://img.shields.io/badge/Visuals-Pygame-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Sobre o Projeto

Este projeto é uma implementação educacional e técnica de Aprendizado por Reforço Evolutivo. O diferencial é que **não utilizamos frameworks de Deep Learning** (como PyTorch, TensorFlow ou Keras). Toda a matemática da Rede Neural (Feedforward, Funções de Ativação, Manipulação de Pesos) foi implementada utilizando apenas álgebra linear com **NumPy**.

O agente "enxerga" o ambiente, processa as informações em sua rede neural e decide a próxima ação. Através da seleção natural, as cobras que jogam melhor sobrevivem e passam seus "genes" (pesos da rede) para as próximas gerações.

---

## ✨ Funcionalidades e Diferenciais

### 🧠 IA "From Scratch"
- **MLP Personalizada:** Rede Neural Feedforward implementada com multiplicação de matrizes.
- **Arquitetura Flexível:** Camadas ocultas configuráveis (Padrão: `8 -> 16 -> 12 -> 3`).
- **Ativações:** `ReLU` nas camadas ocultas e `Tanh` na saída para decisão de direção.

### 🧬 Algoritmo Genético Robusto
- **Evolução Contínua:** Seleção por torneio e elitismo (preserva os top 5%).
- **Diversidade Genética:** Operadores de Crossover e Mutação Gaussiana ajustável.
- **Heurística de Fitness Dinâmica:** O critério de sucesso muda conforme a cobra cresce:
  - *Fase Jovem:* Foco agressivo em comer maçãs.
  - *Fase Adulta:* Foco em sobrevivência, evitar becos sem saída e maximizar tempo de vida.

### 👀 Sensores Avançados (Input)
A cobra não vê a tela como nós (pixels). Ela percebe o mundo através de 8 sensores normalizados:
1.  **Perigo Imediato (3):** Paredes ou corpo à Frente, Esquerda e Direita.
2.  **Direção da Comida (1):** Ângulo relativo entre a cabeça e a maçã.
3.  **Tamanho (1):** Comprimento atual normalizado.
4.  **Instinto de Sobrevivência (3):** Utiliza o algoritmo de **Dijkstra** para calcular se existe um caminho livre até a própria cauda em cada direção possível. Isso evita que a IA entre em "becos sem saída" (espaços fechados de onde não conseguirá sair).

### 📊 Dashboard e Visualização
- **Painel em Tempo Real:** Acompanhe 9 jogos simultâneos enquanto a IA treina.
- **Gráficos:** Plotagem ao vivo da curva de aprendizado (Fitness Médio x Melhor Fitness).
- **Snapshots:** O sistema salva automaticamente o "cérebro" (modelo .npy) das melhores cobras.

---

## 📈 Resultados Recentes

O sistema demonstra convergência consistente. Em treinamentos recentes, observamos:
- **Geração 0:** Movimentos aleatórios, colisão imediata.
- **Geração 50:** Já aprende a buscar comida e evitar paredes simples.
- **Geração 150+:** Domina a estratégia de sobrevivência, circulando o mapa quando encurralada e planejando rotas.

*Exemplo de Log de Treinamento (Gen 149):*
- **Melhor Fitness:** ~14.99
- **Média da População:** ~4.48 (Crescimento constante)

---

## 🚀 Instalação e Execução

### Pré-requisitos
- Python 3.8+
- Pip

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```
*As libs principais são apenas `numpy`, `pygame`, `matplotlib` e `tqdm`.*

### 2. Treinar a IA
Para iniciar um novo experimento evolutivo:
```bash
python main_train.py
```
*Uma janela de configuração abrirá permitindo ajustar o tamanho do grid, população, velocidade, etc.*

### 3. Assistir ao Melhor Agente
Para ver o resultado final de um treinamento (substitua o arquivo pelo seu modelo gerado):
```bash
python play_best.py --model models/best_overall.npy
```

---

## 📂 Estrutura do Código

```
Snake/
├── main_train.py           # Orquestrador do treinamento
├── snake_ai/
│   ├── agents/
│   │   ├── neural_net.py   # O "cérebro" (Matemática da MLP)
│   │   └── genetic_algorithm.py # O "motor" da evolução
│   ├── env/
│   │   ├── snake_env.py    # Regras do jogo
│   │   └── state_encoding.py # Sensores (Dijkstra, Visão)
│   ├── training/
│   │   └── evaluation.py   # Função de Fitness Dinâmica
│   └── visualization/      # Dashboard Pygame e Plots
└── models/                 # Onde os .npy salvos ficam
```

---

## 🛠️ Tecnologias
- **Linguagem:** Python
- **Core Logic:** NumPy
- **Game Engine:** Pygame
- **Data Viz:** Matplotlib

---

## 📝 Licença
Este projeto foi desenvolvido para fins de estudo em Inteligência Artificial e Engenharia de Software. Sinta-se livre para usar, modificar e compartilhar!
