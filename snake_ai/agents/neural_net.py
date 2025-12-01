import numpy as np

class NeuralNetwork:
    """
    Rede Neural Artificial (MLP) para o agente Snake.
    Arquitetura configurável, mas fixa em: Input -> [Hidden Layers] -> Output.
    Ativação: ReLU nas ocultas, Tanh na saída.
    """

    def __init__(self, layer_sizes: list[int]):
        """
        Args:
            layer_sizes (list[int]): Lista com tamanhos das camadas. 
                                     Ex: [10, 16, 8, 3] (10 inputs, 2 hidden, 3 outputs).
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Inicialização aleatória dos pesos e biases
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            
            # He initialization para ReLU (ou Xavier para Tanh/Sigmoid, mas vamos simplificar com normal)
            scale = np.sqrt(2.0 / n_in)
            W = np.random.randn(n_in, n_out) * scale
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Realiza a passagem direta (forward pass).
        Args:
            x (np.ndarray): Vetor de entrada (shape: (input_size,) ou (1, input_size)).
        Returns:
            np.ndarray: Vetor de saída (scores das ações).
        """
        # Garantir que x seja (1, input_size)
        if x.ndim == 1:
            a = x.reshape(1, -1)
        else:
            a = x
            
        # Propagação pelas camadas ocultas
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            
        # Camada de saída (última)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self.tanh(z_out)
        
        return output.flatten()

    def forward_debug(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Realiza o forward pass e retorna as ativações de cada camada para visualização.
        """
        activations = []
        
        if x.ndim == 1:
            a = x.reshape(1, -1)
        else:
            a = x
            
        activations.append(a.flatten()) # Input
        
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            activations.append(a.flatten()) # Hidden
            
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = self.tanh(z_out)
        activations.append(output.flatten()) # Output
        
        return output.flatten(), activations

    def get_weights_flat(self) -> np.ndarray:
        """Retorna todos os pesos e biases concatenados em um vetor 1D."""
        flat_params = []
        for W, b in zip(self.weights, self.biases):
            flat_params.append(W.flatten())
            flat_params.append(b.flatten())
        return np.concatenate(flat_params)

    def set_weights_flat(self, genome: np.ndarray) -> None:
        """Reconstroi pesos e biases a partir de um vetor 1D (genoma)."""
        start = 0
        new_weights = []
        new_biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i+1]
            
            # Extrair Pesos W
            w_size = n_in * n_out
            end = start + w_size
            W = genome[start:end].reshape(n_in, n_out)
            start = end
            
            # Extrair Bias b
            b_size = n_out
            end = start + b_size
            b = genome[start:end].reshape(1, n_out)
            start = end
            
            new_weights.append(W)
            new_biases.append(b)
            
        self.weights = new_weights
        self.biases = new_biases

