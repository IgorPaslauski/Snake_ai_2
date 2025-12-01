import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_curves(csv_path: str, output_path: str) -> None:
    """
    Gera gráfico de evolução do fitness a partir do log CSV (Estático).
    """
    try:
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['generation'], df['best_fitness'], label='Best Fitness')
        plt.plot(df['generation'], df['mean_fitness'], label='Mean Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Snake AI')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráfico: {e}")

class LivePlotter:
    """
    Gerencia um gráfico Matplotlib atualizado em tempo real.
    """
    def __init__(self):
        plt.ion() # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title("Training Progress (Live)")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.grid(True)
        
        self.line_best, = self.ax.plot([], [], 'g-', label="Best Fitness")
        self.line_mean, = self.ax.plot([], [], 'b-', label="Mean Fitness")
        self.ax.legend()
        
        self.generations = []
        self.best_fits = []
        self.mean_fits = []
        
    def update(self, generation: int, best_fit: float, mean_fit: float):
        self.generations.append(generation)
        self.best_fits.append(best_fit)
        self.mean_fits.append(mean_fit)
        
        self.line_best.set_data(self.generations, self.best_fits)
        self.line_mean.set_data(self.generations, self.mean_fits)
        
        self.ax.relim()
        self.ax.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        plt.ioff()
        plt.close(self.fig)
