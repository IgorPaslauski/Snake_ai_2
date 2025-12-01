import tkinter as tk
from tkinter import ttk

class ConfigScreen:
    def __init__(self):
        self.config = {}
        self.root = tk.Tk()
        self.root.title("Snake AI Training Config")
        self.root.geometry("400x550")
        
        # Style
        style = ttk.Style()
        style.configure("TLabel", padding=5, font=("Arial", 10))
        style.configure("TButton", padding=5, font=("Arial", 10, "bold"))
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Configurações de Treinamento", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Generations
        self.create_entry(main_frame, "Gerações:", "generations", "1000")
        
        # Population
        self.create_entry(main_frame, "Tamanho da População:", "population_size", "150")
        
        # Grid Size
        self.create_entry(main_frame, "Largura do Tabuleiro:", "width", "10")
        self.create_entry(main_frame, "Altura do Tabuleiro:", "height", "10")
        
        # Vincular atualização da energia ao mudar tamanho do grid
        self.entry_width.bind("<KeyRelease>", self.update_energy)
        self.entry_height.bind("<KeyRelease>", self.update_energy)
        
        # Energy
        self.create_entry(main_frame, "Energia Inicial:", "initial_energy", "100")
        
        # Mutation
        self.create_entry(main_frame, "Taxa de Mutação (0-1):", "mutation_rate", "0.1")
        
        # Checkbox Grow
        self.grow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Cobra Cresce ao Comer?", variable=self.grow_var).pack(pady=10, anchor="w")
        
        # Checkboxes Viz
        self.live_dash_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Visualizar Dashboard ao Vivo?", variable=self.live_dash_var).pack(pady=5, anchor="w")
        
        # FPS Slider
        ttk.Label(main_frame, text="Velocidade de Visualização (FPS):").pack(anchor="w")
        self.fps_scale = ttk.Scale(main_frame, from_=10, to=120, orient=tk.HORIZONTAL)
        self.fps_scale.set(50)
        self.fps_scale.pack(fill=tk.X, pady=5)
        
        # Start Button
        ttk.Button(main_frame, text="Iniciar Treinamento", command=self.on_start).pack(pady=20, fill=tk.X)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_entry(self, parent, label_text, key, default_val):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, pady=5)
        
        ttk.Label(container, text=label_text).pack(side=tk.LEFT)
        entry = ttk.Entry(container, width=10)
        entry.insert(0, default_val)
        entry.pack(side=tk.RIGHT)
        
        # Store entry ref using key
        setattr(self, f"entry_{key}", entry)

    def update_energy(self, event=None):
        """Atualiza automaticamente a energia baseada no tamanho do grid."""
        try:
            w = int(self.entry_width.get() or 0)
            h = int(self.entry_height.get() or 0)
            if w > 0 and h > 0:
                new_energy = w * h
                self.entry_initial_energy.delete(0, tk.END)
                self.entry_initial_energy.insert(0, str(new_energy))
        except ValueError:
            pass # Ignora se não for número enquanto digita

    def on_start(self):
        try:
            self.config = {
                "generations": int(self.entry_generations.get()),
                "population_size": int(self.entry_population_size.get()),
                "width": int(self.entry_width.get()),
                "height": int(self.entry_height.get()),
                "initial_energy": int(self.entry_initial_energy.get()),
                "mutation_rate": float(self.entry_mutation_rate.get()),
                "grow_on_eat": self.grow_var.get(),
                "live_dashboard": self.live_dash_var.get(),
                "fps": int(self.fps_scale.get())
            }
            self.root.quit() # Para o mainloop mas mantém a janela até destroy
        except ValueError:
            tk.messagebox.showerror("Erro", "Por favor, insira valores válidos.")

    def on_close(self):
        self.config = None # Sinaliza cancelamento
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        if self.config:
            self.root.destroy()
        return self.config

