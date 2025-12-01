import os

class TrainingLogger:
    def __init__(self, log_path: str, headers: list[str]):
        self.log_path = log_path
        self.headers = headers
        
        # Criar arquivo e escrever cabeçalho se não existir
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(",".join(headers) + "\n")
                
    def log(self, data: dict):
        """
        Registra uma linha no CSV.
        data deve conter chaves correspondentes aos headers.
        """
        row = []
        for h in self.headers:
            row.append(str(data.get(h, "")))
        
        with open(self.log_path, "a") as f:
            f.write(",".join(row) + "\n")

