import torch.nn as nn

# Modelo 1: Red simple - 1 capa oculta pequeña
# Baseline, para ver si el problema es resolvible con poca capacidad (1 capa oculta)
class MLP_Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Modelo 2: Red mediana - 2 capas ocultas con Dropout
# Se agrega otra capa y regularizacion para reducir overfitting (el dropout de 30%)
class MLP_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Modelo 3: Red profunda - 3 capas ocultas con BatchNorm y Dropout
# Mayor capacidad expresiva, usando BatchNorm se estabiliza el entrenamiento
class MLP_Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)