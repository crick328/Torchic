import torch
import torch.nn as nn
import torch.optim as optim

# Definizione della rete neurale
class ScalarNet(nn.Module):
    def __init__(self):
        super(ScalarNet, self).__init__()
        self.fc1 = nn.Linear(1, 16)  # Primo layer, input di dimensione 1
        self.fc2 = nn.Linear(16, 32) # Secondo layer
        self.fc3 = nn.Linear(32, 1)  # Output layer, di dimensione 1 per un problema di regressione

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Nessuna attivazione per l'output finale
        return x

# Inizializzazione della rete
model = ScalarNet()

# Parametri di addestramento
criterion = nn.MSELoss()  # Funzione di perdita
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dati di esempio
# Supponiamo di avere dei dati di input scalari e valori target
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)  # input scalari
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)  # target per una semplice relazione lineare

# Ciclo di addestramento
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()               # Resetta i gradienti
    outputs = model(inputs)              # Forward pass
    loss = criterion(outputs, targets)   # Calcola la loss
    loss.backward()                      # Backward pass
    optimizer.step()                     # Aggiorna i pesi

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Test del modello con un nuovo input scalare
test_input = torch.tensor([[5.0]], dtype=torch.float32)
test_output = model(test_input)
print("Output per input 5.0:", test_output.item())
