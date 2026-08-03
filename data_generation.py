import numpy as np
import torch

def metropolis_step(lattice, beta):
    # executes one Monte Carlo sweep over the lattice using Metropolis-Hastings
    L = lattice.shape[0]
    for _ in range(L * L):
        i, j = np.random.randint(0, L, size=2)
        s = lattice[i,j]
        # sum of nearest neighbors with periodic boundary conditions
        neighbors = (lattice[(i + 1) % L, j] + lattice[(i-1) % L, j] + lattice[i, (j+1) % L] + lattice[i, (j-1)%L])
        dE = 2 * s * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE * beta):
            lattice[i,j] = -s
    return lattice

def generate_ising_dataset(L=32, samples_per_temp=100, temps=None):
    if temps is None:
        temps = np.linspace(1.0, 3.5, 26)

    Tc = 2.269
    X_data, y_data, temp_data = [], [], []

    print("Generating Ising configurations...")
    for T in temps:
        beta = 1.0/T
        # random initial lattice
        lattice = np.random.choice([-1,1], size=(L,L))

        # thermalization sweeps
        for _ in range(500):
            lattice = metropolis_step(lattice, beta)

        # sampling sweeps
        for _ in range(samples_per_temp):
            lattice = metropolis_step(lattice, beta)
            # label: 0 for ferromagnetic, 1 for paramagnetic
            label = 0 if T < Tc else 1

            X_data.append(lattice.copy())
            y_data.append(label)
            temp_data.append(T)

    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.long)
    t_tensor = torch.tensor(np.array(temp_data), dtype=torch.float32)

    return X_tensor, y_tensor, t_tensor

if __name__ == "__main__":
    X, y, T = generate_ising_dataset(L=32, samples_per_temp=100)
    torch.save({'X': X, 'y': y, 'T':T}, 'ising_data.pt')
    print(f"Dataset saved with shape: {X.shape}")
