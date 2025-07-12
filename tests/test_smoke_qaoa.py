import numpy as np, pytest
from src.qaoa.hamiltonians import maxcut_cost_hamiltonian

def test_build():
    W = np.zeros((4,4))
    cost = maxcut_cost_hamiltonian(W)
    assert cost.num_qubits == 4
