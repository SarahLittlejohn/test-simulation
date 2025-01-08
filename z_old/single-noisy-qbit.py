from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, pauli_error
import matplotlib.pyplot as plt
 
# Create a circuit with a register of 1 qubit
circ = QuantumCircuit(1, 1)
circ.delay(100, 0)
circ.measure(0, 0)

# define a Pauli
noise_model = NoiseModel()
bit_flip = pauli_error([('X', 0.33), ('I', 0.67)])
noise_model.add_all_qubit_quantum_error(bit_flip, ['delay'])

simulator = Aer.get_backend('qasm_simulator')
job = execute(circ, simulator, noise_model=noise_model, shots=1000)
result = job.result()
# Get the counts of the measurements (how many times 0 and 1 were observed)
counts = result.get_counts(circ)

# Plot a histogram of the results
plot_histogram(counts)
plt.show()


# # Draw the circuit
# circ.draw('mpl')
# # Show the circuit
# plt.show()