from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
 
# Create a circuit with a register of 1 qubit
circ = QuantumCircuit(1, 1)
# H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
circ.h(0)
circ.measure(0, 0)
simulator = Aer.get_backend('qasm_simulator')
job = execute(circ, simulator, shots=1000)
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