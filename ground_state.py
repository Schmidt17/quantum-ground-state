import numpy as np
import matplotlib.pyplot as plt
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

def dist(a, b):
	diff = b - a
	return np.sqrt(np.sum(diff*diff))

def plot_configuration(conf, pos):
	for var_key, p in zip(conf, pos):
		if conf[var_key]:
			plt.plot([p[0]], [p[1]], 'o', color='black', markerfacecolor='blue')
		else:
			plt.plot([p[0]], [p[1]], 'o', color='black', markerfacecolor='none')
	plt.show()

N_grid = 5
N_electrons = 5
eps = .1

x = np.linspace(0., 1., N_grid)
site_positions = np.array([[x[i], x[j]] for i in range(N_grid) for j in range(N_grid)])

N_sites = len(site_positions)

connection_weights = np.zeros((N_sites, N_sites))
# connection_weights = -np.identity(N_sites)
for i in range(N_sites-1):
	for j in range(i+1, N_sites):
# 		connection_weights[i, j] = 5.5/9.
		connection_weights[i, j] = 1. / dist(site_positions[i], site_positions[j]) / eps

# connection_weights = (connection_weights + connection_weights.T)

model = dimod.BinaryQuadraticModel.from_numpy_matrix(connection_weights)
model.update(dimod.generators.combinations(model.variables, N_electrons, strength=1000.))

model.normalize()

# QPC
pause_time = 1.
# schedule = [[0.0,0.0], [50.0, 0.5], [50. + pause_time, 0.5], [100. + pause_time, 1.0]]
solver = EmbeddingComposite(DWaveSampler())
result = solver.sample(model, num_reads=10000)

# classical brute-force
# solver = dimod.ExactSolver()
# result = solver.sample(model)

print(result.lowest())
plot_configuration(result.first.sample, site_positions)