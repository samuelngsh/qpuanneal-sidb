from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# Set the problem for Huff's OR gate
linear = {
        ('x0', 'x0'): -0.27,
        ('x1', 'x1'): -0.27,
        ('x2', 'x2'): -0.27,
        ('x3', 'x3'): -0.27,
        ('x4', 'x4'): -0.27,
        ('x5', 'x5'): -0.27,
        ('x6', 'x6'): -0.27
        }
quadratic =  {
        ('x0', 'x1'): 0.190507,
        ('x0', 'x2'): 0.0651353,
        ('x0', 'x3'): 0.0452766,
        ('x0', 'x4'): 0.0533672,
        ('x0', 'x5'): 0.0376638,
        ('x0', 'x6'): 0.018388,
        ('x1', 'x2'): 0.123117,
        ('x1', 'x3'): 0.0651353,
        ('x1', 'x4'): 0.106198,
        ('x1', 'x5'): 0.0651353,
        ('x1', 'x6'): 0.027172,
        ('x2', 'x3'): 0.190507,
        ('x2', 'x4'): 0.106198,
        ('x2', 'x5'): 0.0651353,
        ('x2', 'x6'): 0.027172,
        ('x3', 'x4'): 0.0533672,
        ('x3', 'x5'): 0.0376638,
        ('x3', 'x6'): 0.018388,
        ('x4', 'x5'): 0.287115,
        ('x4', 'x6'): 0.061307,
        ('x5', 'x6'): 0.102661
        }

Q = dict(linear)
Q.update(quadratic)

response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=1000)

for datum in response.data(['sample', 'energy', 'num_occurrences']):
    print(datum.sample, datum.energy, "Occurrences: ", datum.num_occurrences)

#for sample, energy, num_occurrences in response.data():
#    print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences)
