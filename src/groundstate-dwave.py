#!/usr/bin/env python
# encoding: utf-8

'''
Parse the SiQADConnector ouput and prepare for D-Wave QPU simulation.
'''

__author__      = 'Samuel Ng'
__copyright__   = 'Apache License 2.0'
__version__     = '0.1'
__date__        = '2019-02-31'  # last update

import argparse
import os.path
import numpy as np
from scipy.spatial import distance

import siqadconn

class GroundStateQPU:
    '''Attempt to find the ground state electron configuration of the given DB 
    configuration.'''

    q0 = 1.602e-19
    eps0 = 8.854e-12
    k_b = 8.617e-5

    #dbs = []                # list of tuples containing all dbs, (x, y)

    def __init__(self, in_file, out_file):
        self.in_file = in_file
        self.out_file = out_file

        self.sqconn = siqadconn.SiQADConnector('Ground State Solver QPU', 
                self.in_file, self.out_file)

        self.precalculations()

    # Import problem parameters and design from SiQAD Connector
    def precalculations(self):
        '''Retrieve variables from SiQADConnector and precompute handy 
        variables.'''

        sq_param = lambda key : self.sqconn.getParameter(key)

        self.repeat_count = int(sq_param('repeat_count'))

        # retrieve DBs and convert to a format that hopping model takes
        db_scale = 1e-10            # DB locations given in angstrom
        dbs = [(db.x*db_scale, db.y*db_scale) for db in self.sqconn.dbCollection()]
        dbs = np.asarray(dbs)

        # retrieve and process simulation parameters
        K_c = 1./(4 * np.pi * float(sq_param('epsilon_r')) * self.eps0)
        debye_length = float(sq_param('debye_length'))
        debye_length *= 1e-9        # debye_length given in nm

        # precompute distances and inter-DB potentials
        db_r = distance.cdist(dbs, dbs, 'euclidean')
        d_threshold = 1e-9 * float(sq_param('d_threshold'))
        self.V_ij = np.divide(self.q0 * K_c * np.exp(-db_r/debye_length), 
                db_r, out=np.zeros_like(db_r), where=db_r!=0)
        self.V_ij_pruned = np.copy(self.V_ij)
        if d_threshold > 0:
            self.V_ij_pruned[db_r>d_threshold] = 0  # prune elements past distance threshold
        print('V_ij=\n{}'.format(self.V_ij))
        print('V_ij_pruned=\n{}'.format(self.V_ij_pruned))

        # local potentials
        self.V_local = np.ones(len(dbs)) * -1 * float(sq_param('global_v0'))

        # TODO estimate qubit resource requirement and whether further pruning is required

        # create graph for QPU
        self.linear = {}
        self.quadratic = {}
        for i in range(len(db_r)):
            key_i = 'db{}'.format(i)
            self.linear[(key_i, key_i)] = self.V_local[i]
            for j in range(i+1,len(db_r[0])):
                if self.V_ij_pruned[i][j] != 0:
                    key_j = 'db{}'.format(j)
                    self.quadratic[(key_i, key_j)] = self.V_ij_pruned[i][j]

        self.edgelist = dict(self.linear)
        self.edgelist.update(self.quadratic)

        print(self.edgelist)

    def invoke_solver(self, embedding_in_path=None, embedding_out_path=None):
        '''Invoke D-Wave's solver using the problem defined in this class. In 
        the future, add user options for using local classical solver rather 
        than D-Wave's QPU.'''

        import networkx as nx
        import dwave_networkx as dnx
        import matplotlib.pyplot as plt
        import minorminer
        import json
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import FixedEmbeddingComposite

        dwave_sampler = DWaveSampler()
        target_edgelist = dwave_sampler.edgelist

        # import embedding if import file specified, else embed using minorminer
        embedding = None
        if embedding_in_path != None:
            embedding = json.load(embedding_in_path)
        else:
            print('Attempting to embed problem to QPU...')
            embedding = minorminer.find_embedding(self.edgelist, target_edgelist)

        # export embedding if export file specified
        if embedding_out_path != None:
            with open(embedding_out_path, 'w') as outfile:
                json.dump(embedding, outfile)

        # Load edges from structure of available solver
        T_nodelist, T_edgelist, T_adjacency = dwave_sampler.structure
        G = dnx.chimera_graph(16,node_list=T_nodelist)
        dnx.draw_chimera_embedding(G, embedding, node_size=8)
        plt.show()

        # plot 3x3 Chimera graph
        #plt.figure(1, figsize=(20,20))
        #G = dnx.chimera_graph(3,3,4)
        #dnx.draw_chimera(G)
        #plt.show()

        annealing_time = int(self.sqconn.getParameter('annealing_time'))
        
        # Invoke simulation
        sampler = FixedEmbeddingComposite(dwave_sampler, embedding)
        self.response = sampler.sample_qubo(self.edgelist,
                annealing_time=annealing_time, num_reads=repeat_count)
        
        # Print results
        for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
            print(datum.sample, datum.energy, 'Occurrences: ', datum.num_occurrences)

    def invoke_classical_solver(self):
        '''Invoke D-Wave's classical solver.'''
        from dwave_qbsolv import QBSolv

        self.response = QBSolv().sample_qubo(self.edgelist, 
                num_repeats=self.repeat_count)

        for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
            print(datum.sample, datum.energy, 'Occurrences: ', datum.num_occurrences)

    def export_results(self):
        '''Export QPU simultion results to SiQADConnector.'''
        dblocs = []
        for db in self.sqconn.dbCollection():
            dblocs.append((str(db.x), str(db.y)))
        print(dblocs)

        charge_configs = []
        for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
            charge_config = ''
            for charge in datum.sample.values():
                charge_config += str(charge)
            charge_configs.append([charge_config, str(self.system_energy(charge_config)), str(datum.num_occurrences)])
        print(charge_configs)

        self.sqconn.export(db_loc=dblocs)
        self.sqconn.export(db_charge=charge_configs)

    def system_energy(self, charge_config):
        '''Return the system energy of the given charge configuration 
        accounting for all Coulombic interactions.'''

        E = 0.
        charges = np.asarray([int(c) for c in charge_config])
        return .5 * np.inner(charges, np.dot(self.V_ij, charges))
        
        #for i in range(len(charges)):
        #    for j in range(i+1,len(charges)):
        #        E += self.V_ij[i][j] * charges[i] * charges[j]

        #return E

def parse_cml_args():
    '''Parse command-line arguments.'''

    def file_must_exist(fpath):
        '''Local method checking if input file exists for argument parser.'''
        if not os.path.exists(fpath):
            raise argparse.ArgumentTypeError('{0} does not exist'.format(fpath))
        return fpath

    parser = argparse.ArgumentParser(description='This script takes the problem '
            'file and attempts to find the ground state electron '
            'configuration on D-Wave\'s QPU.')
    parser.add_argument(dest='in_file',
            type=file_must_exist,
            help='Path to the problem file.',
            metavar='IN_FILE')
    parser.add_argument(dest='out_file', 
            help='Path to the output file.',
            metavar='OUT_FILE')
    parser.add_argument('--solver',
            dest='solver', 
            default='qpu', 
            const='qpu', 
            nargs='?',
            choices=['qpu', 'classical'], 
            help='Specify which solver to use.')
    parser.add_argument('--import-embedding', 
            dest='embedding_in_file',
            type=argparse.FileType('r'), 
            help='Path to the input file for embedding import.', 
            metavar='EMBEDDING_IMPORT_PATH')
    parser.add_argument('--export-embedding', 
            dest='embedding_out_file',
            help='Path to the output file for embedding export.',
            metavar='EMBEDDING_EXPOERT_PATH')
    return parser.parse_args()

if __name__ == '__main__':
    cml_args = parse_cml_args()
    gs_qpu = GroundStateQPU(cml_args.in_file, cml_args.out_file)

    if cml_args.solver == 'qpu':
        print('QPU solver')
        gs_qpu.invoke_solver(cml_args.embedding_in_file, cml_args.embedding_out_file)
    elif cml_args.solver == 'classical':
        print('Classical solver')
        gs_qpu.invoke_classical_solver()
    else:
        raise ValueError('Unknown solver name {}'.format(cml_args.solver))

    gs_qpu.export_results()
