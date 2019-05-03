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
import itertools
import time

import siqadconn

class GroundStateQPU:
    '''Attempt to find the ground state electron configuration of the given DB 
    configuration.'''

    q0 = 1.602e-19
    eps0 = 8.854e-12
    k_b = 8.617e-5

    #dbs = []                # list of tuples containing all dbs, (x, y)

    def __init__(self, eng_name, in_file, out_file, verbose=False):
        self.in_file = in_file
        self.out_file = out_file
        self.verbose = verbose
        self.cpu_time = None
        self.timing_info = {}

        self.sqconn = siqadconn.SiQADConnector(eng_name, self.in_file, self.out_file)

        self.precalculations()

    # Import problem parameters and design from SiQAD Connector
    def precalculations(self):
        '''Retrieve variables from SiQADConnector and precompute handy 
        variables.'''

        print('Performing pre-calculations...')
        sq_param = lambda key : self.sqconn.getParameter(key)

        self.repeat_count = int(sq_param('repeat_count'))

        # retrieve DBs and convert to a format that hopping model takes
        db_scale = 1e-10            # DB locations given in angstrom
        dbs = [(round(db.x,2)*db_scale, round(db.y,2)*db_scale) for db in self.sqconn.dbCollection()]
        dbs = np.asarray(dbs)

        # retrieve and process simulation parameters
        K_c = 1./(4 * np.pi * float(sq_param('epsilon_r')) * self.eps0)
        debye_length = float(sq_param('debye_length'))
        debye_length *= 1e-9        # debye_length given in nm

        # precompute distances and inter-DB potentials
        db_r = distance.cdist(dbs, dbs, 'euclidean')
        d_threshold = 1e-9 * float(sq_param('d_threshold'))
        self.v_ij = np.divide(self.q0 * K_c * np.exp(-db_r/debye_length), 
                db_r, out=np.zeros_like(db_r), where=db_r!=0)
        self.v_ij_pruned = np.copy(self.v_ij)
        if d_threshold > 0:
            self.v_ij_pruned[db_r>d_threshold] = 0  # prune elements past distance threshold
        if self.verbose:
            print('v_ij=\n{}'.format(self.v_ij))
            print('v_ij_pruned=\n{}'.format(self.v_ij_pruned))

        # local potentials
        self.mu = float(sq_param('global_v0'))
        self.V_local = np.ones(len(dbs)) * -1 * self.mu

        # TODO estimate qubit resource requirement and whether further pruning is required

        # create graph for QPU
        self.linear = {}
        self.quadratic = {}
        for i in range(len(db_r)):
            key_i = 'db{}'.format(i)
            self.linear[(key_i, key_i)] = self.V_local[i]
            for j in range(i+1,len(db_r[0])):
                if self.v_ij_pruned[i][j] != 0:
                    key_j = 'db{}'.format(j)
                    self.quadratic[(key_i, key_j)] = self.v_ij_pruned[i][j]

        self.edgelist = dict(self.linear)
        self.edgelist.update(self.quadratic)

        if self.verbose:
            print(self.edgelist)

        print('Pre-calculations complete.')

    def invoke_solver(self, timing_info_out_path=None, embedding_plot_path=None, 
            embedding_in_path=None, embedding_out_path=None):
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

        print('Embedding problem...')

        dwave_sampler = DWaveSampler()
        target_edgelist = dwave_sampler.edgelist

        # import embedding if import file specified, else embed using minorminer
        self.embedding_time = None
        embedding = None
        if embedding_in_path != None:
            print('Loading embedding from {}'.format(embedding_in_path))
            embedding = json.load(embedding_in_path)
        else:
            print('Attempting to embed problem to QPU...')
            # TODO might want to time the embedding function call
            time_start = time.process_time()
            embedding = minorminer.find_embedding(self.edgelist, target_edgelist)
            self.embedding_time = time.process_time() - time_start

        # export embedding if export file specified
        if embedding_out_path != None:
            with open(embedding_out_path, 'w') as outfile:
                print('Exporting embedding to {}.'.format(embedding_out_path))
                json.dump(embedding, outfile)

        # Load edges from structure of available solver and plot embedding
        if embedding_plot_path != None:
            print('Generating embedding graph.')
            plt.figure(figsize=(16,16))
            T_nodelist, T_edgelist, T_adjacency = dwave_sampler.structure
            G = dnx.chimera_graph(16,node_list=T_nodelist)
            dnx.draw_chimera_embedding(G, embedding, node_size=8, cmap='rainbow')
            plt.savefig(embedding_plot_path)

        print('Embedding complete.')

        # plot 3x3 Chimera graph
        #plt.figure(1, figsize=(20,20))
        #G = dnx.chimera_graph(3,3,4)
        #dnx.draw_chimera(G)
        #plt.show()

        print('Invoking QPU...')

        annealing_time = int(self.sqconn.getParameter('annealing_time'))
        
        # Invoke simulation
        sampler = FixedEmbeddingComposite(dwave_sampler, embedding)
        self.response = sampler.sample_qubo(self.edgelist,
                annealing_time=annealing_time, num_reads=self.repeat_count)

        print('QPU finished.')

        # Dump response information (mostly timing related information)
        self.timing_info['time_s_cpu_minorminer'] = self.embedding_time
        self.timing_info['time_us_qpu'] = self.response.info['timing']
        if timing_info_out_path != None:
            with open(timing_info_out_path, 'w') as outfile:
                json.dump(self.timing_info, outfile)

        # Print results
        if self.verbose:
            for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
                print(datum.sample, datum.energy, 'Occurrences: ', datum.num_occurrences)

    def invoke_classical_solver(self):
        '''Invoke D-Wave's classical solver.'''
        from dwave_qbsolv import QBSolv

        time_start = time.process_time()
        self.response = QBSolv().sample_qubo(self.edgelist, 
                num_repeats=self.repeat_count)
        self.timing_info['time_s_cpu_qbsolv'] = time.process_time() - time_start
        print('CPU time: {}'.format(self.cpu_time))

        if self.verbose:
            for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
                print(datum.sample, datum.energy, 'Occurrences: ', datum.num_occurrences)

    def export_results(self):
        '''Export QPU simultion results to SiQADConnector.'''

        def flatten(d, parent_key='', sep='_'):
            '''Flatten multi-layer dictionary.
            From: https://stackoverflow.com/questions/6027558/
            '''
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)


        print('Exporting results...')
        # DB locations
        dblocs = []
        for db in self.sqconn.dbCollection():
            dblocs.append((str(db.x), str(db.y)))
        self.sqconn.export(db_loc=dblocs)

        # charge configurations
        charge_configs = []
        for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
            charge_config = ''
            for charge in datum.sample.values():
                charge_config += str(charge)
            charge_configs.append([charge_config, 
                str(self.system_energy(charge_config)), 
                str(datum.num_occurrences),
                str(self.physically_valid(charge_config))])
        if self.verbose:
            print('Export charge configurations:')
            print(charge_configs)
        self.sqconn.export(db_charge=charge_configs)

        # export timing information
        if self.timing_info:
            # flatten timing information (esp. QPU response)
            flattened_timing_info = flatten(self.timing_info)
            export_timing_info = []
            for key, val in flattened_timing_info.items():
                export_timing_info.append([key, val])
            self.sqconn.export(misc=export_timing_info)

        #if self.cpu_time is not None:
        #    self.sqconn.export(misc=[ ['cpu_time', self.cpu_time] ])

        print('Export complete.')

    def system_energy(self, charge_config):
        '''Return the system energy of the given charge configuration 
        accounting for all Coulombic interactions.'''

        charges = np.asarray([int(c) for c in charge_config])
        return .5 * np.inner(charges, np.dot(self.v_ij, charges))

    def physically_valid(self, charge_config):
        '''Return whether the configuration is physically valid.'''

        charges = np.asarray([int(c) for c in charge_config])

        # check if all sites meet energy constraints
        for i in range(len(charge_config)):
            v_local = 0
            #v_local -= self.v_ext[i]   # TODO add v_ext support
            for j in range(len(charge_config)):
                if i==j:
                    continue
                v_local += self.v_ij[i][j] * charges[j]

            if (charges[i] == 1 and v_local > self.mu) or \
                    (charges[i] == 0 and v_local < self.mu):
                # constraints not met
                if self.verbose:
                    print('Config {} is invalid, failed at index {} with v_local={}'
                            .format(charge_config, i, v_local))
                return 0

        print('Config {} is valid'.format(charge_config))
        return 1
            

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

    if cml_args.solver == 'qpu':
        print('QPU solver')
        gs_qpu = GroundStateQPU('QPUAnneal', cml_args.in_file, cml_args.out_file)
        embedding_plot_path = os.path.join(os.path.dirname(cml_args.out_file), 
                'embedding.pdf')
        timing_info_path = os.path.join(os.path.dirname(cml_args.out_file),
                'timing_info.json')
        gs_qpu.invoke_solver(
                timing_info_out_path=timing_info_path,
                embedding_plot_path=embedding_plot_path,
                embedding_in_path=cml_args.embedding_in_file, 
                embedding_out_path=cml_args.embedding_out_file)
        gs_qpu.export_results()
    elif cml_args.solver == 'classical':
        print('Classical solver')
        gs_qpu = GroundStateQPU('QPUAnneal Classical Solver', cml_args.in_file, cml_args.out_file)
        gs_qpu.invoke_classical_solver()
        gs_qpu.export_results()
    else:
        raise ValueError('Unknown solver name {}'.format(cml_args.solver))

