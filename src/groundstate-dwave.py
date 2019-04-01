#!/usr/bin/env python
# encoding: utf-8

'''
Parse the SiQADConnector ouput and prepare for D-Wave QPU simulation.
'''

__author__      = 'Samuel Ng'
__copyright__   = 'Apache License 2.0'
__version__     = '0.1'
__date__        = '2019-02-31'  # last update

from argparse import ArgumentParser
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

        self.annealing_time = int(sq_param('annealing_time'))
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
        self.V_ij_pruned = self.V_ij
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

        #print(self.linear)
        #print(self.quadratic)

        self.edgelist = dict(self.linear)
        self.edgelist.update(self.quadratic)

        print(self.edgelist)

    def invoke_solver(self):
        '''Invoke D-Wave's solver using the problem defined in this class. In 
        the future, add user options for using local classical solver rather 
        than D-Wave's QPU.'''

        import networkx as nx
        import dwave_networkx as dnx
        import matplotlib.pyplot as plt
        import minorminer
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import FixedEmbeddingComposite

        dwave_sampler = DWaveSampler()
        target_edgelist = dwave_sampler.edgelist

        embedding = minorminer.find_embedding(self.edgelist, target_edgelist)
        print(embedding)
        dnx.draw_chimera(embedding)
        #G = dnx.chimera_graph(m=7, n=6, t=4)
        #dnx.draw_chimera(G)
        plt.show()
        
        #sampler = FixedEmbeddingComposite(dwave_sampler, embedding)
        #self.response = sampler.sample_qubo(self.edgelist,
        #        annealing_time=self.annealing_time, num_reads=self.repeat_count)
        #
        #for datum in self.response.data(['sample', 'energy', 'num_occurrences']):
        #    print(datum.sample, datum.energy, "Occurrences: ", datum.num_occurrences)

    def invoke_classical_solver(self):
        '''Invoke D-Wave's classical solver.'''
        from dwave_qbsolv import QBSolv

        self.response_classical = QBSolv().sample_qubo(self.edgelist, 
                num_repeats=self.repeat_count)

        for datum in self.response_classical.data(['sample', 'energy', 'num_occurrences']):
            print(datum.sample, datum.energy, "Occurrences: ", datum.num_occurrences)

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
        charges = [int(c) for c in charge_config]
        
        for i in range(len(charges)):
            for j in range(i+1,len(charges)):
                E += self.V_ij[i][j] * charges[i] * charges[j]

        return E

    ## Run simulation
    #def runSimulation(self):
    #    '''Run the simulation'''

    #    # check simulation type ('animation' or 'line_scan')
    #    if (self.sqconn.getParameter('simulation_type') == 'line_scan'):
    #        self.runLineScan()
    #    else:
    #        self.runAnimation()

    #def runLineScan(self):
    #    # for now, only 1D line scan is supported, all y values will be discarded
    #    # TODO 2D support
    #    X = []
    #    for dbloc in self.dbs:
    #        X.append(dbloc[0])
    #    X.sort()
    #    print(X)

    #    # call the AFM simulation
    #    self.afm = AFMLine(X)
    #    self.afm.setScanType(int(self.sqconn.getParameter('scan_type')),
    #            float(self.sqconn.getParameter('write_strength')))
    #    self.afm.setBias(float(self.sqconn.getParameter('bias')))
    #    self.afm.run(Nel=int(self.sqconn.getParameter('num_electrons')),
    #            nscans=int(self.sqconn.getParameter('num_scans')),
    #            pad=[int(self.sqconn.getParameter('lattice_padding_l')),
    #                int(self.sqconn.getParameter('lattice_padding_r'))]
    #            )

    #def runAnimation(self):
    #    import sys

    #    model = HoppingModel(self.dbs, self.sqconn.getParameter('hopping_model'))
    #    model.fixElectronCount(int(self.sqconn.getParameter('num_electrons')))

    #    model.addChannel('bulk')
    #    model.addChannel('clock', enable=False)
    #    model.addChannel('tip', enable=False)

    #    app = QApplication(sys.argv)
    #    mw = MainWindow(model)

    #    mw.show()
    #    mw.animator.start()
    #    sys.exit(app.exec_())


def parse_cml_args():
    '''Parse command-line arguments.'''

    def file_must_exist(fpath):
        '''Local method checking if input file exists for argument parser.'''
        if not os.path.exists(fpath):
            raise argparse.ArgumentTypeError('{0} does not exist'.format(fpath))
        return fpath

    parser = ArgumentParser(description='This script takes the problem '
            'file and attempts to find the ground state electron '
            'configuration on D-Wave\'s QPU.')
    parser.add_argument(dest='in_file', type=file_must_exist,
            help='Path to the problem file.',
            metavar='IN_FILE')
    parser.add_argument(dest='out_file', help='Path to the output file.',
            metavar='OUT_FILE')
    return parser.parse_args()

if __name__ == '__main__':
    cml_args = parse_cml_args()
    gs_qpu = GroundStateQPU(cml_args.in_file, cml_args.out_file)
    print('Classical solver')
    gs_qpu.invoke_classical_solver()
    print('QPU solver')
    gs_qpu.invoke_solver()
    #gs_qpu.export_results()
