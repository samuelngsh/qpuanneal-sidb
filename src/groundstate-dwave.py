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

        # retrieve DBs and convert to a format that hopping model takes
        dbs = []
        db_scale = 1e-10    # DB locations given in angstrom
        for db in self.sqconn.dbCollection():
            dbs.append((db.x*db_scale, db.y*db_scale))

        # retrieve and process simulation parameters
        K_c = 1./(4 * np.pi * float(self.sqconn.getParameter('epsilon_r')) * self.eps0)
        debye_length = float(self.sqconn.getParameter('debye_length'))
        debye_length *= 1e-9 # debye_length given in nm

        # precompute distances and inter-DB potentials
        dbs = np.asarray(dbs)
        db_r = distance.cdist(dbs, dbs, 'euclidean')
        self.V_ij = np.divide(self.q0 * K_c * np.exp(-db_r/debye_length), 
                db_r, out=np.zeros_like(db_r), where=db_r!=0)
        print('V_ij=\n{}'.format(self.V_ij))

        # local potentials
        self.V_local = np.ones_like(dbs) * -1 * float(self.sqconn.getParameter('global_v0'))

        # TODO prune potentials that are too far away (let user set threshold)

        # TODO estimate qubit resource requirement and whether further pruning is required

    def invoke_solver(self):
        '''Invoke D-Wave's solver using the problem defined in this class. In 
        the future, add user options for using local classical solver rather 
        than D-Wave's QPU.'''

        print('To be implemented.')

        return

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
    gs_qpu.invoke_solver()
