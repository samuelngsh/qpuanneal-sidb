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

class DBSimConnector:
    '''This class serves as a connector between the C++ physics engine connector and
    the AFMMarcus Python classes'''

    q0 = 1.602e-19
    eps0 = 8.854e-12
    k_b = 8.617e-5

    dbs = []                # list of tuples containing all dbs, (x, y)

    def parse_cml_args(self):
        parser = ArgumentParser(description="This script takes the problem "
                "file and attempts to find the ground state electron "
                "configuration on D-Wave's QPU.")
        parser.add_argument(dest="in_file", type=self.file_must_exist,
                help="Path to the problem file.",
                metavar="IN_FILE")
        parser.add_argument(dest="out_file", help="Path to the output file.",
                metavar="OUT_FILE")
        self.args = parser.parse_args()

    def file_must_exist(self, fpath):
        '''Check if input file exists for argument parser'''
        if not os.path.exists(fpath):
            raise argparse.ArgumentTypeError("{0} does not exist".format(fpath))
        return fpath

    # Import problem parameters and design from SiQAD Connector
    def init_problem(self):
        '''Read the problem from SiQADConnector.'''
        self.sqconn = siqadconn.SiQADConnector("Ground State Solver QPU", 
                self.args.in_file, self.args.out_file)

        # retrieve DBs and convert to a format that hopping model takes
        for db in self.sqconn.dbCollection():
            self.dbs.append((db.x, db.y))

        K_c = 1./(4 * np.pi * float(self.sqconn.getParameter('epsilon_r')) * self.eps0)
        debye_length = 1e-9 * float(self.sqconn.getParameter('debye_length'))

        self.dbs = np.asarray(self.dbs)
        self.db_r = distance(self.dbs, self.dbs, 'euclidean')
        # TODO distance scale
        self.V_ij = self.q0 * K_c * np.exp(-db_r/debye_length) / db_r
        # TODO deal with divide by 0

        # TODO prune potentials that are too far away (let user set threshold)

        # TODO estimate qubit resource requirement and whether further pruning is required

    def invoke_dwave_simulation(self):

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

if __name__ == "__main__":
    # TODO maybe move this to animator.py
    connector = DBSimConnector()
    connector.parse_cml_args()
    connector.init_problem()
    connector.invoke_dwave_simulation()
