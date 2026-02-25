import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.functions.functions_parallel import check_OpenMPI

# Interface to OpenBabel for running implemented theories (e.g. UFF)
# TODO: Move other OpenBabel functionality to this file

class OpenBabelTheory():
    def __init__(self, forcefield="UFF", chargemodel=None, label="OpenBabelTheory", printlevel=2, user_atomcharges=None):
        self.label = label
        self.printlevel = printlevel
        self.theorytype = 'QM'
        self.theorynamelabel = 'OpenBabel'
        self.forcefield=forcefield #UFF, GAFF, MMF94, Ghemical, etc. See https://openbabel.org/docs/dev/Forcefields/FF.html for options
        self.chargemodel=chargemodel #gasteiger, mmff94, qeq, qtpie. See https://openbabel.org/docs/dev/Forcefields/ChargeModels.html for options

        self.user_atomcharges=user_atomcharges
        from openbabel import openbabel as ob
        from openbabel import pybel

    def cleanup(self):
        print("No cleanup implemented")

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time = time.time()
        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        from openbabel import openbabel as ob
        from openbabel import pybel

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Create an OBMol object and populate it with the current geometry
        mol = ob.OBMol()
        for elem, coord in zip(qm_elems, current_coords):
            # Create the atom object
            atom = mol.NewAtom() 
            atomic_num = ob.GetAtomicNum(elem)
            atom.SetAtomicNum(atomic_num)
            atom.SetVector(coord[0], coord[1], coord[2])

        print("Determining connectivity and bond orders for FF...")
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

        # Turn off auto charges
        #mol.SetAutomaticPartialCharge(False)
        #mol.SetPartialChargesPerceived()
        #mol.SetAutomaticFormalCharge(False)

        def print_charges(mol):
            # Print charges for each atom
            for i in range(1, mol.NumAtoms() + 1):
                atom = mol.GetAtom(i)
                # Get charge from your dict, default to 0.0 if not found
                charge = atom.GetPartialCharge()
                print(f"Atom {i} charge: {charge}")
        def set_charges(mol,usercharges):
            # Set charges for each atom
            for i in range(1, mol.NumAtoms() + 1):
                atom = mol.GetAtom(i)
                atom.SetPartialCharge(usercharges[i-1])
        print("Initial charges in mol (before applying any charge model):")
        print_charges(mol)

        # Charge model
        if self.chargemodel is not None:
            print("Charge model is active")
            if self.user_atomcharges is not None:
                print("Setting charges to user-atomcharges")
                set_charges(mol,self.user_atomcharges)
            else:
                self.cm = ob.OBChargeModel.FindType(self.chargemodel)
                print("Computing charges using OpenBabel charge model:", self.chargemodel)
                success = self.cm.ComputeCharges(mol)
                if not success:
                    raise RuntimeError("Failed to compute charges")
            print("Charges (after applying charge model):")
            print_charges(mol)
            self.ff = ob.OBForceField.FindForceField(self.forcefield)
            self.ff.Setup(mol)
            self.ff.GetPartialCharges(mol)
            # NOTE: still not working
        else:
            self.ff = ob.OBForceField.FindForceField(self.forcefield)
            success = self.ff.Setup(mol)

        print("Computing regular FF energy:")
        self.energy = self.ff.Energy() / ash.constants.hartokj 
        print("FF energy:", self.energy)
        elec_energy = self.ff.E_Electrostatic()
        print("Electrostatic energy:", elec_energy)
        if Grad:
            self.gradient = np.zeros((len(qm_elems), 3))
            for i in range(len(qm_elems)):
                atom = mol.GetAtom(i + 1)
                f = self.ff.GetGradient(atom)
                self.gradient[i, 0] = f.GetX()*-1
                self.gradient[i, 1] = f.GetY()*-1
                self.gradient[i, 2] = f.GetZ()*-1
            self.gradient = self.gradient * 0.00020155
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Returning energy and gradient
        if Grad is True:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy, self.gradient
        # Returning energy without gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy