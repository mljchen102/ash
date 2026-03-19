import os
import glob
import shutil
import copy
import time
import itertools
#import ash
from ash.functions.functions_general import frange, BC, natural_sort, print_line_with_mainheader,print_line_with_subheader1,print_time_rel, ashexit
import ash.functions.functions_parallel
from ash.modules.module_coords import check_charge_mult, write_CIF_file, write_POSCAR_file, write_XSF_file
from ash.modules.module_results import ASH_Results
from ash.interfaces.interface_geometric_new import geomeTRICOptimizer,GeomeTRICOptimizerClass
from ash.modules.module_theory import NumGradclass


# New rewritten calc_surface function
def calc_surface(
    fragment=None, theory=None, charge=None, mult=None,
    scantype='UNRELAXED', resultfile='surface_results.txt',
    keepoutputfiles=True, keepmofiles=False,
    runmode='serial', coordsystem='dlc', maxiter=250,
    NumGrad=False, extraconstraints=None,
    convergence_setting=None, conv_criteria=None,
    subfrctor=1, force_noPBC=False,
    numcores=1, ActiveRegion=False, actatoms=None,
    PBC_format_option="CIF",
    # ---- New N-dimensional interface ----
    RC_list=None,
    # ---- Legacy 1D/2D interface (kept for backward compatibility) ----
    RC1_range=None, RC1_type=None, RC1_indices=None,
    RC2_range=None, RC2_type=None, RC2_indices=None,
):
    """Calculate an N-dimensional potential energy surface (1D, 2D, 3D, …).
 
    The preferred interface is *RC_list*, a list of reaction-coordinate dicts::
 
        RC_list=[
            {'type': 'bond',  'indices': [[0, 1]],    'range': [1.0, 2.0, 0.1]},
            {'type': 'angle', 'indices': [[0, 1, 2]], 'range': [90, 180, 10]},
        ]
 
    The legacy ``RC1_*`` / ``RC2_*`` keyword arguments continue to work unchanged.
 
    Args:
        fragment           : ASH Fragment object
        theory             : ASH Theory object
        charge, mult       : charge and multiplicity
        scantype           : 'UNRELAXED' or 'RELAXED'
        resultfile         : filename for surface results
        keepoutputfiles    : copy QM output files per point
        keepmofiles        : copy MO files per point
        runmode            : 'serial' or 'parallel'
        numcores           : number of cores for parallel mode
        coordsystem        : coordinate system for geomeTRIC
        maxiter            : max optimisation iterations
        NumGrad            : use numerical gradients
        extraconstraints   : additional constraints dict
        convergence_setting: geomeTRIC convergence preset
        conv_criteria      : explicit convergence criteria dict
        subfrctor          : subfrctor for geomeTRIC
        force_noPBC        : disable PBC in optimiser
        ActiveRegion       : use active region in optimisation
        actatoms           : list of active atoms
        PBC_format_option  : 'CIF', 'XSF', or 'POSCAR'
        RC_list            : list of RC dicts (new interface)
        RC1_*/RC2_*        : legacy 1D/2D parameters
 
    Returns:
        ASH_Results with surfacepoints dict
    """
    module_init_time = time.time()
    print_line_with_mainheader("CALC_SURFACE FUNCTION")
 
    # -- NumGrad wrapping ---------------------------------------------------
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        theory = NumGradclass(theory=theory)
 
    # -- Charge/mult check --------------------------------------------------
    charge, mult = check_charge_mult(
        charge, mult, theory.theorytype, fragment, "calc_surface", theory=theory,
    )
 
    # -- Build RC_list (legacy compat) --------------------------------------
    if RC_list is None:
        RC_list = _legacy_to_rc_list(
            RC1_type, RC1_indices, RC1_range,
            RC2_type, RC2_indices, RC2_range,
        )
    RC_list = _normalise_rc_list(RC_list)
    dimension = len(RC_list)
    print(f"Number of reaction coordinates (dimension): {dimension}")
    # -- Build value lists and total point count ----------------------------
    RC_value_lists = _build_rc_value_lists(RC_list)
    totalnumpoints = 1
    for vl in RC_value_lists:
        totalnumpoints *= len(vl)
    for i, vl in enumerate(RC_value_lists):
        print(f"RCvalue{i + 1}_list: {vl}")
    print(f"Number of surfacepoints to calculate: {totalnumpoints}")
 
    # -- Read existing results ----------------------------------------------
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary:", surfacedictionary)
 
    # -- Output-file policy -------------------------------------------------
    keepoutputfiles, keepmofiles = _silence_outputfiles_for_special_theories(
        theory, keepoutputfiles, keepmofiles,
    )
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)
 
    # -- PBC setup ----------------------------------------------------------
    if getattr(theory, "periodic", False):
        print(
            "Warning: Theory is periodic. Constrained geometry optimizations by "
            "geomeTRIC Optimizer will optimize both atom and cell parameters"
        )
        print("Set force_noPBC=True if you do not want cell-parameter optimisation.")
        print(f"PBC_format_option: {PBC_format_option}")
    convert_to_pbcfile = _select_pbc_converter(PBC_format_option)
 
    # -- Create/reset output directories ------------------------------------
    _setup_directories(theory)
 
    # -----------------------------------------------------------------------
    # PARALLEL MODE
    # -----------------------------------------------------------------------
    if runmode == 'parallel':
        print("Parallel runmode. Number of cores:", numcores)
        if numcores == 1:
            print("Error: numcores must be > 1 for parallel runmode. Exiting.")
            ashexit()
 
        surfacepointfragments_list = []
 
        if scantype.upper() == 'UNRELAXED':
            # Geometry-setting pass with ZeroTheory
            zerotheory = ash.ZeroTheory()
            pointcount = 0
            for rc_values in itertools.product(*RC_value_lists):
                pointcount += 1
                key = _point_key(rc_values)
                label = _point_label(rc_values)
                print(f"======= Surfacepoint {pointcount}/{totalnumpoints}: {label} =======")
                if key in surfacedictionary:
                    continue
                allconstraints = set_constraints_nd(RC_list, rc_values, extraconstraints)
                print("allconstraints:", allconstraints)
                geomeTRICOptimizer(
                    fragment=fragment, theory=zerotheory, maxiter=maxiter,
                    coordsystem=coordsystem, constraints=allconstraints,
                    constrainvalue=True, convergence_setting=convergence_setting,
                    conv_criteria=conv_criteria, subfrctor=subfrctor,
                    ActiveRegion=ActiveRegion, actatoms=actatoms,
                    result_write_to_disk=False, force_noPBC=force_noPBC,
                    PBC_format_option=PBC_format_option,
                )
                newfrag = copy.copy(fragment)
                newfrag.label = key
                xyzname = f"{label}.xyz"
                newfrag.write_xyzfile(xyzfilename=xyzname)
                shutil.move(xyzname, f"surface_xyzfiles/{xyzname}")
                _handle_pbc(theory, newfrag, label, convert_to_pbcfile)
                surfacepointfragments_list.append(newfrag)

            result_surface = ash.functions.functions_parallel.Job_parallel(
                fragments=surfacepointfragments_list, theories=[theory], numcores=numcores,
            )
            surfacedictionary = result_surface.energies_dict

        elif scantype.upper() == 'RELAXED':
            print("Warning: Relaxed scans in parallel mode are experimental")
            optimizer = GeomeTRICOptimizerClass(
                maxiter=maxiter, coordsystem=coordsystem,
                convergence_setting=convergence_setting, conv_criteria=conv_criteria,
                subfrctor=subfrctor, ActiveRegion=ActiveRegion, actatoms=actatoms,
                force_noPBC=force_noPBC, PBC_format_option=PBC_format_option,
            )
            pointcount = 0
            for rc_values in itertools.product(*RC_value_lists):
                pointcount += 1
                key = _point_key(rc_values)
                label = _point_label(rc_values)
                print(f"======= Surfacepoint {pointcount}/{totalnumpoints}: {label} =======")
                if key in surfacedictionary:
                    continue
                allconstraints = set_constraints_nd(RC_list, rc_values, extraconstraints)
                print("allconstraints:", allconstraints)
                newfrag = copy.copy(fragment)
                newfrag.label = key
                newfrag.constraints = allconstraints
                surfacepointfragments_list.append(newfrag)
 
            result_surface = ash.functions.functions_parallel.Job_parallel(
                fragments=surfacepointfragments_list, theories=[theory],
                numcores=numcores, Opt=True, optimizer=optimizer,
            )
            # Copy optimised XYZ files to surface_xyzfiles/
            for rc_values in itertools.product(*RC_value_lists):
                key = _point_key(rc_values)
                label = _point_label(rc_values)
                d = result_surface.worker_dirnames[key]
                shutil.copy(
                    d + "/Fragment-optimized.xyz",
                    f"surface_xyzfiles/{label}.xyz",
                )
            surfacedictionary = result_surface.energies_dict
 
        print("Parallel calculation done!")
        print("surfacedictionary:", surfacedictionary)
        if len(surfacedictionary) != totalnumpoints:
            print(
                f"Warning: Dictionary incomplete! "
                f"Got {len(surfacedictionary)}, expected {totalnumpoints}"
            )
 
    # -----------------------------------------------------------------------
    # SERIAL MODE
    # -----------------------------------------------------------------------
    elif runmode == 'serial':
        print("Serial runmode")
        zerotheory = ash.ZeroTheory()
        pointcount = 0
 
        for rc_values in itertools.product(*RC_value_lists):
            pointcount += 1
            key = _point_key(rc_values)
            label = _point_label(rc_values)
 
            print("=" * 50)
            print(f"Surfacepoint: {pointcount} / {totalnumpoints}")
            print(f"  {label}")
            if scantype.upper() == 'UNRELAXED':
                print("  Unrelaxed scan: using ZeroTheory + geomeTRIC to set geometry.")
            else:
                print("  Relaxed scan: relaxing geometry with theory + constraints.")
            print("=" * 50)
 
            if key in surfacedictionary:
                print(f"{label} already in dict. Skipping.")
                continue
 
            allconstraints = set_constraints_nd(RC_list, rc_values, extraconstraints)
            print("allconstraints:", allconstraints)
 
            if scantype.upper() == 'UNRELAXED':
                geomeTRICOptimizer(
                    fragment=fragment, theory=zerotheory, maxiter=maxiter,
                    coordsystem=coordsystem, constraints=allconstraints,
                    constrainvalue=True, convergence_setting=convergence_setting,
                    conv_criteria=conv_criteria, subfrctor=subfrctor,
                    charge=charge, mult=mult,
                    ActiveRegion=ActiveRegion, actatoms=actatoms,
                    result_write_to_disk=False, force_noPBC=force_noPBC,
                    PBC_format_option=PBC_format_option,
                )
                result = ash.Singlepoint(
                    fragment=fragment, theory=theory, charge=charge, mult=mult,
                )
 
            else:  # RELAXED
                result = geomeTRICOptimizer(
                    fragment=fragment, theory=theory, maxiter=maxiter,
                    coordsystem=coordsystem, constraints=allconstraints,
                    constrainvalue=True, convergence_setting=convergence_setting,
                    conv_criteria=conv_criteria, subfrctor=subfrctor,
                    charge=charge, mult=mult,
                    ActiveRegion=ActiveRegion, actatoms=actatoms,
                    result_write_to_disk=False, force_noPBC=force_noPBC,
                    PBC_format_option=PBC_format_option,
                )
 
            energy = float(result.energy)
            print(f"  {label}  Energy: {energy}")
 
            # -- File I/O ---------------------------------------------------
            fragment.write_xyzfile(xyzfilename="surface_traj.xyz", writemode='a')
            xyzname = f"{label}.xyz"
            fragment.write_xyzfile(xyzfilename=xyzname)
            shutil.move(xyzname, f"surface_xyzfiles/{xyzname}")
            _handle_output_files(theory, label, keepoutputfiles, keepmofiles)
            _handle_pbc(theory, fragment, label, convert_to_pbcfile)
 
            surfacedictionary[key] = float(energy)
            write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)
 
        print("surfacedictionary:", surfacedictionary)
 
    else:
        print(f"Error: Unknown runmode '{runmode}'. Use 'serial' or 'parallel'.")
        ashexit()
 
    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)
 
    # Combine all per-point XYZ files into a single trajectory
    xyzfile_list = glob.glob("surface_xyzfiles/*.xyz")

    with open("surface_traj_final.xyz", 'w') as outfile:
        for xyzfile in natural_sort(xyzfile_list):
            with open(xyzfile) as infile:
                outfile.write(infile.read())
 
    print_time_rel(module_init_time, modulename='calc_surface', moduleindex=0)
 
    result = ASH_Results(label="Surface calc", surfacepoints=surfacedictionary)
    try:
        result.write_to_disk(filename="ASH_surface.result")
    except TypeError as e:
        print("Problem writing ASH_surface.result to disk. Skipping.")
        print("Error:", e)
    return result

# FROM XYZ
def calc_surface_fromXYZ(
    xyzdir=None, multixyzfile=None, theory=None, charge=None, mult=None,
    dimension=None, resultfile='surface_results.txt',
    scantype='UNRELAXED', runmode='serial',
    coordsystem='dlc', maxiter=250, extraconstraints=None,
    convergence_setting=None, conv_criteria=None, subfrctor=1, NumGrad=False,
    numcores=None,
    keepoutputfiles=True, force_noPBC=False,
    keepmofiles=False, read_mofiles=False, mofilesdir=None,
    # New ND interface:
    RC_list=None,
    # Legacy 1D/2D interface (kept for backward compatibility):
    RC1_type=None, RC1_indices=None,
    RC2_type=None, RC2_indices=None,
):
    """Calculate an N-dimensional surface from a directory of XYZ files.

    XYZ filenames must follow the convention produced by calc_surface::

        RC1_<val1>-RC2_<val2>-...-RCN_<valN>.xyz

    RC information is only required for RELAXED scans (to rebuild constraints).
    For UNRELAXED scans all RC arguments may be omitted.

    Preferred interface uses RC_list (same format as calc_surface, but 'range'
    is ignored and may be omitted since the grid is defined by the XYZ files)::

        calc_surface_fromXYZ(
            xyzdir='surface_xyzfiles', theory=theory, charge=0, mult=1,
            scantype='Relaxed', dimension=2,
            RC_list=[
                {'type': 'bond',  'indices': [[0, 1], [0, 2]]},
                {'type': 'angle', 'indices': [[1, 0, 2]]},
            ],
        )

    Legacy 1D/2D keyword arguments (RC1_type, RC1_indices, RC2_type,
    RC2_indices) continue to work unchanged.

    Args:
        xyzdir           : directory containing XYZ files
        dimension        : number of RC coordinates; inferred from RC_list if
                           not provided, or from the first filename as fallback
        theory           : ASH Theory object
        charge, mult     : charge and multiplicity
        scantype         : 'UNRELAXED' or 'RELAXED'
        runmode          : 'serial' or 'parallel'
        numcores         : cores for parallel mode
        RC_list          : list of RC dicts (new ND interface)
        RC1_type/indices : legacy 1D/2D constraint specification
        RC2_type/indices : legacy 2D constraint specification
        read_mofiles     : read MO files from mofilesdir
        mofilesdir       : directory containing MO files

    Returns:
        ASH_Results with surfacepoints dict
    """
    module_init_time = time.time()
    print_line_with_mainheader("CALC_SURFACE_FROMXYZ FUNCTION")

    # -- NumGrad wrapping ---------------------------------------------------
    if NumGrad:
        print("NumGrad flag detected. Wrapping theory object into NumGrad class")
        theory = NumGradclass(theory=theory)

    # -- Basic argument checks ----------------------------------------------
    if charge is None or mult is None:
        print(BC.FAIL, "Error: charge and mult must be defined for calc_surface_fromXYZ", BC.END)
        ashexit()
    if xyzdir is None:
        print("Error: xyzdir must be provided")
        ashexit()
    if read_mofiles and mofilesdir is None:
        print("Error: mofilesdir not set but read_mofiles=True. Exiting.")
        ashexit()

    # -- Build RC_list from legacy kwargs if needed -------------------------
    if RC_list is None and RC1_type is not None:
        # Legacy path: build RC_list without 'range' (not needed here)
        RC_list = [{'type': RC1_type, 'indices': RC1_indices}]
        if RC2_type is not None:
            RC_list.append({'type': RC2_type, 'indices': RC2_indices})

    # Normalise indices to list-of-lists
    if RC_list is not None:
        RC_list = _normalise_rc_list(RC_list)

    # For RELAXED scans RC_list is mandatory
    if scantype.upper() == 'RELAXED' and not RC_list:
        print(
            "Error: RC_list (or legacy RC1_type/RC1_indices) is required for "
            "RELAXED scans in calc_surface_fromXYZ"
        )
        ashexit()

    # -- Discover XYZ files -------------------------------------------------
    xyzfile_list = sorted(glob.glob(xyzdir + '/*.xyz'))
    totalnumpoints = len(xyzfile_list)
    if totalnumpoints == 0:
        print(f"Found no XYZ-files in directory '{xyzdir}'. Exiting")
        ashexit()

    # -- Infer dimension ----------------------------------------------------
    if dimension is None:
        if RC_list is not None:
            dimension = len(RC_list)
        else:
            # Infer from first filename: count how many 'RC' tokens appear
            first_file = os.path.basename(xyzfile_list[0])
            dimension = first_file.replace('.xyz', '').count('RC')
        print(f"Inferred dimension={dimension}")

    print("XYZdir:", xyzdir)
    print("Theory:", theory)
    print("Dimension:", dimension)
    print("Scan type:", scantype)
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)
    print("read_mofiles:", read_mofiles)
    print("mofilesdir:", mofilesdir)
    print("runmode:", runmode)
    print("totalnumpoints:", totalnumpoints)

    # -- Read existing results ----------------------------------------------
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary:", surfacedictionary)

    if len(surfacedictionary) == totalnumpoints:
        print(
            f"Surface dictionary size {len(surfacedictionary)} matches "
            f"total number of XYZ files {totalnumpoints}. All data present."
        )
        result = ASH_Results(label="Surface calc XYZ", surfacepoints=surfacedictionary)
        result.write_to_disk(filename="ASH_surface_xyz.result")
        return result

    # -- Output-file policy -------------------------------------------------
    keepoutputfiles, keepmofiles = _silence_outputfiles_for_special_theories(
        theory, keepoutputfiles, keepmofiles,
    )
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)

    # -- Directory setup ----------------------------------------------------
    if scantype.upper() == 'RELAXED':
        if os.path.exists('surface_xyzfiles'):
            print(BC.FAIL, "surface_xyzfiles directory already exists. Please remove it.", BC.END)
            ashexit()
        os.mkdir('surface_xyzfiles')

    if runmode == 'serial':
        shutil.rmtree("surface_outfiles", ignore_errors=True)
        os.makedirs("surface_outfiles", exist_ok=True)
        shutil.rmtree("surface_mofiles", ignore_errors=True)
        os.makedirs("surface_mofiles", exist_ok=True)

    # -----------------------------------------------------------------------
    # Helper: parse RC values from filename
    # Handles filenames like RC1_1.45-RC2_90.0-RC3_0.0.xyz
    # -----------------------------------------------------------------------
    def parse_rc_values(relfile):
        base = relfile.replace('.xyz', '')
        # Split on '-RC' to get ['RC1_1.45', '2_90.0', '3_0.0']
        parts = base.split('-RC')
        vals = []
        for part in parts:
            # Each part is like 'RC1_1.45' or '2_90.0' — value is after last '_'
            vals.append(float(part.split('_')[-1]))
        return tuple(vals[:dimension])

    # -----------------------------------------------------------------------
    # Helper: build geomeTRIC constraints for a given point
    # -----------------------------------------------------------------------
    def build_constraints(rc_vals):
        if not RC_list:
            return {}
        return set_constraints_nd(RC_list, rc_vals, extraconstraints)

    # -----------------------------------------------------------------------
    # PARALLEL
    # -----------------------------------------------------------------------
    if runmode == 'parallel':
        if numcores is None:
            print("Error: numcores argument required for parallel runmode")
            ashexit()

        surfacepointfragments_list = []
        for file in xyzfile_list:
            relfile = os.path.basename(file)
            rc_vals = parse_rc_values(relfile)
            key = _point_key(rc_vals)
            if key in surfacedictionary:
                continue
            newfrag = ash.Fragment(xyzfile=file, label=key, charge=charge, mult=mult)
            if scantype.upper() == 'RELAXED':
                newfrag.constraints = build_constraints(rc_vals)
            surfacepointfragments_list.append(newfrag)

        if scantype.upper() == 'UNRELAXED':
            kwargs = dict(
                fragments=surfacepointfragments_list,
                theories=[theory],
                numcores=numcores,
            )
            if read_mofiles:
                kwargs['mofilesdir'] = mofilesdir
            results = ash.functions.functions_parallel.Job_parallel(**kwargs)

        else:  # RELAXED
            optimizer = GeomeTRICOptimizerClass(
                maxiter=maxiter, coordsystem=coordsystem,
                convergence_setting=convergence_setting, conv_criteria=conv_criteria,
                subfrctor=subfrctor, result_write_to_disk=False, force_noPBC=force_noPBC,
            )
            kwargs = dict(
                fragments=surfacepointfragments_list,
                theories=[theory],
                numcores=numcores,
                Opt=True,
                optimizer=optimizer,
            )
            if read_mofiles:
                kwargs['mofilesdir'] = mofilesdir
            results = ash.functions.functions_parallel.Job_parallel(**kwargs)

        print("Parallel calculation done!")
        surfacedictionary = {k: float(v) for k, v in results.energies_dict.items()}
        if len(surfacedictionary) != totalnumpoints:
            print(
                f"Warning: Dictionary incomplete! "
                f"Got {len(surfacedictionary)}, expected {totalnumpoints}"
            )

    # -----------------------------------------------------------------------
    # SERIAL
    # -----------------------------------------------------------------------
    elif runmode == 'serial':
        for count, file in enumerate(xyzfile_list):
            relfile = os.path.basename(file)
            rc_vals = parse_rc_values(relfile)
            key = _point_key(rc_vals)
            label = _point_label(rc_vals)

            print("=" * 66)
            print(f"Surfacepoint: {count + 1} / {totalnumpoints}")
            print(f"XYZ-file: {relfile}  ({label})")
            print("=" * 66)

            if read_mofiles:
                mofile = f"{mofilesdir}/{theory.filename}_{label}.gbw"
                print(f"Will read MO-file: {mofile}")
                if theory.__class__.__name__ == "ORCATheory":
                    theory.moreadfile = mofile

            if key in surfacedictionary:
                print(f"{label} already in dict. Skipping.")
                continue

            mol = ash.Fragment(xyzfile=file)

            if scantype.upper() == 'UNRELAXED':
                result = ash.Singlepoint(
                    theory=theory, fragment=mol, charge=charge, mult=mult,
                )

            else:  # RELAXED
                allconstraints = build_constraints(rc_vals)
                print("allconstraints:", allconstraints)
                result = geomeTRICOptimizer(
                    fragment=mol, theory=theory, maxiter=maxiter,
                    coordsystem=coordsystem, constraints=allconstraints,
                    constrainvalue=True, convergence_setting=convergence_setting,
                    conv_criteria=conv_criteria, subfrctor=subfrctor,
                    charge=charge, mult=mult, result_write_to_disk=False,
                    force_noPBC=force_noPBC,
                )
                xyzname = f"{label}.xyz"
                mol.write_xyzfile(xyzfilename=xyzname)
                shutil.move(xyzname, f"surface_xyzfiles/{xyzname}")

            energy = float(result.energy)
            print(f"Energy of {relfile}: {energy} Eh")
            _handle_output_files(theory, label, keepoutputfiles, keepmofiles)
            surfacedictionary[key] = energy
            # Write after every point so partial results are never lost
            write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)

    else:
        print(f"Error: Unknown runmode '{runmode}'. Use 'serial' or 'parallel'.")
        ashexit()

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    write_surfacedict_to_file(surfacedictionary, resultfile, dimension=dimension)
    print("Final surfacedictionary:", surfacedictionary)
    print_time_rel(module_init_time, modulename='calc_surface_fromXYZ', moduleindex=0)

    result = ASH_Results(label="Surface calc XYZ", surfacepoints=surfacedictionary)
    result.write_to_disk(filename="ASH_surface_xyz.result")
    return result



# HELPER FUNCTIONS

def read_surfacedict_from_file(resultfile, dimension=None):
    """Read surface dictionary from resultfile.

    Returns an empty dict if the file does not exist.
    Keys are tuples of floats (uniform for all dimensions).
    """
    surfacedictionary = {}
    if not os.path.isfile(resultfile):
        return surfacedictionary
    print(f"Found existing resultfile: {resultfile}. Reading entries.")
    with open(resultfile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            try:
                energy = float(tokens[-1])
                rc_vals = tuple(float(t) for t in tokens[:-1])
                if dimension == 1:
                    # Legacy: 1D keys stored as bare float in old files
                    key = rc_vals[0] if len(rc_vals) == 1 else rc_vals
                else:
                    key = rc_vals
                surfacedictionary[key] = float(energy)
            except (ValueError, IndexError):
                print(f"Warning: could not parse line: {line!r}")
    return surfacedictionary

def write_surfacedict_to_file(surfacedictionary, resultfile, dimension=None):
    """Write surface dictionary to resultfile.

    Each line: RC1_val [RC2_val ...] energy
    """
    with open(resultfile, 'w') as f:
        f.write("# Surface scan results\n")
        f.write("# RC1 [RC2 ...] Energy\n")
        print("surfacedictionary.items():", surfacedictionary.items())
        for key, energy in sorted(surfacedictionary.items()):
            if isinstance(key, tuple):
                rc_str = '  '.join(str(v) for v in key)
            else:
                rc_str = str(key)
            f.write(f"{rc_str}  {energy}\n")


# SUPPORT FUNCTIONS (not to be called by user)

def _silence_outputfiles_for_special_theories(theory, keepoutputfiles, keepmofiles):
    name = theory.__class__.__name__
    if name in ("ZeroTheory", "ORCA_CC_CBS_Theory"):
        return False, False
    return keepoutputfiles, keepmofiles

def _select_pbc_converter(PBC_format_option):
    opt = PBC_format_option.upper()
    if opt == "CIF":
        return write_CIF_file
    elif opt == "XSF":
        return write_XSF_file
    elif opt == "POSCAR":
        return write_POSCAR_file
    else:
        print(f"Warning: Unknown PBC_format_option '{PBC_format_option}', defaulting to CIF")
        return write_CIF_file

def _legacy_to_rc_list(RC1_type, RC1_indices, RC1_range,
                        RC2_type, RC2_indices, RC2_range):
    if RC1_type is None or RC1_indices is None:
        print("Error: RC1_type and RC1_indices are required")
        ashexit()
    RC_list = [{'type': RC1_type, 'indices': RC1_indices, 'range': RC1_range}]
    if RC2_type is not None:
        RC_list.append({'type': RC2_type, 'indices': RC2_indices, 'range': RC2_range})
    return RC_list

def _normalise_rc_list(RC_list):
    """Ensure every RC dict has 'indices' as a list-of-lists."""
    out = []
    for rc in RC_list:
        rc = dict(rc)  # shallow copy so we don't mutate caller's data
        indices = rc['indices']
        if not any(isinstance(el, list) for el in indices):
            indices = [indices]
        rc['indices'] = indices
        out.append(rc)
    return out

def _build_rc_value_lists(RC_list):
    """Return a list of value-lists, one per RC dimension."""
    result = []
    for rc in RC_list:
        r = rc['range']
        vals = list(frange(r[0], r[1], r[2]))
        vals.append(float(r[1]))  # always include the endpoint
        result.append(vals)
    return result

def _setup_directories(theory):
    """Create/reset the standard surface output directories."""
    for d in ("surface_xyzfiles", "surface_outfiles", "surface_mofiles"):
        shutil.rmtree(d, ignore_errors=True)
        os.mkdir(d)
    try:
        os.remove("surface_traj.xyz")
    except FileNotFoundError:
        pass
    if getattr(theory, "periodic", False):
        shutil.rmtree("surface_pbcfiles", ignore_errors=True)
        os.mkdir("surface_pbcfiles")
        print("Created directory: surface_pbcfiles")

def _point_key(rc_values):
    """Dictionary key for a surface point.
 
    A 1-tuple behaves exactly like the old scalar key for 1D surfaces,
    but we keep it as a tuple throughout so the logic is uniform.
    Callers that need the old scalar key for 1D can unpack themselves.
    """
    return tuple(rc_values)

def _point_label(rc_values):
    """Human-readable label: 'RC1_1.5-RC2_120.0-RC3_2.0' etc."""
    return '-'.join(f'RC{i + 1}_{v}' for i, v in enumerate(rc_values))

def set_constraints_nd(RC_list, rc_values, extraconstraints=None):
    """Build a geomeTRIC constraints dict for any number of reaction coordinates.
 
    Args:
        RC_list   : list of RC dicts (already normalised, indices are list-of-lists)
        rc_values : tuple of current values, one per RC
        extraconstraints : optional additional constraints dict
 
    Returns:
        dict suitable for geomeTRICOptimizer's ``constraints`` argument
    """
    allconstraints = {}
    for rc, val in zip(RC_list, rc_values):
        rc_type = rc['type']
        allconstraints.setdefault(rc_type, [])
        for indices in rc['indices']:
            allconstraints[rc_type].append([*indices, val])
    if extraconstraints:
        for k, v in extraconstraints.items():
            allconstraints.setdefault(k, []).extend(v)
    return allconstraints

def _handle_pbc(theory, fragment, pointlabel, convert_to_pbcfile):
    """Move PBC coordinate file to surface_pbcfiles/ if theory is periodic."""
    if not getattr(theory, "periodic", False):
        return
    pbcfile = convert_to_pbcfile(
        fragment.coords, fragment.elems,
        cellvectors=theory.periodic_cell_vectors,
    )
    ext = pbcfile.split('.')[-1]
    shutil.move(pbcfile, f"surface_pbcfiles/{pointlabel}.{ext}")

def _handle_output_files(theory, pointlabel, keepoutputfiles, keepmofiles):
    """Copy QM output / MO files to their surface subdirectories."""
    if not hasattr(theory, 'theorytype') or theory.theorytype != "QM":
        if keepoutputfiles or keepmofiles:
            print("Warning: For hybrid theories, outputfiles and MO-files are not kept")
        return
    if keepoutputfiles:
        try:
            shutil.copyfile(
                theory.filename + '.out',
                f'surface_outfiles/{theory.filename}_{pointlabel}.out',
            )
        except TypeError:
            print("Theory has no outputfile, probably. ignoring")
            pass
        except FileNotFoundError:
            pass
    if keepmofiles:
        try:
            shutil.copyfile(
                theory.filename + '.gbw',
                f'surface_mofiles/{theory.filename}_{pointlabel}.gbw',
            )
        except FileNotFoundError:
            pass