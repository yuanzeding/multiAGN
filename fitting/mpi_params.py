import time, sys
import matplotlib.pyplot as plt
import numpy as np
from sedpy.observate import load_filters
import astropy.units as u
from astropy.io import fits,ascii
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from astropy.cosmology import Planck18
import my_module

# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

KBSSpath="/disk/bifrost/yuanze/KBSS"
multiAGNpath="/disk/bifrost/yuanze/multiAGN"
stab = ascii.read(multiAGNpath+"/sources.list",format="ipac")

obj=1
sourcename="UGC2369_{}".format(obj)
field="UGC2369"


sentry = stab[stab["Field"]==field]

ra = np.mean(sentry["RA"].value)
dec = np.mean(sentry["Dec"].value)

z=sentry["z_sys"][obj-1]

run_params={'verbose': True,
            'debug': False,
            'outfile': 'demo_galphot',
            'output_pickles': False,
            # Optimization parameters
            'emcee': False,
            'dynesty': False,
            'optimize': True,
            "min_method":'lm',
            'do_powell': False,
            'ftol': 0.5e-5, 
            'maxfev': 5000,
            'do_levenberg': True,
            'nmin': 6,
            # emcee fitting parameters
            'nwalkers': 128,
            'nburn': [16, 32, 64],
            'niter': 512,
            'interval': 0.25,
            'initial_disp': 0.1,
            # dynesty Fitter parameters
            'nested_bound': 'multi',  # bounding method
            'nested_sample': 'unif',  # sampling method
            'nested_nlive_init': 100,
            'nested_nlive_batch': 100,
            'nested_bootstrap': 0,
            'nested_dlogz_init': 0.05,
            'nested_weight_kwargs': {"pfrac": 1.0},
            'nested_target_n_effective': 10000,
            # Obs data parameters
            'objid': 0,
            'phottable': 'demo_photometry.dat',
            'ldist': 1e-5,  # in Mpc
            # Model parameters
            'add_neb': True,
            'add_duste': False,
            # SPS parameters
            'zcontinuous': 1,
            }

# --------------
# Model Definition
# --------------



# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (my_module.build_obs(**kwargs), my_module.build_model(**kwargs),
            my_module.build_sps(**kwargs), build_noise(**kwargs))


if __name__ == '__main__':
    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_redshift', type=float, default=None,
                        help=("Redshift for the model"))
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    parser.add_argument('--luminosity_distance', type=float, default=1e-5,
                        help=("Luminosity distance in Mpc. Defaults to 10pc "
                              "(for case of absolute mags)"))
    parser.add_argument('--phottable', type=str, default="demo_photometry.dat",
                        help="Names of table from which to get photometry.")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")

    args = parser.parse_args()
    run_params = vars(args)


    run_params["ldist"] = Planck18.luminosity_distance(sentry["z_sys"][obj-1]).to(u.Mpc).value
    run_params["zcontinuous"] = 1
    run_params["object_redshift"] = None#sentry["z_sys"][obj-1]  letting redshift free to vary,0.0317
    run_params["fixed_metallicity"] = 0.0 #fix metalicity to solar value
    run_params["add_duste"] = False #no dust emission
    run_params["add_neb"] = True #add nebular emission through FSPS
    run_params["add_dispersion"] = True #add stellar+instrumental velocity dispersion 

    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    

    
    run_params["dynesty"] = True
    run_params["optmization"] = False
    run_params["emcee"] = False
    run_params["nested_method"] = "auto"
    run_params["nlive_init"] = 400
    run_params["nlive_batch"] = 200
    run_params["nested_dlogz_init"] = 0.05
    run_params["nested_posterior_thresh"] = 0.05
    run_params["nested_maxcall"] = int(1e7)

    print(run_params)
    print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))

    if args.debug:
        sys.exit()

    # Set up MPI. Note that only model evaluation is parallelizable in dynesty,
    # and many operations (e.g. new point proposal) are still done in serial.
    # This means that single-core fits will always be more efficient for large
    # samples. having a large ratio of (live points / processors) helps efficiency
    # Scaling is: S = K ln(1 + M/K), where M = number of processes and K = number of live points
    # Run as: mpirun -np <number of processors> python demo_mpi_params.py
    try:
        import mpi4py
        from mpi4py import MPI
        from schwimmbad import MPIPool

        mpi4py.rc.threads = False
        mpi4py.rc.recv_mprobe = False

        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        withmpi = comm.Get_size() > 1
    except ImportError:
        print('Failed to start MPI; are mpi4py and schwimmbad installed? Proceeding without MPI.')
        withmpi = False

    # Evaluate SPS over logzsol grid in order to get necessary data in cache/memory
    # for each MPI process. Otherwise, you risk creating a lag between the MPI tasks
    # caching data depending which can slow down the parallelization
    if (withmpi) & ('logzsol' in model.free_params):
        dummy_obs = dict(filters=None, wavelength=None)

        logzsol_prior = model.config_dict["logzsol"]['prior']
        lo, hi = logzsol_prior.range
        logzsol_grid = np.around(np.arange(lo, hi, step=0.1), decimals=2)

        sps.update(**model.params)  # make sure we are caching the correct IMF / SFH / etc
        for logzsol in logzsol_grid:
            model.params["logzsol"] = np.array([logzsol])
            _ = model.predict(model.theta, obs=dummy_obs, sps=sps)

    # ensure that each processor runs its own version of FSPS
    # this ensures no cross-over memory usage
    from prospect.fitting import lnprobfn
    from functools import partial
    lnprobfn_fixed = partial(lnprobfn, sps=sps)

    if withmpi:
        with MPIPool() as pool:

            # The subprocesses will run up to this point in the code
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nprocs = pool.size
            output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)
    else:
        output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn_fixed, **run_params)



    hfile = "{0}_{1}_dynesty.h5".format(args.outfile, int(time.time()))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
