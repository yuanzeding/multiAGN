import numpy as np
import h5py

import ppxf.ppxf_util as util

import astropy.units as u
from astropy import constants
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from astropy.cosmology import Planck18

import os, sys
os.environ["SPS_HOME"]="/disk/bifrost/yuanze/software/fsps"


from prospect.fitting import fit_model,lnprobfn


KBSSpath="/disk/bifrost/yuanze/KBSS"
multiAGNpath="/disk/bifrost/yuanze/multiAGN"
stab = ascii.read(multiAGNpath+"/sources.list",format="ipac")

obj=1
sourcename="UGC2369_{}".format(obj)
field="UGC2369"


sentry = stab[stab["Field"]==field]

ra = np.mean(sentry["RA"].value)
dec = np.mean(sentry["Dec"].value)
sc=SkyCoord(ra=ra,dec=dec,unit="deg")

z=sentry["z_sys"][obj-1]

def build_obs(snr=10, ldist=10.0, **extras):
    """Build a dictionary of observational data.  In this example 
    the data consist of photometry for UGC 2369
    
    :param snr:
        The S/N assumed to the photometry
        
    :param ldist:
        The luminosity distance to assume for translating absolute magnitudes 
        into apparent magnitudes.
        
    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    from prospect.utils.obsutils import fix_obs
    import sedpy

    # The obs dictionary, empty for now
    obs = {}

    # These are the names of the relevant filters, 
    # in the same order as the photometric data (see below)
    #galex = ['galex_FUV', 'galex_NUV']
    spitzer = ['spitzer_irac_ch'+n for n in ['1','2','3','4']]
    #sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]
    filternames = spitzer
    # And here we instantiate the `Filter()` objects using methods in `sedpy`,
    # and put the resultinf list of Filter objects in the "filters" key of the `obs` dictionary
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # In this example we use a row of absolute AB magnitudes from Johnson et al. 2013 (NGC4163)
    # We then turn them into apparent magnitudes based on the supplied `ldist` meta-parameter.
    # You could also, e.g. read from a catalog.
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    #M_AB = np.array([-11.93, -12.37, -13.37, -14.22, -14.61, -14.86, 
    #                 -14.94, -14.09, -13.62, -13.23, -12.78])
    #dm = 25 + 5.0 * np.log10(ldist)
    #mags = M_AB + dm
    #1 maggie is the flux density in Janskys divided by 3631
    obs["maggies"] = None#10**(-0.4*mags)

    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the supplied `snr` meta-parameter.
    obs["maggies_unc"] = None#(1./snr) * obs["maggies"]

    # Now we need a mask, which says which flux values to consider in the likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit, 
    # and *False* for values you want to ignore.  Here we mask the spitzer bands.
    obs["phot_mask"] = None#np.array(['spitzer' in f.name for f in obs["filters"]])

    # This is an array of effective wavelengths for each of the filters.  
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = None#np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    specinstru="MUSE"#,"OSIRIS"]
    
    datapath=os.path.join(multiAGNpath,field)
    path_out = os.path.join(datapath, '{}_1D'.format(specinstru))
    
    data = np.loadtxt(os.path.join(path_out, "{}_nuclear_spec_obj{}.txt".format(specinstru,obj)))
    lam = data[:,0]  # OBS wavelength [A]
    flux = data[:,1]*1e-20  # OBS flux [erg/s/cm^2/A]
    err = data[:,2]*1e-20  # 1 sigma error
    
    specmaggies = 3.33564095e4*flux*lam**2 / 3631
    specunc = 3.33564095e4*err*lam**2 / 3631
    
    obs["wavelength"] = lam
    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = specmaggies
    # (spectral uncertainties are given here)
    obs['unc'] = specunc
    # (again, to ignore a particular wavelength set the value of the 
    #  corresponding elemnt of the mask to *False*)
    mask0 = util.determine_mask(np.log(lam), [lam[0],lam[-1]], z, width=1000)
    #zf=1+z
    mask = (specmaggies>0.001*np.median(specmaggies))  &((lam<5400)| (lam>6120)) & mask0#&((lam<6500*zf)| (lam>6765*zf))
    
    obs['mask'] = mask

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs

def build_model(object_redshift=None, ldist=10.0, fixed_metallicity=None,
                add_duste=False,add_neb=False, add_dispersion=False,**extras):
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed 
        frame (apparent) photometry will be appropriate for this luminosity distance.
        
    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
        
    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """
    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]
    model_params.update(TemplateLibrary["burst_sfh"])
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units":"Mpc"}
    
    # Let's make some changes to initial values appropriate for our objects and data
    model_params["zred"]["init"] = z
    model_params["dust2"]["init"] = 1.0
    model_params["dust_type"]["init"] = 2
    model_params["imf_type"]["init"] = 0
    model_params["logzsol"]["init"] = -0.5
    
    model_params["tage"]["init"] = 13.6
    model_params["fage_burst"]["init"] = 0.9 #burst time 
    model_params["fburst"]["init"] = 0.3
    model_params["mass"]["init"] = 1e10

    model_params["tage"]["isfree"] = False # fix the total age to 13.6 Gyr
    model_params["fage_burst"]["isfree"] = True
    #model_params["tburst"]["isfree"] = False
    model_params["fburst"]["isfree"] = True
    
    # These are starburst galaxies, so lets also adjust the metallicity prior,
    # the tau parameter upward, and the mass prior downward
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.5, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e8, maxi=1e11)
    model_params["fburst"]["prior"] = priors.TopHat(mini=0.0, maxi=0.5)
    #model_params["tburst"]["prior"] = priors.LogUniform(mini=1e-3, maxi=1)
    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["mass"]["init_disp"] = 1e6
    model_params["tau"]["init_disp"] = 1.0
    model_params["tage"]["init_disp"] = 1.0
    
    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity 

    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift
    else:
        model_params["zred"]["isfree"] = True
        model_params["zred"]["prior"] = priors.TopHat(mini=0.95*z, maxi=1.05*z)

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        # Since `model_params` is a dictionary of parameter specifications, 
        # and `TemplateLibrary` returns dictionaries of parameter specifications, 
        # we can just update `model_params` with the parameters described in the 
        # pre-packaged `dust_emission` parameter set.
        model_params.update(TemplateLibrary["dust_emission"])
    if add_neb:
        # Add nebular emission
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]['isfree'] = False
        model_params["gas_logu"]["init"] = -2.92

    if add_dispersion:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"]["init"] = 150
        model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50, maxi=200)
        #model_params["gas_logu"]["prior"] = priors.LogUniform(mini=-4, maxi=0)
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SpecModel(model_params)

    return model
def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous: 
        A vlue of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None


if __name__ == '__main__':
    run_params = {}
    run_params["ldist"] = Planck18.luminosity_distance(sentry["z_sys"][obj-1]).to(u.Mpc).value
    run_params["zcontinuous"] = 1
    run_params["object_redshift"] = sentry["z_sys"][obj-1] # letting redshift free to vary,0.0317
    run_params["fixed_metallicity"] = 0.0 #fix metalicity to solar value
    run_params["add_duste"] = False #no dust emission
    run_params["add_neb"] = True #add nebular emission through FSPS
    run_params["add_dispersion"] = True #add stellar+instrumental velocity dispersion 

    obs = build_obs(**run_params)
    model = build_model(**run_params)
    sps = build_sps(**run_params)
    noise = build_noise(**run_params)
    #print(model)
    #print("\nInitial free parameter vector theta:\n  {}\n".format(model.theta))
    #print("Initial parameter dictionary:\n{}".format(model.params))
    #print(sps.ssp.libraries)


    # Set this to False if you don't want to do another optimization
    # before emcee sampling (but note that the "optimization" entry 
    # in the output dictionary will be (None, 0.) in this case)
    # If set to true then another round of optmization will be performed 
    # before sampling begins and the "optmization" entry of the output
    # will be populated.
    run_params["dynesty"] = True
    run_params["optmization"] = True
    run_params["emcee"] = False
    run_params["nested_method"] = "auto"
    run_params["nlive_init"] = 400
    run_params["nlive_batch"] = 200
    run_params["nested_dlogz_init"] = 0.05
    run_params["nested_posterior_thresh"] = 0.05
    run_params["nested_maxcall"] = int(1e6)

'''
    'nested_bound': 'multi',  # bounding method
    'nested_sample': 'unif',  # sampling method
    'nested_nlive_init': 100,
    'nested_nlive_batch': 100,
    'nested_bootstrap': 0,
    'nested_dlogz_init': 0.05,
    'nested_weight_kwargs': {"pfrac": 1.0},
    'nested_target_n_effective': 10000,
'''
    # Now we instantiate the dynesty sampler

    import mpi4py
    from mpi4py import MPI
    from schwimmbad import MPIPool
    from functools import partial
    import time
    from prospect.io import write_results as writer
    lnprobfn_fixed = partial(lnprobfn, sps=sps)
    mpi4py.rc.threads = False
    mpi4py.rc.recv_mprobe = False

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    withmpi = comm.Get_size() > 1

    with MPIPool() as pool:
    # The subprocesses will run up to this point in the code
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        nprocs = pool.size
        output = fit_model(obs, model, sps, noise, pool=pool, queue_size=nprocs, lnprobfn=lnprobfn_fixed, **run_params)


    hfile = "obj{0}_{1}_emcee.h5".format(obj, int(time.time()))
    writer.write_hdf5(hfile, run_params, model, obs,
                        output["sampling"][0], output["optimization"][0],
                        tsample=output["sampling"][1],
                        toptimize=output["optimization"][1],
                        sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
