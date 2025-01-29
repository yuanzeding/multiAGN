import subprocess
from astropy.io import ascii
from astropy import constants, units as u
# Variables
field = "UGC2369"
instru="OSIRIS"
fname="datacube.fits"
# Path to the data
root_directory="/disk/bifrost/yuanze/multiAGN"
#wdir=f"{root_directory}/CubEx_run/Q{field}"
wdir=f"{root_directory}/{field}/{instru}"
stab = ascii.read(root_directory+"/sources.list",format="ipac")
sentry = stab[stab["Field"]==field]

redshift=0.03145#sentry["z_sys"][1]


line="H2_21_S1"
wave={"Lya":1215.67,"CIV":1549.06,"MgII":2799.12,"HeII":1640.4,"OVI":1031,"SiIV":1402.77,"CII":1334.53,"NV":1240,"OI":1304,"Brd":1945.09,"Brg":2166.121,"H2_10_S1":2121.8,"H2_21_S1":2247.7} #Angstroms, but for IR line it is micron
sigma_v={"Lya":3.e3,"CIV":3e3,"SiIV":3e3,"CII":3e3,"NV":3e3,"OI":3e3,"OVI":3e3,"HeII":3e3,"Brd":1.5e3,"Brg":1.5e3,"H2_10_S1":1.5e3,"H2_21_S1":1.5e3} #km/s turbulent velocity
c=constants.c.to(u.km/u.s).value #km/s speed of light
lmin=int((redshift+1.)*wave[line]*(1-sigma_v[line]/c)) #Lyman-alpha wavelength range
lmax=int((redshift+1.)*wave[line]*(1+sigma_v[line]/c)) #Lyman-alpha wavelength range, optical
print("with redshift",redshift,"min wavelength",lmin,"max wavelength",lmax)

RescaleVar = ".true."
RescaleVarArea = '"15 85 15 85"'
RescaleVarFR = 5
RescalingVarOutFile = '"Revar.out"'

MinNVox=3
MinArea = 1
MinDz = 2
SN_Threshold = 3.0
ApplyFilter = ".false."
ApplyFilterVar = ".false."
FilterXYRad = 1
FilterZRad = 0
XYedge = 1
ReliabCheck = ".false."
# Path to the CubEx executable
cubex_path = "/disk/bifrost/yuanze/software/CubEx/exe_files"

# Construct the new InpFile line


Inp_file = f'InpFile = "{wdir}/{fname}"'
Catalogue = f'Catalogue = "{wdir}/{field}_{line}.cat"'
Var_file = f'VarFile = "{wdir}/varcube.fits"'
CheckCube = f'CheckCube = "{wdir}/checkcube_{line}.fits"'
#Catalogue = f'Catalogue = "/disk/bifrost/yuanze/KBSS/Q{field}/QSO/kcwi_oned/q{field}-QSO.cat"'
# Read in the current contents of the file

with open(f'{wdir}/par.in', 'r') as file:
    lines = file.readlines()

# Replace the line containing InpFile
with open(f'{wdir}/par.in', 'w') as file:
    for line in lines:
        if line.startswith('RescaleVar ='):
            line = f'RescaleVar = {RescaleVar}\n'
        if line.startswith('RescaleVarArea ='):
            line = f'RescaleVarArea = {RescaleVarArea}\n'
        if line.startswith('RescaleVarFR ='):
            line = f'RescaleVarFR = {RescaleVarFR}\n'
        if line.startswith('RescalingVarOutFile ='):
            line = f'RescalingVarOutFile = {RescalingVarOutFile}\n'
        if line.startswith('FilterZRad ='):
            line = f'FilterZRad = {FilterZRad}\n'
        if line.startswith('ApplyFilterVar ='):
            line = f'ApplyFilterVar = {ApplyFilterVar}\n'
        if line.startswith('FilterXYRad ='):
            line = f'FilterXYRad = {FilterXYRad}\n'
        if line.startswith('ApplyFilter ='):
            line = f'ApplyFilter = {ApplyFilter}\n'
        if line.startswith('SN_Threshold ='):
            line = f'SN_Threshold = {SN_Threshold}\n'
        if line.startswith('InpFile ='):
            line = Inp_file + '\n'
        if line.startswith('Catalogue ='):
            line = Catalogue + '\n'
        if line.startswith('VarFile ='):
            line = Var_file + '\n'
        if line.startswith('lmin ='):
            line = f'lmin = {lmin}\n'
        if line.startswith('lmax ='):
            line = f'lmax = {lmax}\n'
        if line.startswith('MinNVox ='):
            line = f'MinNVox = {MinNVox}\n'
        if line.startswith('MinArea ='):
            line = f'MinArea = {MinArea}\n'
        if line.startswith('MinDz ='):
            line = f'MinDz = {MinDz}\n'
        if line.startswith('CheckCube ='):
            line = CheckCube + '\n'
        if line.startswith('XYedge ='):
            line = f'XYedge = {XYedge}\n'
        if line.startswith('ReliabCheck ='):
            line = f'ReliabCheck = {ReliabCheck}\n'
        file.write(line)

# Execute the CubEx command
#subprocess.run([f"{cubex_path}/CubEx", "par.in"])