#!/usr/bin/env python3
"""
Convert a CICE4 restart file to a CICE6 NetCDF restart.
Missing variables required by CICE6 are added and initialized.
Ice and snow energies are converted to enthalpy and
remapped from the CICE4 vertical discretization to the CICE6
layer structure using a conservative remapping scheme that
preserves total column energy.
"""

import os
import numpy as np
import sys
import time
import xarray as xr
from yaml import safe_load
import argparse
import mod_cice6rest as mc6rest

# ==============================================================================
# CICE / Icepack Thermodynamic and Other Constants
# ==============================================================================
puny      = 1.e-11
c0        = 0.0
c1        = 1.0
c2        = 2.0
p5        = 0.5
Lsub      = 2.835e6           # Latent heat sublimation fw (J/kg)
Lvap      = 2.501e6           # Latent heat vaporization fw (J/kg)
Lfresh    = Lsub - Lvap       # Latent heat of melting of fresh ice (J/kg)
cp_ice    = 2106.             # Specific heat of fresh ice (J/ kg/K)
rhos      = 330.              # Density of snow (kg/m3)
hs_min    = 1.e-4             # Min snow thickness for computing Tsno (m)
nsal      = 0.407
msal      = 0.573
min_salin = 0.1               # Threshold for brine pocket treatment
saltmax   = 3.2               # Max S at ice base
spval     = 1.e30             # Bad values
Tsn_min   = -100.             # minimum snow T

class CICE:
    def __init__(self, nx, ny, ncat, nilyr, nslyr):
        self.ncat = ncat
        self.nilyr = nilyr
        self.nslyr = nslyr
        self.nx = nx
        self.ny = ny

        self.ntilyr = self.ncat * self.nilyr
        self.ntslyr = self.ncat * self.nslyr

def check_depth_file(pthdpth, dpthfl):
    """
    Checks if input topography file is netcdf or unformatted binary *.a.
    """
    fldptha  = fldpthb = None
    topo_nc = topo_ab = False

    if dpthfl.endswith('.nc'):
        fldpthnc = pthdpth
        fdpthin = os.path.join(pthdpth, dpthfl)
        topo_nc = True
    elif dpthfl.endswith('.a'):
        fldptha = dpthfl
        fldpthb = fldptha.replace('.a', '.b')
        ftopo   = fldptha.removesuffix('.a')
        topo_ab = True
    else:
        raise ValueError(f"topo file {dpthfl} not recognized, expected *.a or *.nc")

    return fldptha, fldpthb, topo_nc, topo_ab

def read_cice6_grid(dirflnc, varnc):
    """Read CICE6 field varnc from a grid  netcdf file."""
    with xr.open_dataset(dirflnc) as dset:
        AA = dset[varnc].data.squeeze()

    return AA  

def read_rest_cice4(fid, nx, ny):
    """
    Read a 2D field from an open CICE4 restart file.
    CICE4 restart files are unformatted sequential binary records
    in big-endian format. 
    """
    recS = np.fromfile(fid, dtype='>i4', count=1)[0]
    A     = np.fromfile(fid, dtype='>f8', count=nx*ny)
    A     = np.reshape(A,(ny,nx), order='C')
    recE = np.fromfile(fid, dtype='>i4', count=1)[0]
    if recS != recE:
       raise ValueError(f"Record length mismatch: {recS} != {recE}")

    return A

def print_minmax(sfld, A):
    """Print min/max statistics of a numpy array."""
    print(f'   {sfld} min/max:  {np.nanmin(A)} / {np.nanmax(A)}')
    return

def read_cice4_layers(fid, nlrs, nx, ny, label):
    """ Read CICE4 fields by layers. """
    print(f'\n Reading {label}:')
    fld = np.zeros((nlrs, ny, nx), dtype=np.float64)

    for k in range(nlrs):
        A = read_rest_cice4(fid, nx, ny)
        fld[k, :, :] = A
        print_minmax(f"{k+1} {label}", A)

    return fld

def read_cice4_2D(fid, nx, ny, varnm):
    """Read CICE4 2D fields."""
    print(f'\nReading {varnm}:')
    A = read_rest_cice4(fid, nx, ny)
    print_minmax(varnm, A)

    return A

def energy_to_enthalpy(aicen, vicen, vsnon, eicen, esnon,
        cice4, rhos, Lfresh, cp_ice, puny, hs_min, Tsn_min):
    """
    Convert ice and snow energy used in CICE4 (J/m2)
    to enthalpy used in CICE6 (J/m3).
    The CICE4 number of snow and ice layers is preserved.
    """
    nilyr = cice4.nilyr
    nslyr = cice4.nslyr
    nx    = cice4.nx
    ny    = cice4.ny
    ncat  = cice4.ncat
    ilyr1 = np.arange(ncat) * nilyr   # ice layer index in eicen array for each cat
    slyr1 = np.arange(ncat) * nslyr   # snow lyaer index in esnon array for each cat
    qT0   = -Lfresh * rhos            # Enthalpy at the melting point
    hs_min_layer = hs_min / nslyr     # min snow thikness in a layer
    qsnon = np.zeros((nslyr, ncat, ny, nx), dtype=np.float64)
    qicen = np.zeros((nilyr, ncat, ny, nx), dtype=np.float64)
    vsnon_out = vsnon.copy()          # Working copy

    for n in range(ncat):               
        vsn    = vsnon[n,:,:].copy()
        ai_cat = aicen[n,:,:].copy()
        mask_noice = ai_cat <= puny
        ai_cat[mask_noice] = puny 
        hsn    = vsn / ai_cat
        vsn    = np.where(vsn < puny, puny, vsn)
        vsn    = np.where(hsn <= hs_min_layer, 0., vsn)
        mask_vol = vsn > puny

        # Snow enthalpy
        for k in range(nslyr):              
            iesnon = slyr1[n] + k
            qsn   = esnon[iesnon,:,:] * nslyr / vsn 
            qsn   = np.minimum(qsn, qT0)
            qsn   = np.where(mask_noice, 0., qsn) 
            qsn   = np.where(hsn <= hs_min_layer, qT0, qsn)

            # Check upper / lower bounds deriving snow T from enthalpy.
            # Clip to the nearest valid value.
            zTsn  = (Lfresh + qsn / rhos) / cp_ice
            Tmax  = np.zeros_like(vsn)
            Tmax[mask_vol] = -qsn[mask_vol] * puny * nslyr / (rhos * cp_ice * vsn[mask_vol])
            qsn_Tmin = (rhos * (cp_ice * Tsn_min - Lfresh))
            qsn_Tmax = (rhos * (cp_ice * Tmax - Lfresh))
            mask_cold = zTsn < Tsn_min
            mask_warm = zTsn > Tmax

            if np.any(mask_cold):
                print(f"cat={n+1} snow layer={k+1}: {np.count_nonzero(mask_cold)} cells " 
                      f"below Tsn_min={Tsn_min}")
                qsn = np.where(mask_cold, qsn_Tmin, qsn)

            if np.any(mask_warm):
                print(f"cat={n+1} snow layer={k+1}: {np.count_nonzero(mask_warm)} cells " 
                      f"above Tmax")
                qsn = np.where(mask_warm, qsn_Tmax, qsn)

            qsnon[k,n,:,:]   = qsn
            vsnon_out[n,:,:] = vsn 

    # Sea ice enthalpy.
    for n in range(ncat):
        ai_cat = aicen[n,:,:]
        vin    = vicen[n,:,:]  
        vin    = np.where(vin < puny, puny, vin)
        for k in range(nilyr):
            iicen  = ilyr1[n]+k
            print(f"cat={n+1} ice layer={k+1}: " f"eicen index={iicen}")

            # Convert ice energy J/m2 --> J/m3.
            qin    = eicen[iicen,:,:] * nilyr / vin 
            qin    = np.where(ai_cat <= puny, 0.0, qin)
            qicen[k,n,:,:] = qin

    return qsnon, qicen, vsnon_out

def set_ncvar(dst, varname, A3d):
    """Update or add netcdf variable to restart dataset."""

    print(f'Updating {varname}') 
    if A3d.shape != dst[varname].shape:
        raise ValueError(
            f"{varname}: shape mismatch "
            f"{A3d.shape} != {dst[varname].shape}"
        )

    new_fld = xr.DataArray(A3d, 
                          dims=dst[varname].dims, 
                          coords=dst[varname].coords)
    dst[varname] = new_fld

    return dst

def sice_lr_BL99(klr, Ni, aicen, puny, Smax=3.2, a=0.407, b=0.573):
    """   
    Compute ice S in a single ice layer for all categories
    following the Bitz & Lipscomb (1999) formulation used in CICE4
    aicen is a 3D array of ice partial areas by categories
    """
    if klr > Ni: 
        raise Exception (f'klr {klr} cannot be > {Ni}')

    z       = (klr - 0.5) / Ni
    sice    = 0.5 * Smax * (1 - np.cos(np.pi * z**(a / (z + b))))
    sice_lr = np.where(aicen < puny, 0.0, sice)     
 
    return sice_lr


def main():
    fyaml = 'restart_cice6.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyaml", 
        help=f"yaml file with paths, filenames, params, default={fyaml}", 
        default=fyaml)
    parser.add_argument("--rdate6", help="Required restart date in CICE6: YYYYMMDD[hh], default hh=0", 
                        required=True, type=int)
    args = parser.parse_args()

    fyaml  = args.fyaml 
    rdate6 = args.rdate6 if args.rdate6 else None

    dnmb6 = mc6rest.dateint2datenum(rdate6)
    YR6, MM6, DD6, HH6 = mc6rest.datevec(dnmb6, round_hrs=True)[:4]

    with open(fyaml) as ff:
      PATHS = safe_load(ff)

    cicerst4 = PATHS["rest_names"]["cice4"]["flnm"]
    cicerstT = PATHS["rest_names"]["tmplt"]["flnm"]
    cicerst6 = PATHS["rest_names"]["cice6"]["flnm"].format(yr=YR6, mm=MM6, dd=DD6, hr=HH6)
    pthrst4  = PATHS["cice_paths"]["cice4"]["pth"]
    pthrstT  = PATHS["cice_paths"]["tmplt"]["pth"]
    pthrst6  = PATHS["cice_paths"]["cice6"]["pth"]

    fl_restart4 = os.path.join(pthrst4, cicerst4)
    fl_restartT = os.path.join(pthrstT, cicerstT)
    fl_restart6 = os.path.join(pthrst6, cicerst6)

    ice_grid4 = PATHS["cice_params"]["cice4"]["grid"]
    ice_grid6 = PATHS["cice_params"]["cice6"]["grid"]

    print(' \n===================================')
    print(f'Creating CICE6 restart for {YR6}/{MM6:02d}/{DD6:02d} {HH6:02d}hr UTC')
    print(f'CICE4 restart:       {fl_restart4}')
    print(f'CICE6 template:      {fl_restartT}')
    print(f'New CICE6 restart:   {fl_restart6}')
    print(f'CICE4 grid:          {ice_grid4}')
    print(f'CICE6 grid:          {ice_grid6}')
    print(' =================================== \n')

    # Grid CICE4 unformatted binary file.
    pthgrd4 = PATHS["grid_topo"]["cice4"]["pthgrid"]
    grdfl4  = PATHS["grid_topo"]["cice4"]["filegrid"]
    fgrdin4 = os.path.join(pthgrd4, grdfl4)

    # Grid and topo CICE6 files.
    pthgrd  = PATHS["grid_topo"]["cice6"]["pthgrid"]
    grdfl   = PATHS["grid_topo"]["cice6"]["filegrid"]
    fgrdin  = os.path.join(pthgrd, grdfl)
    pthdpth = PATHS["grid_topo"]["cice6"]["pthtopo"]
    dpthfl  = PATHS["grid_topo"]["cice6"]["filedepth"]

    fldptha, fldpthb, topo_nc, topo_ab = check_depth_file(pthdpth, dpthfl)

    # Create object with CICE4 grid parameters.
    nx    = PATHS["cice_params"]["cice4"]["nx"]
    ny    = PATHS["cice_params"]["cice4"]["ny"]
    ncat  = PATHS["cice_params"]["cice4"]["ncat"]
    nilyr = PATHS["cice_params"]["cice4"]["nilyr"]
    nslyr = PATHS["cice_params"]["cice4"]["nslyr"]
    cice4 = CICE(nx, ny, ncat, nilyr, nslyr)

    # Create object with CICE6 grid parameters.
    nx    = PATHS["cice_params"]["cice6"]["nx"]
    ny    = PATHS["cice_params"]["cice6"]["ny"]
    ncat  = PATHS["cice_params"]["cice6"]["ncat"]
    nilyr = PATHS["cice_params"]["cice6"]["nilyr"]
    nslyr = PATHS["cice_params"]["cice6"]["nslyr"]
    cice6 = CICE(nx, ny, ncat, nilyr, nslyr)

    # Read fields from the CICE4 restart file.
    if not os.path.exists(fl_restart4):
        raise FileNotFoundError(f"Does not exist: {fl_restart4}")

    print(f'Reading restart: {fl_restart4}')
    with open(fl_restart4, 'rb') as fid:
        # Read the 1st sequential record of CICE4 restart file.
        # recS:     4-byte record-length marker (start marker)
        # istep:    current model step
        # runtime:  total elapsed model time (s)
        # frtime:   elapsed time since the last forcing update (s)
        # recE:     record-length marker (end marker)
        recS    = np.fromfile(fid, dtype='>i4', count=1)[0]
        istep   = np.fromfile(fid, dtype='>i4', count=1)[0]
        runtime = np.fromfile(fid, dtype='>f8', count=1)[0]
        frtime  = np.fromfile(fid, dtype='>f8', count=1)[0]
        recE    = np.fromfile(fid, dtype='>i4', count=1)[0]

        if recS != recE:
          raise ValueError(f"Record length mismatch: {recS} != {recE}")

        print(
        f"Restart: step={istep}, "
        f"total time={runtime/(3600*24*365.25):.2f} yr, "
        f"forcing update={frtime/3600:.1f} hr ago"
        )

        nx     = cice4.nx
        ny     = cice4.ny
        ncat   = cice4.ncat
        ntilyr = cice4.ntilyr  # total # of icelrs * cat 
        ntslyr = cice4.ntslyr

        aicen = np.zeros((ncat,ny,nx), dtype='float64')
        vicen = np.zeros((ncat,ny,nx), dtype='float64')
        vsnon = np.zeros((ncat,ny,nx), dtype='float64')
        trcrn = np.zeros((ncat,ny,nx), dtype='float64')

        for n in range(ncat):
            print(f" Category {n+1}")
            for varname, arr in [
                ("ice area", aicen),
                ("ice vol",  vicen),
                ("snow vol", vsnon),
                ("surf T",   trcrn),
            ]:
                A = read_rest_cice4(fid, nx, ny)
                arr[n, :, :] = A
                print_minmax(varname, A)

        eicen = read_cice4_layers(fid, ntilyr, nx, ny, "eicen")
        esnon = read_cice4_layers(fid, ntslyr, nx, ny, "esnon")
        uvel  = read_cice4_2D(fid, nx, ny, 'uvel')
        vvel  = read_cice4_2D(fid, nx, ny, 'vvel')
        uvelE = None
        vvelN = None
        if ice_grid6 == 'C':
            uvelE, vvelN = mc6rest.interp_uvelE_vvelN(uvel, vvel, aicen) 

        scale_factor = read_cice4_2D(fid, nx, ny, 'scale factor')
        swvdr        = read_cice4_2D(fid, nx, ny, 'sh/wave vis direct')
        swvdf        = read_cice4_2D(fid, nx, ny, 'sh/wave vis diff')
        swidr        = read_cice4_2D(fid, nx, ny, 'sh/wave IR dir')
        swidf        = read_cice4_2D(fid, nx, ny, 'sh/wave IR diff')
        strocnxT     = read_cice4_2D(fid, nx, ny, 'ocean stress x-comp')
        strocnyT     = read_cice4_2D(fid, nx, ny, 'ocean stress y-comp')

        stressp = {}
        for fld in ["stressp_1", "stressp_3", "stressp_2", "stressp_4"]:
            stressp[fld] = read_cice4_2D(fid, nx, ny, fld)
        stressm = {}
        for fld in ["stressm_1", "stressm_3", "stressm_2", "stressm_4"]:
            stressm[fld] = read_cice4_2D(fid, nx, ny, fld)
        stress12 = {}
        for fld in ["stress12_1", "stress12_3", "stress12_2", "stress12_4"]:
            stress12[fld] = read_cice4_2D(fid, nx, ny, fld)

        iceumask = read_cice4_2D(fid, nx, ny, 'ice umask')             
        sst      = read_cice4_2D(fid, nx, ny, 'ocean mixed layer sst')
        frzmlt   = read_cice4_2D(fid, nx, ny, 'frzmlt')


    # Mask out land points:
    print(' Masking out fields ')
    maskval = 0.5 * spval
    for A in [
        aicen, vicen, vsnon, trcrn, eicen, esnon,
        uvel, vvel, scale_factor, swvdr, swvdf, swidr, swidf,
        strocnxT, strocnyT, sst, frzmlt
    ]:
        A[A > maskval] = 0.

    for A in stressp.values():
        A[A > maskval] = 0.

    for A in stressm.values():
        A[A > maskval] = 0.

    for A in stress12.values():
        A[A > maskval] = 0.

    # Read  CICE4 grid coordinates.
    ulati4 = mc6rest.read_cice4_grid(fgrdin4, 'ulati', IDM=cice4.nx, JDM=cice4.ny)
    uloni4 = mc6rest.read_cice4_grid(fgrdin4, 'uloni', IDM=cice4.nx, JDM=cice4.ny)

    # Read CICE6 coordinates from restart template.
    ulati6 = read_cice6_grid(fgrdin, 'ulat')
    uloni6 = read_cice6_grid(fgrdin, 'ulon')

    # Check grids in CICE4 and CICE6, expected to match.
    mc6rest.check_cice_grids(ulati4, uloni4, ulati6, uloni6)

    dnmb_new   = mc6rest.datenum([YR6, MM6, DD6, HH6])
    coszen_new = mc6rest.compute_coszen(ulati6, uloni6, dnmb_new, time_zone=0)

    # Make unknown fields 0, Level ice area and volume make 1 where aicen>0.
    fsnow = np.zeros((ny,nx), dtype=np.float64)
    iage  = np.zeros_like(aicen)
    apnd  = np.zeros_like(aicen)
    hpnd  = np.zeros_like(aicen)
    ipnd  = np.zeros_like(aicen)
    dhs   = np.zeros_like(aicen)
    ffrac = np.zeros_like(aicen)
    alvl = (aicen > 0.0).astype(np.float64)
    vlvl = (vicen > 0.0).astype(np.float64)

    # Ice and snow enthalpy derived from ice and snow energy in CICE4 layers.
    qsnon, qicen, vsnon_out =  energy_to_enthalpy(aicen, vicen, vsnon, eicen, esnon,
        cice4, rhos, Lfresh, cp_ice, puny, hs_min, Tsn_min)

    # Interpolate from nilyr=4 in CICE4 to nilyr=7 ice layers in CICE6
    if not cice6.nilyr == cice4.nilyr:
        qicen = mc6rest.remap_enthalpy_bins(qicen, cice4.nilyr, cice6.nilyr)

    # Snow enthaply interpolation
    if not cice6.nslyr == cice4.nslyr:
        qsnon = mc6rest.remap_enthalpy_bins(qsnon, cice4.nslyr, cice6.nslyr)


    # Collect updated fields with names and corresponding arrays
    updated_vars = {
    'uvel': uvel,
    'vvel': vvel,
    'scale_factor': scale_factor,
    'swvdr': swvdr,
    'swvdf': swvdf,
    'swidr': swidr,
    'swidf': swidf,
    'strocnxT': strocnxT,
    'strocnyT': strocnyT,
    'iceumask': iceumask,
    'fsnow': fsnow,
    'aicen': aicen,
    'vicen': vicen,
    'vsnon': vsnon,
    'iage': iage,
    'alvl': alvl,
    'vlvl': vlvl,
    'apnd': apnd,
    'hpnd': hpnd,
    'ipnd': ipnd,
    'dhs': dhs,
    'ffrac': ffrac,
    'Tsfcn': trcrn,
    'coszen': coszen_new,
    }

    # Add uvelE and vvelN only if they were calculated
    if uvelE is not None:
        updated_vars['uvelE'] = uvelE
    if vvelN is not None:
        updated_vars['vvelN'] = vvelN

    # Add all stress fields
    updated_vars.update(stressp)
    updated_vars.update(stressm)
    updated_vars.update(stress12)

    print(' \n\n -------------')
    print('Creating CICE6 restart')

    if not os.path.isfile(fl_restartT):
        raise FileNotFoundError(f"CICE6 restart template not found {fl_restartT}")

    dst = xr.open_dataset(fl_restartT)

    # Update existing CICE6 fields.
    for varname, new_data in updated_vars.items():
        if varname in dst.variables:
            dst = set_ncvar(dst, varname, new_data)
        else:
            print(f"WARNING: {varname} is not in {fl_restartT}")

    #  4D fields written by layers as 3D (ncat,nj,ni).
    # Ice salinity by layers 
    for ik in range(1, cice6.nilyr+1): 
      sice_lr = sice_lr_BL99(ik, cice6.nilyr, aicen, puny)
      varname = f'sice{ik:03d}'
      dst = set_ncvar(dst, varname, sice_lr)

    # Ice enthalpy by layers:
    for ik in range(1,cice6.nilyr+1): 
      qice_lr = qicen[ik-1,:,:,:]
      varname = f'qice{ik:03d}'
      dst = set_ncvar(dst, varname, qice_lr)

    # Snow enthalpy by layers
    for ik in range(1,cice6.nslyr+1):
      qsnon_lr = qsnon[ik-1,:,:,:]
      varname = f'qsno{ik:03d}'
      dst = set_ncvar(dst, varname, qsnon_lr)

    # Change restart date:
    print(
        f'Changing global attributes: restart time to '
        f'{YR6}/{MM6:02d}/{DD6:02d} {HH6*3600} sec'
    )

    dst.attrs['myear']  = np.int32(YR6)
    dst.attrs['mmonth'] = np.int32(MM6)
    dst.attrs['mday']   = np.int32(DD6)
    dst.attrs['msec']   = np.int32(HH6 * 3600)
    dst.attrs['info1']  = f"Restart created from CICE4: {cicerst4}"

    print(f"Saving cice restart ---> {fl_restart6}")
    dst.to_netcdf(
        fl_restart6,
        encoding={
            var: {'_FillValue': None}
            for var in dst.data_vars
        },
        format='NETCDF3_64BIT'
    )

    dst.close()

    if not os.path.isfile(fl_restart6):
      raise Exception (f'ERR: CICE6 restart was NOT CREATED: {fl_restart6}')

    print(f'Created CICE6 restart: {fl_restart6}\n')

if __name__ == "__main__":
    main()





