#!/usr/bin/env python3

"""
Convert CICE4 restart to CICE6 restart file
netcdf format.
"""

import os
import numpy as np
import sys
import time
import xarray as xr
from yaml import safe_load
import argparse
import mod_time as mtime
import mod_cice6utils as mc6util

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
    print(f'   {sfld} min/max:  {np.nanmin(A)}/{np.nanmax(A)}')
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
      vsn    = vsnon[n,:,:]
      ai_cat = aicen[n,:,:]
      mask_noice = ai_cat <= puny
      ai_cat[mask_noice] = puny 
      hsn    = vsn / ai_cat
      vsn    = np.where(vsn < puny, puny, vsn)
      vsn    = np.where(hsn <= hs_min_layer, 0., vsn)
      mask_vol = vsn > puny

      """Snow enthalpy"""
      for k in range(nslyr):              
        iesnon = slyr1[n]+k
        qsn   = esnon[iesnon,:,:] * nslyr / vsn 
        qsn   = np.minimum(qsn, qT0)
        qsn   = np.where(mask_noice, 0., qsn) 
        qsn   = np.where(hsn <= hs_min_layer, qT0, qsn)

        """
        Check upper / lower bounds deriving snow T from enthalpy.
        Clip to the nearest valid value.
        """
        zTsn  = (Lfresh + qsn / rhos) / cp_ice
        Tmax  = np.zeros_like(vsn)
        Tmax[mask_vol] = -qsn[mask_vol] * puny * nslyr / (rhos * cp_ice * vsn[mask_vol])

        qsn_Tmin = (rhos * (cp_ice * Tsn_min - Lfresh))
        qsn_Tmax = (rhos * (cp_ice * Tmax - Lfresh))

        mask_cold = zTsn < Tsn_min
        mask_warm = zTsn > Tmax

        if np.any(mask_cold):
            print(f"cat={n+1} snow layer={k+1}: {np.count_nonzero(mask_cold)} cells " 
                  f"below Tsn_min={Tsn_min}"
             qsn = np.where(mask_cold, qsn_Tmin, qsn)

        if np.any(mask_warm):
            print(f"cat={n+1} snow layer={k+1}: {np.count_nonzero(mask_warm)} cells " 
                  f"above Tmax"
            qsn = np.where(mask_warm, qsn_Tmax, qsn)

STOPPED HERE


        """Check max snow T from max enthalpy, should be ~0."""
        JJ,II = np.where(zTsn > 1.e-11)
        if len(JJ) > 0:
            qsn[JJ,II] = qT0 

        """Zap entire snow volume if T is out of bounds."""
        print(f'Snow enthalpy: lr = {k+1}, Tmax = {np.max(Tmax)}')
        print(f'cat={n} slyr={k+1} min/max snow T: {np.min(zTsn)}/{np.max(zTsn)}')
        J1,I1 = np.where(zTsn < Tsn_min)
        J2,I2 = np.where(zTsn > Tmax)
        if len(J1) > 0 or len(J2) > 0:
            print('Tsnow is out of bound zeroing snow volume ')
            vsn = np.where(zTsn < Tsn_min, 0., vsn)
            qsn = np.where(zTsn < Tsn_min, 0., qsn)
            vsn = np.where(zTsn > Tmax, 0., vsn)
            qsn = np.where(zTsn > Tmax, 0., qsn)

        qsnon[k,n,:,:] = qsn
        vsnon[n,:,:]   = vsn 

      """Sea ice enthalpy."""
      for k in range(nilyr):
          ai_cat = aicen[n,:,:]
          iicen  = ilyr1[n]+k
          print('eicen index: iicen = {0}'.format(iicen))
          vin    = vicen[n,:,:]               # ice volume per m2 in cat=n
          vin    = np.where(vin < 1.e-30, 1.e-30, vin)
          qin    = eicen[iicen,:,:]*float(nilyr)/vin  # J/m2 --> J/m3
          qin    = np.where(ai_cat <= puny, 0.0, qin)
          qicen[k,n,:,:] = qin

    return qsnon, qicen, vsnon


def salin_profile(cice4, saltamx, min_salin):
NOT NEEDED?
    """
    Determine S profile for CICE4: isosaline or cos-shaped.
    """ 
    salin = np.zeros((cice4.nilyr+1))
    if saltmax > min_salin:
      l_brine = True

      for k in range(cice4.nilyr):
        zn = (float(k+1)-0.5)/(float(cice4.nilyr))
        salin[k] = (saltmax/2.)*(1.-np.cos(np.pi*zn**(nsal/(msal+zn))))
      salin[k+1] = saltmax

    else:
      l_brine = False
      salin = 0.0

    return l_brine, salin

def main():
    fyaml = 'restart_cice6.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("--fyaml", help=f"yaml file with paths, filenames, params, default={fyaml}", type=str)
    parser.add_argument("--rdate6", help="Required restart date in CICE6: YYYYMMDDhh", required=True, type=int)
    args = parser.parse_args()

    fyaml  = args.fyaml if args.fyaml else fyaml
    rdate6 = args.rdate6 if args.rdate6 else None

    dnmb6 = mtime.dateint2datenum(rdate6)
    YR6, MM6, DD6, HH6, _ = mtime.datevec(dnmb6, round_hrs=True)

    with open(fyaml) as ff:
      PATHS = safe_load(ff)

    cicerst4 = PATHS["rest_names"]["cice4"]["flnm"]
    cicerstT = PATHS["rest_names"]["tmplt"]["flnm"]
    cicerst6 = PATHS["rest_names"]["cice6"]["flnm"].format(yr=YR6, mm=MM6, dd=DD6, hr=HH6)
    pthrst4  = PATHS["cice_paths"][node_nm]["cice4"]["pth"]
    pthrstT  = PATHS["cice_paths"][node_nm]["tmplt"]["pth"]
    pthrst6  = PATHS["cice_paths"][node_nm]["cice6"]["pth"]

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

    """Grid CICE4 unformatted binary file."""
    pthgrd4 = PATHS["grid_topo"]["cice4"]["pthgrid"]
    grdfl4  = PATHS["grid_topo"]["cice4"]["filegrid"]
    fgrdin4 = os.path.join(pthgrd4, grdfl4)

    """ Grid and topo CICE6 files."""
    pthgrd  = PATHS["grid_topo"]["cice6"]["pthgrid"]
    grdfl   = PATHS["grid_topo"]["cice6"]["filegrid"]
    fgrdin  = os.path.join(pthgrd, grdfl)
    pthdpth = PATHS["grid_topo"]["cice6"]["pthtopo"]
    dpthfl  = PATHS["grid_topo"]["cice6"]["filedepth"]

    fldptha, fldpthb, topo_nc, topo_ab = check_depth_file(dpthfl, pthdpth)

    """ Create object with CICE4 grid parameters."""
    nx    = PATHS["cice_params"]["cice4"]["nx"]
    ny    = PATHS["cice_params"]["cice4"]["ny"]
    ncat  = PATHS["cice_params"]["cice4"]["ncat"]
    nilyr = PATHS["cice_params"]["cice4"]["nilyr"]
    nslyr = PATHS["cice_params"]["cice4"]["nslyr"]
    cice4 = mc6util.CICE(nx, ny, ncat, nilyr, nslyr)

    """ Create object with CICE6 grid parameters."""
    nx    = PATHS["cice_params"]["cice6"]["nx"]
    ny    = PATHS["cice_params"]["cice6"]["ny"]
    ncat  = PATHS["cice_params"]["cice6"]["ncat"]
    nilyr = PATHS["cice_params"]["cice6"]["nilyr"]
    nslyr = PATHS["cice_params"]["cice6"]["nslyr"]
    cice6 = mc6util.CICE(nx, ny, ncat, nilyr, nslyr)

    """ Read fields from the CICE4 restart file. """
    if not os.path.exists(fl_restart4):
        raise FileNotFoundError(f"Does not exist: {fl_restart4}")

    print(f'Reading restart: {fl_restart4}')
    with open(fl_restart4, 'rb') as fid:
        fid.seek(0)
        """
        Read the 1st sequential record of CICE4 restart file.
        recS:     4-byte record-length marker (start marker)
        istep:    current model step
        runtime:  total elapsed model time (s)
        frtime:   elapsed time since the last forcing update (s)
        recE:     record-length marker (end marker)
        """
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
            uvelE, vvelN = mc6util.interp_uvelE_vvelN(uvel, vvel, aicen) 

        scale_factor = read_cice4_2D(fid, nx, ny, 'scale factor')
        swvdr        = read_cice4_2D(fid, nx, ny, 'sh/wave vis direct')
        swvdf        = read_cice4_2D(fid, nx, ny, 'sh/wave vis diff')
        swidr        = read_cice4_2D(fid, nx, ny, 'sh/wave IR dir')
        swidf        = read_cice4_2D(fid, nx, ny, 'sh/wave IR diff')
        strocnxT     = read_cice4_2D(fid, nx, ny, 'ocean stress x-comp')
        strocnxY     = read_cice4_2D(fid, nx, ny, 'ocean stress y-comp')

        stressp = {}
        for fld in ["stressp_1", "stressp_3", "stressp_2", "stressp_4"]:
            stressp[fld] = read_cice4_2D(fid, nx, ny, fld)
        for fld in ["stressm_1", "stressm_3", "stressm_2", "stressm_4"]:
            stressp[fld] = read_cice4_2D(fid, nx, ny, fld)
        for fld in ["stress12_1", "stress12_3", "stress12_2", "stress12_4"]:
            stressp[fld] = read_cice4_2D(fid, nx, ny, fld)

        iceumask = read_cice4_2D(fid, nx, ny, 'ice umask')             
        sst      = read_cice4_2D(fid, nx, ny, 'ocean mixed layer sst')
        frzmly   = read_cice4_2D(fid, nx, ny, 'frzmlt')


    # Mask out land points:
    print(' Masking out fields ')
    maskval = 0.5 * spval
    for A in [
        aicen, vicen, vsnon, trcrn, eicen, esnon,
        uvel, vvel, scale_factor, swvdr, swvdf, swidr, swidf,
        strocnxT, strocnyT, stressp_1, stressp_3, stressp_2, stressp_4,
        stressm_1, stressm_3, stressm_2, stressm_4, 
        stress12_1, stress12_3, stress12_2, stress12_4,
        sst, frzmlt
    ]:
        A[A > maskval] = 0.

    """lon/lat of CICE4 grid"""
    ulati4 = mc6util.read_cice4_grid(fgrdin4, 'ulati', IDM=nx, JDM=ny)
    uloni4 = mc6util.read_cice4_grid(fgrdin4, 'uloni', IDM=nx, JDM=ny)

    """Read lon/lat from CICE6 restart template."""
    ulati6 = read_cice6_grid(fgrdin, 'ulat')
    uloni6 = read_cice6_grid(fgrdin, 'ulon')

    """Check grids in CICE4 and CICE6, expected to match."""
    mc6util.check_cice_grids(ulati4, uloni4, ulati6, uloni6)

    dnmb_new   = mtime.datenum([int(YR6), int(MM6), int(DD6), int(HH6)])
    coszen_new = mc6util.compute_coszen(ulati6, uloni6, dnmb_new, time_zone=0)

    """Make unknown fields 0, Level ice area and volume make 1 where aicen>0."""
    fsnow = np.zeros((ny,nx), dtype=np.float64)
    iage  = np.zeros_like(aicen)
    apnd  = np.zeros_like(aicen)
    hpnd  = np.zeros_like(aicen)
    ipnd  = np.zeros_like(aicen)
    dhs   = np.zeros_like(aicen)
    ffrac = np.zeros_like(aicen)
    alvl = (aicen > 0.0).astype(np.float64)
    vlvl = (vicen > 0.0).astype(np.float64)

    """Ice and snow enthalpy derived from ice and snow energy in CICE4."""

# Interpolate from nilyr=4 in CICE4 to nilyr=7 ice layers in CICE6
if not cice6.nilyr == cice4.nilyr:
  qicen = mc6util.remap_enthalpy_bins(qicen, cice4.nilyr, cice6.nilyr)

# Snow enthaply interpolation has not been used
# but probably should work fine
if not cice6.nslyr == cice4.nslyr:
  print('!!! Need to check snow enthalpy interpolation \n\n!!!!')
  qsnon = mc6util.remap_enthalpy_bins(qsnon, cice4.nilyr, cice6.nilyr)

# =======================================================
# 
#  scale_factor - scaling factor for shortwave radiation components
# use from CICE6 template? CICE4 scale_factor 
# is different
# scale_factor: netsw scaling factor (new netsw / old netsw)
# see: icepack_shortwave.F90
#
# Collect updated fields:
updated_vars = {
  'uvel':         uvel,
  'vvel':         vvel,
  'uvelE':        uvelE,
  'vvelN':        vvelN,
  'scale_factor': scale_factor,
  'swvdr':        swvdr,
  'swvdf':        swvdf,
  'swidr':        swidr,
  'swidf':        swidf,
  'strocnxT':     strocnxT,
  'strocnyT':     strocnyT,
  'stressp_1':    stressp_1, 
  'stressp_2':    stressp_2, 
  'stressp_3':    stressp_3, 
  'stressp_4':    stressp_4, 
  'stressm_1':    stressm_1, 
  'stressm_2':    stressm_2, 
  'stressm_3':    stressm_3, 
  'stressm_4':    stressm_4, 
  'stress12_1':   stress12_1, 
  'stress12_2':   stress12_2, 
  'stress12_3':   stress12_3, 
  'stress12_4':   stress12_4, 
  'iceumask':     iceumask,
  'fsnow':        fsnow,
  'aicen':        aicen,
  'vicen':        vicen,
  'vsnon':        vsnon,
  'Tsfcn':        trcrn,
  'coszen':       coszen_new,
  'iage':         iage,
  'alvl':         alvl,
  'vlvl':         vlvl,
  'apnd':         apnd,
  'hpnd':         hpnd,
  'ipnd':         ipnd,
  'dhs':          dhs,
  'ffrac':        ffrac
}

print(' \n\n -------------\n Creating CICE6 restart')
dst = xr.open_dataset(fl_restartT)

for varname, new_data in updated_vars.items():
  if varname in dst:
    dst[varname] = xr.DataArray(
      new_data,
      dims=dst[varname].dims,
      coords=dst[varname].coords
    )
  else:
    print(f"{varname} is not in {fl_restartT}")

# Add a new variable to restart file:
def add_newvar(dst, varname, A3d):
  new_fld = xr.DataArray(A3d, 
                        dims=dst[varname].dims, 
                        coords=dst[varname].coords)
  dst[varname] = new_fld

  return dst

#  4D fields:
# sice - ice bulk salinity
# sice - 4D field, written by layers as 3D (ncat,nj,ni)
# ufs-weather-model/CICE-interface/CICE/cicecore/cicedynB/infrastructure/io/io_netcdf
# ice_restart.F90
# ufs-weather-model/CICE-interface/CICE/cicecore/shared/ice_restart_column.F90
#aice = np.sum(aicen, axis=0)
#
# Ice salinity by layers - compute S profile using BZ99 formulation:
for ik in range(1,cice6.nilyr+1): 
  sice_lr = mc6util.sice_lr_cice4(ik, cice6.nilyr, aicen)
  varname = f'sice{ik:03d}'
  print(f'Updating {varname}')
  dst = add_newvar(dst, varname, sice_lr)

# Ice enthalpy by layers:
for ik in range(1,cice6.nilyr+1): 
  qice_lr = qicen[ik-1,:,:,:]
  varname = f'qice{ik:03d}'
  print(f'Updating {varname}')
  dst = add_newvar(dst, varname, qice_lr)
  #mc6util.modify_fld_nc(fl_restart6,fldout,qice_lr)

# Snow enthalpy by layers
for ik in range(1,cice6.nslyr+1):
  qsnon_lr = qsnon[ik-1,:,:,:]
  varname = f'qsno{ik:03d}'
  print(f'Updating {varname}')
  dst = add_newvar(dst, varname, qsnon_lr)
  #mc6util.modify_fld_nc(fl_restart6, fldout, qsnon_lr)

#
# Change restart date:
print(f'Changing global attributes: restart time to {YR6}/{MM6:02d}/{DD6:02d} {HH6*3600} sec')
dst.attrs['myear']  = np.int32(YR6)
dst.attrs['mmonth'] = np.int32(MM6)
dst.attrs['mday']   = np.int32(DD6)
dst.attrs['msec']   = np.int32(HH6*3600)
dst.attrs['info1']  = f"Restart created from CICE4: {cicerst4}"
dst.attrs['info2']  = f"code: {btx}"

print(f"Saving cice restart ---> {fl_restart6}")
dst.to_netcdf(fl_restart6, encoding={var: {'_FillValue': None} for var in dst.data_vars}, \
              format='NETCDF3_64BIT')

dst.close()
#ds6.close()

print(f'Created CICE6 restart: {fl_restart6}\n')

if not os.path.isfile(fl_restart6):
  raise Exception (f'ERR: CICE6 restart was NOT CREATED: {fl_restart6}')


