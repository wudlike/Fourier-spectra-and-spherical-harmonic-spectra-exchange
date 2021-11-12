import numpy as np
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import ICRS
import astropy.units as u
import healpy as hp
from settings import Settings
import matplotlib.pyplot as plt

st = Settings()

def Cl2map(n_realization):
    '''
    map realization
    '''
    maps = []
    for _ in range(n_realization):
        maps.append(hp.synfast(Cl_tt, nside=st.nside))
    return np.array(maps)

#Cl2map
powspect = np.loadtxt('data/test_totCls.dat')
ell_tt = powspect[:, 0]
Dl_tt = powspect[:, 1]
Cl_tt = Dl_tt*2*np.pi/(ell_tt*(ell_tt+1))
lmax = Cl_tt.size
nside = st.nside

#Cl2map, map realization
n_realization = 1 # generate only one cmb map
tmaps_cmb = Cl2map(n_realization=n_realization)

# first map
tmaps_cmb = tmaps_cmb[0]

# samples with a ring
samples = 4000

# time interval and how many rings
rings = 1000

# lat, height and lon for site
b1_Ali = EarthLocation.from_geodetic(-80.0305 *
                                     u.deg, 32.3105*u.deg, 5200*u.m)
start_observing = Time('2020-12-21 1:00')
delta_time = np.arange(rings)/24/60 # time interval
observing_time = start_observing + delta_time
observing_time = observing_time.reshape(rings, 1)

alt = 90 - st.theta  # zenith=50 (theta=50)
az = np.zeros((rings, samples))
for i in range(rings):
    az[i] = np.linspace(0, 360, samples)

aa = AltAz(az=az*u.deg, alt=alt*u.deg, location=b1_Ali, obstime=observing_time)
ra = aa.transform_to(ICRS).ra.deg
dec = aa.transform_to(ICRS).dec.deg
pix = hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True)
np.save('pix_theta_'+str(st.theta)+'_deg', pix)
# pix = np.load('pix.npy')
ring_tt = tmaps_cmb[pix]
print('ring_shape :', np.shape(ring_tt))
# np.save('ring_tt_cmb', ring_tt)

# test: show the scanning trace on map
trace_map = np.zeros(12*nside**2)
trace_map[pix] = trace_map[pix] - 10
hp.mollview(trace_map, title='trace of ring')
hp.mollview(tmaps_cmb, title='CMB map')

# tod of cmb
plt.figure(figsize=(10, 6))
plt.plot(np.arange(samples), ring_tt[0])
plt.title('CMB TOD of a ring')

plt.show()
