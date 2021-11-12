import numpy as np
import matplotlib.pyplot as plt
from pyshtools.expand import spharm
import healpy as hp
from settings import Settings
import copy
from time import time

st = Settings()

def get_ring_value(nside,  theta):
    '''
    Getting ring value for a given theta.
    Note: scan along the galactic latitude
    '''
    idx = hp.ang2pix(nside, theta, np.radians(30))
    pix_ring = hp.pix2ring(nside, np.array([idx]))
    start_pix, len_pix = hp.ringinfo(nside, pix_ring)[:2]
    return start_pix[0], len_pix[0]

def Cl2map(n_realization,Cl):
    '''
    map realization
    '''
    maps = []
    for i in range(n_realization):
        print(i)
        maps.append(hp.synfast(Cl, nside=st.nside))
    return np.array(maps)

def ring2gamma(ring):
    samples = ring.shape[-1]
    yf = np.fft.rfft(ring)
    # normalization and averaging
    Gamma_m = np.mean((np.abs(yf)/samples)**2, axis=0)
    return Gamma_m

def plm_matrix():
    matrix_plm = spharm(ell_len-1+m_start, theta, 0, normalization='ortho',
                        degrees=False, kind='complex', csphase=1)[0].real.T
    return matrix_plm


def Gamma2Cl(mat_plm, B, gamma):
    return np.matmul(np.matmul(B, np.linalg.pinv(np.matmul(mat_plm, B))), gamma)


def Cl2Gamma(mat_plm, Cl_plt):
    return np.matmul(mat_plm, Cl_plt)


def bin_matrix(q, rt):
    '''
    compute bin_matrix B
    bin_size = round(rt*q**i)
    '''
    k = [round(rt*q**i) for i in range(ell_len)]
    w = np.cumsum(k)
    idx = abs(w-ell_len).argmin()
    if w[idx] < ell_len:
        bi = idx + 1
    elif w[idx] >= ell_len:
        bi = idx
    bin_mat = np.zeros((ell_len, bi+1))
    h = np.arange(ell_len).tolist()
    a = 0
    bin = []
    for i in range(bi+1):
        s = round(rt*q**i)
        bin_mat[a:a+s, i] = 1
        bin.append(int(np.mean(h[a:a+s])))
        a += s
    return bin_mat, bin


if __name__ == '__main__':

    s_time = time()
    # -------- load theoretical Cl data -------------------------------
    m_start = 0
    powspect = np.loadtxt('data/test_totCls.dat')  # tensor_to_scalar: r=0
    ell_tt = powspect[m_start:2000, 0]  # ell
    Dl_tt = powspect[m_start:2000, 1]   # TT
    Cl_tt = Dl_tt*2*np.pi/(ell_tt*(ell_tt+1))
    theta = np.radians(st.theta)  # elevation = 90 - theta deg,
    ell_len = ell_tt.size
    lmax = max(ell_tt).astype(int)

    # --------- Cl to Gamma_m --------------------------------------
    # smooth Cl with 7 arcmin
    # bl = hp.gauss_beam(np.radians(7/60), lmax=lmax)[m_start+2:]

    # without smoothing
    bl = 1
    mat_plm_in = (plm_matrix()**2)[m_start:, m_start:]
    Gamma_m_th = Cl2Gamma(mat_plm_in, Cl_tt*bl**2)  # theory gamma_m

    # --------- Gamma_m from ring scanning ----------------------
    # cmb map realization
    n_realization = 50
    map_cmb = Cl2map(n_realization=n_realization,Cl=Cl_tt)
    np.save('map_cmb',map_cmb)
    # map_cmb = np.load('map_cmb.npy')
    start_pix, len_pix = get_ring_value(st.nside, theta)
    ring_tt = map_cmb[:, start_pix:start_pix+len_pix]
    Gamma_m = ring2gamma(ring_tt)[m_start:m_start+ell_len]

    # -------- compute bin_matrix B ---------------------------
    rt = 20
    q = 1.03
    bin_mat, b = bin_matrix(q, rt)
    ell_bin = np.sum(bin_mat*b, axis=1)

    # --------------- Gamma_m to Cl ---------------------------
    Cl_r = Gamma2Cl(mat_plm_in, bin_mat, Gamma_m)
    Dl_r = ell_bin*(ell_bin+1)/2/np.pi*Cl_r
    ell_r = np.arange(Cl_r.size)
    Dl_bin1 = ell_tt*(ell_tt+1)/2/np.pi*Cl_r

    #bin
    Dl_bin_stack = []
    ell_bin_stack = []
    w = np.zeros(max(b))
    k = list(copy.copy(b))
    k.insert(0, 0)
    for i in np.arange(len(k)-1):
        temp = np.mean(Dl_bin1[k[i]:k[i+1]])
        Dl_bin_stack.append(temp)
        ell_bin_stack.append((k[i]+k[i+1]-1)/2)
        w[k[i]:k[i+1]] = temp

    # -------------- plot Cl2Gamma ------------------------
    m_tt = np.arange(m_start, 2000, 1)
    m = np.arange(m_start,len(Gamma_m),1)
    plt.plot(m, 2*m*Gamma_m, label='ring_scan')
    plt.plot(m_tt, 2*m_tt*Gamma_m_th, 'r', label='CMB_th')
    plt.legend()
    plt.xlabel('$m$')
    plt.ylabel('$2m\Gamma_m$')
    plt.title('TT')

    # ------------ plot Gamma2Cl ---------------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(w)), w, s=3, c='k')
    plt.scatter(ell_bin_stack, Dl_bin_stack, s=5, c='r', label='Dl_bin')
    plt.plot(ell_tt, Dl_tt*bl**2, 'b', label='Dl')
    plt.title('$\Gamma_m to C_\ell$, bin_size = round(' +
              str(rt)+'*'+str(q)+'^i); m_start='+str(m_start))
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$')
    plt.ylim(-1000, 6500)
    plt.xlim(-10, 2000)
    plt.legend()

    plt.show()
