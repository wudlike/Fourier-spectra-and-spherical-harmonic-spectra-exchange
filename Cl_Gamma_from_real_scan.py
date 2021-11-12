import numpy as np
import matplotlib.pyplot as plt
from pyshtools.expand import spharm
import healpy as hp
from settings import Settings
import copy
from time import time

st = Settings()

def Cl2map(Cl):
    '''
    CMB map realization
    '''
    return hp.synfast(Cl, nside=st.nside)


def ring2gamma(ring):
    samples = ring.shape[-1]
    yf = np.fft.rfft(ring)
    # normalization and averaging
    Gamma_m = np.mean((np.abs(yf)/samples)**2, axis=0)
    return Gamma_m


def plm_matrix(theta):
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
    powspect = np.loadtxt('../data/Cl/test_totCls.dat')  # tensor_to_scalar: r=0 (ell start from 2)
    # powspect = np.loadtxt('CMB_th_totCl.dat')
    ell_tt = powspect[m_start:2000, 0]  # ell
    # Cl_tt = powspect[m_start:2000, 1]   # TT
    # Dl_tt = Cl_tt*(ell_tt*(ell_tt+1))/(2*np.pi)

    Dl_tt = powspect[m_start:2000, 1]   # TT
    Cl_tt = Dl_tt*2*np.pi/(ell_tt*(ell_tt+1))

    theta = np.radians(80)  # elevation 40 deg,
    ell_len = ell_tt.size
    lmax = max(ell_tt).astype(int)

    # --------- Cl to Gamma_m --------------------------------------
    # smooth Cl with 7 arcmin
    # bl = hp.gauss_beam(np.radians(7/60), lmax=lmax)[m_start+2:]

    # without smoothing
    bl = 1
    mat_plm_in = (plm_matrix(theta)**2)[m_start:, m_start:]
    Gamma_m_th = Cl2Gamma(mat_plm_in, Cl_tt*bl**2)  # theory gamma_m

    # --------- Gamma_m from real ring scanning ----------------------
    map_cmb = Cl2map(Cl=Cl_tt)
    np.save('map_cmb_TT',map_cmb)

    # pixel index can be obtained by running scan_ring.py
    # pix = np.load('pix.npy')
    pix = np.load('pix_theta_80_deg.npy')
    ring_tt = map_cmb[pix]
    Gamma_m = ring2gamma(ring_tt)[m_start:m_start+ell_len]

    # -------- compute bin_matrix B ---------------------------
    rt = 10
    q = 1.03
    bin_mat, b = bin_matrix(q, rt)
    ell_bin = np.sum(bin_mat*b, axis=1)

    # --------------- Gamma_m to Cl ---------------------------
    Cl_r = Gamma2Cl(mat_plm_in, bin_mat, Gamma_m)
    Dl_r = ell_bin*(ell_bin+1)/2/np.pi*Cl_r
    ell_r = np.arange(Cl_r.size)
    Dl_bin1 = ell_tt*(ell_tt+1)/2/np.pi*Cl_r

    Dl_bin_stack = []
    ell_bin_stack = []
    w = np.zeros(max(b))
    k = list(copy.copy(b))
    k.insert(0,0)
    for i in np.arange(len(k)-1):
        temp = np.mean(Dl_bin1[k[i]:k[i+1]])
        Dl_bin_stack.append(temp)
        ell_bin_stack.append((k[i]+k[i+1]-1)/2)
        w[k[i]:k[i+1]] = temp

    e_time = time()
    print('time cost (second)>>> ', (e_time-s_time))

    # ------------ plot Gamma2Cl ---------------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(w)),w,s=3,c='k')
    plt.scatter(ell_bin_stack, Dl_bin_stack, s=5, c='r', label='Dl_bin')
    plt.plot(ell_tt, Dl_tt*bl**2, 'b', label='Dl')
    plt.title('$\Gamma_m to C_\ell$, bin_size = round(' +
              str(rt)+'*'+str(q)+'^i); m_start='+str(m_start))
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$')
    plt.ylim(-1000, 6500)
    plt.xlim(2, 2000)
    plt.legend()

    # -------------- plot Cl2Gamma ------------------------
    plt.figure(figsize=(12,6))
    m_tt = np.arange(m_start, 2000, 1)
    m = np.arange(m_start, m_start+len(Gamma_m), 1)
    plt.scatter(m, 2*m*Gamma_m,s=1,label='ring_scan')
    plt.plot(m_tt, 2*m_tt*Gamma_m_th,'r',label='CMB_th')
    plt.legend()
    plt.xlabel('$m$')
    plt.ylabel('$2m\Gamma_m$')
    plt.title('TT')

    plt.show()
