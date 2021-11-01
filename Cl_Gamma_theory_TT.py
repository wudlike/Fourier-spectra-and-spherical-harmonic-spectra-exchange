import numpy as np
import matplotlib.pyplot as plt
from pyshtools.expand import spharm
import healpy as hp
from time import time

def plm_matrix():
    matrix_plm = spharm(ell_len-1+m_start, theta, 0, normalization='ortho',
                        degrees=False, kind='complex', csphase=1)[0].real.T
    return matrix_plm


def Gamma2Cl(mat_plm, B, gamma):
    return np.matmul(np.matmul(B, np.linalg.pinv(np.matmul(mat_plm, B))), gamma)


def Cl2Gamma(mat_plm, Cl_plt):
    return np.matmul(mat_plm, Cl_plt)


def bin_matrix(q,rt):
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
    return bin_mat,bin 

if __name__ == '__main__':

    s_time = time()
    # -------- load theoretical Cl data -------------------------------
    m_start = 0
    powspect = np.loadtxt('data/test_totCls.dat')  # tensor_to_scalar: r=0
    ell_tt = powspect[m_start:2000, 0]  # ell
    Dl_tt = powspect[m_start:2000, 1]   # TT
    Cl_tt = Dl_tt*2*np.pi/(ell_tt*(ell_tt+1))
    theta = np.radians(80)  
    ell_len = ell_tt.size
    lmax = max(ell_tt).astype(int)

    # --------- Cl to Gamma_m --------------------------------------
    # smooth Cl with 7 arcmin
    # bl = hp.gauss_beam(np.radians(20/60), lmax=lmax)[m_start+2:]
    
    # without smoothing
    bl = 1
    mat_plm_in = (plm_matrix()**2)[m_start:, m_start:]
    Gamma_m = Cl2Gamma(mat_plm_in, Cl_tt*bl**2)  # theory gamma_m

    # -------- compute bin_matrix B ---------------------------
    rt = 3
    q = 1.02
    bin_mat,b = bin_matrix(q,rt)
    ell_bin = np.sum(bin_mat*b, axis=1)

    # --------------- Gamma_m to Cl ---------------------------
    Cl_r = Gamma2Cl(mat_plm_in, bin_mat, Gamma_m)
    Dl_r = ell_bin*(ell_bin+1)/2/np.pi*Cl_r
    ell_r = np.arange(Cl_r.size)
    Dl_bin = ell_tt*(ell_tt+1)/2/np.pi*Cl_r
    Dl_bin = Dl_bin[b]
    ell_tt_bin = ell_tt[b]
    e_time = time()
    print(Dl_bin.shape)
    print('time cost (second)>>> ', (e_time-s_time))

    # ------------ plot Gamma2Cl ---------------------------------
    plt.figure(figsize=(14, 8))
    plt.scatter(ell_tt_bin, Dl_bin, s=3, c='r', label='Dl_bin')
    plt.plot(ell_tt, Dl_tt*bl**2, 'b', label='Dl')
    plt.title('$\Gamma_m to C_\ell$, bin_size = round(' +
            str(rt)+'*'+str(q)+'^i); m_start='+str(m_start))
    plt.xlabel('$\ell$')
    plt.ylabel('$D_\ell$')
    plt.ylim(-1000, 6500)
    plt.xlim(-10, 2000)
    plt.legend()

    # -------------- plot Cl2Gamma ------------------------
    m_tt = np.arange(m_start, 2000, 1)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('$C_\ell to \Gamma_m$')
    ax[1].semilogx(m_tt, np.sqrt(2*m_tt*Gamma_m))
    ax[1].set_xlabel('$m$')
    ax[1].set_ylabel('$(2m\Gamma_m)^{1/2}$')
    ax[0].semilogx(ell_tt, np.sqrt(Dl_tt/2*bl**2), label='Cl')
    ax[0].set_xlabel('$\ell$')
    ax[0].set_ylabel('$(\ell(2\ell+1)C_\ell/4\pi)^{1/2}$')

    plt.show()
