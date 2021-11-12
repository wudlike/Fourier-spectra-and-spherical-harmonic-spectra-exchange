import os
import quaternionic
import spherical
import matplotlib.pyplot as plt
import time
from scipy import special
import numpy as np

start_time = time.time()
root_dir=os.path.dirname(os.path.abspath('.'))
#   ell start from 2, delete the column of ell=0 and ell=1
m_start = 2
lmax=2000

#-----download data-----
# powspect = np.loadtxt(root_dir+'/data/test_totCls.dat')
# ell_l = powspect[m_start-2:lmax-1, 0]
# Gamma_12 = np.loadtxt(root_dir+'/data/gamma/gamma12_theory_70degree.txt')
# Gamma_13 = np.loadtxt(root_dir+'/data/gamma/gamma13_theory_70degree.txt')

powspect = np.loadtxt('../data/Cl/test_totCls.dat')
ell_l = powspect[m_start-2:lmax-1, 0]
Gamma_12 = np.loadtxt(
	'data/gamma12_theory_70degree.txt')
Gamma_13 = np.loadtxt(
	'data/gamma13_theory_70degree.txt')

Gamma_12 = Gamma_12[m_start-2:lmax-1]
Gamma_13 = Gamma_13[m_start-2:lmax-1]
#Gamma_12 = Gamma_12[m_start:lmax+1]
#Gamma_13 = Gamma_13[m_start:lmax+1]

theta = np.radians(70)
phi = np.radians(0)
wigner = spherical.Wigner(lmax)
R = quaternionic.array.from_spherical_coordinates(theta,phi)
Y0 = wigner.sYlm( 0,R)
Ya = wigner.sYlm( 2,R)
Yb = wigner.sYlm(-2,R)

plm1 = np.zeros((lmax-m_start+1,lmax-m_start+1))
plm2 = np.zeros((lmax-m_start+1,lmax-m_start+1))
plm3 = np.zeros((lmax-m_start+1,lmax-m_start+1))

for ell in range(m_start,lmax+1):
	for m in range(m_start,ell+1):
		plm1[m-m_start][ell-m_start] = Y0[wigner.Yindex(ell,m).real]
		plm2[m-m_start][ell-m_start] = Ya[wigner.Yindex(ell,m).real]
		plm3[m-m_start][ell-m_start] = Yb[wigner.Yindex(ell,m).real]
		m = m+1
	ell = ell+1

plm12 = plm1*plm2
plm13 = plm1*plm3

# bin_matrix B
q = 1.03 # bin strategy
rt = 10  # bin strategy
k = [round(rt*q**i) for i in range(lmax)]
w = np.cumsum(k)
idx1 = abs(w-m_start).argmin() 
if w[idx1] < m_start:
    bi1 = idx1 + 1
elif w[idx1] >= m_start:
    bi1 = idx1
idx2 = abs(w-lmax).argmin() 
if w[idx2] < lmax:
    bi2 = idx2 + 1
elif w[idx2] >= lmax:
    bi2 = idx2
bi = bi2 - bi1
bin_mat = np.zeros((lmax-m_start+1, bi+1))
h = np.arange(lmax).tolist()
a = 0
b = []
for i in range(bi1,bi2+1):
    s = round(rt*q**i)
    bin_mat[a:a+s, i-bi1] = 1
    b.append(int(np.mean(h[a+m_start:a+s+m_start])))
    a = a+s
ell_bin = np.sum(bin_mat*b, axis=1)

plm12bin = np.matmul(plm12, bin_mat)
plm13bin = np.matmul(plm13, bin_mat)
plm_mid = np.matmul(bin_mat, np.linalg.pinv(np.matmul(plm12bin.T, plm12bin)+np.matmul(plm13bin.T, plm13bin)))

Cl_TE =-np.matmul(plm_mid, np.matmul(plm12bin.T, Gamma_12)+np.matmul(plm13bin.T, Gamma_13))

Dl_TE = ell_bin*(ell_bin+1)/(2*np.pi)*Cl_TE

#-----Plot these points-----
Dl_TE_bin = []
ell_l_bin = []
for i in range(np.shape(b)[0]):
	Dl_TE_bin.append(Dl_TE[b[i]-m_start])
	ell_l_bin.append(ell_l[b[i]-m_start]) 
