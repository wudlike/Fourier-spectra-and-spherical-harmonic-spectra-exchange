import os
import numpy as np
import quaternionic
import spherical

root_dir=os.path.dirname(os.path.abspath('.'))
#   ell start from 2, delete the column of ell=0 and ell=1
m_start = 2
lmax=2000

#-----load data-----
powspect = np.loadtxt(root_dir+'/data/test_totCls.dat')
Gamma_22 = np.loadtxt(root_dir+'/data/gamma/gamma22_theory_70degree.txt')
Gamma_33 = np.loadtxt(root_dir+'/data/gamma/gamma33_theory_70degree.txt')
Gamma_23 = np.loadtxt(root_dir+'/data/gamma/gamma23_theory_70degree.txt')

ell_l = powspect[m_start-2:lmax-1, 0]
Gamma_22 = Gamma_22[m_start-2:lmax-1]
Gamma_33 = Gamma_33[m_start-2:lmax-1]
Gamma_23 = Gamma_23[m_start-2:lmax-1]
#Gamma_22 = Gamma_22[m_start:lmax+1]
#Gamma_33 = Gamma_33[m_start:lmax+1]
#Gamma_23 = Gamma_23[m_start:lmax+1]

theta = np.radians(70)
phi = np.radians(0)
wigner = spherical.Wigner(lmax)
R = quaternionic.array.from_spherical_coordinates(theta,phi)
Ya = wigner.sYlm( 2,R)
Yb = wigner.sYlm(-2,R)

plm2 = np.zeros((lmax-m_start+1,lmax-m_start+1))
plm3 = np.zeros((lmax-m_start+1,lmax-m_start+1))

for ell in range(m_start,lmax+1):
	for m in range(m_start,ell+1):
		plm2[m-m_start][ell-m_start] = Ya[wigner.Yindex(ell,m).real]
		plm3[m-m_start][ell-m_start] = Yb[wigner.Yindex(ell,m).real]
		m = m+1
	ell = ell+1

plm22 = plm2**2
plm33 = plm3**2
plm23 = plm2*plm3

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

plm22bin = np.matmul(plm22, bin_mat)
plm33bin = np.matmul(plm33, bin_mat)
plm23bin = np.matmul(plm23, bin_mat)
plm_mid = np.matmul(bin_mat, np.linalg.pinv(np.matmul(plm22bin.T, plm22bin)+np.matmul(plm33bin.T, plm33bin)))
Cl_mid_1 = np.matmul(plm_mid, 0.5*np.matmul(plm22bin.T, Gamma_22)+0.5*np.matmul(plm33bin.T, Gamma_33))
Cl_mid_2 = 0.5*np.matmul(np.matmul(np.matmul(bin_mat, np.linalg.pinv(np.matmul(plm23bin.T, plm23bin))), plm23bin.T), Gamma_23)

Cl_EE = Cl_mid_1 + Cl_mid_2
Cl_BB = Cl_mid_1 - Cl_mid_2

Dl_EE = ell_bin*(ell_bin+1)/(2*np.pi)*Cl_EE
Dl_BB = ell_bin*(ell_bin+1)/(2*np.pi)*Cl_BB

#-----Plot these points-----
Dl_EE_bin = []
Dl_BB_bin = []
ell_l_bin = []
for i in range(np.shape(b)[0]):
	Dl_EE_bin.append(Dl_EE[b[i]-m_start])
	Dl_BB_bin.append(Dl_BB[b[i]-m_start])
	ell_l_bin.append(ell_l[b[i]-m_start]) 


