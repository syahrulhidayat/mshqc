import numpy as np

# Let's test a small RHF system (e.g., na=2, va=3, nbf=5) with random symmetric 2-electron integrals and H_core
np.random.seed(42)
na = 2
va = 3
nbf = na + va

# Create random symmetric h_ao and 8-fold symmetric eri_ao (pq|rs)
A = np.random.randn(nbf, nbf)
h_ao = 0.5 * (A + A.T)
# Make occupied orbitals lower energy
for i in range(na):
    h_ao[i, i] -= 5.0
for a in range(na, nbf):
    h_ao[a, a] += 5.0

M = np.random.randn(nbf*nbf, nbf*nbf) * 0.1
M = 0.5 * (M + M.T)
eri_ao = M.reshape(nbf, nbf, nbf, nbf)
# symmetrize pq and rs
eri_ao = 0.25 * (eri_ao + eri_ao.transpose(1,0,2,3) + eri_ao.transpose(0,1,3,2) + eri_ao.transpose(1,0,3,2))
eri_ao = 0.5 * (eri_ao + eri_ao.transpose(2,3,0,1))

# Start with random orthogonal C (slightly rotated from diagonal h_ao)
K_init = np.random.randn(nbf, nbf) * 0.1
K_init = 0.5 * (K_init - K_init.T)
from scipy.linalg import expm
C_init = expm(K_init)

def eval_omp3_mshqc(C):
    # Pseudocanonicalize C like MSHQC
    P_a = C[:, :na] @ C[:, :na].T
    P_b = P_a.copy()
    J = np.einsum('pqrs,rs->pq', eri_ao, P_a + P_b)
    K = np.einsum('prqs,rs->pq', eri_ao, P_a)
    F_ao = h_ao + J - K
    e_hf = np.sum(P_a * (h_ao + F_ao))
    
    F_mo = C.T @ F_ao @ C
    ea_o, U_o = np.linalg.eigh(F_mo[:na, :na])
    ea_v, U_v = np.linalg.eigh(F_mo[na:, na:])
    
    C_can = C.copy()
    C_can[:, :na] = C[:, :na] @ U_o
    C_can[:, na:] = C[:, na:] @ U_v
    ea = np.concatenate([ea_o, ea_v])
    
    Cao = C_can[:, :na]
    Cav = C_can[:, na:]
    
    # Transform integrals
    V_ovov = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao, Cav, Cao, Cav)
    V_oovv = np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Cao, Cao, Cav, Cav)
    V_vvvv = np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cav, Cav, Cav, Cav)
    V_oooo = np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Cao, Cao, Cao, Cao)
    
    # T2_aa_ijab: wait! How is t2_aa_dense defined in MSHQC OMP2?
    # Let's check: in MSHQC, D = ea[i] + ea[j] - ea[na+a] - ea[na+b]
    D = ea[:na, None, None, None] + ea[None, :na, None, None] - ea[None, None, na:, None] - ea[None, None, None, na:]
    T2 = np.einsum('iajb->ijab', V_ovov) / D
    
    # MP2 energy
    e_mp2 = np.sum(np.einsum('iajb->ijab', V_ovov) * (2.0 * T2 - T2.transpose(0,1,3,2)))
    
    # MP3 W in MSHQC (lines 436-460)
    W = np.zeros((na, na, va, va))
    W += np.einsum('ijef,eafb->ijab', T2, V_vvvv)
    W += np.einsum('mnab,minj->ijab', T2, V_oooo)
    
    W += 2.0 * np.einsum('iakc,kjcb->ijab', V_ovov, T2)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', V_oovv, T2)
    W += 2.0 * np.einsum('ikac,kcjb->ijab', T2, V_ovov)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', T2, V_oovv)
    W += -1.0 * np.einsum('ikca,kcjb->ijab', T2, V_ovov)
    W += -1.0 * np.einsum('iakc,kjbc->ijab', V_ovov, T2)
    W += -1.0 * np.einsum('ikbc,kjac->ijab', V_oovv, T2)
    W += -1.0 * np.einsum('ikcb,jkac->ijab', T2, V_oovv)
    
    L2 = W / D
    e_mp3 = np.sum(W * (2.0 * T2 - T2.transpose(0,1,3,2)))
    
    return e_hf, e_mp2, e_mp3, C_can, ea, T2, L2, F_ao

e_hf, e_mp2, e_mp3, C_can, ea, T2, L2, F_ao = eval_omp3_mshqc(C_init)
print("e_hf:", e_hf, "e_mp2:", e_mp2, "e_mp3:", e_mp3)

# Let's implement MSHQC's analytical gradient from mp3.cc (restricted) and compare with finite difference!
e_hf, e_mp2, e_mp3, C_can, ea, T2, L2, F_ao = eval_omp3_mshqc(C_init)

Cao = C_can[:, :na]
Cav = C_can[:, na:]

T2_tilde = 2.0 * T2 - T2.transpose(0,1,3,2)
L2_tilde = 2.0 * L2 - L2.transpose(0,1,3,2)

# build_opdm_alpha (lines 803-809)
G_oo_mp2 = -1.0 * np.einsum('ikab,jkab->ij', T2, T2_tilde)
G_oo_mp3 = -0.5 * np.einsum('ikab,jkab->ij', T2, L2_tilde) - 0.5 * np.einsum('ikab,jkab->ij', L2, T2_tilde)
G_oo = G_oo_mp2 + G_oo_mp3

G_vv_mp2 = 1.0 * np.einsum('ijac,ijbc->ab', T2, T2_tilde)
G_vv_mp3 = 0.5 * np.einsum('ijac,ijbc->ab', T2, L2_tilde) + 0.5 * np.einsum('ijac,ijbc->ab', L2, T2_tilde)
G_vv = G_vv_mp2 + G_vv_mp3

# build_generalized_fock (lines 895-1122)
Teff = T2_tilde + L2_tilde
V_ovvv = np.einsum('pqrs,pi,qa,rb,sc->iabc', eri_ao, Cao, Cav, Cav, Cav)
V_ooov = np.einsum('pqrs,pi,qj,rk,sa->ijka', eri_ao, Cao, Cao, Cao, Cav)

Z_mat = np.zeros((va, na))
Z_mat += 1.0 * np.einsum('kcab,ikbc->ai', V_ovvv, Teff)
Z_mat += -1.0 * np.einsum('jikc,jkac->ai', V_ooov, Teff)

# 2. Explicit 3rd order TPDM (lines 976-1025)
G_vvvv = 1.0 * np.einsum('ijab,ijef->eafb', T2_tilde, T2)
Z_mat += -1.0 * np.einsum('xafb,mafb->xm', G_vvvv, V_ovvv)
Z_mat += -1.0 * np.einsum('exfb,mefb->xm', G_vvvv, V_ovvv)
Z_mat += -1.0 * np.einsum('eaxb,mbea->xm', G_vvvv, V_ovvv)
Z_mat += -1.0 * np.einsum('eafx,mfae->xm', G_vvvv, V_ovvv)

G_oooo = 1.0 * np.einsum('ijab,mnab->minj', T2_tilde, T2)
Z_mat += 1.0 * np.einsum('minj,njix->xm', G_oooo, V_ooov)
Z_mat += 1.0 * np.einsum('imnj,njix->xm', G_oooo, V_ooov)
Z_mat += 1.0 * np.einsum('inmj,injx->xm', G_oooo, V_ooov)
Z_mat += 1.0 * np.einsum('injm,injx->xm', G_oooo, V_ooov)

G_ovov = np.zeros((na, va, na, va))
G_oovv = np.zeros((na, na, va, va))
G_ovov += 1.0 * np.einsum('kjcb,ijab->iakc', T2, T2_tilde)
G_oovv += -1.0 * np.einsum('kjcb,ijab->ikac', T2, T2_tilde)

G_ovov += -1.0 * np.einsum('kjbc,ijab->iakc', T2, T2_tilde)
G_ovov += 2.0 * np.einsum('miea,mkec->iakc', T2, T2_tilde)
G_ovov += -1.0 * np.einsum('miae,mkec->iakc', T2, T2_tilde)

G_oovv += -1.0 * np.einsum('miea,mkec->ikac', T2, T2_tilde)
G_oovv += -1.0 * np.einsum('kmec,imea->ikac', T2, T2_tilde)
G_oovv += -1.0 * np.einsum('mkce,miae->ikac', T2, T2_tilde)

Z_mat += 1.0 * np.einsum('makc,kcxa->xm', G_ovov, V_ovvv)
Z_mat += -1.0 * np.einsum('ixkc,imkc->xm', G_ovov, V_ooov)
Z_mat += 1.0 * np.einsum('iamc,iaxc->xm', G_ovov, V_ovvv)
Z_mat += -1.0 * np.einsum('iakx,kmia->xm', G_ovov, V_ooov)

Z_mat += 1.0 * np.einsum('mkac,kxca->xm', G_oovv, V_ovvv)
Z_mat += 1.0 * np.einsum('imac,ixac->xm', G_oovv, V_ovvv)
Z_mat += -1.0 * np.einsum('ikxc,ikmc->xm', G_oovv, V_ooov)
Z_mat += -1.0 * np.einsum('ikax,ikma->xm', G_oovv, V_ooov)

G_full = np.zeros((nbf, nbf))
G_full[:na, :na] = G_oo
G_full[na:, na:] = G_vv
P_corr = C_can @ G_full @ C_can.T

J_corr = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_corr)
K_corr = np.einsum('prqs,rs->pq', eri_ao, P_corr)
G_gamma_ao = J_corr - K_corr
G_gamma_mo = C_can.T @ G_gamma_ao @ C_can

F_HF_mo = C_can.T @ F_ao @ C_can
F_gen = F_HF_mo + G_gamma_mo
F_HF_vo = F_HF_mo[na:, :na]
L_sep = F_HF_vo @ G_oo - G_vv @ F_HF_vo
F_gen[na:, :na] += L_sep + Z_mat

grad_ana = -4.0 * F_gen[na:, :na] # shape (va, na)

# Finite difference from C_can
eps = 1e-6
grad_num_tot = np.zeros((va, na))
grad_num_mp2 = np.zeros((va, na))
grad_num_mp3 = np.zeros((va, na))
for i in range(na):
    for a in range(va):
        U = np.eye(nbf)
        U[i, na+a] = eps
        U[na+a, i] = -eps
        # Exact orthogonal rotation exp(K)
        K = np.zeros((nbf, nbf))
        K[i, na+a] = eps
        K[na+a, i] = -eps
        eh_p, em2_p, em3_p, *_ = eval_omp3_mshqc(C_can @ expm(K))
        eh_m, em2_m, em3_m, *_ = eval_omp3_mshqc(C_can @ expm(-K))
        grad_num_tot[a, i] = ((eh_p + em2_p + em3_p) - (eh_m + em2_m + em3_m)) / (2 * eps)
        grad_num_mp2[a, i] = ((eh_p + em2_p) - (eh_m + em2_m)) / (2 * eps)
        grad_num_mp3[a, i] = (em3_p - em3_m) / (2 * eps)

print("Grad Ana Tot:\n", grad_ana)
print("Grad Num Tot:\n", grad_num_tot)
print("Diff:\n", grad_ana - grad_num_tot)

# Let's separate the MP2 analytical gradient and MP3 analytical gradient!
# First check if MP2 analytical gradient matches grad_num_mp2 to 1e-10!
G_full_mp2 = np.zeros((nbf, nbf))
G_full_mp2[:na, :na] = G_oo_mp2
G_full_mp2[na:, na:] = G_vv_mp2
P_corr_mp2 = C_can @ G_full_mp2 @ C_can.T
G_gamma_ao_mp2 = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_corr_mp2) - np.einsum('prqs,rs->pq', eri_ao, P_corr_mp2)
G_gamma_mo_mp2 = C_can.T @ G_gamma_ao_mp2 @ C_can

Z_mat_mp2 = np.einsum('kcab,ikbc->ai', V_ovvv, T2_tilde) - np.einsum('jikc,jkac->ai', V_ooov, T2_tilde)
L_sep_mp2 = F_HF_vo @ G_oo_mp2 - G_vv_mp2 @ F_HF_vo
F_gen_mp2_vo = F_HF_vo + G_gamma_mo_mp2[na:, :na] + L_sep_mp2 + Z_mat_mp2
grad_ana_mp2 = -4.0 * F_gen_mp2_vo

print("MP2 Grad Diff (Ana - Num):\n", grad_ana_mp2 - grad_num_mp2)
print("\nMP3 Grad Ana:\n", grad_ana - grad_ana_mp2)
print("MP3 Grad Num:\n", grad_num_mp3)
print("MP3 Grad Diff (Ana - Num):\n", (grad_ana - grad_ana_mp2) - grad_num_mp3)

# Let's test the two parts of d(e_mp3)/d(kappa) separately by finite difference!
# Part 1: Keep V_vvvv, V_oooo, V_ovov, V_oovv FIXED at C_can, and only vary T2(C)!
# Part 2: Keep T2 FIXED at C_can, and only vary V_vvvv(C), V_oooo(C), V_ovov(C), V_oovv(C)!

 Cao = C_can[:, :na]
Cav = C_can[:, na:]
V_ovov_0 = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao, Cav, Cao, Cav)
V_oovv_0 = np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Cao, Cao, Cav, Cav)
V_vvvv_0 = np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cav, Cav, Cav, Cav)
V_oooo_0 = np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Cao, Cao, Cao, Cao)

def compute_W(T2_in, V_vvvv, V_oooo, V_ovov, V_oovv):
    W = np.zeros((na, na, va, va))
    W += np.einsum('ijef,eafb->ijab', T2_in, V_vvvv)
    W += np.einsum('mnab,minj->ijab', T2_in, V_oooo)
    W += 2.0 * np.einsum('iakc,kjcb->ijab', V_ovov, T2_in)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', V_oovv, T2_in)
    W += 2.0 * np.einsum('ikac,kcjb->ijab', T2_in, V_ovov)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', T2_in, V_oovv)
    W += -1.0 * np.einsum('ikca,kcjb->ijab', T2_in, V_ovov)
    W += -1.0 * np.einsum('iakc,kjbc->ijab', V_ovov, T2_in)
    W += -1.0 * np.einsum('ikbc,kjac->ijab', V_oovv, T2_in)
    W += -1.0 * np.einsum('ikcb,jkac->ijab', T2_in, V_oovv)
    return W

# Wait! In eval_omp3_mshqc, C_can has a gauge choice from eigh(F_oo) and eigh(F_vv)!
# Because eigh(F_oo) rotates within OO and VV subspaces, T2_0 and V_0 must be compared in the SAME orbital basis,
# OR we must evaluate the invariant trace!
# Wait: how does OMP2 avoid gauge dependence of pseudocanonical orbitals?
# In OMP2, T2 satisfies: F_oo T2 + T2 F_vv - ... = V_ijab in ANY OO/VV basis!
# Let's write the basis-independent (non-pseudocanonical) T2 and W equations so we can test Part 1 and Part 2 without OO/VV eigenvector sign/rotation ambiguity!

def solve_T2_general(C):
    P_a = C[:, :na] @ C[:, :na].T
    J = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_a)
    K = np.einsum('prqs,rs->pq', eri_ao, P_a)
    F_ao = h_ao + J - K
    F_mo = C.T @ F_ao @ C
    
    # Transform to diagonal basis of F_oo and F_vv, solve, then transform back to C basis!
    ea_o, U_o = np.linalg.eigh(F_mo[:na, :na])
    ea_v, U_v = np.linalg.eigh(F_mo[na:, na:])
    Cao_c = C[:, :na] @ U_o
    Cav_c = C[:, na:] @ U_v
    V_ovov_c = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao_c, Cav_c, Cao_c, Cav_c)
    D = ea_o[:, None, None, None] + ea_o[None, :, None, None] - ea_v[None, None, :, None] - ea_v[None, None, None, :]
    T2_c = np.einsum('iajb->ijab', V_ovov_c) / D
    # Transform T2_c back to original C basis (without U_o, U_v rotation)
    T2_orig = np.einsum('ijab,pi,qj,ra,sb->pqrs', T2_c, U_o, U_o, U_v, U_v)
    return T2_orig

T2_0 = solve_T2_general(C_can) # Since C_can is already pseudocanonical, U_o = I, U_v = I

grad_num_part1 = np.zeros((va, na)) # varying T2 only
grad_num_part2_vvvv = np.zeros((va, na))
grad_num_part2_oooo = np.zeros((va, na))
grad_num_part2_ring = np.zeros((va, na))

for i in range(na):
    for a in range(va):
        K = np.zeros((nbf, nbf))
        K[i, na+a] = eps
        K[na+a, i] = -eps
        Cp = C_can @ expm(K)
        Cm = C_can @ expm(-K)
        
        T2_p = solve_T2_general(Cp)
        T2_m = solve_T2_general(Cm)
        
        # Part 1: vary T2, keep V fixed at C_can
        ep_1 = np.sum(compute_W(T2_p, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0) * (2.0 * T2_p - T2_p.transpose(0,1,3,2)))
        em_1 = np.sum(compute_W(T2_m, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0) * (2.0 * T2_m - T2_m.transpose(0,1,3,2)))
        grad_num_part1[a, i] = (ep_1 - em_1) / (2 * eps)
        
        # Part 2: keep T2 fixed at T2_0, vary V(C)
        def get_Vs(C_curr):
            Co = C_curr[:, :na]
            Cv = C_curr[:, na:]
            return (
                np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cv, Cv, Cv, Cv),
                np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Co, Co, Co, Co),
                np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Co, Cv, Co, Cv),
                np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Co, Co, Cv, Cv)
            )
        V_vvvv_p, V_oooo_p, V_ovov_p, V_oovv_p = get_Vs(Cp)
        V_vvvv_m, V_oooo_m, V_ovov_m, V_oovv_m = get_Vs(Cm)
        
        T2_0_t = 2.0 * T2_0 - T2_0.transpose(0,1,3,2)
        ep_vvvv = np.sum(np.einsum('ijef,eafb->ijab', T2_0, V_vvvv_p) * T2_0_t)
        em_vvvv = np.sum(np.einsum('ijef,eafb->ijab', T2_0, V_vvvv_m) * T2_0_t)
        grad_num_part2_vvvv[a, i] = (ep_vvvv - em_vvvv) / (2 * eps)

        ep_oooo = np.sum(np.einsum('mnab,minj->ijab', T2_0, V_oooo_p) * T2_0_t)
        em_oooo = np.sum(np.einsum('mnab,minj->ijab', T2_0, V_oooo_m) * T2_0_t)
        grad_num_part2_oooo[a, i] = (ep_oooo - em_oooo) / (2 * eps)

        ep_ring = np.sum(compute_W(T2_0, np.zeros_like(V_vvvv_0), np.zeros_like(V_oooo_0), V_ovov_p, V_oovv_p) * T2_0_t)
        em_ring = np.sum(compute_W(T2_0, np.zeros_like(V_vvvv_0), np.zeros_like(V_oooo_0), V_ovov_m, V_oovv_m) * T2_0_t)
        grad_num_part2_ring[a, i] = (ep_ring - em_ring) / (2 * eps)

print("Sum of parts vs total MP3 num grad diff:\n", (grad_num_part1 + grad_num_part2_vvvv + grad_num_part2_oooo + grad_num_part2_ring) - grad_num_mp3)

Cao = C_can[:, :na]
Cav = C_can[:, na:]
V_ovov_0 = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao, Cav, Cao, Cav)
V_oovv_0 = np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Cao, Cao, Cav, Cav)
V_vvvv_0 = np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cav, Cav, Cav, Cav)
V_oooo_0 = np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Cao, Cao, Cao, Cao)

def compute_W(T2_in, V_vvvv, V_oooo, V_ovov, V_oovv):
    W = np.zeros((na, na, va, va))
    W += np.einsum('ijef,eafb->ijab', T2_in, V_vvvv)
    W += np.einsum('mnab,minj->ijab', T2_in, V_oooo)
    W += 2.0 * np.einsum('iakc,kjcb->ijab', V_ovov, T2_in)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', V_oovv, T2_in)
    W += 2.0 * np.einsum('ikac,kcjb->ijab', T2_in, V_ovov)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', T2_in, V_oovv)
    W += -1.0 * np.einsum('ikca,kcjb->ijab', T2_in, V_ovov)
    W += -1.0 * np.einsum('iakc,kjbc->ijab', V_ovov, T2_in)
    W += -1.0 * np.einsum('ikbc,kjac->ijab', V_oovv, T2_in)
    W += -1.0 * np.einsum('ikcb,jkac->ijab', T2_in, V_oovv)
    return W

def solve_T2_general(C):
    P_a = C[:, :na] @ C[:, :na].T
    J = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_a)
    K = np.einsum('prqs,rs->pq', eri_ao, P_a)
    F_ao = h_ao + J - K
    F_mo = C.T @ F_ao @ C
    
    ea_o, U_o = np.linalg.eigh(F_mo[:na, :na])
    ea_v, U_v = np.linalg.eigh(F_mo[na:, na:])
    Cao_c = C[:, :na] @ U_o
    Cav_c = C[:, na:] @ U_v
    V_ovov_c = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao_c, Cav_c, Cao_c, Cav_c)
    D = ea_o[:, None, None, None] + ea_o[None, :, None, None] - ea_v[None, None, :, None] - ea_v[None, None, None, :]
    T2_c = np.einsum('iajb->ijab', V_ovov_c) / D
    T2_orig = np.einsum('ijab,pi,qj,ra,sb->pqrs', T2_c, U_o, U_o, U_v, U_v)
    return T2_orig

T2_0 = solve_T2_general(C_can)

grad_num_part1 = np.zeros((va, na))
grad_num_part2_vvvv = np.zeros((va, na))
grad_num_part2_oooo = np.zeros((va, na))
grad_num_part2_ring = np.zeros((va, na))

for i in range(na):
    for a in range(va):
        K = np.zeros((nbf, nbf))
        K[i, na+a] = eps
        K[na+a, i] = -eps
        Cp = C_can @ expm(K)
        Cm = C_can @ expm(-K)
        
        T2_p = solve_T2_general(Cp)
        T2_m = solve_T2_general(Cm)
        
        ep_1 = np.sum(compute_W(T2_p, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0) * (2.0 * T2_p - T2_p.transpose(0,1,3,2)))
        em_1 = np.sum(compute_W(T2_m, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0) * (2.0 * T2_m - T2_m.transpose(0,1,3,2)))
        grad_num_part1[a, i] = (ep_1 - em_1) / (2 * eps)
        
        def get_Vs(C_curr):
            Co = C_curr[:, :na]
            Cv = C_curr[:, na:]
            return (
                np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cv, Cv, Cv, Cv),
                np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Co, Co, Co, Co),
                np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Co, Cv, Co, Cv),
                np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Co, Co, Cv, Cv)
            )
        V_vvvv_p, V_oooo_p, V_ovov_p, V_oovv_p = get_Vs(Cp)
        V_vvvv_m, V_oooo_m, V_ovov_m, V_oovv_m = get_Vs(Cm)
        
        T2_0_t = 2.0 * T2_0 - T2_0.transpose(0,1,3,2)
        ep_vvvv = np.sum(np.einsum('ijef,eafb->ijab', T2_0, V_vvvv_p) * T2_0_t)
        em_vvvv = np.sum(np.einsum('ijef,eafb->ijab', T2_0, V_vvvv_m) * T2_0_t)
        grad_num_part2_vvvv[a, i] = (ep_vvvv - em_vvvv) / (2 * eps)

        ep_oooo = np.sum(np.einsum('mnab,minj->ijab', T2_0, V_oooo_p) * T2_0_t)
        em_oooo = np.sum(np.einsum('mnab,minj->ijab', T2_0, V_oooo_m) * T2_0_t)
        grad_num_part2_oooo[a, i] = (ep_oooo - em_oooo) / (2 * eps)

        ep_ring = np.sum(compute_W(T2_0, np.zeros_like(V_vvvv_0), np.zeros_like(V_oooo_0), V_ovov_p, V_oovv_p) * T2_0_t)
        em_ring = np.sum(compute_W(T2_0, np.zeros_like(V_vvvv_0), np.zeros_like(V_oooo_0), V_ovov_m, V_oovv_m) * T2_0_t)
        grad_num_part2_ring[a, i] = (ep_ring - em_ring) / (2 * eps)

print("Sum of parts vs total MP3 num grad diff:\n", (grad_num_part1 + grad_num_part2_vvvv + grad_num_part2_oooo + grad_num_part2_ring) - grad_num_mp3)

Cao = C_can[:, :na]
Cav = C_can[:, na:]
V_ovov_0 = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao, Cav, Cao, Cav)
V_oovv_0 = np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Cao, Cao, Cav, Cav)
V_vvvv_0 = np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cav, Cav, Cav, Cav)
V_oooo_0 = np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Cao, Cao, Cao, Cao)

def compute_W(T2_in, V_vvvv, V_oooo, V_ovov, V_oovv):
    W = np.zeros((na, na, va, va))
    W += np.einsum('ijef,eafb->ijab', T2_in, V_vvvv)
    W += np.einsum('mnab,minj->ijab', T2_in, V_oooo)
    W += 2.0 * np.einsum('iakc,kjcb->ijab', V_ovov, T2_in)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', V_oovv, T2_in)
    W += 2.0 * np.einsum('ikac,kcjb->ijab', T2_in, V_ovov)
    W += -1.0 * np.einsum('ikac,kjcb->ijab', T2_in, V_oovv)
    W += -1.0 * np.einsum('ikca,kcjb->ijab', T2_in, V_ovov)
    W += -1.0 * np.einsum('iakc,kjbc->ijab', V_ovov, T2_in)
    W += -1.0 * np.einsum('ikbc,kjac->ijab', V_oovv, T2_in)
    W += -1.0 * np.einsum('ikcb,jkac->ijab', T2_in, V_oovv)
    return W

def solve_T2_general(C):
    P_a = C[:, :na] @ C[:, :na].T
    J = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_a)
    K = np.einsum('prqs,rs->pq', eri_ao, P_a)
    F_ao = h_ao + J - K
    F_mo = C.T @ F_ao @ C
    
    ea_o, U_o = np.linalg.eigh(F_mo[:na, :na])
    ea_v, U_v = np.linalg.eigh(F_mo[na:, na:])
    Cao_c = C[:, :na] @ U_o
    Cav_c = C[:, na:] @ U_v
    V_ovov_c = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao_c, Cav_c, Cao_c, Cav_c)
    D = ea_o[:, None, None, None] + ea_o[None, :, None, None] - ea_v[None, None, :, None] - ea_v[None, None, None, :]
    T2_c = np.einsum('iajb->ijab', V_ovov_c) / D
    T2_orig = np.einsum('ijab,pi,qj,ra,sb->pqrs', T2_c, U_o, U_o, U_v, U_v)
    return T2_orig

T2_0 = solve_T2_general(C_can)

grad_num_part1 = np.zeros((va, na))
grad_num_part2_vvvv = np.zeros((va, na))
grad_num_part2_oooo = np.zeros((va, na))
grad_num_part2_ring = np.zeros((va, na))

for i in range(na):
    for a in range(va):
        K = np.zeros((nbf, nbf))
        K[i, na+a] = eps
        K[na+a, i] = -eps
        Cp = C_can @ expm(K)
        Cm = C_can @ expm(-K)
        
        T2_p = solve_T2_general(Cp)
        T2_m = solve_T2_general(Cm)
        
        ep_1 = np.sum(compute_W(T2_p, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0) * (2.0 * T2_p - T2_p.transpose(0,1,3,2)))
        em_1 = np.sum(compute_W(T2_m, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0) * (2.0 * T2_m - T2_m.transpose(0,1,3,2)))
        grad_num_part1[a, i] = (ep_1 - em_1) / (2 * eps)
        
        def get_Vs(C_curr):
            Co = C_curr[:, :na]
            Cv = C_curr[:, na:]
            return (
                np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cv, Cv, Cv, Cv),
                np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Co, Co, Co, Co),
                np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Co, Cv, Co, Cv),
                np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Co, Co, Cv, Cv)
            )
        V_vvvv_p, V_oooo_p, V_ovov_p, V_oovv_p = get_Vs(Cp)
        V_vvvv_m, V_oooo_m, V_ovov_m, V_oovv_m = get_Vs(Cm)
        
        T2_0_t = 2.0 * T2_0 - T2_0.transpose(0,1,3,2)
        ep_vvvv = np.sum(np.einsum('ijef,eafb->ijab', T2_0, V_vvvv_p) * T2_0_t)
        em_vvvv = np.sum(np.einsum('ijef,eafb->ijab', T2_0, V_vvvv_m) * T2_0_t)
        grad_num_part2_vvvv[a, i] = (ep_vvvv - em_vvvv) / (2 * eps)

        ep_oooo = np.sum(np.einsum('mnab,minj->ijab', T2_0, V_oooo_p) * T2_0_t)
        em_oooo = np.sum(np.einsum('mnab,minj->ijab', T2_0, V_oooo_m) * T2_0_t)
        grad_num_part2_oooo[a, i] = (ep_oooo - em_oooo) / (2 * eps)

        ep_ring = np.sum(compute_W(T2_0, np.zeros_like(V_vvvv_0), np.zeros_like(V_oooo_0), V_ovov_p, V_oovv_p) * T2_0_t)
        em_ring = np.sum(compute_W(T2_0, np.zeros_like(V_vvvv_0), np.zeros_like(V_oooo_0), V_ovov_m, V_oovv_m) * T2_0_t)
        grad_num_part2_ring[a, i] = (ep_ring - em_ring) / (2 * eps)

print("Sum of parts vs total MP3 num grad diff:\n", (grad_num_part1 + grad_num_part2_vvvv + grad_num_part2_oooo + grad_num_part2_ring) - grad_num_mp3)

# Let's check each of the 4 parts in mp3.cc against the numerical gradient of that part!

# 1. Part 1 (varying T2): in mp3.cc, this comes from L2_tilde in Z_mat, plus G_oo_mp3 and G_vv_mp3 in G_gamma_mo and L_sep!
G_full_mp3 = np.zeros((nbf, nbf))
G_full_mp3[:na, :na] = G_oo_mp3
G_full_mp3[na:, na:] = G_vv_mp3
P_corr_mp3 = C_can @ G_full_mp3 @ C_can.T
G_gamma_ao_mp3 = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_corr_mp3) - np.einsum('prqs,rs->pq', eri_ao, P_corr_mp3)
G_gamma_mo_mp3 = C_can.T @ G_gamma_ao_mp3 @ C_can

Z_mat_part1 = np.einsum('kcab,ikbc->ai', V_ovvv, L2_tilde) - np.einsum('jikc,jkac->ai', V_ooov, L2_tilde)
L_sep_mp3 = F_HF_vo @ G_oo_mp3 - G_vv_mp3 @ F_HF_vo
grad_ana_part1 = -4.0 * (G_gamma_mo_mp3[na:, :na] + L_sep_mp3 + Z_mat_part1)

# 2. Part 2 VVVV
Z_vvvv = np.zeros((va, na))
Z_vvvv += -1.0 * np.einsum('xafb,mafb->xm', G_vvvv, V_ovvv)
Z_vvvv += -1.0 * np.einsum('exfb,mefb->xm', G_vvvv, V_ovvv)
Z_vvvv += -1.0 * np.einsum('eaxb,mbea->xm', G_vvvv, V_ovvv)
Z_vvvv += -1.0 * np.einsum('eafx,mfae->xm', G_vvvv, V_ovvv)
grad_ana_vvvv = -4.0 * Z_vvvv

# 3. Part 2 OOOO
Z_oooo = np.zeros((va, na))
Z_oooo += 1.0 * np.einsum('minj,njix->xm', G_oooo, V_ooov)
Z_oooo += 1.0 * np.einsum('imnj,njix->xm', G_oooo, V_ooov)
Z_oooo += 1.0 * np.einsum('inmj,injx->xm', G_oooo, V_ooov)
Z_oooo += 1.0 * np.einsum('injm,injx->xm', G_oooo, V_ooov)
grad_ana_oooo = -4.0 * Z_oooo

# 4. Part 2 Ring
Z_ring = np.zeros((va, na))
Z_ring += 1.0 * np.einsum('makc,kcxa->xm', G_ovov, V_ovvv)
Z_ring += -1.0 * np.einsum('ixkc,imkc->xm', G_ovov, V_ooov)
Z_ring += 1.0 * np.einsum('iamc,iaxc->xm', G_ovov, V_ovvv)
Z_ring += -1.0 * np.einsum('iakx,kmia->xm', G_ovov, V_ooov)

Z_ring += 1.0 * np.einsum('mkac,kxca->xm', G_oovv, V_ovvv)
Z_ring += 1.0 * np.einsum('imac,ixac->xm', G_oovv, V_ovvv)
Z_ring += -1.0 * np.einsum('ikxc,ikmc->xm', G_oovv, V_ooov)
Z_ring += -1.0 * np.einsum('ikax,ikma->xm', G_oovv, V_ooov)
grad_ana_ring = -4.0 * Z_ring

print("=== PART 1 (T2 response) ===")
print("Ana:\n", grad_ana_part1)
print("Num:\n", grad_num_part1)
print("Ratio Ana/Num:\n", grad_ana_part1 / grad_num_part1)

print("\n=== PART 2 VVVV ===")
print("Ana:\n", grad_ana_vvvv)
print("Num:\n", grad_num_part2_vvvv)
print("Ratio Ana/Num:\n", grad_ana_vvvv / grad_num_part2_vvvv)

print("\n=== PART 2 OOOO ===")
print("Ana:\n", grad_ana_oooo)
print("Num:\n", grad_num_part2_oooo)
print("Ratio Ana/Num:\n", grad_ana_oooo / grad_num_part2_oooo)

print("\n=== PART 2 RING ===")
print("Ana:\n", grad_ana_ring)
print("Num:\n", grad_num_part2_ring)
print("Ratio Ana/Num:\n", grad_ana_ring / grad_num_part2_ring)

# Let's check the derivative of E_mp3(T) = sum_{ijab} (2 T_{ijab} - T_{ijba}) W_{ijab}(T) with respect to T_{ijab}!
def E_mp3_of_T(T_in):
    W_in = compute_W(T_in, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0)
    return np.sum(W_in * (2.0 * T_in - T_in.transpose(0,1,3,2)))

# Numerical derivative d(E_mp3)/d(T_{ijab})
dE_dT_num = np.zeros_like(T2_0)
delta = 1e-6
for i in range(na):
    for j in range(na):
        for a in range(va):
            for b in range(va):
                Tp = T2_0.copy()
                Tm = T2_0.copy()
                # Note: T2_0 has symmetry T_{ijab} = T_{jiba}. Let's first differentiate wrt unconstrained T_{ijab}, then symmetrize!
                Tp[i,j,a,b] += delta
                Tm[i,j,a,b] -= delta
                dE_dT_num[i,j,a,b] = (E_mp3_of_T(Tp) - E_mp3_of_T(Tm)) / (2 * delta)

# Symmetrize wrt (i,a) <-> (j,b) since T_{ijab} = T_{jiba}
dE_dT_num_sym = 0.5 * (dE_dT_num + dE_dT_num.transpose(1,0,3,2))

W_0 = compute_W(T2_0, V_vvvv_0, V_oooo_0, V_ovov_0, V_oovv_0)
W_0_tilde = 2.0 * W_0 - W_0.transpose(0,1,3,2)

print("Max diff between 0.5 * dE_dT_num_sym and W_0_tilde:", np.max(np.abs(0.5 * dE_dT_num_sym - W_0_tilde)))

print("grad_ana_part1 + 0.25*(vvvv+oooo+ring) vs grad_num_mp3:")
test_grad = grad_ana_part1 + 0.25 * (grad_ana_vvvv + grad_ana_oooo + grad_ana_ring)
print("Test:\n", test_grad)
print("Num:\n", grad_num_mp3)
print("Diff:\n", test_grad - grad_num_mp3)

# Let's isolate the 3 terms of E_mp3:
# E_mp3 = E_mp3_vvvv + E_mp3_oooo + E_mp3_ring
# And compare the EXACT total orbital derivative of each of the 3 terms with its analytical formula!

def eval_mp3_terms(C):
    P_a = C[:, :na] @ C[:, :na].T
    J = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_a)
    K = np.einsum('prqs,rs->pq', eri_ao, P_a)
    F_ao = h_ao + J - K
    F_mo = C.T @ F_ao @ C
    ea_o, U_o = np.linalg.eigh(F_mo[:na, :na])
    ea_v, U_v = np.linalg.eigh(F_mo[na:, na:])
    C_can = C.copy()
    C_can[:, :na] = C[:, :na] @ U_o
    C_can[:, na:] = C[:, na:] @ U_v
    ea = np.concatenate([ea_o, ea_v])
    Cao = C_can[:, :na]
    Cav = C_can[:, na:]
    
    V_ovov = np.einsum('pqrs,pi,qa,rj,sb->iajb', eri_ao, Cao, Cav, Cao, Cav)
    V_oovv = np.einsum('pqrs,pi,qj,ra,sb->ijab', eri_ao, Cao, Cao, Cav, Cav)
    V_vvvv = np.einsum('pqrs,pe,qa,rf,sb->eafb', eri_ao, Cav, Cav, Cav, Cav)
    V_oooo = np.einsum('pqrs,pm,qi,rn,sj->minj', eri_ao, Cao, Cao, Cao, Cao)
    
    D = ea[:na, None, None, None] + ea[None, :na, None, None] - ea[None, None, na:, None] - ea[None, None, None, na:]
    T2 = np.einsum('iajb->ijab', V_ovov) / D
    T2_t = 2.0 * T2 - T2.transpose(0,1,3,2)
    
    W_vvvv = np.einsum('ijef,eafb->ijab', T2, V_vvvv)
    W_oooo = np.einsum('mnab,minj->ijab', T2, V_oooo)
    W_ring = np.zeros((na, na, va, va))
    W_ring += 2.0 * np.einsum('iakc,kjcb->ijab', V_ovov, T2)
    W_ring += -1.0 * np.einsum('ikac,kjcb->ijab', V_oovv, T2)
    W_ring += 2.0 * np.einsum('ikac,kcjb->ijab', T2, V_ovov)
    W_ring += -1.0 * np.einsum('ikac,kjcb->ijab', T2, V_oovv)
    W_ring += -1.0 * np.einsum('ikca,kcjb->ijab', T2, V_ovov)
    W_ring += -1.0 * np.einsum('iakc,kjbc->ijab', V_ovov, T2)
    W_ring += -1.0 * np.einsum('ikbc,kjac->ijab', V_oovv, T2)
    W_ring += -1.0 * np.einsum('ikcb,jkac->ijab', T2, V_oovv)
    
    return np.sum(W_vvvv * T2_t), np.sum(W_oooo * T2_t), np.sum(W_ring * T2_t), W_vvvv/D, W_oooo/D, W_ring/D

_, _, _, L2_vvvv, L2_oooo, L2_ring = eval_mp3_terms(C_can)

num_vvvv = np.zeros((va, na))
num_oooo = np.zeros((va, na))
num_ring = np.zeros((va, na))

for i in range(na):
    for a in range(va):
        K = np.zeros((nbf, nbf))
        K[i, na+a] = eps
        K[na+a, i] = -eps
        ev_p, eo_p, er_p, *_ = eval_mp3_terms(C_can @ expm(K))
        ev_m, eo_m, er_m, *_ = eval_mp3_terms(C_can @ expm(-K))
        num_vvvv[a, i] = (ev_p - ev_m) / (2 * eps)
        num_oooo[a, i] = (eo_p - eo_m) / (2 * eps)
        num_ring[a, i] = (er_p - er_m) / (2 * eps)

def ana_part1_from_L2(L2_part):
    L2_p_t = 2.0 * L2_part - L2_part.transpose(0,1,3,2)
    G_oo_p = -0.5 * np.einsum('ikab,jkab->ij', T2, L2_p_t) - 0.5 * np.einsum('ikab,jkab->ij', L2_part, T2_tilde)
    G_vv_p = 0.5 * np.einsum('ijac,ijbc->ab', T2, L2_p_t) + 0.5 * np.einsum('ijac,ijbc->ab', L2_part, T2_tilde)
    G_full_p = np.zeros((nbf, nbf))
    G_full_p[:na, :na] = G_oo_p
    G_full_p[na:, na:] = G_vv_p
    P_corr_p = C_can @ G_full_p @ C_can.T
    G_gamma_ao_p = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_corr_p) - np.einsum('prqs,rs->pq', eri_ao, P_corr_p)
    G_gamma_mo_p = C_can.T @ G_gamma_ao_p @ C_can
    Z_p = np.einsum('kcab,ikbc->ai', V_ovvv, L2_p_t) - np.einsum('jikc,jkac->ai', V_ooov, L2_p_t)
    L_sep_p = F_HF_vo @ G_oo_p - G_vv_p @ F_HF_vo
    return -4.0 * (G_gamma_mo_p[na:, :na] + L_sep_p + Z_p)

print("VVVV Diff (with 0.25 on Z_vvvv):\n", (ana_part1_from_L2(L2_vvvv) + 0.25 * grad_ana_vvvv) - num_vvvv)
print("OOOO Diff (with 0.25 on Z_oooo):\n", (ana_part1_from_L2(L2_oooo) + 0.25 * grad_ana_oooo) - num_oooo)
print("RING Diff (with 0.25 on Z_ring):\n", (ana_part1_from_L2(L2_ring) + 0.25 * grad_ana_ring) - num_ring)

# Let's check why ana_part1_from_L2(L2_vvvv) + 0.25 * grad_ana_vvvv had a 2e-6 difference from num_vvvv!
# Let's compute the exact derivative of T2(C) with respect to K_{ia}!
# Wait: E_vvvv(C) = sum_{ijabef} (2 T_{ijab} - T_{ijba}) T_{ijef} V_{eafb}
# Notice that E_vvvv(C) is invariant under U_o and U_v because U_o and U_v cancel out in all contractions!
# Let's verify that E_vvvv(C) is invariant and check the exact derivative of V_ovov and F_oo, F_vv wrt K_{ia}!

# Wait: when C -> C exp(K), how do F_mo[:na, :na], F_mo[na:, na:], and V_ovov change?
# Let's check the numerical derivative of E_mp2(C) vs E_vvvv(C) when F_ao is held FIXED vs when F_ao varies!
# Wait: in G_oo_mp3 and G_vv_mp3:
# Look at lines 804-805 and 808-809 of mp3.cc:
# G_oo_mp3 = -0.5 * einsum('ikab,jkab->ij', T2, L2_tilde) - 0.5 * einsum('ikab,jkab->ij', L2, T2_tilde)
# Wait: why is there a 0.5 here?!
# Let's check! In MP2:
# E_MP2 = sum_{ijab} V_{iajb} T2_tilde_{ijab} = sum_{ijab} D_{ijab} T2_{ijab} T2_tilde_{ijab}
# When we differentiate E_MP2 wrt F_{ij} (which is in D_{ijab} = f_{ii} + f_{jj} - f_{aa} - f_{bb}):
# d(E_MP2)/d(f_{ij}) = -1.0 * sum_{kab} T2_{ikab} T2_tilde_{jkab} = G_oo_mp2_{ij} !
# Now, what is d(E_vvvv)/d(f_{ij})?!
# In E_vvvv = sum_{ijab} W_vvvv_{ijab}[T2] T2_tilde_{ijab}:
# T2 appears TWICE in E_vvvv!
# So when f_{ij} changes by df_{ij}, T2 changes by:
# d(T2_{ijab}) = - (df_{ik} T2_{kjab} + df_{jk} T2_{ikab}) / D_{ijab}
# And since d(E_vvvv)/d(T2_{ijab}) = 2 * W_vvvv_tilde_{ijab} = 2 * D_{ijab} * L2_vvvv_tilde_{ijab}:
# Substitute d(T2_{ijab}) into d(E_vvvv):
# d(E_vvvv) = sum_{ijab} 2 * D_{ijab} * L2_vvvv_tilde_{ijab} * ( - (df_{ik} T2_{kjab} + df_{jk} T2_{ikab}) / D_{ijab} )
#           = - 2 * sum_{ijkab} df_{ik} (L2_vvvv_tilde_{ijab} T2_{kjab} + L2_vvvv_tilde_{kjab} T2_{ijab}) !
# WAIT A SECOND!
# Look at the coefficient in front of L2 * T2_tilde:
# For MP2, T2 appeared ONCE in V * T2 (or with a factor of -1 in -D * T2 * T2_tilde), so G_oo_mp2 had -1.0 * T2 * T2_tilde!
# For MP3, d(E_MP3)/d(T2) = 2 * D * L2_tilde (with a factor of 2 because E_MP3 is QUADRATIC in T2!),
# OR in Z_p, d(V_{iajb})/dK contracts with d(E_MP3)/d(V_{iajb}) = 2 * L2_tilde!
# Wait: why did MP2 have 1.0 * T2_tilde in Z_mat_mp2 and -1.0 * T2 * T2_tilde in G_oo_mp2?
# Because in MP2:
# d(E_MP2)/d(V_{iajb}) = 2 * T2_tilde_{ijab} (since E_MP2 = sum V_{iajb} T2_tilde_{ijab} and V = D*T2 is also quadratic in V!),
# AND in Z_mat_mp2, 1.0 * T2_tilde is multiplied by -4.0 at the end!
# Now for MP3:
# E_MP3 = sum_{ijab} W_{ijab}[T2] T2_tilde_{ijab}
# Since W is linear in T2, E_MP3 is QUADRATIC in T2 (and therefore QUADRATIC in V_{iajb}, just like E_MP2!)
# Wait: let's check the factor in Z_p, G_oo_p, G_vv_p by testing when V_vvvv is held FIXED!

def eval_mp3_vvvv_fixedV(C):
    # Transform C into C_can0 basis so V_vvvv_0 is fixed in MO basis?
    # Wait: if V_vvvv is in MO basis, V_vvvv depends on C!
    pass

# Wait: why did VVVV Diff have 2e-6?! Let's check what 2e-6 is!
print("ana_part1_from_L2(L2_vvvv):\n", ana_part1_from_L2(L2_vvvv))
print("0.25 * grad_ana_vvvv:\n", 0.25 * grad_ana_vvvv)
print("num_vvvv:\n", num_vvvv)
def ana_part1_from_L2_factor2(L2_part):
    L2_p_t = 2.0 * L2_part - L2_part.transpose(0,1,3,2)
    # Factor of -1.0 and +1.0 instead of -0.5 and +0.5!
    G_oo_p = -1.0 * np.einsum('ikab,jkab->ij', T2, L2_p_t) - 1.0 * np.einsum('ikab,jkab->ij', L2_part, T2_tilde)
    G_vv_p = 1.0 * np.einsum('ijac,ijbc->ab', T2, L2_p_t) + 1.0 * np.einsum('ijac,ijbc->ab', L2_part, T2_tilde)
    G_full_p = np.zeros((nbf, nbf))
    G_full_p[:na, :na] = G_oo_p
    G_full_p[na:, na:] = G_vv_p
    P_corr_p = C_can @ G_full_p @ C_can.T
    G_gamma_ao_p = np.einsum('pqrs,rs->pq', eri_ao, 2.0 * P_corr_p) - np.einsum('prqs,rs->pq', eri_ao, P_corr_p)
    G_gamma_mo_p = C_can.T @ G_gamma_ao_p @ C_can
    Z_p = np.einsum('kcab,ikbc->ai', V_ovvv, L2_p_t) - np.einsum('jikc,jkac->ai', V_ooov, L2_p_t)
    L_sep_p = F_HF_vo @ G_oo_p - G_vv_p @ F_HF_vo
    return -4.0 * (G_gamma_mo_p[na:, :na] + L_sep_p + Z_p)

print("VVVV Diff with factor 1.0 in G_oo/G_vv and 0.25 in Z_vvvv:\n", (ana_part1_from_L2_factor2(L2_vvvv) + 0.25 * grad_ana_vvvv) - num_vvvv)
print("OOOO Diff with factor 1.0 in G_oo/G_vv and 0.25 in Z_oooo:\n", (ana_part1_from_L2_factor2(L2_oooo) + 0.25 * grad_ana_oooo) - num_oooo)