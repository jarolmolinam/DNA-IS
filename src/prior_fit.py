import numpy as np
import os
import pandas as pd
import yaml
from tqdm import tqdm
from scipy.optimize import curve_fit
from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd
import math as m


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


kB = 0.0019872041 # kcal/mol/K

# Input:  counts and bin limits (n+1)
# Output: counts normalized by spheric shell volumes, and bin centers (n)
def renorm(counts, bins):
    h = bins[1]-bins[0]
    R = .5*(bins[1:]+bins[:-1])  # bin centers
    vols = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)
    ncounts = counts / vols
    return np.vstack([R,ncounts])


# Bonded fit

def harmonic(x,x0,k,V0):
    return k*(x-x0)**2+V0

# Coarse Graining repulsion fit

def LJ(r,eps,sigma,V0):
    sigma_over_r = sigma/r
    V = 4*eps*(sigma_over_r**12 - sigma_over_r**6) + V0
    return V

def CG(r,eps,sigma,V0):
    sigma_over_r = sigma/r
    V = 4*eps*(sigma_over_r**6) + V0
    return V

def exp12(r,A,C,lamb,R,V0):
    return A/(r - R)**6 - C*np.exp(-(r - R)/lamb) + V0

def morse(r,D,a,r0,V0):
    return D*(0.1**6/r**6 + np.exp(-2*a*(r-r0)) - 2*np.exp(-a*(r-r0))) + V0

def morse1(r,D,a,r0,V0):
    return D*(1 - np.exp(-a*(r-r0)))**2 + V0

def cosine(cos, k, V0):
    return k*(1-cos)+V0

def get_param_bonded(mol, bond_range, Temp):
    bonds_types = {}
    for bond in mol.bonds:
        btype = tuple(mol.atomtype[bond])
        if btype in bonds_types:
            bonds_types[btype].append(bond)
        elif tuple([btype[1], btype[0]]) in bonds_types:
            bonds_types[tuple([btype[1], btype[0]])].append(bond)
        else:
            bonds_types[btype] = [bond]

    prior_bond = {}
    for bond in bonds_types.keys():
        dists = []
        for idx0, idx1 in bonds_types[bond]:
            dists.append(np.linalg.norm(mol.coords[idx0,:,:] - mol.coords[idx1,:,:], axis=0))

        dist = np.concatenate(dists, axis=0)

        yb, bins= np.histogram(dist, bins=40,  range=bond_range)
        RR, ncounts = renorm(yb, bins)

        # Drop zero counts
        RR_nz = RR[ncounts>0]
        ncounts_nz = ncounts[ncounts>0]
        dG_nz = -kB*Temp*np.log(ncounts_nz)


        # Fit may fail, better to try-catch. p0 usually not necessary if function is reasonable.
        popt, _ = curve_fit(harmonic, RR_nz, dG_nz, p0=[1.36, 40, -1])

        # Just a hard-coded example, the full code requires more changes
        bname=f"({bond[0]}, {bond[1]})"
        prior_bond[bname]={'req': popt[0].tolist(),
                            'k0':  popt[1].tolist() }


        # popt now has the function parameters
        plt.plot(RR_nz, dG_nz, 'o')
        plt.plot(RR_nz, harmonic(RR_nz, *popt))
        plt.title(f'{bond[0]}-{bond[1]}')
        plt.show()

    angles_types = {}
    for angle in mol.angles:
        atype = tuple(mol.atomtype[angle])
        if atype in angles_types:
            angles_types[atype].append(angle)
        elif tuple([atype[1], atype[0]]) in angles_types:
            angles_types[tuple([btype[1], btype[0]])].append(angle)
        else:
            angles_types[atype] = [angle]

    prior_angle = {}
    for angle in angles_types.keys():
        coss = []
        for idx0, idx1, idx2 in angles_types[angle]:
            a = mol.coords[idx1,:,:] - mol.coords[idx0,:,:]
            b = mol.coords[idx2,:,:] - mol.coords[idx1,:,:]

            unit_a = a / np.linalg.norm(a, axis=0)
            unit_b = b / np.linalg.norm(b, axis=0)
            coss.append(np.tensordot(unit_a,unit_b, axes=([0],[0])))

        cos = np.concatenate(coss, axis=0)

        yb, bins= np.histogram(cos, bins=40,  range=[0.9,1])
        RR, ncounts = renorm(yb, bins)

        # Drop zero counts
        RR_nz = RR[ncounts>0]
        ncounts_nz = ncounts[ncounts>0]
        dG_nz = -kB*Temp*np.log(ncounts_nz)


        # p0 usually not necessary if function is reasonable.
        popt, _ = curve_fit(cosine, RR_nz, dG_nz)

        # Just a hard-coded example, the full code requires more changes
        aname=f"({angle[0]}, {angle[1]}, {angle[2]})"
        prior_angle[aname]={'k_theta':  popt[0].tolist() }


        # popt now has the function parameters
        plt.plot(RR_nz, dG_nz, 'o')
        plt.plot(RR_nz, cosine(RR_nz, *popt))
        plt.title(f'{angle[0]}-{angle[1]}-{angle[2]}')
        plt.show()

    return prior_bond, prior_angle


def get_param_nonbonded(mol, fit_range, Temp):
    nb_types = {}
    for i in range(mol.numAtoms):
        for j in range(mol.numAtoms):
            if j is not i:
                nbtype = tuple(mol.atomtype[[i,j]])
                if nbtype in nb_types:
                    nb_types[nbtype].append([i,j])
                else:
                    nb_types[nbtype] = [[i,j]]

    prior_lj = {}
    for ij in nb_types.keys():
        dists = []
        for idx in nb_types[ij]:
            if ij[0] != ij[1]:
                dists.append(np.linalg.norm(mol.coords[idx[0], :, :] - mol.coords[idx[1], :, :], axis=0))

        if len(dists) > 1:
            dist = np.concatenate(dists, axis=0)
            # save only below 6A
            dist = dist[dist < fit_range[1]]

            yb, bins = np.histogram(dist, bins=50, range=fit_range)  ### Adjust the range if needed
            RR, ncounts = renorm(yb, bins)

            RR_nz = RR[ncounts > 0]
            ncounts_nz = ncounts[ncounts > 0]
            dG_nz = -kB * Temp * np.log(ncounts_nz)

            popt, _ = curve_fit(morse, RR_nz, dG_nz)

            # Just a hard-coded example, the full code requires more changes
            # bname = tuple(ij)
            bname = f"({ij[0]}, {ij[1]})"
            prior_lj[bname] = {'D': popt[0].tolist(),
                                'a': popt[1].tolist(),
                                'r0':popt[2].tolist()}
            print('V0 ', popt[3].tolist())

            plt.plot(RR_nz, dG_nz, 'o')
            plt.plot(RR_nz, morse(RR_nz, *popt))
            plt.title(f'{ij}')
            plt.show()

    return prior_lj

def get_param_nonbonded_rep(mol, fit_range, Temp):
    atom_types = {}
    for at in set(mol.atomtype):
        atom_types[at] = np.where(mol.atomtype == at)[0]

    prior_lj = {}

    for at in atom_types.keys():
        dists = []
        for idx in atom_types[at]:
            bonded = []
            for bond in mol.bonds:
                if idx in bond:
                    bonded.append(bond[0])
                    bonded.append(bond[1])

            for idx2 in list(set(mol.bonds.flatten())):
                if idx2 not in bonded:
                    dists.append(np.linalg.norm(mol.coords[idx, :, :] - mol.coords[idx2, :, :], axis=0))

        dist = np.concatenate(dists, axis=0)
        # save only below 6A
        dist = dist[dist < fit_range[at][1]]

        yb, bins = np.histogram(dist, bins=30, range=fit_range[at])  ### Adjust the range if needed
        RR, ncounts = renorm(yb, bins)

        RR_nz = RR[ncounts > 0]
        ncounts_nz = ncounts[ncounts > 0]
        dG_nz = -kB * Temp * np.log(ncounts_nz)

        popt, _ = curve_fit(LJ, RR_nz, dG_nz, p0=[5.0, 5.0, -1.0])

        # Just a hard-coded example, the full code requires more changes
        bname = at
        prior_lj[bname] = {'epsilon': popt[0].tolist(),
                           'sigma': popt[1].tolist()}

        plt.plot(RR_nz, dG_nz, 'o')
        plt.plot(RR_nz, LJ(RR_nz, *popt))
        plt.title(f'{at}')
        plt.show()

    return prior_lj