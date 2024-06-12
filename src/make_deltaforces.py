import numpy as np
from tqdm import tqdm


def make_deltaforces(
    mol,
    forces_npz,
    delta_forces_npz,
    forcefield,
):
    atom_types = mol.atomtype
    natoms = mol.numAtoms
    bonds = mol.bonds
    angles = mol.angles
    coords = mol.coords

    ### test if any bonds are longer than 10A
    print("Check for broken coords.")

    broken_frames = []
    for bond in tqdm(mol.bonds):
        crds = coords[bond, :, :]
        crds_dif = crds[ 0, :, :] - crds[1, :, :]
        dists = np.linalg.norm(crds_dif, axis=0)
        broken = dists > 10.0
        broken_frames.append(broken)

    broken_frames = np.stack(broken_frames)
    broken_frames = broken_frames.any(axis=0)

    if broken_frames.any():
        print("Removing broken coords with distances larger than 10A.")

        coords_good = coords[:, :, ~broken_frames]  # remove broken frames
        print(coords_good.shape)
        # np.save(coords_npz.replace(".", "_fix."), coords_good)

        broken_coords = coords[:, :, broken_frames]
        print(broken_coords.shape)
        # np.save(coords_npz.replace(".", "_broken."), broken_coords)

        coords = coords_good

    else:
        print("No broken frames")

    all_forces = forces_npz[~broken_frames, :, :]
    coords = np.moveaxis(coords,2,0)

    print("Producing delta forces")
    prior_forces = []
    frame_forces = np.zeros_like(all_forces[0,:,:])

    k_bond = forcefield['bonds']['(DNA, DNA)']['k0']
    r_eq = forcefield['bonds']['(DNA, DNA)']['req']
    k_theta =  forcefield['angles']['(DNA, DNA, DNA)']['k_theta']
    D_morse = forcefield['morse']['(DNA, SPD)']['D']
    r_morse = forcefield['morse']['(DNA, SPD)']['r0']
    a_morse = forcefield['morse']['(DNA, SPD)']['a']
    Da_morse = 2.0*D_morse*a_morse

    for co in tqdm(coords):
        for idx0, idx1 in bonds:
            r = co[idx0,:] - co[idx1,:]
            r_norm = np.linalg.norm(r)
            f_aux = k_bond*(r_norm - r_eq)/r_norm
            frame_forces[idx0,:] += r*f_aux
            frame_forces[idx1,:] -= r*f_aux

        for idx0, idx1, idx2 in angles:
            a = co[idx0,:] - co[idx1,:]
            b = co[idx2,:] - co[idx1,:]
            r_a = np.linalg.norm(a, axis=0)
            r_b = np.linalg.norm(b, axis=0)

            cos = np.dot(r_a,r_b)/(r_b*r_a)
            k11 = k_theta*cos/(r_a*r_a)
            k12 = -k_theta/(r_b*r_a)
            k22 =  k_theta*cos/(r_b*r_b)

            f_a = k11*a + k12*b
            f_b = k22*b + k12*a

            frame_forces[idx0,:] += f_a
            frame_forces[idx1,:] -= f_a + f_b
            frame_forces[idx2,:] += f_b

        for idx0 in range(natoms - 1):
            for idx1 in range(idx0 + 1,natoms):
                if atom_types[idx0] != atom_types[idx1]:
                    r = co[idx1,:] - co[idx0,:]
                    r_norm = np.linalg.norm(r)
                    dr = r_norm - r_morse
                    dexp = np.exp(-a_morse*dr)
                    f_aux = Da_morse*(dexp*dexp - dexp)/r_norm

                    frame_forces[idx0,:] += r*f_aux
                    frame_forces[idx1,:] -= r*f_aux
        
        prior_forces.append(frame_forces)

    prior_forces = np.array(prior_forces)
    delta_forces = all_forces - prior_forces

    np.save(delta_forces_npz, delta_forces)
