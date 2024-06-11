import numpy as np
import MDAnalysis as mda
from utilities import conv_inputs

u = mda.Universe("/exports/home/jarol.molina/DNA-1/MD.gro", "/exports/home/jarol.molina/DNA-1/MD.trr")
dna=u.select_atoms("nucleic")

pos = dna.positions
print(pos[-1])

# Creando archivos de coordenadas y fuerzas                              
xyz = []
fxyz = []
i = 1
for t in u.trajectory:
   print('Frame     ', i)
   i =  i+1
   xyz.append(dna.positions)
   fxyz.append(dna.forces)
with open('data/DNA1_xyz.npy', 'wb') as f:
   np.save(f,xyz)
with open('data/DNA1_Fxyz.npy', 'wb') as g:
   np.save(g,fxyz)


# Creacion de array de nombres                                           
an = conv_inputs('exports/home/jarol.molina/20dp_dna/DNA_1.pdb')
with open('data//DNA1_Z.npy', 'wb') as f:
   np.save(f,an.title())
print(len(an.title()))
