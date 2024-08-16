import sys
from importlib.machinery import SourceFileLoader
import os
import numpy as np
import logging
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis import transformations
from tqdm import tqdm
import src.cgmap as cg



def load_and_preprocess_dna(path, name):
    """
    Carga un sistema molecular desde archivos GRO y TRR, selecciona los átomos de ADN y ajusta la 
    trayectoria para un análisis posterior.

    Args:
        path (str): Ruta base donde se encuentran los archivos GRO y TRR.
        name (str): Nombre base de los archivos GRO y TRR (sin extensión).

    Returns:
        tuple: Una tupla que contiene:
            - u (MDAnalysis Universe): El objeto Universe que representa el sistema molecular cargado.
            - dna (MDAnalysis AtomGroup): Un AtomGroup que contiene solo los átomos de ADN seleccionados.

    Raises:
        FileNotFoundError: Si alguno de los archivos GRO o TRR no se encuentra.
        OSError: Si ocurre algún otro error al cargar los archivos.
    """

    top_file = os.path.join(path + name + '.gro')
    trr_file = os.path.join(path + name + '.trr')

    try:
        u = mda.Universe(top_file, trr_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo encontrar el archivo GRO o TRR en la ruta '{path}' con el nombre base '{name}'.")
    except OSError as e:
        raise OSError(f"Ocurrió un error al cargar los archivos: {e}")

    dna = u.select_atoms("nucleic")
    transform = transformations.wrap(dna)
    u.trajectory.add_transformations(transform)

    return u, dna

##########################################################################

def write_dna_trajectory(u, dna, name, output_path): 
    """
    Escribe la trayectoria del ADN en formato DCD y la topología en formato PDB
    solo del dna

    Args:
        u (MDAnalysis Universe): El objeto Universe que contiene la trayectoria completa.
        dna (MDAnalysis AtomGroup): El AtomGroup que representa los átomos de ADN.
        name (str): Nombre base para los archivos de salida.
        output_path (str): La ruta base donde se escribirán los archivos de salida.
    """

    print("Escribiendo trayectoria y topología del ADN...")

    dcd_file = os.path.join(output_path, name + '_wrap.dcd')
    pdb_file = os.path.join(output_path, name + '_wrap.pdb')

    try:
        with mda.Writer(dcd_file, dna.n_atoms) as dcd_writer:
            for ts in u.trajectory:
                dcd_writer.write(dna)

        # Escribe la topología del ADN en formato PDB (solo el primer cuadro)
        dna.write(pdb_file) 

    except OSError as e:
        print(f"Error al escribir los archivos: {e}")

##########################################################################

def create_cg_trajectory(path, f_name, maps, basepairs_per_bead):
    """
    Crea y guarda una trayectoria coarse-grained (CG) a partir de una trayectoria all-atom.

    Args:
        path (str): Ruta base donde se encuentran los archivos de entrada y se guardarán los de salida.
        f_name (str): Nombre base de los archivos de entrada y salida.
        maps (list): Lista de selecciones de átomos para cada bead CG.
        basepairs_per_bead (int): Número de pares de bases representados por cada bead CG (0 para all-atom).

    Returns:
        tuple: Una tupla que contiene:
            - index_list (list): Lista de índices de átomos para cada bead CG.
            - label_list (list): Lista de etiquetas para cada bead CG.

    Raises:
        Exception: Si ocurre algún error durante la carga, procesamiento o guardado de las trayectorias.
    """

    # Construye las rutas completas de los archivos de entrada y salida
    dcd_file = os.path.join(path, f_name + '_wrap.dcd')
    pdb_file = os.path.join(path, f_name + '_wrap.pdb')
    output_pdb = os.path.join(path, f_name + '_CG.pdb')
    output_trj = os.path.join(path, f_name + '_CG.dcd')
    output_npy = os.path.join(path, f_name + '_CG.npy')

    try:
        # Carga la trayectoria all-atom
        trj = md.load(dcd_file, top=pdb_file) 

        if basepairs_per_bead == 0:
            # Caso all-atom: cada residuo es un bead CG
            label_list = [res.name for res in trj.topology.residues]  
        else:
            # Caso coarse-grained: crea etiquetas "DNA" según el número de beads
            num_beads = int(len(trj.topology.residues) / basepairs_per_bead)
            label_list = ["DNA"] * num_beads

        # Obtiene los índices de átomos para cada bead CG
        index_list = [trj.top.select(sel) for sel in maps]

        # Crea la trayectoria coarse-grained
        cg_trj = cg.cg_by_index(trj, index_list, label_list)

        # Guarda la trayectoria CG en formato DCD
        print("Guardando trayectoria CG...")
        cg_trj.save(output_trj)

        # Guarda la topología CG en formato PDB (solo el primer frame)
        print("Guardando PDB de CG...")
        cg_trj[0].save(output_pdb)

        # Guarda las coordenadas de la trayectoria CG en formato NumPy
        with open(output_npy, 'wb') as f:
            np.save(f, cg_trj.xyz)

        # Imprime las etiquetas de los beads CG (opcional, para verificación)
        print(label_list)

        return index_list, label_list

    except Exception as e:
        # Captura cualquier excepción y re-lanza para manejo en niveles superiores
        print(f"Error al procesar o guardar la trayectoria CG: {e}")
        raise

###########################################################################

def mapping_forces(u, dna, index, output_path, f_name): # Agregamos output_path y f_name
    """
    Mapea las fuerzas atómicas de una trayectoria all-atom a una representación coarse-grained.

    Args:
        u (MDAnalysis Universe): El objeto Universe que contiene la trayectoria all-atom.
        dna (MDAnalysis AtomGroup): El AtomGroup que representa los átomos de ADN.
        index (list): Lista de índices de átomos para cada bead CG.
        output_path (str): Ruta base donde se guardará el archivo de salida NumPy.
        f_name (str): Nombre base para el archivo de salida.

    Returns:
        None (Las fuerzas CG se guardan en un archivo NumPy)
    """

    output_npy = os.path.join(output_path, f_name + '_Force_CG.npy')

    n = len(index)  # Número de beads CG
    nf = u.trajectory.n_frames  # Número de frames en la trayectoria

    # Crea un array vacío para almacenar las fuerzas CG
    forces_cg = np.empty((nf, n, 3)) 

    print("Mapeando fuerzas...")
    for t, frame in enumerate(u.trajectory):  # Itera sobre cada frame de la trayectoria
        f_frame = dna.forces  # Obtiene las fuerzas atómicas del frame actual

        for i in range(n):  # Itera sobre cada bead CG
            bead_indexes = index[i]  # Obtiene los índices de los átomos del bead actual
            forces_cg_i = cg.map_forces(f_frame, bead_indexes)  # Mapea las fuerzas atómicas a las fuerzas CG
            forces_cg[t, i, :] = forces_cg_i  # Almacena las fuerzas CG en el array

    # Guarda las fuerzas CG en un archivo NumPy
    np.save(output_npy, forces_cg)

############################################################################

def embedding(label_list):
    Bead2INT = {'DA':1,
            'DT':2,
            'DG':3,
            'DC':4
            }

    emb = np.array([Bead2INT[x] for x in label_list])
    np.save('data/DNA1_embeddings.npy', emb)

    embeddings = np.tile(emb, [forces_cg_np.shape[0], 1])
    print("Embeddings size: {}".format(embeddings.shape))

#############################################################################

# Configuración del registro 
logging.basicConfig(filename='dna_analysis.log', level=logging.INFO)

# Configuración y rutas
output_path = '/exports/home/jarol.molina/DNA-IS/data/'
input_path = '/exports/home/jarol.molina/DNA-1/'
input_name = 'MD'
output_name = 'DNA'
basepairs_per_bead = 0  # 0 indica representación all-atom

try:
    # 1. Cargar y preprocesar el ADN
    logging.info("Cargando y preprocesando ADN...") 
    u, dna = load_and_preprocess_dna(input_path, input_name)

    # 2. Escribir la trayectoria y topología del ADN (opcional, si se desea guardar la trayectoria all-atom)
    logging.info("Escribiendo trayectoria y topología del ADN...")
    write_dna_trajectory(u, dna, output_name, output_path)

    # 3. Generar mapas CG (asegúrate de tener cg.DNA_CG_map definido)
    maps = cg.DNA_CG_map(basepairs_per_bead)

    # 4. Mapeo a CG y guardado
    logging.info("Mapeando a CG y guardando...")
    index, label = create_cg_trajectory(output_path, output_name, maps, basepairs_per_bead)

    # 5. Mapeo de fuerzas
    logging.info("Mapeando fuerzas...")
    mapping_forces(u, dna, index, output_path, output_name)

    logging.info("Análisis completado con éxito.")

except Exception as e:
    # Capturar y registrar cualquier error inesperado
    logging.error(f"Ocurrió un error durante el análisis: {e}")
#####################
