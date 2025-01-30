import modal

glamour_image = (
    modal.Image.micromamba(python_version="3.8")
    .apt_install("build-essential")
    .micromamba_install('numpy==1.20.0', 'cython', channels=['conda-forge'])  # Pin numpy early and add cython
    .micromamba_install('pandas==1.2.4', channels=['conda-forge'])  # Add pandas early with compatible version
    .micromamba_install('pytorch==1.13', 'blas=*=mkl', channels=['conda-forge'])
    .micromamba_install('cudatoolkit=11.0', channels=['conda-forge'])
    .micromamba_install('rdkit==2020.09.5', channels=['conda-forge'])
    .micromamba_install('networkx<2.8', channels=['conda-forge'])
    .micromamba_install('scikit-learn==0.23.2', channels=['conda-forge'])
    .micromamba_install('svglib', 'umap-learn<0.5.4', 'matplotlib==3.2.2', channels=['conda-forge'])
    .micromamba_install('dgl-cuda11.0==0.8.0post2', channels=['dglteam', 'conda-forge'])
    .micromamba_install('dgllife', channels=['conda-forge'])
    .pip_install('captum')
    .pip_install('seaborn')
    .apt_install("git")
    .run_commands("git clone https://github.com/learningmatter-mit/GLAMOUR")
    .run_commands("pip install --force-reinstall --no-deps numpy==1.21.0 && pip install --force-reinstall --no-deps pandas==1.2.4")
    .run_commands("pip install --force-reinstall --no-deps scipy==1.6.0")
    .pip_install('grakel==0.1.8')
)

app = modal.App("glamour-app-oof")

@app.function(image=glamour_image)
def proof_of_concept():
    import sys
    sys.path.append('/GLAMOUR')

    import random
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole

    from utils.load_networkx import networkx_feat
    from utils.macro_dataset import MacroDataset
    from utils import macro_unsupervised as unsup
    from utils.macro_supervised import MacroSupervised
    from utils.macro_attribution import Attribution
    # from utils import plot_utils

    MON_SMILES = '/GLAMOUR/tables/SMILES_glycans_monomer.txt'
    BOND_SMILES = '/GLAMOUR/tables/SMILES_glycans_bond.txt'

    TXT_DATA_PATH = '/GLAMOUR/dataset/classification/'
    DF_PATH = '/GLAMOUR/tables/glycobase_immuno.txt'
    MODEL_PATH = '/GLAMOUR/'
    FIG_PATH = '/GLAMOUR/'

    FEAT = 'fp'
    FP_RADIUS_MON = 3
    FP_BITS_MON = 128
    FP_RADIUS_BOND = 3
    FP_BITS_BOND = 16

    SEED = 108
    TASK = 'classification'
    MODEL = 'AttentiveFP'
    LABELNAME = 'Immunogenic'
    SPLIT = '0.6,0.2,0.2'
    NORM = 'qt'

    NUM_EPOCHS = 10
    NUM_WORKERS = 1

    SAVE_MODEL = False
    SAVE_OPT = False
    SAVE_CONFIG = False

    PLOT_TYPE = 'val'

    CUSTOM_PARAMS = {}

    NX_GRAPHS = networkx_feat(
        TXT_DATA_PATH = TXT_DATA_PATH, 
        MON_SMILES = MON_SMILES, 
        BOND_SMILES = BOND_SMILES, 
        FEAT = FEAT, 
        FP_RADIUS_MON = FP_RADIUS_MON, 
        FP_RADIUS_BOND = FP_RADIUS_BOND, 
        FP_BITS_MON = FP_BITS_MON, 
        FP_BITS_BOND = FP_BITS_BOND
    )
    res = unsup.edit_distance(NX_GRAPHS['GBID10271'], NX_GRAPHS['GBID10330'], node_attr = 'h', edge_attr = 'e', upper_bound = 100)

    return res

@app.local_entrypoint()
def main():
    print(proof_of_concept.remote())
