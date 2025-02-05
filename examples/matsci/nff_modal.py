import modal

conda_requirements = [
    'ase=3.22.1=pyhd8ed1ab_1',
    'debugpy=1.6.0=py39h5a03fae_0',
    'django=4.0=pyhd8ed1ab_0',
    'matplotlib-base=3.4.3=py39h2fa2bec_2',
    'matplotlib-inline=0.1.3=pyhd8ed1ab_0',
    'numpy=1.22.4=py39hc58783e_0',
    'openbabel=3.1.1=py39hee2736e_3',
    'pandas=1.4.2=py39h1832856_2',
    'pip=22.1.1=pyhd8ed1ab_0',
    'plotly=5.8.0=pyhd8ed1ab_1',
    'pre-commit',
    'py=1.10.0=pyhd3deb0d_0',
    'pymatgen=2022.5.26=py39hf939315_0',
    'pyyaml=6.0=py39hb9d737c_4',
    'scikit-learn=1.1.1=py39h4037b75_0',
    'scipy=1.8.1=py39he49c0e8_0',
    'setuptools=62.3.2=py39hf3d152e_0',
    'sqlite=3.38.5=h4ff8645_0',
    'sympy=1.10.1=py39hf3d152e_0',
    'toml=0.10.2=pyhd8ed1ab_0',
    'torchaudio=0.11.0=py39_cu113',
    'torchvision=0.12.0=py39_cu113',
    'tqdm=4.64.0=pyhd8ed1ab_0',
    'typing_extensions=4.2.0=pyha770c72_1',
    'wheel=0.37.1=pyhd8ed1ab_0',
    'yaml=0.2.5=h7f98852_2',
]

volume = modal.Volume.from_name("nff-weights", create_if_missing=True)


nff_image = (
    modal.Image.micromamba(python_version="3.9")
    .apt_install("build-essential", "git")
    .micromamba_install(*conda_requirements, channels=['pytorch', 'conda-forge', 'defaults', 'omnia'])
    .pip_install('nglview==3.0.3', 'periodictable==1.6.1', 'seaborn==0.11.2', 'torch==2.2.0', 'e3fp==1.2.5')
    .run_commands("git clone https://github.com/WillEngler/NeuralForceField")
    .run_commands('pip install /NeuralForceField')
    .apt_install("git-lfs")
    .run_commands("git clone https://huggingface.co/willengler-uc/nff")
    .run_commands("ls -la /")
)

app = modal.App("nff")

@app.function(image=nff_image, gpu='A10G')
def get_forces(p):
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    from torch.utils.data import DataLoader

    from nff.io.ase import AtomsBatch
    from nff.io.ase_calcs import NeuralFF
    from nff.data import Dataset
    from nff.train import load_model, evaluate
    import nff.utils.constants as const

    nxyz = torch.tensor(p)
    atoms = AtomsBatch(positions=nxyz[:, 1:], numbers=nxyz[:, 0])
    nff_ase = NeuralFF.from_file("/nff/nff_weights", device="cuda:0")
    atoms.set_calculator(nff_ase)

    results = atoms.get_forces()

    return results

@app.local_entrypoint()
def test_nff():
    p = [
        [6.0, 0.005520567763596773, 0.5914900302886963, -0.0008138174889609218],
        [6.0, -1.2536393404006958, -0.2553567886352539, -0.029800591990351677],
        [8.0, 1.087830662727356, -0.3075546622276306, 0.04822981357574463],
        [1.0, 0.06282120943069458, 1.283752679824829, -0.842788577079773],
        [1.0, 0.006056669633835554, 1.2303121089935303, 0.8853464126586914],
        [1.0, -2.2182207107543945, 0.1898050457239151, -0.058160122483968735],
        [1.0, -0.9109717011451721, -1.0539263486862183, -0.7815958261489868],
        [1.0, -1.192009449005127, -0.7424768209457397, 0.9219667315483093],
        [1.0, 1.8487985134124756, -0.028632404282689095, -0.5256902575492859]
    ]
    result = get_forces.remote(p)
    print(result)
    print(type(result))
    return result