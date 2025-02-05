import modal

app = modal.App("mace-comparison")

mace_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("ase==3.23.0", "mace-torch==0.3.6", "pymatgen==2024.8.9")
    .apt_install("curl")
    .run_commands("python3 -c \"MODEL_NAME = 'https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model';from mace.calculators import mace_mp; mace_mp(model=MODEL_NAME, device='cpu', default_dtype='float64')\"") # This caches the weights in the container
)

with mace_image.imports():
    import numpy as np
    from ase import Atoms
    from mace.calculators import mace_mp


@app.function(image=mace_image, cpu=8)
def get_forces(atoms_nxyz):
    MODEL_NAME = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model"
    mace_calc = mace_mp(model=MODEL_NAME, device="cpu", default_dtype="float64")
    nxyz = np.array(atoms_nxyz)
    
    atoms = Atoms(
        numbers=nxyz[:, 0],
        positions=nxyz[:, 1:],
    )

    atoms.set_calculator(mace_calc)

    forces = atoms.get_forces()
    return forces

@app.local_entrypoint()
def main():
    atoms_nxyz = [
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
    forces = get_forces.remote(atoms_nxyz)
    print(forces)
