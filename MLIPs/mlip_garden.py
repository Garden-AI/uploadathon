from __future__ import annotations

import modal

# --- Basic Modal Setup ---
app = modal.App("mlip-app")
CACHE_DIR = "/root/.cache/matbench-discovery"
WEIGHTS_CACHE_DIR = "/root/.cache/mace"
WEIGHTS_VOLUME = modal.Volume.from_name(
    "mlip-weights", create_if_missing=True
)

GPU_CHOICE = "T4"

MLIP_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Install PyTorch and the CUEQ libraries for GPU acceleration
    .pip_install(
        "torch==2.4.0",
        "cuequivariance==0.1.0",
        "cuequivariance-torch==0.1.0",
        "cuequivariance-ops-torch-cu12==0.1.0",
        gpu=GPU_CHOICE,
    )
    # Install MACE with CUEQ support and TorchSim
    .pip_install(
        # This specific version of mace-torch is built to use cuequivariance
        "mace-torch[cueq] @ git+https://github.com/ACEsuit/mace.git@v0.3.10",
        # The main library for running the simulations
        "torch-sim-atomistic",
        gpu=GPU_CHOICE,
    )
    # Install the remaining packages
    .pip_install("ase")
)

with MLIP_IMAGE.imports():
    import torch
    import torch_sim as ts
    from ase.io import read, write
    from torch_sim import generate_force_convergence_fn

@app.cls(
    image=MLIP_IMAGE,
    volumes={WEIGHTS_CACHE_DIR: WEIGHTS_VOLUME},
    gpu=GPU_CHOICE,
    timeout=1800, # 30 minute timeout
    max_containers=3
)
class MACE:
    @modal.enter()
    def setup(self):
        """
        This method is run once when the container starts. It initializes
        the MACE model, which will be reused by sub-batches.
        """
        from mace.calculators import mace_mp
        from torch_sim.models.mace import MaceModel

        self.device = torch.device("cuda")

        print("Pre-loading MACE model...")
        # Get the raw model directly, bypassing the ASE Calculator wrapper
        raw_model = mace_mp(
            model="medium",
            device=self.device,
            enable_cueq=True,
            default_dtype="float64",
            return_raw_model=True,
        )
        self.mace_model = MaceModel(model=raw_model, device=self.device)
        print("MACE model loaded and ready.")
        WEIGHTS_VOLUME.commit()

    # Your SDK class will need to call MatbenchDiscover.relax
    @modal.method()
    def relax(self, xyz_file_contents: str) -> str:
        from io import StringIO

        string_file = StringIO(xyz_file_contents)
        initial_atoms_list = read(string_file, index=":", format='extxyz')
        
        # Extract material IDs (create if not present) and atom counts - matching original pattern
        mat_ids = []
        for idx, atoms in enumerate(initial_atoms_list):
            if 'material_id' in atoms.info:
                mat_ids.append(atoms.info['material_id'])
                print(atoms.info['material_id'])
            else:
                mat_id = f"material_{idx}"
                atoms.info['material_id'] = mat_id
                mat_ids.append(mat_id)
        
        atom_counts = [len(atoms) for atoms in initial_atoms_list]
        
        print(f"  - Found {len(initial_atoms_list)} materials")
        print(f"  - Atom count range: {min(atom_counts)} - {max(atom_counts)}")
        
        # Create lists for each size category with (index, atoms, mat_id) tuples
        small_materials = []  # <10 atoms
        medium_materials = [] # 10-19 atoms
        large_materials = []  # 20+ atoms
        
        # Use the same loop structure as the original working code
        for idx, (atoms, mat_id, n_atoms) in enumerate(zip(initial_atoms_list, mat_ids, atom_counts)):
            material_data = (idx, atoms, mat_id)
            if n_atoms < 10:
                small_materials.append(material_data)
            elif n_atoms < 20:
                medium_materials.append(material_data)
            else:
                large_materials.append(material_data)
        
        print(f"\n📊 Size distribution:")
        print(f"  - Small (<10 atoms): {len(small_materials)} materials")
        print(f"  - Medium (10-19 atoms): {len(medium_materials)} materials")
        print(f"  - Large (20+ atoms): {len(large_materials)} materials")

        # Define batch sizes for each category
        batch_sizes = {
            'small': 80,
            'medium': 40,
            'large': 16
        }
        
        # Process each size category
        all_results = []
        
        # Helper function to process a category
        def process_category(materials, category_name, batch_size):
            """Process all materials in a size category."""
            if not materials:
                print('nothing in ' + category_name)
                return

            n_batches = (len(materials) + batch_size - 1) // batch_size
            print(f"\n🔄 Processing {category_name} materials: {len(materials)} materials in {n_batches} batches")

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(materials))
                batch_materials = materials[start_idx:end_idx]
                
                batch_atoms = [item[1] for item in batch_materials]
                batch_mat_ids = [item[2] for item in batch_materials]
                batch_original_indices = [item[0] for item in batch_materials]
                
                print(f"  - Batch {batch_idx + 1}/{n_batches}: {len(batch_atoms)} materials")
                
                # Run relaxation
                initial_state = ts.initialize_state(batch_atoms, device=self.device, dtype=torch.float64)
                optimizer_builder = ts.frechet_cell_fire
                optimizer_callable = lambda model, **_kwargs: optimizer_builder(model, md_flavor="ase_fire")
                
                relaxed_state = ts.optimize(
                    system=initial_state,
                    model=self.mace_model,
                    optimizer=optimizer_callable,
                    max_steps=500,
                    convergence_fn=generate_force_convergence_fn(force_tol=0.05),
                )
                
                # Collect results
                relaxed_atoms_list = relaxed_state.to_atoms()
                final_energies = relaxed_state.energy.cpu().tolist()
                
                batch_results = [
                    {
                        "material_id": mat_id,
                        "final_energy": energy,
                        "relaxed_atoms": atoms,
                        "original_index": orig_idx
                    }
                    for mat_id, atoms, energy, orig_idx in zip(
                        batch_mat_ids, relaxed_atoms_list, final_energies, batch_original_indices
                    )
                ]
                all_results.extend(batch_results)
        
        # Process each category
        process_category(small_materials, "small", batch_sizes['small'])
        process_category(medium_materials, "medium", batch_sizes['medium'])
        process_category(large_materials, "large", batch_sizes['large'])
        
        # Sort results back to original order
        all_results.sort(key=lambda x: x['original_index'])
        
        # Prepare relaxed atoms for XYZ output
        relaxed_atoms_list = []
        for result in all_results:
            atoms = result['relaxed_atoms']  # Already an ASE Atoms object
            # Add material_id and energy to info
            atoms.info['material_id'] = result['material_id']
            atoms.info['final_energy'] = result['final_energy']
            relaxed_atoms_list.append(atoms)
        
        # Write all relaxed structures to XYZ format string
        output_string = StringIO()
        write(output_string, relaxed_atoms_list, format='extxyz')
        xyz_content = output_string.getvalue()
        
        print(f"\n✅ Relaxation complete! Returning XYZ content for {len(relaxed_atoms_list)} structures")
        return xyz_content


@app.local_entrypoint()
def main():
    sample_file_contents = '''32
Lattice="7.16 0.0 0.0 0.0 7.16 0.0 0.0 0.0 7.16" Properties=species:S:1:pos:R:3 pbc="T T T"
Cu       0.00000000       0.00000000       0.00000000
Cu       0.00000000       1.79000000       1.79000000
Cu       1.79000000       0.00000000       1.79000000
Cu       1.79000000       1.79000000       0.00000000
Cu       0.00000000       0.00000000       3.58000000
Cu       0.00000000       1.79000000       5.37000000
Cu       1.79000000       0.00000000       5.37000000
Cu       1.79000000       1.79000000       3.58000000
Cu       0.00000000       3.58000000       0.00000000
Cu       0.00000000       5.37000000       1.79000000
Cu       1.79000000       3.58000000       1.79000000
Cu       1.79000000       5.37000000       0.00000000
Cu       0.00000000       3.58000000       3.58000000
Cu       0.00000000       5.37000000       5.37000000
Cu       1.79000000       3.58000000       5.37000000
Cu       1.79000000       5.37000000       3.58000000
Cu       3.58000000       0.00000000       0.00000000
Cu       3.58000000       1.79000000       1.79000000
Cu       5.37000000       0.00000000       1.79000000
Cu       5.37000000       1.79000000       0.00000000
Cu       3.58000000       0.00000000       3.58000000
Cu       3.58000000       1.79000000       5.37000000
Cu       5.37000000       0.00000000       5.37000000
Cu       5.37000000       1.79000000       3.58000000
Cu       3.58000000       3.58000000       0.00000000
Cu       3.58000000       5.37000000       1.79000000
Cu       5.37000000       3.58000000       1.79000000
Cu       5.37000000       5.37000000       0.00000000
Cu       3.58000000       3.58000000       3.58000000
Cu       3.58000000       5.37000000       5.37000000
Cu       5.37000000       3.58000000       5.37000000
Cu       5.37000000       5.37000000       3.58000000
'''
    print(MACE().relax.remote(sample_file_contents))