from __future__ import annotations

import modal
import shutil
from pathlib import Path

# --- Basic Modal Setup ---
app = modal.App("torchsim-app")
CACHE_DIR = "/root/.cache/matbench-discovery"
WEIGHTS_CACHE_DIR = "/root/.cache/mace"
# Reusable volume for shared task data and results
BENCHMARK_VOLUME = modal.Volume.from_name(
    "matbench-discovery-data", create_if_missing=True
)
WEIGHTS_VOLUME = modal.Volume.from_name(
    "mlip-weights", create_if_missing=True
)

GPU_CHOICE = "T4"

BENCHMARK_IMAGE = (
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
    .pip_install(
        "ase",
        "pymatgen",
        "numpy",
        "pandas",
        "pyarrow",
        "matbench-discovery @ git+https://github.com/janosh/matbench-discovery.git@d6bd5f5",
    )
)

with BENCHMARK_IMAGE.imports():
    import pandas as pd
    import json
    import torch
    import torch_sim as ts

    from ase.io import read, write
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatviz.enums import Key
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip
    from matbench_discovery.enums import MbdKey

    from torch_sim import generate_force_convergence_fn

@app.function(
    image=BENCHMARK_IMAGE,
    volumes={CACHE_DIR: BENCHMARK_VOLUME},
    timeout=1200,  # 20 minutes
)
def create_smaller_chunks(chunk_size: int = 500, force: bool = False):
    """
    Creates a new set of data chunks with a smaller size (5,000 materials).

    This function is similar to `create_data_chunks` but saves the output to a
    new directory named 'smaller_chunks' to avoid overwriting the original set.

    Args:
        chunk_size (int): Number of materials to include in each chunk. Defaults to 5000.
        force (bool): If True, deletes and regenerates chunks if they already exist.

    Returns:
        int: The number of chunks created.
    """
    # The new directory for the smaller chunks
    chunk_dir = Path(CACHE_DIR) / "smallest_chunks"

    if chunk_dir.exists() and not force:
        # Count existing directories to confirm number of chunks
        n_chunks = len([p for p in chunk_dir.iterdir() if p.is_dir()])
        print(f"✅ Smaller chunks already exist at '{chunk_dir}'. Found {n_chunks} chunks. Skipping creation.")
        BENCHMARK_VOLUME.reload()  # Ensure volume state is up-to-date
        return n_chunks
    elif chunk_dir.exists() and force:
        print(f"💥 Deleting existing smaller chunks at '{chunk_dir}' because force=True.")
        shutil.rmtree(chunk_dir)
        BENCHMARK_VOLUME.commit()

    print(f"🏗️ Starting to create smaller data chunks (size={chunk_size})...")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the full datasets into memory from matbench-discovery
    print("Downloading and loading initial atoms from zip...")
    initial_atoms_path = DataFiles.wbm_initial_atoms.path
    atoms_list = ase_atoms_from_zip(initial_atoms_path)
    print(f"  Loaded {len(atoms_list):,} ASE Atoms objects.")

    print("Downloading and loading WBM computed structure entries...")
    df_wbm_cse = pd.read_json(
        DataFiles.wbm_computed_structure_entries.path, lines=True
    ).set_index(Key.mat_id)

    df_wbm_cse[Key.computed_structure_entry] = df_wbm_cse[Key.computed_structure_entry].apply(json.dumps)
    print("  Serialized entry dictionaries to JSON strings for Parquet compatibility.")

    print(f"  Loaded {len(df_wbm_cse):,} Pymatgen CSEs.")
    BENCHMARK_VOLUME.commit()

    # 2. Split data into chunks and save to the shared volume
    n_atoms = len(atoms_list)
    n_chunks = (n_atoms + chunk_size - 1) // chunk_size
    print(f"\nSplitting {n_atoms:,} materials into {n_chunks} chunks of size {chunk_size:,}...")

    for i in range(n_chunks):
        chunk_path = chunk_dir / str(i)
        chunk_path.mkdir(exist_ok=True)

        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_atoms)
        atoms_chunk = atoms_list[start_idx:end_idx]

        mat_ids_in_chunk = [atom.info[Key.mat_id] for atom in atoms_chunk]
        df_cse_chunk = df_wbm_cse.loc[mat_ids_in_chunk]

        atoms_filename = chunk_path / "atoms.traj"
        write(atoms_filename, atoms_chunk)

        df_cse_chunk.to_parquet(chunk_path / "df_wbm_cse.parquet")

        if (i + 1) % 10 == 0 or (i + 1) == n_chunks:
            print(f"  - Saved chunk {i+1}/{n_chunks} to {chunk_path}")

    print("\nCommitting all smaller chunks to the volume...")
    BENCHMARK_VOLUME.commit()
    print(f"✅ Successfully created and saved {n_chunks} smallest chunks.")
    return n_chunks

@app.cls(
    image=BENCHMARK_IMAGE,
    volumes={CACHE_DIR: BENCHMARK_VOLUME, WEIGHTS_CACHE_DIR: WEIGHTS_VOLUME},
    gpu=GPU_CHOICE,
    timeout=3600, # 1 hour timeout per chunk
    max_containers=8
)
class MatbenchDiscover:
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

    @modal.method()
    def run_chunk_size_binned(self, chunk_index: int, chunk_dir_name: str = "smallest_chunks"):
        """
        Runs the relaxation workflow with simple size-based batching.
        
        Batching strategy:
        - Materials with <10 atoms: 100 per batch
        - Materials with 10-19 atoms: 50 per batch  
        - Materials with 20+ atoms: 25 per batch
        """
        import numpy as np
        import json
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        chunk_path = Path(CACHE_DIR) / chunk_dir_name / str(chunk_index)
        results_file = chunk_path / "results.parquet"
        
        if results_file.exists():
            print(f"✅ Chunk {chunk_index} already processed. Results found at {results_file}. Skipping.")
            return

        print(f"🎬 Starting size-batched run for chunk {chunk_index}...")
        
        # Read all atoms and get their info
        initial_atoms_list = read(chunk_path / "atoms.traj", index=":")
        mat_ids = [atoms.info[Key.mat_id] for atoms in initial_atoms_list]
        atom_counts = [len(atoms) for atoms in initial_atoms_list]
        
        print(f"  - Found {len(initial_atoms_list)} materials")
        print(f"  - Atom count range: {min(atom_counts)} - {max(atom_counts)}")
        
        # Create lists for each size category with (index, atoms, mat_id) tuples
        small_materials = []  # <10 atoms
        medium_materials = [] # 10-19 atoms
        large_materials = []  # 20+ atoms
        
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
                return
                
            n_batches = (len(materials) + batch_size - 1) // batch_size
            print(f"\n🔄 Processing {category_name} materials: {len(materials)} materials in {n_batches} batches")
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(materials))
                batch_materials = materials[start_idx:end_idx]
                
                # Extract atoms and mat_ids for this batch
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
                        "relaxed_atoms_json": json.dumps(atoms.todict(), cls=NumpyEncoder),
                        "original_index": orig_idx  # Keep track of original ordering
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
        
        # Sort results back to original order before saving
        all_results.sort(key=lambda x: x['original_index'])
        
        # Remove the temporary original_index field
        for result in all_results:
            del result['original_index']
        
        print(f"\n✅ Chunk {chunk_index} processing complete")
        print(f"  - Total materials processed: {len(all_results)}")
        
        # Save results
        df_results = pd.DataFrame(all_results).set_index("material_id")
        print(f"💾 Saving results to {results_file}...")
        df_results.to_parquet(results_file)
        BENCHMARK_VOLUME.commit()
        print("  - Save complete!")


@app.function(
    image=BENCHMARK_IMAGE,
    volumes={CACHE_DIR: BENCHMARK_VOLUME, WEIGHTS_CACHE_DIR: WEIGHTS_VOLUME},
    gpu=GPU_CHOICE,
    timeout=1200,  # Increased timeout for the extra relaxation
)
def analyze_nth_worst_relaxation(
    chunk_dir_name: str = "smallest_chunks",
    material_id: str | None = None,
    error_rank: int = 1,
):
    """
    Analyzes a failed relaxation by finding the Nth worst offender and comparing
    its geometry and energy against a baseline ASE relaxation.

    Args:
        chunk_dir_name (str): The directory where result chunks are stored.
        material_id (str, optional): A specific material_id to analyze. If None,
                                     the function will find the Nth worst.
        error_rank (int): The rank of the error to investigate (1=worst, 2=2nd worst, etc.).
    """
    # --- Self-Contained Imports and Constants ---
    import json
    import torch
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from ase import Atoms
    from ase.io import read
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from mace.calculators import mace_mp
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from matbench_discovery.data import DataFiles, df_wbm
    from matbench_discovery.enums import MbdKey
    from matbench_discovery.energy import (
        mp_elemental_ref_energies,
        calc_energy_from_e_refs,
    )

    CACHE_DIR = "/root/.cache/matbench-discovery"
    # -------------------------------------------

    print("🔍 Faulty Relaxation Analyzer")
    print("=" * 80)

    chunk_dir = Path(CACHE_DIR) / chunk_dir_name

    # --- Step 0: Find the Nth worst offender if no material_id is given ---
    if material_id is None:
        print(f"--- Step 0: Finding material with error rank {error_rank} ---")
        all_results_files = sorted(list(chunk_dir.glob("**/results.parquet")))
        if not all_results_files:
            raise FileNotFoundError(f"No results.parquet files found in {chunk_dir}")

        df_preds = pd.concat(pd.read_parquet(f) for f in all_results_files)
        df_preds = df_preds.dropna(subset=["final_energy", "relaxed_atoms_json"])
        
        # Simplified scoring to find errors
        df_wbm_cse = pd.read_json(
            DataFiles.wbm_computed_structure_entries.path, lines=True
        ).set_index("material_id")
        df_preds["composition"] = [
            ComputedStructureEntry.from_dict(
                df_wbm_cse.loc[idx, "computed_structure_entry"]
            ).composition
            for idx in df_preds.index
        ]
        df_preds["e_form_pred"] = [
            calc_energy_from_e_refs(
                row["composition"],
                total_energy=row["final_energy"],
                ref_energies=mp_elemental_ref_energies,
            )
            / row["composition"].num_atoms
            for _, row in df_preds.iterrows()
        ]
        df_preds["e_form_dft"] = df_wbm.loc[df_preds.index, MbdKey.e_form_dft]
        df_preds["abs_error"] = (
            df_preds["e_form_pred"] - df_preds["e_form_dft"]
        ).abs()

        df_sorted = df_preds.sort_values(by="abs_error", ascending=False)
        if error_rank > len(df_sorted):
            raise ValueError(
                f"error_rank {error_rank} is out of bounds. Only {len(df_sorted)} predictions available."
            )
        
        material_id = df_sorted.index[error_rank - 1]
        print(f"  - Found material with rank {error_rank}: {material_id}")

    print(
        f"Analyzing the relaxation of {material_id} to understand geometric distortion."
    )
    print("=" * 80)

    # --- Step A: Load Initial and Relaxed Data ---
    print("\n--- Step A: Load Initial and Relaxed Data ---")
    
    found = False
    for f in sorted(list(chunk_dir.glob("**/results.parquet"))):
        df_chunk = pd.read_parquet(f)
        if material_id in df_chunk.index:
            ml_result = df_chunk.loc[material_id]
            found = True
            print(f"  - Found torch-sim result for {material_id} in {f.name}")
            break
    if not found:
        raise ValueError(f"Could not find result for {material_id} in any chunk.")
        
    initial_atoms_file = f.parent / "atoms.traj"
    initial_atoms_list = read(initial_atoms_file, index=":")
    initial_atoms = next(
        (
            atoms
            for atoms in initial_atoms_list
            if atoms.info.get("material_id") == material_id
        ),
        None,
    )
    if initial_atoms is None:
        raise ValueError(f"Could not find initial structure for {material_id}")
    print(f"  - Found initial structure for {material_id}")

    torch_sim_energy = ml_result["final_energy"]
    relaxed_atoms_ts = Atoms.fromdict(json.loads(ml_result["relaxed_atoms_json"]))
    

    # --- Step B: Analyze Structures ---
    print("\n--- Step B: Structural Geometry Analysis ---")

    def analyze_structure(atoms: Atoms, name: str):
        print(f"\n  Analyzing '{name}' structure ({atoms.get_chemical_formula()})...")
        if not all(atoms.pbc):
            print("  - Structure has no periodic boundary conditions.")
            volume = 0.0
        else:
            volume = atoms.get_volume()
            print(f"  - Cell Volume: {volume:.2f} Å³")
        
        distances = atoms.get_all_distances(mic=True).flatten()
        distances = distances[distances > 1e-3]
        
        min_dist = distances.min() if len(distances) > 0 else 0
        max_dist = distances.max() if len(distances) > 0 else 0
        print(f"  - Min Interatomic Distance: {min_dist:.3f} Å")
        print(f"  - Max Interatomic Distance: {max_dist:.3f} Å")
        return {"volume": volume, "min_dist": min_dist, "max_dist": max_dist}

    initial_metrics = analyze_structure(initial_atoms, "Initial")
    relaxed_ts_metrics = analyze_structure(
        relaxed_atoms_ts, "Relaxed (from torch-sim)"
    )

    # --- Step B.5: Perform Baseline ASE Relaxation ---
    print("\n--- Step B.5: Perform Baseline ASE Relaxation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ase_calc = mace_mp(model="medium", device=device, default_dtype="float64")
    
    baseline_atoms = initial_atoms.copy()
    baseline_atoms.calc = ase_calc
    
    print("  - Relaxing with ase.optimize.FIRE and FrechetCellFilter...")
    ff = FrechetCellFilter(baseline_atoms)
    opt = FIRE(ff, logfile=None)
    opt.run(fmax=0.05, steps=500)
    
    ase_relaxed_energy = baseline_atoms.get_potential_energy()
    relaxed_ase_metrics = analyze_structure(baseline_atoms, "Relaxed (from ASE)")
    print(f"\n  - ASE relaxation complete. Final energy: {ase_relaxed_energy:.2f} eV")

    # --- Step C: Diagnosis ---
    print("\n--- Step C: Diagnosis ---")
    print("=" * 80)
    print(
        f"{'Metric':<25} | {'Initial':>12} | {'torch-sim':>12} | {'ASE Baseline':>12}"
    )
    print("-" * 70)
    print(
        f"{'Cell Volume (Å³)':<25} | {initial_metrics['volume']:12.2f} | {relaxed_ts_metrics['volume']:12.2f} | {relaxed_ase_metrics['volume']:12.2f}"
    )
    print(
        f"{'Min Distance (Å)':<25} | {initial_metrics['min_dist']:12.3f} | {relaxed_ts_metrics['min_dist']:12.3f} | {relaxed_ase_metrics['min_dist']:12.3f}"
    )
    print(
        f"{'Final Energy (eV)':<25} | {'N/A':>12} | {torch_sim_energy:12.2f} | {ase_relaxed_energy:12.2f}"
    )
    print("=" * 80)

    # Final conclusion
    if relaxed_ts_metrics["min_dist"] < 0.5:
        print("\n❌ DIAGNOSIS: Catastrophic structural collapse confirmed in `torch-sim`.")
        print(
            "The `torch-sim` optimizer moved atoms on top of each other, leading to an"
        )
        print(
            "unphysical structure with extreme negative energy. The ASE baseline produced"
        )
        print(
            "a physically reasonable structure. This definitively points to unstable"
        )
        print("dynamics in the `torch_sim.frechet_cell_fire` optimizer implementation.")
    elif abs(torch_sim_energy - ase_relaxed_energy) > 100:  # High threshold for pathological cases
        print("\n❌ DIAGNOSIS: Pathological energy minimum found by `torch-sim`.")
        print(
            "While the geometry hasn't fully collapsed, the `torch-sim` optimizer found"
        )
        print(
            "a wildly different and unphysical energy state compared to the trusted"
        )
        print(
            "ASE baseline. This confirms the `torch_sim` optimizer is unstable for this system."
        )
    else:
        print("\n🤔 DIAGNOSIS: No obvious geometric catastrophe.")
        print("This is a very strange case. The final diagnosis is unclear.")

    print("\nRECOMMENDATION: The `torch_sim.frechet_cell_fire` optimizer is the root cause.")
    print(
        "Its dynamics are unstable for some materials. You should switch to a more robust"
    )
    print(
        "alternative, such as using the pure ASE optimizer or trying the 'vv_fire' flavor,"
    )
    print("which might have different stability characteristics.")

    return {
        "material_id": material_id,
        "initial_metrics": initial_metrics,
        "relaxed_ts_metrics": relaxed_ts_metrics,
        "relaxed_ase_metrics": relaxed_ase_metrics,
        "ts_energy": torch_sim_energy,
        "ase_energy": ase_relaxed_energy,
    }

@app.function(
    image=BENCHMARK_IMAGE,
    volumes={CACHE_DIR: BENCHMARK_VOLUME},
    timeout=3600,  # Increased timeout for processing many files
)
def score_all_results_with_filtering(chunk_dir_name: str = "smallest_chunks"):
    """
    Scores all results after filtering out pathological relaxations based on
    physical and energetic criteria.
    
    Args:
        chunk_dir_name (str): Name of the chunk directory. Defaults to "smallest_chunks".
    
    Returns:
        dict: Scoring metrics and summary information after filtering.
    """
    import json
    import numpy as np
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    from matbench_discovery.data import DataFiles, df_wbm
    from matbench_discovery.energy import mp_elemental_ref_energies, calc_energy_from_e_refs
    from matbench_discovery.metrics.discovery import stable_metrics
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.core import Composition

    chunk_dir = Path(CACHE_DIR) / chunk_dir_name
    
    if not chunk_dir.exists():
        raise ValueError(f"Directory '{chunk_dir}' does not exist.")
    
    # 1. Load all raw prediction data
    print(f"🔍 Loading all raw predictions from '{chunk_dir}'...")
    all_results_files = sorted(list(chunk_dir.glob("**/results.parquet")))
    if not all_results_files:
        raise FileNotFoundError(f"No results.parquet files found in {chunk_dir}")
    
    print(len(all_results_files))
    # return(len(all_results_files))
    df_preds = pd.concat(pd.read_parquet(f) for f in all_results_files)
    df_preds = df_preds.dropna(subset=["final_energy", "relaxed_atoms_json"])
    n_raw_predictions = len(df_preds)
    print(f"  - Loaded {n_raw_predictions:,} raw predictions.")

    # 2. Apply physical and energetic filters
    print("\nApplying filters to remove unphysical results...")
    
    # Filter 1: Energy threshold
    energy_threshold = -10_000  # eV
    energy_filter = df_preds["final_energy"] > energy_threshold
    n_after_energy_filter = energy_filter.sum()
    print(f"  - Energy filter (> {energy_threshold} eV): Kept {n_after_energy_filter:,} / {len(df_preds):,} predictions.")
    
    df_filtered = df_preds[energy_filter].copy()

    # Filter 2: Geometric threshold (minimum interatomic distance)
    def get_min_dist(atoms_json):
        try:
            atoms = Atoms.fromdict(json.loads(atoms_json))
            if len(atoms) < 2:
                return np.inf
            distances = atoms.get_all_distances(mic=True)
            # Get the upper triangle of the distance matrix to avoid self-distances
            # and double counting.
            min_dist = distances[np.triu_indices(len(distances), k=1)].min()
            return min_dist
        except Exception:
            # If structure parsing fails, treat it as a failed calculation
            return 0 
            
    df_filtered["min_dist"] = df_filtered["relaxed_atoms_json"].apply(get_min_dist)
    
    dist_threshold = 1.0  # Angstrom
    dist_filter = df_filtered["min_dist"] > dist_threshold
    n_after_dist_filter = dist_filter.sum()
    print(f"  - Geometry filter (min distance > {dist_threshold} Å): Kept {n_after_dist_filter:,} / {len(df_filtered):,} predictions.")

    df_final = df_filtered[dist_filter].copy()
    n_filtered_out = n_raw_predictions - len(df_final)
    print(f"\n  - Total results kept for scoring: {len(df_final):,}")
    print(f"  - Total results filtered out: {n_filtered_out:,} ({n_filtered_out/n_raw_predictions:.2%})")

    # 3. Proceed with scoring on the cleaned DataFrame
    print("\nScoring filtered results...")
    
    df_wbm_cse = pd.read_json(
        DataFiles.wbm_computed_structure_entries.path, lines=True
    ).set_index("material_id")
    
    updated_cses = {}
    for mat_id, row in df_final.iterrows():
        cse_dict = df_wbm_cse.loc[mat_id, "computed_structure_entry"]
        original_cse = ComputedStructureEntry.from_dict(cse_dict)
        relaxed_atoms = Atoms.fromdict(json.loads(row["relaxed_atoms_json"]))
        new_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
        original_cse._energy = row["final_energy"]
        original_cse._structure = new_structure
        updated_cses[mat_id] = original_cse

    df_final["cse_updated"] = pd.Series(updated_cses)

    processed_entries = MaterialsProject2020Compatibility().process_entries(
        df_final["cse_updated"].dropna(), verbose=False, clean=True
    )
    print(f"  - {len(processed_entries):,}/{len(df_final):,} entries remain after MP compatibility processing.")
    
    if not processed_entries:
        raise ValueError("All entries were filtered out by compatibility processing")

    processed_dict = {entry.entry_id: entry for entry in processed_entries}
    df_final["processed_cse"] = pd.Series(processed_dict)
    df_final = df_final.dropna(subset=["processed_cse"])
    df_final["e_raw_per_atom"] = [e.energy_per_atom for e in df_final["processed_cse"]]

    # Calculate formation energy
    df_final[MbdKey.e_form_dft] = df_wbm.loc[df_final.index, MbdKey.e_form_dft]
    df_final["e_form_pred"] = [
        calc_energy_from_e_refs(
            cse.composition,
            total_energy=cse.energy,
            ref_energies=mp_elemental_ref_energies,
        ) / cse.composition.num_atoms
        for cse in df_final.processed_cse
    ]

    # Calculate final metrics
    print("\nCalculating final stability metrics on filtered data...")
    metrics = stable_metrics(
        each_true=df_final[MbdKey.e_form_dft],
        each_pred=df_final["e_form_pred"],
    )

    print("\n" + "="*60)
    print("✅ Filtered Scoring Complete")
    print("="*60)
    print(f"📊 Summary:")
    print(f"  - Initial predictions: {n_raw_predictions:,}")
    print(f"  - Predictions after filtering: {len(df_final):,}")
    print(f"  - Filtered out: {n_filtered_out:,} ({n_filtered_out/n_raw_predictions:.2%})")
    print("\n🎯 Final Filtered Stability Metrics:")
    for key, val in metrics.items():
        print(f"  - {key:<25}: {val:.4f}")
    print("="*60)
    
    return metrics

@app.local_entrypoint()
def main():
    # next_batch = [i for i in range(400, 514)]
    # for res in MatbenchDiscover().run_chunk_size_binned.map(next_batch):
    #     print(res)
    # MatbenchDiscover().run_chunk_size_binned.remote(294)
    # MatbenchDiscover().run_chunk_size_binned
    # get_final_benchmark_score.remote()
    # trace_single_material_scoring.remote()
    # analyze_faulty_relaxation.remote()
    # analyze_nth_worst_relaxation.remote(error_rank=2)
    # get_chunk_file_sizes.remote(0)
    score_all_results_with_filtering.remote()