from __future__ import annotations

import modal

app = modal.App("mlip-app")

#
# VOLUMES
#
# Unfortunately, each model sets its weights cache at a specific location we can't override.
# So we need to mount separate volumes for each model.

WEIGHTS_CACHE_DIR = "/root/.cache/mace"
WEIGHTS_VOLUME = modal.Volume.from_name(
    "mlip-weights", create_if_missing=True
)

SEVENNET_WEIGHTS_VOLUME = modal.Volume.from_name(
    "sevennet-weights", create_if_missing=True
)
SEVENNET_CACHE_PATH = "/usr/local/lib/python3.11/site-packages/sevenn/pretrained_potentials/SevenNet_MF_ompa"

MATTERSIM_WEIGHTS_VOLUME = modal.Volume.from_name(
    "mattersim-weights", create_if_missing=True
)
# The default path used by MatterSim
MATTERSIM_CACHE_PATH = "/root/.local/mattersim/pretrained_models"

# Matbench Discovery data, used to verify the models are working
BENCHMARK_CACHE_DIR = "/root/.cache/matbench-discovery"
BENCHMARK_VOLUME = modal.Volume.from_name(
    "matbench-discovery-data", create_if_missing=True
)

GPU_CHOICE = "T4"

#
# Containers
#
# MACE has some dependencies that are incompatible with the other models.
# But Mattersim and 7net can use the same container.

# MACE ENV

MACE_IMAGE = (
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
        "pandas",
        "numpy",
        "pyarrow",
    )
    .pip_install(
        # Just needed for ad hoc benchmarking
        "matbench-discovery @ git+https://github.com/janosh/matbench-discovery.git@d6bd5f5",
        "plotly"
    )
)


### MatterSim and 7net ENV

UNIFIED_BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        gpu=GPU_CHOICE,
    )
    .pip_install(
        # Core torch-sim and geometric dependencies
        "torch-sim-atomistic",
        "torch-geometric",
        "pymatgen",
        "dgl",
        gpu=GPU_CHOICE,
    )
    .pip_install(
        # Equivariance dependencies (needed for multiple models)
        "cuequivariance==0.1.0", 
        "cuequivariance-torch==0.1.0",
        "cuequivariance-ops-torch-cu12==0.1.0",
        gpu=GPU_CHOICE,
    )
    .apt_install("git")  # Moved git install before pip installs that need it
    .pip_install(
        # SevenNet dependencies
        "sevenn",
        "plotly",  # Required for SevenNet
        gpu=GPU_CHOICE,
    )
    .pip_install(
        # MatterSim dependencies (install last as it's from GitHub)
        "git+https://github.com/microsoft/mattersim.git",
        gpu=GPU_CHOICE,
    )
    .pip_install(
        # Common utilities
        "pymatviz",
        gpu=GPU_CHOICE,
    )
    .pip_install(
        # Just needed for ad hoc benchmarking
        "matbench-discovery @ git+https://github.com/janosh/matbench-discovery.git@d6bd5f5",
        "plotly"
    )
)

def validate_relaxation_params(relax_params: dict | None, frechet_supported: bool = True) -> tuple[dict, str | None]:
    """
    Validate and normalize relaxation parameters.
    
    Args:
        relax_params: Dictionary of relaxation parameters from user
        frechet_supported: Whether frechet_cell_fire optimizer is supported
        
    Returns:
        Tuple of (validated_params, error_message). If error_message is not None,
        validated_params should be ignored.
    """
    if relax_params is None:
        relax_params = {}
        
    # Create a copy to avoid modifying the original
    validated = {}
    
    # Validate optimizer_type
    optimizer_type = relax_params.get("optimizer_type", "frechet_cell_fire")
    if optimizer_type not in ["frechet_cell_fire", "fire"]:
        return {}, f"Invalid optimizer_type '{optimizer_type}'. Must be 'frechet_cell_fire' or 'fire'."
    
    if optimizer_type == "frechet_cell_fire" and not frechet_supported:
        return {}, "The 'frechet_cell_fire' optimizer is not supported in this context. Use 'fire' instead."
        
    validated["optimizer_type"] = optimizer_type
    
    # Validate md_flavor
    md_flavor = relax_params.get("md_flavor", "ase_fire")
    if md_flavor not in ["ase_fire", "vv_fire"]:
        return {}, f"Invalid md_flavor '{md_flavor}'. Must be 'ase_fire' or 'vv_fire'."
    validated["md_flavor"] = md_flavor
    
    # Validate frechet-specific parameters
    frechet_only_bool_params = ["hydrostatic_strain", "constant_volume"]
    frechet_only_float_params = ["scalar_pressure"]
    
    # Handle boolean frechet params
    for param_name in frechet_only_bool_params:
        if param_name in relax_params:
            param_value = relax_params[param_name]
            
            # If user is using fire but passed frechet-specific params, error
            if optimizer_type == "fire":
                return {}, f"Parameter '{param_name}' is only valid for 'frechet_cell_fire' optimizer, but 'fire' optimizer was specified."
                
            # Validate boolean type
            if not isinstance(param_value, bool):
                return {}, f"Parameter '{param_name}' must be a boolean (True/False), got {type(param_value).__name__}."
            validated[param_name] = param_value
        elif optimizer_type == "frechet_cell_fire":
            # Set defaults for frechet_cell_fire
            validated[param_name] = False
    
    # Handle float frechet params
    for param_name in frechet_only_float_params:
        if param_name in relax_params:
            param_value = relax_params[param_name]
            
            # If user is using fire but passed frechet-specific params, error
            if optimizer_type == "fire":
                return {}, f"Parameter '{param_name}' is only valid for 'frechet_cell_fire' optimizer, but 'fire' optimizer was specified."
                
            # Validate float type (allow int too, will be converted)
            if not isinstance(param_value, (float, int)):
                return {}, f"Parameter '{param_name}' must be a number, got {type(param_value).__name__}."
            validated[param_name] = float(param_value)
        elif optimizer_type == "frechet_cell_fire":
            # Set defaults for frechet_cell_fire
            if param_name == "scalar_pressure":
                validated[param_name] = 0.0
    
    # Copy through other known parameters with their defaults
    other_params = {
        "fmax": 0.05,
        "max_steps": 500,
    }
    
    for param_name, default_value in other_params.items():
        if param_name in relax_params:
            validated[param_name] = relax_params[param_name]
        else:
            validated[param_name] = default_value
    
    return validated, None


def _perform_batch_relaxation(
    xyz_file_contents: str,
    model,
    device,
    validated_params: dict,
    dtype: str = "float64",
) -> str:
    """
    Common helper function to perform batch relaxation using any torch-sim compatible model.
    
    Args:
        xyz_file_contents: String content of XYZ file
        model: The torch-sim model to use for relaxation
        device: The torch device to use
        validated_params: Pre-validated dictionary of relaxation parameters
        dtype: Data type to use ("float64" or "float32")
        
    Returns:
        String content of relaxed structures in XYZ format
    """
    from io import StringIO
    import torch

    # --- Resolve the optimizer based on validated parameters ---
    optimizer_type = validated_params["optimizer_type"]
    if optimizer_type == "frechet_cell_fire":
        optimizer_builder = ts.frechet_cell_fire
        # These parameters are only valid for frechet_cell_fire
        optimizer_kwargs = {
            "hydrostatic_strain": validated_params["hydrostatic_strain"],
            "constant_volume": validated_params["constant_volume"],
            "scalar_pressure": validated_params["scalar_pressure"],
            "md_flavor": validated_params["md_flavor"],
        }
        convergence_fn = generate_force_convergence_fn(
            force_tol=validated_params["fmax"], include_cell_forces=True
        )
    elif optimizer_type == "fire":
        optimizer_builder = ts.optimizers.fire
        optimizer_kwargs = {"md_flavor": validated_params["md_flavor"]}
        convergence_fn = generate_force_convergence_fn(
            force_tol=validated_params["fmax"], include_cell_forces=False
        )
    else:
        # This should never happen with validated params, but keep for safety
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    string_file = StringIO(xyz_file_contents)
    initial_atoms_list = read(string_file, index=":", format='extxyz')

    # --- Resolve the dtype string inside the container ---
    if dtype == "float64":
        torch_dtype = torch.float64
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype string: {dtype}")

    # Extract material IDs (create if not present) and atom counts - matching original pattern
    mat_ids = []
    for idx, atoms in enumerate(initial_atoms_list):
        if 'material_id' in atoms.info:
            mat_ids.append(atoms.info['material_id'])
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
            initial_state = ts.initialize_state(batch_atoms, device=device, dtype=torch_dtype)
            optimizer_callable = lambda model, **_kwargs: optimizer_builder(model, **optimizer_kwargs)
            
            relaxed_state = ts.optimize(
                system=initial_state,
                model=model,
                optimizer=optimizer_callable,
                max_steps=validated_params.get("max_steps", 500),
                convergence_fn=convergence_fn,
            )
            
            relaxed_atoms_list = relaxed_state.to_atoms()
            energies_tensor = relaxed_state.energy

            # Ensure energies are always in a list
            if energies_tensor.dim() == 0:
                # This is a 0-d tensor, so convert to a list with one item
                final_energies = [energies_tensor.item()]
            else:
                # This is a 1-d or higher tensor, tolist() works correctly
                final_energies = energies_tensor.cpu().tolist()

            # Also ensure the atoms list is always a list for consistency
            if not isinstance(relaxed_atoms_list, list):
                relaxed_atoms_list = [relaxed_atoms_list]
            
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

def _get_benchmark_data(n_samples: int = 2500) -> tuple[str, dict]:
    """
    Loads cached benchmark structures from the full 2500-structure XYZ file
    and fetches their ground truth energies from df_wbm at runtime. If n_samples
    is provided, it returns the first n structures from the file.
    """
    from pathlib import Path
    from ase.io import read, write
    from io import StringIO
    from pymatviz.enums import Key
    from matbench_discovery.data import df_wbm
    from matbench_discovery.enums import MbdKey

    full_benchmark_size = 2500
    structures_path = Path(BENCHMARK_CACHE_DIR) / f"wbm_subset_{full_benchmark_size}_structures.xyz"

    if not structures_path.exists():
        raise FileNotFoundError(
            f"Cached benchmark structures not found at {structures_path}. "
            f"Please run 'modal run mlip_garden.py::cache_matbench_subset --n-samples {full_benchmark_size}' first."
        )

    # Read all Atoms objects from the canonical file
    full_atoms_list = read(structures_path, index=":")
    
    if n_samples < len(full_atoms_list):
        atoms_list = full_atoms_list[:n_samples]
    else:
        atoms_list = full_atoms_list
    
    string_buffer = StringIO()
    write(string_buffer, atoms_list, format='extxyz')
    xyz_contents = string_buffer.getvalue()
    
    # Get material IDs from the (potentially smaller) list of structures
    mat_ids = [atoms.info[Key.mat_id] for atoms in atoms_list]

    # Fetch ground truth energies on demand
    ground_truth_e_form = df_wbm.loc[mat_ids, MbdKey.e_form_dft].to_dict()

    return xyz_contents, ground_truth_e_form

def _score_benchmark(
    model_class_name: str,
    relaxed_xyz: str,
    ground_truth: dict,
    wall_time: float,
    n_samples: int,
) -> dict:
    """
    Scores the results of a benchmark run, calculating MAE and F1 score.
    This version includes the essential patch for the matbench-discovery
    UnicodeDecodeError and does not calculate RMSD or CPS.
    """
    import yaml
    original_yaml_load = yaml.load

    # Define a new, robust load function
    def patched_yaml_load(stream, Loader):
        # The stream can be a string, a byte string, or a file object.
        # The error only happens when it's a file object opened with the wrong encoding.
        # We check if the stream has a 'name' attribute, which file objects do.
        if hasattr(stream, 'name'):
            # Re-open the file stream with the correct encoding
            with open(stream.name, 'r', encoding='utf-8') as f:
                return original_yaml_load(f, Loader)
        # If it's not a file object (e.g., a string), load it normally
        return original_yaml_load(stream, Loader)

    yaml.load = patched_yaml_load

    from io import StringIO
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor
    from sklearn.metrics import mean_absolute_error
    import pandas as pd
    from matbench_discovery.metrics.discovery import stable_metrics
    from matbench_discovery.energy import (
        calc_energy_from_e_refs,
        mp_elemental_ref_energies,
    )

    relaxed_atoms_list = read(StringIO(relaxed_xyz), index=":", format="extxyz")

    true_e_form_per_atom = []
    pred_e_form_per_atom = []
    material_ids = []

    for atoms in relaxed_atoms_list:
        mat_id = atoms.info.get("material_id")
        if mat_id and mat_id in ground_truth:
            material_ids.append(mat_id)
            structure = AseAtomsAdaptor.get_structure(atoms)
            composition = structure.composition
            e_form = calc_energy_from_e_refs(
                composition,
                total_energy=atoms.info["final_energy"],
                ref_energies=mp_elemental_ref_energies,
            )
            pred_e_form_per_atom.append(e_form / len(atoms))
            true_e_form_per_atom.append(ground_truth[mat_id])

    if not true_e_form_per_atom:
        print("Warning: No matching ground truth energies found for scoring.")
        mae, f1_score = float("nan"), float("nan")
    else:
        mae = mean_absolute_error(true_e_form_per_atom, pred_e_form_per_atom)
        s_true = pd.Series(true_e_form_per_atom, index=material_ids)
        s_pred = pd.Series(pred_e_form_per_atom, index=material_ids)
        metrics = stable_metrics(
            each_true=s_true, each_pred=s_pred
        )
        f1_score = metrics.get("F1", 0)

    materials_per_second = n_samples / wall_time
    cost_per_second = 0.000164 + 0.0000131
    cost_per_1000_materials = (1000 / materials_per_second) * cost_per_second

    return {
        "model_class": model_class_name,
        "n_samples": n_samples,
        "wall_time_seconds": round(wall_time, 2),
        "materials_per_second": round(materials_per_second, 2),
        "cost_per_1000_materials_usd": round(cost_per_1000_materials, 4),
        "mae_e_form_per_atom": round(mae, 4),
        "f1_score": round(f1_score, 4),
    }

@app.cls(
    image=UNIFIED_BASE_IMAGE,
    volumes={
        SEVENNET_CACHE_PATH: SEVENNET_WEIGHTS_VOLUME,
    },
    gpu=GPU_CHOICE
)
class SEVENNET:
    @modal.enter()
    def initialize_calculator(self):
        """Initialize the SevenNet calculator and model on GPU."""
        from sevenn.calculator import SevenNetCalculator
        from torch_sim.models.sevennet import SevenNetModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modal = 'mpa'

        print("Initializing SevenNetCalculator...")
        # On first run, this will download weights directly into the mounted volume.
        # On subsequent runs, it will find them there.
        self.sevennet_calculator = SevenNetCalculator(
            model="7net-mf-ompa",
            modal=self.modal,
            device=self.device
        )

        self.sevennet_model = SevenNetModel(
            model=self.sevennet_calculator.model,
            modal=self.modal,
            device=self.device
        )

        # Commit the volume to save the weights after the first download.
        print("Committing volume to save weights...")
        SEVENNET_WEIGHTS_VOLUME.commit()
        print("Initialization complete.")

    @modal.method()
    def relax(self, xyz_file_contents: str, relax_params: dict | None = None) -> str:
        """Relax materials using SevenNet model."""
        validated_params, error_msg = validate_relaxation_params(relax_params, frechet_supported=True)
        if error_msg:
            raise ValueError(f"Invalid relaxation parameters: {error_msg}")
        return _perform_batch_relaxation(xyz_file_contents, self.sevennet_model, self.device, validated_params, dtype="float32")
    
    @modal.method()
    def benchmark(self, n_samples: int = 2500) -> dict:
        """Runs a cost and accuracy benchmark for the model."""
        import time
        # 1. Get benchmark data
        benchmark_xyz, ground_truth = _get_benchmark_data(n_samples)

        # 2. Run relaxation and time it
        start_time = time.perf_counter()
        
        validated_params, _ = validate_relaxation_params(None, frechet_supported=True)
        relaxed_xyz = _perform_batch_relaxation(benchmark_xyz, self.sevennet_model, self.device, validated_params)
        
        end_time = time.perf_counter()
        wall_time = end_time - start_time

        # 3. Score and return results
        return _score_benchmark(
            self.__class__.__name__, relaxed_xyz, ground_truth, wall_time, n_samples
        )

with MACE_IMAGE.imports():
    import torch
    import torch_sim as ts
    from ase.io import read, write
    from torch_sim import generate_force_convergence_fn

@app.cls(
    image=MACE_IMAGE,
    volumes={
        WEIGHTS_CACHE_DIR: WEIGHTS_VOLUME,
    },
    gpu=GPU_CHOICE,
    timeout=3600, # 30 minute timeout
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
            model="medium-mpa-0",
            device=self.device,
            enable_cueq=True,
            default_dtype="float64",
            return_raw_model=True,
        )
        self.mace_model = MaceModel(model=raw_model, device=self.device)
        print("MACE model loaded and ready.")
        WEIGHTS_VOLUME.commit()

    @modal.method()
    def relax(self, xyz_file_contents: str, relax_params: dict | None = None) -> str:
        """Relax materials using MACE model."""
        validated_params, error_msg = validate_relaxation_params(relax_params, frechet_supported=True)
        if error_msg:
            raise ValueError(f"Invalid relaxation parameters: {error_msg}")
        return _perform_batch_relaxation(xyz_file_contents, self.mace_model, self.device, validated_params)

    @modal.method()
    def benchmark(self, n_samples: int = 2500) -> dict:
        """Runs a cost and accuracy benchmark for the model."""
        import time

        # 1. Get benchmark data
        benchmark_xyz, ground_truth = _get_benchmark_data(n_samples)

        # 2. Run relaxation and time it
        start_time = time.perf_counter()
        
        validated_params, _ = validate_relaxation_params(None, frechet_supported=True)
        relaxed_xyz = _perform_batch_relaxation(benchmark_xyz, self.mace_model, self.device, validated_params)
        
        end_time = time.perf_counter()
        wall_time = end_time - start_time

        # 3. Score and return results
        return _score_benchmark(
            self.__class__.__name__, relaxed_xyz, ground_truth, wall_time, n_samples
        )

@app.cls(
    image=UNIFIED_BASE_IMAGE,
    # Mount the volume to the MatterSim cache directory
    volumes={
        MATTERSIM_CACHE_PATH: MATTERSIM_WEIGHTS_VOLUME, 
    },
    gpu=GPU_CHOICE,
)
class MATTERSIM:
    @modal.enter()
    def initialize_calculator(self):
        """Initialize the MatterSim model using the torch-sim wrapper."""
        import torch
        from mattersim.forcefield.potential import Potential
        # Import the official torch-sim wrapper
        from torch_sim.models.mattersim import MatterSimModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Initializing MatterSim model (will download if not cached)...")
        # Load the pretrained Potential object. This handles the download.
        potential = Potential.from_checkpoint(
            load_path="mattersim-v1.0.0-5m", 
            device=str(self.device)
        )
        print("MatterSim model loaded.")

        # Pass the loaded potential into the official torch-sim wrapper
        self.mattersim_model = MatterSimModel(model=potential)
        print("torch-sim MatterSimModel wrapper initialized.")

        # Commit the volume to save weights after the first download
        print("Committing volume to save weights...")
        MATTERSIM_WEIGHTS_VOLUME.commit()
        print("Initialization complete.")

    @modal.method()
    def relax(self, xyz_file_contents: str, relax_params: dict | None = None) -> str:
        """Relax materials using the MatterSim model."""
        # MatterSim models use float32 and do not compute stress
        if relax_params is None:
            relax_params = {}
        
        # Set default optimizer_type to fire if not specified, but don't override user's choice
        if "optimizer_type" not in relax_params:
            relax_params = {**relax_params, "optimizer_type": "fire"}
        
        validated_params, error_msg = validate_relaxation_params(relax_params, frechet_supported=False)
        if error_msg:
            raise ValueError(f"Invalid relaxation parameters: {error_msg}")
        
        return _perform_batch_relaxation(
            xyz_file_contents,
            self.mattersim_model,
            self.device,
            validated_params,
            dtype="float32"
        )

    @modal.method()
    def benchmark(self, n_samples: int = 2500) -> dict:
        """Runs a cost and accuracy benchmark for the model."""
        import time
        # 1. Get benchmark data
        benchmark_xyz, ground_truth = _get_benchmark_data(n_samples)

        # 2. Run relaxation and time it
        start_time = time.perf_counter()
        
        validated_params, _ = validate_relaxation_params({"optimizer_type": "fire"}, frechet_supported=False)
        relaxed_xyz = _perform_batch_relaxation(
            benchmark_xyz,
            self.mattersim_model,
            self.device,
            validated_params,
            dtype="float32"
        )
        
        end_time = time.perf_counter()
        wall_time = end_time - start_time

        # 3. Score and return results
        return _score_benchmark(
            self.__class__.__name__, relaxed_xyz, ground_truth, wall_time, n_samples
        )


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
    print(MACE().relax.remote(sample_file_contents, relax_params={"max_steps": 100}))
    print(MATTERSIM().relax.remote(sample_file_contents, relax_params={"max_steps": 100}))
    print(SEVENNET().relax.remote(sample_file_contents, relax_params={"max_steps": 100}))
