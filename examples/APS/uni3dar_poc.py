import modal
from typing import Any, Dict, List, Tuple, Union

def cache_model_weights():
    """Download and cache the Uni-3DAR model weights from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    import os
    
    # Create directory for model weights if it doesn't exist
    os.makedirs("/models", exist_ok=True)
    
    # Download the model weights file
    model_path = hf_hub_download(
        repo_id="dptech/Uni-3DAR",
        filename="mp20_pxrd.pt",
        local_dir="/models"
    )
    
    print(f"Model weights downloaded to: {model_path}")
    return model_path

def setup_uni3dar():
    """Clone the Uni-3DAR repository and set it up"""
    import os
    import subprocess
    import sys
    
    # Clone the Uni-3DAR repository
    subprocess.run(["git", "clone", "https://github.com/dptech-corp/Uni-3DAR.git", "/uni3dar"], check=True)
    
    # Add to Python path
    if "/uni3dar" not in sys.path:
        sys.path.append("/uni3dar")
        
    # Create an empty __init__.py file if it doesn't exist
    if not os.path.exists("/uni3dar/uni3dar/__init__.py"):
        with open("/uni3dar/uni3dar/__init__.py", "w") as f:
            pass
    

aps_image = (
    modal.Image.debian_slim(python_version="3.11")    
    .pip_install(
        "torch==2.6.0", 
        "torchvision", 
        "torchaudio", 
        "pymatgen",
        "numba",
        "huggingface_hub[hf_transfer]"  # install fast Rust download client
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        extra_options="--no-deps"  # Don't install dependencies from PyPI
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
    .apt_install("git", "git-lfs")
    .run_commands("pip install git+https://github.com/dptech-corp/Uni-Core.git")
    .run_function(cache_model_weights)
    .run_function(setup_uni3dar)
    .pip_install("scikit-learn")
)

app = modal.App("aps-garden")

@app.function(image=aps_image, gpu="T4", timeout=1500)
def proof_of_concept(cif_as_string):
    
    # test_cif_path = "./experimental_xrd/wn6225Isup2.rtv.combined.cif"

    # Take the cif_as_string and convert it to a file path that can be read by read_experimental_cif
    # Create a temporary directory
    import tempfile
    import numpy as np
    import os
    
    from pymatgen.core.periodic_table import Element

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp.cif")
        with open(temp_file_path, "w") as f:
            f.write(cif_as_string)
            
        # Validate CIF content
        with open(temp_file_path, "r") as f:
            content = f.read()
            if not content.strip():
                raise ValueError("CIF file is empty")
            if "loop_" not in content:
                raise ValueError("CIF file does not contain any data loops")
            if "_pd_" not in content:
                raise ValueError("CIF file does not contain XRD data (_pd_ fields)")
                
        (
            source_file_name,
            cif_str,
            pretty_formula,
            frac_coords,
            atom_Zs,
            spacegroup_number,
            two_theta_vals,
            q_vals,
            i_vals,
            exp_wavelength,
            exp_2theta_min,
            exp_2theta_max,
        ) = read_experimental_cif(
            filepath=temp_file_path,
        )
    
    model, args = load_model('/models/mp20_pxrd.pt')

    # Normalize intensity array between 0 and 100
    intensity_array = 100 * i_vals / np.max(i_vals)
    intensity_array[intensity_array < 0] = 0.0

    # Filter data to keep only values between 15 and 120 degrees
    mask = (two_theta_vals >= 15) & (two_theta_vals <= 120)
    two_theta_vals = two_theta_vals[mask]
    intensity_array = intensity_array[mask]
    atom_type = list(set([Element.from_Z(z).symbol for z in atom_Zs]))
    inference_data = {
        "pxrd_x": two_theta_vals,
        "pxrd_y": intensity_array,
        "atom_type": atom_type,
    }
    atom_constraint = None  # np.array(atom_Zs) - 1

    model, args = load_model("/models/mp20_pxrd.pt")
    crystals, scores = sample_from_model(
        model, args, inference_data, atom_constraint=atom_constraint, num_samples=3
    )
    print(crystals)
    print(scores)

    return 'success'

DEFAULT_CFG: Dict[str, Any] = {
    # high‑level definition
    # "user_dir": "/uni3dar",
    "task": "uni3dar",
    "task_name": "uni3dar_pxrd",
    "loss": "ar",
    "arch": "uni3dar_sampler",
    # model hyper‑parameters
    "layer": 24,
    "emb_dim": 1024,
    "num_head": 16,
    "merge_level": 8,
    "head_dropout": 0.1,
    # dataloader & batching
    "batch_size": 256,
    "batch_size_valid": 512,
    "num_workers": 8,
    "data_buffer_size": 32,
    "fixed_validation_seed": 11,
    # sampling temperatures / ranking
    "tree_temperature": 0.15,
    "atom_temperature": 0.3,
    "xyz_temperature": 0.3,
    "count_temperature": 1.0,
    "rank_ratio": 0.8,
    "rank_by": "atom+xyz",
    # crystal pxrd specific toggles
    "data_type": "crystal",
    "grid_len": 0.24,
    "xyz_resolution": 0.01,
    "recycle": 1,
    "atom_type_key": "atom_type",
    "atom_pos_key": "atom_pos",
    "lattice_matrix_key": "lattice_matrix",
    "allow_atoms": "all",
    "crystal_pxrd": 4,
    "crystal_pxrd_step": 0.1,
    "crystal_pxrd_noise": 0.1,
    "crystal_component": 1,
    "crystal_component_sqrt": True,
    "crystal_component_noise": 0.1,
    "crystal_pxrd_threshold": 5,
    "max_num_atom": 128,
    # misc
    "seed": 42,
    "bf16": True,
    "gzip": True,
    "ddp_backend": "c10d",
}


def _make_arg_list(cfg: Dict[str, Any]) -> List[str]:
    """Convert a python dict into the argv‑style list expected by unicore."""
    argv: List[str] = ["dummy_data_placeholder"]
    for k, v in cfg.items():
        if k == "data":
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend(map(str, v))
        else:
            argv.extend([flag, str(v)])
    return argv


def load_model(
    checkpoint: str,
    *,
    overrides: Dict[str, Any] | None = None,
):

    import numpy as np
    import torch
    from unicore import tasks, utils, options
    import warnings
    import sys
    import importlib
    
    # Make sure uni3dar is in path and importable
    if "/uni3dar" not in sys.path:
        sys.path.append("/uni3dar")
    
    # Try to manually import and register the task
    try:
        import uni3dar
        print(f"Imported uni3dar successfully. Contents: {dir(uni3dar)}")
        
        # Try directly importing task module to register it
        try:
            import uni3dar.tasks
            print(f"Imported uni3dar.tasks successfully. Contents: {dir(uni3dar.tasks)}")
        except Exception as e:
            print(f"Failed to import uni3dar.tasks: {e}")
    except Exception as e:
        print(f"Failed to import uni3dar module: {e}")
    
    # Print all registered tasks
    print(f"Available tasks before user_dir: {list(tasks.TASK_REGISTRY.keys())}")
    
    cfg = {**DEFAULT_CFG, **(overrides or {}), "finetune_from_model": checkpoint}
    arg_list = _make_arg_list(cfg)
    
    print(f"Using arguments: {arg_list}")
    
    parser = options.get_training_parser()
    
    # Check if task was registered
    print(f"Available tasks after user_dir: {list(tasks.TASK_REGISTRY.keys())}")

    args = options.parse_args_and_arch(parser, input_args=arg_list)  # failing here. --task uni3dar is not accepted

    utils.import_user_module(args)
    utils.set_jit_fusion_options()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    task = tasks.setup_task(args)
    model = task.build_model(args)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["ema"]["params"], strict=False)
    if missing or unexpected:
        warnings.warn(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    model.eval()
    model = model.cuda().bfloat16()
    return model, args


@app.local_entrypoint()
def main():
    # cif_path = "https://raw.githubusercontent.com/gabeguo/cdvae_xrd/main/data/experimental_xrd/cif_files/av5088sup4.rtv.combined.cif"
    cif_path = "https://raw.githubusercontent.com/gabeguo/cdvae_xrd/main/data/experimental_xrd/cif_files/wh5012phaseIIIsup3.rtv.combined.cif"
    import urllib.request
    with urllib.request.urlopen(cif_path) as response:
        cif_str = response.read().decode('utf-8')
    print(proof_of_concept.remote(cif_str))




    # TODO: can I get a copy of this?
    # Pull one of these down:https://github.com/gabeguo/cdvae_xrd/tree/main/data/experimental_xrd/cif_files
    # test_cif_path = "./experimental_xrd/wn6225Isup2.rtv.combined.cif"
    # (
    #     source_file_name,
    #     cif_str,
    #     pretty_formula,
    #     frac_coords,
    #     atom_Zs,
    #     spacegroup_number,
    #     two_theta_vals,
    #     q_vals,
    #     i_vals,
    #     exp_wavelength,
    #     exp_2theta_min,
    #     exp_2theta_max,
    # ) = read_experimental_cif(
    #     filepath=test_cif_path,
    # )

    # # Normalize intensity array between 0 and 100
    # intensity_array = 100 * i_vals / np.max(i_vals)
    # intensity_array[intensity_array < 0] = 0.0

    # # Filter data to keep only values between 15 and 120 degrees
    # mask = (two_theta_vals >= 15) & (two_theta_vals <= 120)
    # two_theta_vals = two_theta_vals[mask]
    # intensity_array = intensity_array[mask]
    # atom_type = list(set([Element.from_Z(z).symbol for z in atom_Zs]))
    # inference_data = {
    #     "pxrd_x": two_theta_vals,
    #     "pxrd_y": intensity_array,
    #     "atom_type": atom_type,
    # }
    # atom_constraint = None  # np.array(atom_Zs) - 1

    # model, args = load_model("mp20_pxrd.pt", device="cuda")
    # crystals, scores = sample_from_model(
    #     model, args, inference_data, atom_constraint=atom_constraint, num_samples=3
    # )
    # print(crystals)
    # print(scores)


# CIF parsing
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import gzip

# from pymatgen.io.cif import CifParser, CifWriter
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pymatgen.analysis.diffraction.xrd import WAVELENGTHS


default_rad_type = "CuKa"
default_min_2theta = 5.0
default_max_2theta = 90.0
possible_2theta_suffixes = [
    "_2theta_corrected",
    "_2theta_scan",
    "_2theta_centroid",
    "_2theta",
]
possible_dspacing_suffixes = [
    "_d_spacing",
]
possible_tof_suffixes = [
    "_time_of_flight",
]
possible_intensity_suffixes = [
    "_intensity_net",
    "_intensity_total",
    "_counts_total",
    "_counts",
    "_pk_height",
    "_intensity",
    "_calc_intensity_total",
    "_calc_intensity_net",
]
possible_bg_suffixes = [
    "_intensity_bkg_calc",
    "_intensity_calc_bkg",
    "_intensity_bkg",
    "_intensity_background",
]


def get_field_value(all_lines, desired_start, is_num=True):
    for i, the_line in enumerate(all_lines):
        if the_line.startswith(desired_start):
            split_line = the_line.split()
            if len(split_line) > 1:
                val = split_line[-1]
                if is_num:
                    try:
                        return float(val)
                    except ValueError:
                        pass
                else:
                    return val
            else:
                ret_val = all_lines[i + 1]
                tokens = ret_val.split()
                for token in tokens:
                    try:
                        return float(token) if is_num else token
                    except ValueError:
                        continue
                if not is_num:
                    return ret_val
                raise ValueError(f"Invalid numeric value for {desired_start}")
    raise ValueError(f"Could not find field '{desired_start}' in CIF lines.")


def find_index_of_xrd_loop(all_lines):
    for i in range(len(all_lines) - 1):
        if all_lines[i].strip() == "loop_" and ("_pd_" in all_lines[i + 1]):
            return i
    raise ValueError("Could not find an XRD data loop (loop_ + _pd_...).")


def find_end_of_xrd(all_lines, start_idx):
    for i in range(start_idx + 1, len(all_lines)):
        line = all_lines[i].strip()
        if (not line) or line.startswith("_") or line.startswith("loop_"):
            return i
    return len(all_lines)


def find_first_by_suffix(field_list, suffix_list):
    lower_field_list = [f.lower() for f in field_list]
    for i, field_name in enumerate(lower_field_list):
        for sfx in suffix_list:
            if field_name.endswith(sfx.lower()):
                return i
    return None


def auto_identify_columns(field_list):
    two_theta_idx = find_first_by_suffix(field_list, possible_2theta_suffixes)
    d_spacing_idx = find_first_by_suffix(field_list, possible_dspacing_suffixes)
    tof_idx = find_first_by_suffix(field_list, possible_tof_suffixes)
    intensity_idx = find_first_by_suffix(field_list, possible_intensity_suffixes)
    bg_idx = find_first_by_suffix(field_list, possible_bg_suffixes)

    return two_theta_idx, d_spacing_idx, tof_idx, intensity_idx, bg_idx


def read_experimental_cif(filepath, plot=False, save_pickle=False, pickle_path=None):
    import os
    import numpy as np
    
    from pymatgen.io.cif import CifParser, CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")

    with open(filepath, "r") as fin:
        all_lines = [x.rstrip("\n") for x in fin]

    xrd_loop_start_idx = find_index_of_xrd_loop(all_lines)

    field_list = []
    line_idx = xrd_loop_start_idx + 1
    while line_idx < len(all_lines):
        line = all_lines[line_idx].strip()
        if not line:
            break
        if line.startswith("_pd_"):
            token = line.split()[0]
            field_list.append(token)
            line_idx += 1
        else:
            break

    data_start_idx = line_idx
    while data_start_idx < len(all_lines) and not all_lines[data_start_idx].strip():
        data_start_idx += 1

    data_end_idx = find_end_of_xrd(all_lines, data_start_idx)
    xrd_lines = all_lines[data_start_idx:data_end_idx]

    (two_theta_idx, d_spacing_idx, tof_idx, intensity_idx, bg_idx) = (
        auto_identify_columns(field_list)
    )

    if intensity_idx is None and (
        two_theta_idx is None and d_spacing_idx is None and tof_idx is None
    ):
        two_theta_idx = 0
        intensity_idx = 1

    try:
        rad_type = get_field_value(
            all_lines, "_diffrn_radiation_type", is_num=False
        ).lower()
        if "neutron" in rad_type:
            raise ValueError(
                "Neutron diffraction data is not supported in this script."
            )
    except ValueError:
        pass

    try:
        exp_wavelength = float(
            get_field_value(all_lines, "_diffrn_radiation_wavelength")
        )
    except ValueError:
        try:
            rad_type_raw = get_field_value(
                all_lines, "_diffrn_radiation_type", is_num=False
            )
        except ValueError:
            rad_type_raw = default_rad_type
        if "Cu" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CuKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CuKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CuKa2"]
            else:
                exp_wavelength = WAVELENGTHS["CuKa"]
        elif "Mo" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["MoKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["MoKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["MoKa2"]
            else:
                exp_wavelength = WAVELENGTHS["MoKa"]
        elif "Cr" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CrKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CrKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CrKa2"]
            else:
                exp_wavelength = WAVELENGTHS["CrKa"]
        elif "Fe" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["FeKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["FeKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["FeKa2"]
            else:
                exp_wavelength = WAVELENGTHS["FeKa"]
        elif "Co" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CoKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CoKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["CoKa2"]
            else:
                exp_wavelength = WAVELENGTHS["CoKa"]
        elif "Ag" in rad_type_raw:
            if "1" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["AgKa1"]
            elif "1" in rad_type_raw and "b" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["AgKb1"]
            elif "2" in rad_type_raw and "a" in rad_type_raw:
                exp_wavelength = WAVELENGTHS["AgKa2"]
            else:
                exp_wavelength = WAVELENGTHS["AgKa"]
        else:
            exp_wavelength = WAVELENGTHS[rad_type_raw]

    if two_theta_idx is not None and len(xrd_lines) > 0:
        try:
            first_val = float(xrd_lines[0].split()[two_theta_idx].split("(")[0])
            last_val = float(xrd_lines[-1].split()[two_theta_idx].split("(")[0])
            exp_2theta_min = min(first_val, last_val)
            exp_2theta_max = max(first_val, last_val)
        except ValueError:
            pass
    else:
        try:
            exp_2theta_min = float(
                get_field_value(all_lines, "_pd_meas_2theta_range_min")
            )
            exp_2theta_max = float(
                get_field_value(all_lines, "_pd_meas_2theta_range_max")
            )
        except ValueError:
            exp_2theta_min = default_min_2theta
            exp_2theta_max = default_max_2theta

    raw_2theta_vals = []
    raw_q_vals = []
    raw_i_vals = []

    fallback_2theta_array = None
    explicit_2theta = two_theta_idx is not None
    if (not explicit_2theta) and (len(xrd_lines) > 1):
        fallback_2theta_array = np.linspace(
            exp_2theta_min, exp_2theta_max, len(xrd_lines)
        )

    for i, line in enumerate(xrd_lines):
        txt = line.strip()
        if not txt:
            continue
        parts = txt.split()
        if len(parts) < 1:
            continue

        if intensity_idx is not None and intensity_idx < len(parts):
            raw_intensity = parts[intensity_idx]
        else:
            raw_intensity = "0.0"

        try:
            intensity_val = float(raw_intensity.split("(")[0])
        except ValueError:
            intensity_val = 0.0

        if bg_idx is not None and bg_idx < len(parts):
            raw_bg = parts[bg_idx]
            try:
                bg_val = float(raw_bg.split("(")[0])
                intensity_val -= bg_val
            except ValueError:
                pass

        if explicit_2theta and two_theta_idx < len(parts):
            try:
                raw_2theta = parts[two_theta_idx]
                two_theta_deg = float(raw_2theta.split("(")[0])
            except ValueError:
                continue
            theta_rad = np.radians(two_theta_deg / 2.0)
            curr_Q = 4.0 * np.pi * np.sin(theta_rad) / exp_wavelength

        elif d_spacing_idx is not None and d_spacing_idx < len(parts):
            try:
                d_val = float(parts[d_spacing_idx].split("(")[0])
            except ValueError:
                continue
            if d_val == 0:
                continue
            curr_Q = 2.0 * np.pi / d_val

            try:
                theta_rad = np.arcsin(exp_wavelength / (2.0 * d_val))
                two_theta_deg = 2.0 * np.degrees(theta_rad)
            except ValueError:
                two_theta_deg = float("nan")

        elif tof_idx is not None and tof_idx < len(parts):
            print(
                "TOF data not supported in this script. Please add your own conversion."
            )
            continue

        else:
            if fallback_2theta_array is not None and i < len(fallback_2theta_array):
                two_theta_deg = fallback_2theta_array[i]
                theta_rad = np.radians(two_theta_deg / 2.0)
                curr_Q = 4.0 * np.pi * np.sin(theta_rad) / exp_wavelength
            else:
                continue

        raw_q_vals.append(curr_Q)
        raw_i_vals.append(intensity_val)
        raw_2theta_vals.append(two_theta_deg)

    if len(raw_q_vals) == 0:
        raise ValueError("No valid Q-intensity data could be parsed from the CIF.")

    raw_q_vals = np.array(raw_q_vals, dtype=float)
    raw_i_vals = np.array(raw_i_vals, dtype=float)
    raw_2theta_vals = np.array(raw_2theta_vals, dtype=float)
    sort_idx = np.argsort(raw_q_vals)
    q_vals = raw_q_vals[sort_idx]
    i_vals = raw_i_vals[sort_idx]
    two_theta_vals = raw_2theta_vals[sort_idx]

    cif_parser = CifParser(filepath)
    structure = cif_parser.get_structures()[0]
    cif_writer = CifWriter(structure)
    cif_str = cif_writer.__str__()

    source_file_name = os.path.basename(filepath)
    pretty_formula = structure.composition.reduced_formula
    frac_coords = structure.frac_coords
    atom_types = structure.atomic_numbers
    sga = SpacegroupAnalyzer(structure)
    spacegroup_number = sga.get_space_group_number()


    result_tuple = (
        source_file_name,
        cif_str,
        pretty_formula,
        frac_coords,
        atom_types,
        spacegroup_number,
        two_theta_vals,
        q_vals,
        i_vals,
        exp_wavelength,
        exp_2theta_min,
        exp_2theta_max,
    )

    return result_tuple


# We need to install unicore from GH
# https://github.com/dptech-corp/Uni-Core

# PXRD Inference script for Uni‑3DAR (https://github.com/dptech-corp/Uni-3DAR)
# Checkpoint: https://huggingface.co/dptech/Uni-3DAR/tree/main
# Example experimental PXRD cif file: https://github.com/gabeguo/cdvae_xrd/tree/main/data/experimental_xrd
# 1. Clone the Uni-3DAR repo, create a conda environment using the uni3dar_env.yml file
# 2. Obtain the ckpt.pt file place it in the top level of the Uni-3DAR repo
# 3. Download the experimental PXRD cif file or use your own
# 4. Copy the uni3dar_inference.py into the top level of the Uni-3DAR repo
# 5. Run the script
# Inspired by https://github.com/dptech-corp/Uni-3DAR/blob/main/uni3dar/inference.py
# Curated by: Xiangyu Yin (xiangyu-yin.com)

# from __future__ import annotations
# import warnings


# import numpy as np
# import torch
# from unicore import tasks, utils, options
# from pymatgen.core.periodic_table import Element
# from parse_cifs import read_experimental_cif


def sample_from_model(
    model,
    args,
    cur_data: Dict[str, Any],
    *,
    num_samples: int = 1,
    atom_constraint = None,
) -> Tuple[List, List[float]]:
    """
    Parameters
    ----------
    model : torch.nn.Module
        The sampler returned by `load_model`.
    args : argparse.Namespace
        Same object returned by `load_model`; only used to know the
        key‑names (`atom_type_key`, `lattice_matrix_key`, …) if you need them.
    cur_data : dict
        A single datapoint formatted exactly as Uni‑3DAR expects
        (same fields that used to be read from LMDB).
    num_samples : int, default 1
        How many structures to return.
    atom_constraint : np.ndarray, optional
        List of Z numbers - 1. e.g., [25,25,25,12,12,7,7,7,7,7] for Fe3Al2O5
        Leave `None` for unconstrained sampling.

    Returns
    -------
    crystals : list[ase.Atoms]
        List of generated structures (length == `num_samples`).
    scores   : list[float]
        The internal model score for each returned structure.
    """
    if atom_constraint is not None:
        assert atom_constraint.ndim == 1, "Provide a flat 1D array."

    crystals, scores = [], []
    while len(crystals) < num_samples:
        c, s = model.generate(data=cur_data, atom_constraint=atom_constraint)
        print(c, s)
        print("NUMBER OF CRYSTALS RETURNED IS ", len(c))
        crystals.extend(c)
        scores.extend(s)
    crystals, scores = crystals[:num_samples], scores[:num_samples]

    return crystals, scores
    