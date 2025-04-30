# modal.Image.debian_slim(python_version="3.11") # NB: we might want to specify ubuntu here
# .apt_install("git", "git-lfs")
# "pytorch-cuda==12.1", 

# Can I clone the git repo to /uni3dar?

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

# def setup_uni3dar():
#     """Clone the Uni-3DAR repository and set it up"""
#     import os
#     import subprocess
    
#     # Clone the Uni-3DAR repository
#     subprocess.run(["git", "clone", "https://github.com/dptech-corp/Uni-3DAR.git", "/uni3dar"], check=True)
    
#     # Create symbolic link to make it accessible in the python path
#     if not os.path.exists("/uni3dar/uni3dar"):
#         os.symlink("/uni3dar/uni3dar", "/uni3dar/uni3dar")

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

# Following Modal's recommendation for flash-attn
cuda_version = "12.1.1"  # Should be compatible with Modal's GPUs
flavor = "devel"         # Includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# aps_image = (
#     modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
#     .apt_install("git", "git-lfs")
#     # Install dependencies required for building flash-attn
#     .pip_install(
#         "ninja",
#         "packaging",
#         "wheel",
#         "setuptools>=42.0.0"
#     )
#     # Install PyTorch and related packages
#     .pip_install(
#         "torch>=2.0.0",
#         "torchvision",
#         "torchaudio"
#     )
#     # Now install flash-attn
#     .pip_install(
#         "flash-attn==2.7.4.post1", 
#         extra_options="--no-build-isolation"
#     )
#     # Install minimal required dependencies
#     .pip_install(
#         "huggingface_hub[hf_transfer]",  # For downloading models
#         "pymatgen",                      # For crystal structure handling
#         "ase",                           # For atomic simulation
#         "numba",                         # For code acceleration
#         "einops>=0.6.0",                 # Required by flash-attn
#         "lmdb"                           # For database operations
#     )
#     # Install Uni-Core
#     .run_commands("pip install git+https://github.com/dptech-corp/Uni-Core.git")
#     .run_function(setup_uni3dar)
#     .run_function(cache_model_weights)
# )

# app = modal.App("aps-garden")

def detect_env():
    # Get Python version
    import os
    import sys
    import subprocess
    import platform
    import requests
    from packaging import version
    import json
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    
    # Get PyTorch version
    import torch
    torch_version = torch.__version__.split('+')[0]  # Remove any CUDA suffix
    torch_major_minor = '.'.join(torch_version.split('.')[:2])  # e.g., "2.2"
    print(f"PyTorch version: {torch_version} (major.minor: {torch_major_minor})")
    
    # Get CUDA version
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cuda_major_minor = cuda_version.split('.')[:2]  # e.g., ["12", "1"]
        cuda_major = cuda_major_minor[0]  # e.g., "12"
        print(f"CUDA version: {cuda_version} (major: {cuda_major})")
    else:
        print("CUDA not available")
        return False
    
    # System platform
    system_platform = "linux_x86_64" if platform.system() == "Linux" else "unknown"
    print(f"System platform: {system_platform}")
    

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
    # .run_function(detect_env)
)



app = modal.App("aps-garden")

@app.function(image=aps_image, gpu="T4")
def proof_of_concept():
    import os
    import sys
    import importlib
    import unicore
    from typing import Dict, Any, List
    
    # Add to Python path again (in case it wasn't preserved)
    if "/uni3dar" not in sys.path:
        sys.path.append("/uni3dar")
    
    print(f"Unicore version: {unicore.__version__}")
    print(f"Files in /uni3dar: {os.listdir('/uni3dar')}")
    print(f"Files in /uni3dar/uni3dar: {os.listdir('/uni3dar/uni3dar')}")
    
    # Directly import the task module to register it
    try:
        import uni3dar
        print("Successfully imported uni3dar package")
    except Exception as e:
        print(f"Error importing uni3dar: {e}")
    
    # Let's make sure the module is properly loaded and print its contents
    try:
        import uni3dar.tasks
        print("Successfully imported uni3dar.tasks")
        print(f"Contents of uni3dar.tasks: {dir(uni3dar.tasks)}")
    except Exception as e:
        print(f"Error importing uni3dar.tasks: {e}")
    
    # Now try to load the model
    try:
        model, args = load_model('/models/mp20_pxrd.pt')#, overrides={
            # "user_dir": "/uni3dar",
            # Try using a different task if uni3dar isn't available
            #"task": "translation" if "uni3dar" not in unicore.tasks.TASK_REGISTRY else "uni3dar"
        #})
        print("Model loaded successfully!")
        return "Success"
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Available tasks: {list(unicore.tasks.TASK_REGISTRY.keys())}")
        return f"Failed: {str(e)}"

    # import unicore
    # import os
    # print(f"Unicore version: {unicore.__version__}")
    # print(f"Files in /uni3dar: {os.listdir('/uni3dar')}")
    
    # # Critical change: Point to the correct user directory containing the custom tasks
    # model, args = load_model('/models/mp20_pxrd.pt', overrides={"user_dir": "/uni3dar"})
    # print("Model loaded successfully!")
    # return "Success"

@app.local_entrypoint()
def main():
    print(proof_of_concept.remote())

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
    # import numpy as np
    # import torch
    # from unicore import tasks, utils, options
    # import warnings


    # cfg = {**DEFAULT_CFG, **(overrides or {}), "finetune_from_model": checkpoint}
    # arg_list = _make_arg_list(cfg)

    # parser = options.get_training_parser()

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
    
    # Import user module before parsing arguments
    # usr_parser = options._get_parser_with_user_dir()
    # usr_args, usr_unknown = usr_parser.parse_known_args(arg_list)
    # utils.import_user_module(usr_args)
    
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
    # if str(device) == "cuda" and torch.cuda.is_available():
    model = model.cuda().bfloat16()
    return model, args

#         checkpoint: str,
#     *,
#     device: Union[str, torch.device] = "cuda",
#     overrides: Dict[str, Any] | None = None,
# ):
    
    # How do I specify the path correctly here?
    # cfg = {**DEFAULT_CFG, **(overrides or {}), "finetune_from_model": "mp20_pxrd.pt"}
    # arg_list = _make_arg_list(cfg)

    # parser = options.get_training_parser()
    # args = options.parse_args_and_arch(parser, input_args=arg_list)

    # utils.import_user_module(args)
    # utils.set_jit_fusion_options()
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    # task = tasks.setup_task(args)
    # model = task.build_model(args)

    # state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # missing, unexpected = model.load_state_dict(state["ema"]["params"], strict=False)
    # if missing or unexpected:
    #     warnings.warn(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")

    # model.eval()
    # if str(device) == "cuda" and torch.cuda.is_available():
    #     model = model.cuda().bfloat16()
    # else:
    #     model = model.float()

    # return model, args
    # return "basic build done"
    # Can I load the model into GPU memory?






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



# Env yml

# name: uni3dar
# channels:
#   - defaults
#   - conda-forge
#   - pytorch
#   - nvidia
# dependencies:
#   - _libgcc_mutex=0.1=main
#   - _openmp_mutex=5.1=1_gnu
#   - bzip2=1.0.8=h5eee18b_6
#   - ca-certificates=2025.2.25=h06a4308_0
#   - ld_impl_linux-64=2.40=h12ee557_0
#   - libffi=3.4.4=h6a678d5_1
#   - libgcc-ng=11.2.0=h1234567_1
#   - libgomp=11.2.0=h1234567_1
#   - libstdcxx-ng=11.2.0=h1234567_1
#   - libuuid=1.41.5=h5eee18b_0
#   - ncurses=6.4=h6a678d5_0
#   - openssl=3.0.16=h5eee18b_0
#   - pip=25.0=py311h06a4308_0
#   - python=3.11.11=he870216_0
#   - readline=8.2=h5eee18b_0
#   - setuptools=75.8.0=py311h06a4308_0
#   - sqlite=3.45.3=h5eee18b_0
#   - tk=8.6.14=h39e8969_0
#   - wheel=0.45.1=py311h06a4308_0
#   - xz=5.6.4=h5eee18b_1
#   - zlib=1.2.13=h5eee18b_1
#   - pip:
#       - absl-py==2.2.2
#       - annotated-types==0.7.0
#       - ase==3.25.0
#       - certifi==2025.1.31
#       - charset-normalizer==3.4.1
#       - click==8.1.8
#       - contourpy==1.3.2
#       - cycler==0.12.1
#       - dill==0.4.0
#       - docker-pycreds==0.4.0
#       - einops==0.8.1
#       - filelock==3.18.0
#       - flash-attn==2.7.4.post1
#       - fonttools==4.57.0
#       - fsspec==2025.3.2
#       - gitdb==4.0.12
#       - gitpython==3.1.44
#       - huggingface-hub==0.30.2
#       - idna==3.10
#       - jinja2==3.1.6
#       - joblib==1.4.2
#       - kiwisolver==1.4.8
#       - latexcodec==3.0.0
#       - llvmlite==0.44.0
#       - lmdb==1.6.2
#       - markupsafe==3.0.2
#       - matplotlib==3.10.1
#       - ml-collections==1.0.0
#       - monty==2025.3.3
#       - mpmath==1.3.0
#       - multiprocess==0.70.17
#       - narwhals==1.35.0
#       - networkx==3.4.2
#       - numba==0.61.2
#       - numpy==2.2.4
#       - nvidia-cublas-cu12==12.4.5.8
#       - nvidia-cuda-cupti-cu12==12.4.127
#       - nvidia-cuda-nvrtc-cu12==12.4.127
#       - nvidia-cuda-runtime-cu12==12.4.127
#       - nvidia-cudnn-cu12==9.1.0.70
#       - nvidia-cufft-cu12==11.2.1.3
#       - nvidia-curand-cu12==10.3.5.147
#       - nvidia-cusolver-cu12==11.6.1.9
#       - nvidia-cusparse-cu12==12.3.1.170
#       - nvidia-cusparselt-cu12==0.6.2
#       - nvidia-nccl-cu12==2.21.5
#       - nvidia-nvjitlink-cu12==12.4.127
#       - nvidia-nvtx-cu12==12.4.127
#       - packaging==24.2
#       - palettable==3.3.3
#       - pandarallel==1.6.5
#       - pandas==2.2.3
#       - pathos==0.3.3
#       - pillow==11.2.1
#       - platformdirs==4.3.7
#       - plotly==6.0.1
#       - pox==0.3.6
#       - ppft==1.7.7
#       - protobuf==5.29.4
#       - psutil==7.0.0
#       - pybtex==0.24.0
#       - pydantic==2.11.3
#       - pydantic-core==2.33.1
#       - pymatgen==2025.4.16
#       - pyparsing==3.2.3
#       - python-dateutil==2.9.0.post0
#       - pytz==2025.2
#       - pyyaml==6.0.2
#       - requests==2.32.3
#       - ruamel-yaml==0.18.10
#       - ruamel-yaml-clib==0.2.12
#       - scikit-learn==1.6.1
#       - scipy==1.15.2
#       - sentry-sdk==2.26.1
#       - setproctitle==1.3.5
#       - six==1.17.0
#       - smact==3.1.0
#       - smmap==5.0.2
#       - spglib==2.6.0
#       - sympy==1.13.1
#       - tabulate==0.9.0
#       - tensorboardx==2.6.2.2
#       - threadpoolctl==3.6.0
#       - tokenizers==0.21.1
#       - torch==2.6.0
#       - tqdm==4.67.1
#       - triton==3.2.0
#       - typing-extensions==4.13.2
#       - typing-inspection==0.4.0
#       - tzdata==2025.2
#       - uncertainties==3.2.2
#       - unicore==0.0.1
#       - urllib3==2.4.0
#       - wandb==0.19.9

# # CIF parsing
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import gzip

# from pymatgen.io.cif import CifParser, CifWriter
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pymatgen.analysis.diffraction.xrd import WAVELENGTHS


# default_rad_type = "CuKa"
# default_min_2theta = 5.0
# default_max_2theta = 90.0
# possible_2theta_suffixes = [
#     "_2theta_corrected",
#     "_2theta_scan",
#     "_2theta_centroid",
#     "_2theta",
# ]
# possible_dspacing_suffixes = [
#     "_d_spacing",
# ]
# possible_tof_suffixes = [
#     "_time_of_flight",
# ]
# possible_intensity_suffixes = [
#     "_intensity_net",
#     "_intensity_total",
#     "_counts_total",
#     "_counts",
#     "_pk_height",
#     "_intensity",
#     "_calc_intensity_total",
#     "_calc_intensity_net",
# ]
# possible_bg_suffixes = [
#     "_intensity_bkg_calc",
#     "_intensity_calc_bkg",
#     "_intensity_bkg",
#     "_intensity_background",
# ]


# def get_field_value(all_lines, desired_start, is_num=True):
#     for i, the_line in enumerate(all_lines):
#         if the_line.startswith(desired_start):
#             split_line = the_line.split()
#             if len(split_line) > 1:
#                 val = split_line[-1]
#                 if is_num:
#                     try:
#                         return float(val)
#                     except ValueError:
#                         pass
#                 else:
#                     return val
#             else:
#                 ret_val = all_lines[i + 1]
#                 tokens = ret_val.split()
#                 for token in tokens:
#                     try:
#                         return float(token) if is_num else token
#                     except ValueError:
#                         continue
#                 if not is_num:
#                     return ret_val
#                 raise ValueError(f"Invalid numeric value for {desired_start}")
#     raise ValueError(f"Could not find field '{desired_start}' in CIF lines.")


# def find_index_of_xrd_loop(all_lines):
#     for i in range(len(all_lines) - 1):
#         if all_lines[i].strip() == "loop_" and ("_pd_" in all_lines[i + 1]):
#             return i
#     raise ValueError("Could not find an XRD data loop (loop_ + _pd_...).")


# def find_end_of_xrd(all_lines, start_idx):
#     for i in range(start_idx + 1, len(all_lines)):
#         line = all_lines[i].strip()
#         if (not line) or line.startswith("_") or line.startswith("loop_"):
#             return i
#     return len(all_lines)


# def find_first_by_suffix(field_list, suffix_list):
#     lower_field_list = [f.lower() for f in field_list]
#     for i, field_name in enumerate(lower_field_list):
#         for sfx in suffix_list:
#             if field_name.endswith(sfx.lower()):
#                 return i
#     return None


# def auto_identify_columns(field_list):
#     two_theta_idx = find_first_by_suffix(field_list, possible_2theta_suffixes)
#     d_spacing_idx = find_first_by_suffix(field_list, possible_dspacing_suffixes)
#     tof_idx = find_first_by_suffix(field_list, possible_tof_suffixes)
#     intensity_idx = find_first_by_suffix(field_list, possible_intensity_suffixes)
#     bg_idx = find_first_by_suffix(field_list, possible_bg_suffixes)

#     return two_theta_idx, d_spacing_idx, tof_idx, intensity_idx, bg_idx


# def read_experimental_cif(filepath, plot=False, save_pickle=False, pickle_path=None):

#     if not os.path.isfile(filepath):
#         raise FileNotFoundError(f"Could not find file: {filepath}")

#     with open(filepath, "r") as fin:
#         all_lines = [x.rstrip("\n") for x in fin]

#     xrd_loop_start_idx = find_index_of_xrd_loop(all_lines)

#     field_list = []
#     line_idx = xrd_loop_start_idx + 1
#     while line_idx < len(all_lines):
#         line = all_lines[line_idx].strip()
#         if not line:
#             break
#         if line.startswith("_pd_"):
#             token = line.split()[0]
#             field_list.append(token)
#             line_idx += 1
#         else:
#             break

#     data_start_idx = line_idx
#     while data_start_idx < len(all_lines) and not all_lines[data_start_idx].strip():
#         data_start_idx += 1

#     data_end_idx = find_end_of_xrd(all_lines, data_start_idx)
#     xrd_lines = all_lines[data_start_idx:data_end_idx]

#     (two_theta_idx, d_spacing_idx, tof_idx, intensity_idx, bg_idx) = (
#         auto_identify_columns(field_list)
#     )

#     if intensity_idx is None and (
#         two_theta_idx is None and d_spacing_idx is None and tof_idx is None
#     ):
#         two_theta_idx = 0
#         intensity_idx = 1

#     try:
#         rad_type = get_field_value(
#             all_lines, "_diffrn_radiation_type", is_num=False
#         ).lower()
#         if "neutron" in rad_type:
#             raise ValueError(
#                 "Neutron diffraction data is not supported in this script."
#             )
#     except ValueError:
#         pass

#     try:
#         exp_wavelength = float(
#             get_field_value(all_lines, "_diffrn_radiation_wavelength")
#         )
#     except ValueError:
#         try:
#             rad_type_raw = get_field_value(
#                 all_lines, "_diffrn_radiation_type", is_num=False
#             )
#         except ValueError:
#             rad_type_raw = default_rad_type
#         if "Cu" in rad_type_raw:
#             if "1" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CuKa1"]
#             elif "1" in rad_type_raw and "b" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CuKb1"]
#             elif "2" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CuKa2"]
#             else:
#                 exp_wavelength = WAVELENGTHS["CuKa"]
#         elif "Mo" in rad_type_raw:
#             if "1" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["MoKa1"]
#             elif "1" in rad_type_raw and "b" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["MoKb1"]
#             elif "2" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["MoKa2"]
#             else:
#                 exp_wavelength = WAVELENGTHS["MoKa"]
#         elif "Cr" in rad_type_raw:
#             if "1" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CrKa1"]
#             elif "1" in rad_type_raw and "b" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CrKb1"]
#             elif "2" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CrKa2"]
#             else:
#                 exp_wavelength = WAVELENGTHS["CrKa"]
#         elif "Fe" in rad_type_raw:
#             if "1" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["FeKa1"]
#             elif "1" in rad_type_raw and "b" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["FeKb1"]
#             elif "2" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["FeKa2"]
#             else:
#                 exp_wavelength = WAVELENGTHS["FeKa"]
#         elif "Co" in rad_type_raw:
#             if "1" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CoKa1"]
#             elif "1" in rad_type_raw and "b" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CoKb1"]
#             elif "2" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["CoKa2"]
#             else:
#                 exp_wavelength = WAVELENGTHS["CoKa"]
#         elif "Ag" in rad_type_raw:
#             if "1" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["AgKa1"]
#             elif "1" in rad_type_raw and "b" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["AgKb1"]
#             elif "2" in rad_type_raw and "a" in rad_type_raw:
#                 exp_wavelength = WAVELENGTHS["AgKa2"]
#             else:
#                 exp_wavelength = WAVELENGTHS["AgKa"]
#         else:
#             exp_wavelength = WAVELENGTHS[rad_type_raw]

#     if two_theta_idx is not None and len(xrd_lines) > 0:
#         try:
#             first_val = float(xrd_lines[0].split()[two_theta_idx].split("(")[0])
#             last_val = float(xrd_lines[-1].split()[two_theta_idx].split("(")[0])
#             exp_2theta_min = min(first_val, last_val)
#             exp_2theta_max = max(first_val, last_val)
#         except ValueError:
#             pass
#     else:
#         try:
#             exp_2theta_min = float(
#                 get_field_value(all_lines, "_pd_meas_2theta_range_min")
#             )
#             exp_2theta_max = float(
#                 get_field_value(all_lines, "_pd_meas_2theta_range_max")
#             )
#         except ValueError:
#             exp_2theta_min = default_min_2theta
#             exp_2theta_max = default_max_2theta

#     raw_2theta_vals = []
#     raw_q_vals = []
#     raw_i_vals = []

#     fallback_2theta_array = None
#     explicit_2theta = two_theta_idx is not None
#     if (not explicit_2theta) and (len(xrd_lines) > 1):
#         fallback_2theta_array = np.linspace(
#             exp_2theta_min, exp_2theta_max, len(xrd_lines)
#         )

#     for i, line in enumerate(xrd_lines):
#         txt = line.strip()
#         if not txt:
#             continue
#         parts = txt.split()
#         if len(parts) < 1:
#             continue

#         if intensity_idx is not None and intensity_idx < len(parts):
#             raw_intensity = parts[intensity_idx]
#         else:
#             raw_intensity = "0.0"

#         try:
#             intensity_val = float(raw_intensity.split("(")[0])
#         except ValueError:
#             intensity_val = 0.0

#         if bg_idx is not None and bg_idx < len(parts):
#             raw_bg = parts[bg_idx]
#             try:
#                 bg_val = float(raw_bg.split("(")[0])
#                 intensity_val -= bg_val
#             except ValueError:
#                 pass

#         if explicit_2theta and two_theta_idx < len(parts):
#             try:
#                 raw_2theta = parts[two_theta_idx]
#                 two_theta_deg = float(raw_2theta.split("(")[0])
#             except ValueError:
#                 continue
#             theta_rad = np.radians(two_theta_deg / 2.0)
#             curr_Q = 4.0 * np.pi * np.sin(theta_rad) / exp_wavelength

#         elif d_spacing_idx is not None and d_spacing_idx < len(parts):
#             try:
#                 d_val = float(parts[d_spacing_idx].split("(")[0])
#             except ValueError:
#                 continue
#             if d_val == 0:
#                 continue
#             curr_Q = 2.0 * np.pi / d_val

#             try:
#                 theta_rad = np.arcsin(exp_wavelength / (2.0 * d_val))
#                 two_theta_deg = 2.0 * np.degrees(theta_rad)
#             except ValueError:
#                 two_theta_deg = float("nan")

#         elif tof_idx is not None and tof_idx < len(parts):
#             print(
#                 "TOF data not supported in this script. Please add your own conversion."
#             )
#             continue

#         else:
#             if fallback_2theta_array is not None and i < len(fallback_2theta_array):
#                 two_theta_deg = fallback_2theta_array[i]
#                 theta_rad = np.radians(two_theta_deg / 2.0)
#                 curr_Q = 4.0 * np.pi * np.sin(theta_rad) / exp_wavelength
#             else:
#                 continue

#         raw_q_vals.append(curr_Q)
#         raw_i_vals.append(intensity_val)
#         raw_2theta_vals.append(two_theta_deg)

#     if len(raw_q_vals) == 0:
#         raise ValueError("No valid Q-intensity data could be parsed from the CIF.")

#     raw_q_vals = np.array(raw_q_vals, dtype=float)
#     raw_i_vals = np.array(raw_i_vals, dtype=float)
#     raw_2theta_vals = np.array(raw_2theta_vals, dtype=float)
#     sort_idx = np.argsort(raw_q_vals)
#     q_vals = raw_q_vals[sort_idx]
#     i_vals = raw_i_vals[sort_idx]
#     two_theta_vals = raw_2theta_vals[sort_idx]

#     cif_parser = CifParser(filepath)
#     structure = cif_parser.get_structures()[0]
#     cif_writer = CifWriter(structure)
#     cif_str = cif_writer.__str__()

#     source_file_name = os.path.basename(filepath)
#     pretty_formula = structure.composition.reduced_formula
#     frac_coords = structure.frac_coords
#     atom_types = structure.atomic_numbers
#     sga = SpacegroupAnalyzer(structure)
#     spacegroup_number = sga.get_space_group_number()

#     if plot:
#         plt.figure(figsize=(12, 6))
#         plt.plot(q_vals, i_vals, color="green", label="Experimental XRD")
#         plt.title(f"PXRD of {pretty_formula} from {source_file_name}")
#         plt.xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)")
#         plt.ylabel("Experimental Intensity")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     result_tuple = (
#         source_file_name,
#         cif_str,
#         pretty_formula,
#         frac_coords,
#         atom_types,
#         spacegroup_number,
#         two_theta_vals,
#         q_vals,
#         i_vals,
#         exp_wavelength,
#         exp_2theta_min,
#         exp_2theta_max,
#     )

#     if save_pickle:
#         if pickle_path is None:
#             pickle_path = filepath.replace(".cif", ".pkl.gz")
#         with gzip.open(pickle_path, "wb") as f:
#             pickle.dump(result_tuple, f)
#         print(f"[INFO] Saved results to compressed pickle: {pickle_path}")

#     return result_tuple


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




# def sample_from_model(
#     model,
#     args,
#     cur_data: Dict[str, Any],
#     *,
#     num_samples: int = 1,
#     atom_constraint: np.ndarray | None = None,
# ) -> Tuple[List["ase.Atoms"], List[float]]:
#     """
#     Parameters
#     ----------
#     model : torch.nn.Module
#         The sampler returned by `load_model`.
#     args : argparse.Namespace
#         Same object returned by `load_model`; only used to know the
#         key‑names (`atom_type_key`, `lattice_matrix_key`, …) if you need them.
#     cur_data : dict
#         A single datapoint formatted exactly as Uni‑3DAR expects
#         (same fields that used to be read from LMDB).
#     num_samples : int, default 1
#         How many structures to return.
#     atom_constraint : np.ndarray, optional
#         List of Z numbers - 1. e.g., [25,25,25,12,12,7,7,7,7,7] for Fe3Al2O5
#         Leave `None` for unconstrained sampling.

#     Returns
#     -------
#     crystals : list[ase.Atoms]
#         List of generated structures (length == `num_samples`).
#     scores   : list[float]
#         The internal model score for each returned structure.
#     """
#     if atom_constraint is not None:
#         assert atom_constraint.ndim == 1, "Provide a flat 1D array."

#     crystals, scores = [], []
#     while len(crystals) < num_samples:
#         c, s = model.generate(data=cur_data, atom_constraint=atom_constraint)
#         crystals.extend(c)
#         scores.extend(s)
#     crystals, scores = crystals[:num_samples], scores[:num_samples]

#     return crystals, scores


# if __name__ == "__main__":
    