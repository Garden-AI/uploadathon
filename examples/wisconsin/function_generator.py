#!/usr/bin/env python3
"""
Simplified Modal Function Generator

This script contains both function specifications and code generation logic.
Run this script to generate the modal_functions.py file.
"""

from textwrap import dedent

# Define your function specifications here
function_specs = [
    {
        "function_name": "predict_bandgap",
        "module_name": "predict_bandgap",
        "weights_dir_name": "model_bandgap-main",
        "expected_dir_name": "model_bandgap",
        "description": "Random forest model to predict the electronic bandgap of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_concrete",
        "module_name": "predict_concrete",
        "weights_dir_name": "model_concrete",
        "expected_dir_name": "model_concrete",
        "description": "Random forest model to predict the compressive strength of concrete mixtures",
        "params": ["concrete_df"],
    },
    {
        "function_name": "predict_piezoelectric",
        "module_name": "predict_piezoelectric",
        "weights_dir_name": "model_piezoelectric-main",
        "expected_dir_name": "model_piezoelectric",
        "description": "Random forest model to predict max piezoelectric displacement of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_dielectric",
        "module_name": "predict_dielectric",
        "weights_dir_name": "model_dielectric",
        "expected_dir_name": "model_dielectric",
        "description": "Random forest model to predict the dielectric constant of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_diffusion",
        "module_name": "predict_diffusion",
        "weights_dir_name": "model_diffusion",
        "expected_dir_name": "model_diffusion",
        "description": "Random forest model to predict the dilute solute activation energy for diffusion",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_double_perovskite_gap",
        "module_name": "predict_double_perovskite_gap",
        "weights_dir_name": "model_double_perovskite_gap",
        "expected_dir_name": "model_double_perovskite_gap",
        "description": "Random forest model to predict the bandgap of double perovskites",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_elastic_tensor",
        "module_name": "predict_elastic_tensor",
        "weights_dir_name": "model_elastic_tensor",
        "expected_dir_name": "model_elastic_tensor",
        "description": "Random forest model to predict the bulk modulus of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_exfoliationE",
        "module_name": "predict_exfoliationE",
        "weights_dir_name": "model_exfoliationE",
        "expected_dir_name": "model_exfoliationE",
        "description": "Random forest model to predict the exfoliation energy of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_hea_hardness",
        "module_name": "predict_hea_hardness",
        "weights_dir_name": "model_hea_hardness",
        "expected_dir_name": "model_hea_hardness",
        "description": "Random forest model to predict the hardness of high entropy alloys",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_heusler",
        "module_name": "predict_heusler",
        "weights_dir_name": "model_heusler",
        "expected_dir_name": "model_heusler",
        "description": "Random forest model to predict the magnetization of Heusler alloys",
        "params": ["composition_list", "numelec_list", "heuslertype_list"],
    },
    {
        "function_name": "predict_Li_conductivity",
        "module_name": "predict_Li_conductivity",
        "weights_dir_name": "model_Li_conductivity",
        "expected_dir_name": "model_Li_conductivity",
        "description": "Random forest model to predict the conductivity of Li in solid electrolytes",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_metallicglass_Dmax",
        "module_name": "predict_metallicglass_Dmax",
        "weights_dir_name": "model_metallicglass_Dmax",
        "expected_dir_name": "model_metallicglass_Dmax",
        "description": "Random forest model to predict the maximum casting diameter of metallic glasses",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_metallicglass_Rc",
        "module_name": "predict_metallicglass_Rc",
        "weights_dir_name": "model_metallicglass_Rc",
        "expected_dir_name": "model_metallicglass_Rc",
        "description": "Random forest model to predict the critical cooling rate of metallic glasses",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_metallicglass_Rc_LLM",
        "module_name": "predict_metallicglass_Rc_LLM",
        "weights_dir_name": "model_metallicglass_Rc_LLM",
        "expected_dir_name": "model_metallicglass_Rc_LLM",
        "description": "Random forest model to predict the critical cooling rate of metallic glasses (data from LLMs)",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_Mg_alloy",
        "module_name": "predict_Mg_alloy",
        "weights_dir_name": "model_Mg_alloy-main",
        "expected_dir_name": "model_Mg_alloy",
        "description": "Random forest model to predict the yield strength of Mg alloys",
        "params": ["mg_alloy_df"],
    },
    {
        "function_name": "predict_thermal_conductivity",
        "module_name": "predict_thermal_conductivity",
        "weights_dir_name": "model_thermal_conductivity",
        "expected_dir_name": "model_thermal_conductivity",
        "description": "Random forest model to predict the thermal conductivity of materials",
        "params": ["composition_list", "temperature_list"],
    },
    {
        "function_name": "predict_superconductivity",
        "module_name": "predict_superconductivity",
        "weights_dir_name": "model_superconductivity",
        "expected_dir_name": "model_superconductivity",
        "description": "Random forest model to predict the superconducting critical temperature of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_steel_yield",
        "module_name": "predict_steel_yield",
        "weights_dir_name": "model_steel_yield",
        "expected_dir_name": "model_steel_yield",
        "description": "Random forest model to predict the yield strength of steel alloys",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_semiconductor_lvls",
        "module_name": "predict_semiconductor_lvls",
        "weights_dir_name": "model_semiconductor_lvls",
        "expected_dir_name": "model_semiconductor_lvls",
        "description": "Random forest model to predict the defect charge state transition levels in semiconductors",
        "params": ["composition_list", "chargefrom_list", "chargeto_list", "lattice_list", "bandgap_list", "is_b_latt_list"],
    },
    {
        "function_name": "predict_perovskite_ASR",
        "module_name": "predict_perovskite_ASR",
        "weights_dir_name": "model_perovskite_ASR",
        "expected_dir_name": "model_perovskite_ASR",
        "description": "Random forest model to predict the area specific resistance (ASR) of perovskite oxides",
        "params": ["composition_list", "temp_list"],
    },
    {
        "function_name": "predict_perovskite_Habs",
        "module_name": "predict_perovskite_Habs",
        "weights_dir_name": "model_perovskite_Habs-main",
        "expected_dir_name": "model_perovskite_Habs",
        "description": "Random forest model to predict the H absorption uptake of perovskites",
        "params": ["composition_list", "temp_list", "ph2o_list"],
    },
    {
        "function_name": "predict_perovskite_Opband",
        "module_name": "predict_perovskite_Opband",
        "weights_dir_name": "model_perovskite_Opband",
        "expected_dir_name": "model_perovskite_Opband",
        "description": "Random forest model to predict the O 2p-band center of perovskite oxides",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_perovskite_tec",
        "module_name": "predict_perovskite_tec",
        "weights_dir_name": "model_perovskite_tec",
        "expected_dir_name": "model_perovskite_tec",
        "description": "Random forest model to predict the thermal expansion coefficient of perovskites",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_perovskite_workfunction",
        "module_name": "predict_perovskite_workfunction",
        "weights_dir_name": "model_perovskite_workfunction",
        "expected_dir_name": "model_perovskite_workfunction",
        "description": "Random forest model to predict the (001) AO-surface work function of perovskites",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_phonon_freq",
        "module_name": "predict_phonon_freq",
        "weights_dir_name": "model_phonon_freq",
        "expected_dir_name": "model_phonon_freq",
        "description": "Random forest model to predict maximum phonon frequency of materials",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_debyeT_aflow",
        "module_name": "predict_debyeT_aflow",
        "weights_dir_name": "model_debyeT_aflow",
        "expected_dir_name": "model_debyeT_aflow",
        "description": "Random forest model to predict the Debye temperature of materials in the AFLOW database",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_thermalexp_aflow",
        "module_name": "predict_thermalexp_aflow",
        "weights_dir_name": "model_thermalexp_aflow",
        "expected_dir_name": "model_thermalexp_aflow",
        "description": "Random forest model to predict the thermal expansion coefficient of materials, trained from AFLOW database",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_thermalcond_aflow",
        "module_name": "predict_thermalcond_aflow",
        "weights_dir_name": "model_thermalcond_aflow",
        "expected_dir_name": "model_thermalcond_aflow",
        "description": "Random forest model to predict the thermal conductivity of materials, trained from AFLOW database",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_perovskite_formationE",
        "module_name": "predict_perovskite_formationE",
        "weights_dir_name": "model_perovskite_formationE",
        "expected_dir_name": "model_perovskite_formationE",
        "description": "Random forest model to predict the formation energy of perovskites",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_perovskite_stability",
        "module_name": "predict_perovskite_stability",
        "weights_dir_name": "model_perovskite_stability",
        "expected_dir_name": "model_perovskite_stability",
        "description": "Random forest model to predict the perovskite stability (as convex hull energy)",
        "params": ["composition_list"],
    },
    {
        "function_name": "predict_RPV_TTS",
        "module_name": "predict_RPV_TTS",
        "weights_dir_name": "model_RPV_TTS",
        "expected_dir_name": "model_RPV_TTS",
        "description": "Random forest model to predict the transition temperature shift (TTS) of reactor pressure vessel (RPV) steels",
        "params": ["perovskite_df"],
    },
    {
        "function_name": "predict_oxide_vacancy",
        "module_name": "predict_oxide_vacancy",
        "weights_dir_name": "model_oxide_vacancy",
        "expected_dir_name": "model_oxide_vacancy",
        "description": "Random forest model to predict the formation energy of O vacancies in oxides",
        "params": ["perovskite_df"],
    },
    {
        "function_name": "predict_perovskite_conductivity",
        "module_name": "predict_perovskite_conductivity",
        "weights_dir_name": "model_perovskite_conductivity",
        "expected_dir_name": "model_perovskite_conductivity",
        "description": "Random forest model to predict the conductivity of perovskite oxides",
        "params": ["perovskite_df", "temp_list", "ph2o_list", "po2_list"],
    },
]

# Output file path
OUTPUT_FILE = "modal_functions.py"

def generate_stage_files_function():
    """Generate the stage_files function that moves model files to the correct locations."""
    moves = []
    
    for spec in function_specs:
        moves.append(f'        shutil.move("/material-property-predictors/{spec["weights_dir_name"]}", "/{spec["expected_dir_name"]}")')
    
    moves_code = "\n".join(moves)
    
    return dedent(f'''
    def stage_files():
        import shutil
        
        # Move the model files
{moves_code}        
    ''')

def generate_function_code(spec):
    """Generate code for a single function based on its specification."""
    param_str = ", ".join(spec["params"])
    
    return dedent(f'''
    @app.function(image=mastml_image)
    def {spec["function_name"]}({param_str}):
        """
        {spec["description"]}
        """
        import sys
        import os
        sys.path.append('/{spec["expected_dir_name"]}')
        os.chdir('/')
        from {spec["module_name"]} import make_predictions
        
        preds = make_predictions({param_str})
        return preds
    ''')

def generate_modal_functions():
    """Generate a Python file with Modal functions based on specifications."""
    
    header = dedent('''
    """
    This file was automatically generated - DO NOT EDIT DIRECTLY.
    Generated by modal_generator.py
    """
    import modal
    
    mastml_image = (
        modal.Image.from_registry("gardenai/base:mastml-optimized")
        .apt_install("git", "git-lfs")
        .run_commands("ls .")
        .run_commands("git clone https://huggingface.co/willengler-uc/material-property-predictors")
        .run_function(stage_files)
    )

    app = modal.App("wisconsin-materials-predictions")
    
    ''')
    
    with open(OUTPUT_FILE, 'w') as f:
        # First, write the stage_files function
        stage_files_code = generate_stage_files_function()
        f.write(stage_files_code)
        f.write("\n\n")
        
        f.write(header)
        
        # Write each function
        for spec in function_specs:
            function_code = generate_function_code(spec)
            f.write(function_code)
            f.write("\n\n")  # Add space between functions
        
        # Add any additional code needed at the end
        footer = dedent('''
        # Add any additional setup code here
        if __name__ == "__main__":
            app.run()
        ''')
        f.write(footer)

if __name__ == "__main__":
    generate_modal_functions()
    print(f"Generated Modal functions in {OUTPUT_FILE}")