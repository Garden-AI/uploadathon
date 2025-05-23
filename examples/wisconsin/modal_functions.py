
def stage_files():
    import shutil

    # Move the model files
    shutil.move("/material-property-predictors/model_bandgap-main", "/model_bandgap")
    shutil.move("/material-property-predictors/model_concrete", "/model_concrete")
    shutil.move("/material-property-predictors/model_piezoelectric-main", "/model_piezoelectric")
    shutil.move("/material-property-predictors/model_dielectric", "/model_dielectric")
    shutil.move("/material-property-predictors/model_diffusion", "/model_diffusion")
    shutil.move("/material-property-predictors/model_double_perovskite_gap", "/model_double_perovskite_gap")
    shutil.move("/material-property-predictors/model_elastic_tensor", "/model_elastic_tensor")
    shutil.move("/material-property-predictors/model_exfoliationE", "/model_exfoliationE")
    shutil.move("/material-property-predictors/model_hea_hardness", "/model_hea_hardness")
    shutil.move("/material-property-predictors/model_heusler", "/model_heusler")
    shutil.move("/material-property-predictors/model_Li_conductivity", "/model_Li_conductivity")
    shutil.move("/material-property-predictors/model_metallicglass_Dmax", "/model_metallicglass_Dmax")
    shutil.move("/material-property-predictors/model_metallicglass_Rc", "/model_metallicglass_Rc")
    shutil.move("/material-property-predictors/model_metallicglass_Rc_LLM", "/model_metallicglass_Rc_LLM")
    shutil.move("/material-property-predictors/model_Mg_alloy-main", "/model_Mg_alloy")
    shutil.move("/material-property-predictors/model_thermal_conductivity", "/model_thermal_conductivity")
    shutil.move("/material-property-predictors/model_superconductivity", "/model_superconductivity")
    shutil.move("/material-property-predictors/model_steel_yield", "/model_steel_yield")
    shutil.move("/material-property-predictors/model_semiconductor_lvls", "/model_semiconductor_lvls")
    shutil.move("/material-property-predictors/model_perovskite_ASR", "/model_perovskite_ASR")
    shutil.move("/material-property-predictors/model_perovskite_Habs-main", "/model_perovskite_Habs")
    shutil.move("/material-property-predictors/model_perovskite_Opband", "/model_perovskite_Opband")
    shutil.move("/material-property-predictors/model_perovskite_tec", "/model_perovskite_tec")
    shutil.move("/material-property-predictors/model_perovskite_workfunction", "/model_perovskite_workfunction")
    shutil.move("/material-property-predictors/model_phonon_freq", "/model_phonon_freq")
    shutil.move("/material-property-predictors/model_debyeT_aflow", "/model_debyeT_aflow")
    shutil.move("/material-property-predictors/model_thermalexp_aflow", "/model_thermalexp_aflow")
    shutil.move("/material-property-predictors/model_thermalcond_aflow", "/model_thermalcond_aflow")
    shutil.move("/material-property-predictors/model_perovskite_formationE", "/model_perovskite_formationE")
    shutil.move("/material-property-predictors/model_perovskite_stability", "/model_perovskite_stability")
    shutil.move("/material-property-predictors/model_RPV_TTS", "/model_RPV_TTS")
    shutil.move("/material-property-predictors/model_oxide_vacancy", "/model_oxide_vacancy")
    shutil.move("/material-property-predictors/model_perovskite_conductivity", "/model_perovskite_conductivity")        



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


@app.function(image=mastml_image)
def predict_bandgap(composition_list):
    """
    Random forest model to predict the electronic bandgap of materials
    """
    import sys
    import os
    sys.path.append('/model_bandgap')
    os.chdir('/')
    from predict_bandgap import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_concrete(concrete_df):
    """
    Random forest model to predict the compressive strength of concrete mixtures
    """
    import sys
    import os
    sys.path.append('/model_concrete')
    os.chdir('/')
    from predict_concrete import make_predictions

    preds = make_predictions(concrete_df)
    return preds



@app.function(image=mastml_image)
def predict_piezoelectric(composition_list):
    """
    Random forest model to predict max piezoelectric displacement of materials
    """
    import sys
    import os
    sys.path.append('/model_piezoelectric')
    os.chdir('/')
    from predict_piezoelectric import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_dielectric(composition_list):
    """
    Random forest model to predict the dielectric constant of materials
    """
    import sys
    import os
    sys.path.append('/model_dielectric')
    os.chdir('/')
    from predict_dielectric import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_diffusion(composition_list):
    """
    Random forest model to predict the dilute solute activation energy for diffusion
    """
    import sys
    import os
    sys.path.append('/model_diffusion')
    os.chdir('/')
    from predict_diffusion import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_double_perovskite_gap(composition_list):
    """
    Random forest model to predict the bandgap of double perovskites
    """
    import sys
    import os
    sys.path.append('/model_double_perovskite_gap')
    os.chdir('/')
    from predict_double_perovskite_gap import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_elastic_tensor(composition_list):
    """
    Random forest model to predict the bulk modulus of materials
    """
    import sys
    import os
    sys.path.append('/model_elastic_tensor')
    os.chdir('/')
    from predict_elastic_tensor import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_exfoliationE(composition_list):
    """
    Random forest model to predict the exfoliation energy of materials
    """
    import sys
    import os
    sys.path.append('/model_exfoliationE')
    os.chdir('/')
    from predict_exfoliationE import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_hea_hardness(composition_list):
    """
    Random forest model to predict the hardness of high entropy alloys
    """
    import sys
    import os
    sys.path.append('/model_hea_hardness')
    os.chdir('/')
    from predict_hea_hardness import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_heusler(composition_list, numelec_list, heuslertype_list):
    """
    Random forest model to predict the magnetization of Heusler alloys
    """
    import sys
    import os
    sys.path.append('/model_heusler')
    os.chdir('/')
    from predict_heusler import make_predictions

    preds = make_predictions(composition_list, numelec_list, heuslertype_list)
    return preds



@app.function(image=mastml_image)
def predict_Li_conductivity(composition_list):
    """
    Random forest model to predict the conductivity of Li in solid electrolytes
    """
    import sys
    import os
    sys.path.append('/model_Li_conductivity')
    os.chdir('/')
    from predict_Li_conductivity import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_metallicglass_Dmax(composition_list):
    """
    Random forest model to predict the maximum casting diameter of metallic glasses
    """
    import sys
    import os
    sys.path.append('/model_metallicglass_Dmax')
    os.chdir('/')
    from predict_metallicglass_Dmax import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_metallicglass_Rc(composition_list):
    """
    Random forest model to predict the critical cooling rate of metallic glasses
    """
    import sys
    import os
    sys.path.append('/model_metallicglass_Rc')
    os.chdir('/')
    from predict_metallicglass_Rc import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_metallicglass_Rc_LLM(composition_list):
    """
    Random forest model to predict the critical cooling rate of metallic glasses (data from LLMs)
    """
    import sys
    import os
    sys.path.append('/model_metallicglass_Rc_LLM')
    os.chdir('/')
    from predict_metallicglass_Rc_LLM import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_Mg_alloy(mg_alloy_df):
    """
    Random forest model to predict the yield strength of Mg alloys
    """
    import sys
    import os
    sys.path.append('/model_Mg_alloy')
    os.chdir('/')
    from predict_Mg_alloy import make_predictions

    preds = make_predictions(mg_alloy_df)
    return preds



@app.function(image=mastml_image)
def predict_thermal_conductivity(composition_list, temperature_list):
    """
    Random forest model to predict the thermal conductivity of materials
    """
    import sys
    import os
    sys.path.append('/model_thermal_conductivity')
    os.chdir('/')
    from predict_thermal_conductivity import make_predictions

    preds = make_predictions(composition_list, temperature_list)
    return preds



@app.function(image=mastml_image)
def predict_superconductivity(composition_list):
    """
    Random forest model to predict the superconducting critical temperature of materials
    """
    import sys
    import os
    sys.path.append('/model_superconductivity')
    os.chdir('/')
    from predict_superconductivity import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_steel_yield(composition_list):
    """
    Random forest model to predict the yield strength of steel alloys
    """
    import sys
    import os
    sys.path.append('/model_steel_yield')
    os.chdir('/')
    from predict_steel_yield import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_semiconductor_lvls(composition_list, chargefrom_list, chargeto_list, lattice_list, bandgap_list, is_b_latt_list):
    """
    Random forest model to predict the defect charge state transition levels in semiconductors
    """
    import sys
    import os
    sys.path.append('/model_semiconductor_lvls')
    os.chdir('/')
    from predict_semiconductor_lvls import make_predictions

    preds = make_predictions(composition_list, chargefrom_list, chargeto_list, lattice_list, bandgap_list, is_b_latt_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_ASR(composition_list, temp_list):
    """
    Random forest model to predict the area specific resistance (ASR) of perovskite oxides
    """
    import sys
    import os
    sys.path.append('/model_perovskite_ASR')
    os.chdir('/')
    from predict_perovskite_ASR import make_predictions

    preds = make_predictions(composition_list, temp_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_Habs(composition_list, temp_list, ph2o_list):
    """
    Random forest model to predict the H absorption uptake of perovskites
    """
    import sys
    import os
    sys.path.append('/model_perovskite_Habs')
    os.chdir('/')
    from predict_perovskite_Habs import make_predictions

    preds = make_predictions(composition_list, temp_list, ph2o_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_Opband(composition_list):
    """
    Random forest model to predict the O 2p-band center of perovskite oxides
    """
    import sys
    import os
    sys.path.append('/model_perovskite_Opband')
    os.chdir('/')
    from predict_perovskite_Opband import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_tec(composition_list):
    """
    Random forest model to predict the thermal expansion coefficient of perovskites
    """
    import sys
    import os
    sys.path.append('/model_perovskite_tec')
    os.chdir('/')
    from predict_perovskite_tec import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_workfunction(composition_list):
    """
    Random forest model to predict the (001) AO-surface work function of perovskites
    """
    import sys
    import os
    sys.path.append('/model_perovskite_workfunction')
    os.chdir('/')
    from predict_perovskite_workfunction import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_phonon_freq(composition_list):
    """
    Random forest model to predict maximum phonon frequency of materials
    """
    import sys
    import os
    sys.path.append('/model_phonon_freq')
    os.chdir('/')
    from predict_phonon_freq import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_debyeT_aflow(composition_list):
    """
    Random forest model to predict the Debye temperature of materials in the AFLOW database
    """
    import sys
    import os
    sys.path.append('/model_debyeT_aflow')
    os.chdir('/')
    from predict_debyeT_aflow import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_thermalexp_aflow(composition_list):
    """
    Random forest model to predict the thermal expansion coefficient of materials, trained from AFLOW database
    """
    import sys
    import os
    sys.path.append('/model_thermalexp_aflow')
    os.chdir('/')
    from predict_thermalexp_aflow import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_thermalcond_aflow(composition_list):
    """
    Random forest model to predict the thermal conductivity of materials, trained from AFLOW database
    """
    import sys
    import os
    sys.path.append('/model_thermalcond_aflow')
    os.chdir('/')
    from predict_thermalcond_aflow import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_formationE(composition_list):
    """
    Random forest model to predict the formation energy of perovskites
    """
    import sys
    import os
    sys.path.append('/model_perovskite_formationE')
    os.chdir('/')
    from predict_perovskite_formationE import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_stability(composition_list):
    """
    Random forest model to predict the perovskite stability (as convex hull energy)
    """
    import sys
    import os
    sys.path.append('/model_perovskite_stability')
    os.chdir('/')
    from predict_perovskite_stability import make_predictions

    preds = make_predictions(composition_list)
    return preds



@app.function(image=mastml_image)
def predict_RPV_TTS(perovskite_df):
    """
    Random forest model to predict the transition temperature shift (TTS) of reactor pressure vessel (RPV) steels
    """
    import sys
    import os
    sys.path.append('/model_RPV_TTS')
    os.chdir('/')
    from predict_RPV_TTS import make_predictions

    preds = make_predictions(perovskite_df)
    return preds



@app.function(image=mastml_image)
def predict_oxide_vacancy(perovskite_df):
    """
    Random forest model to predict the formation energy of O vacancies in oxides
    """
    import sys
    import os
    sys.path.append('/model_oxide_vacancy')
    os.chdir('/')
    from predict_oxide_vacancy import make_predictions

    preds = make_predictions(perovskite_df)
    return preds



@app.function(image=mastml_image)
def predict_perovskite_conductivity(perovskite_df, temp_list, ph2o_list, po2_list):
    """
    Random forest model to predict the conductivity of perovskite oxides
    """
    import sys
    import os
    sys.path.append('/model_perovskite_conductivity')
    os.chdir('/')
    from predict_perovskite_conductivity import make_predictions

    preds = make_predictions(perovskite_df, temp_list, ph2o_list, po2_list)
    return preds


@app.local_entrypoint()
def main():
    example_input = ['SiO2', 'SrTiO3', 'Al2O3']
    results = predict_bandgap.remote(example_input)
    print(results)

    comps_list = ['V2ScAl', 'V2ScGa', 'V2ScIn']
    numelec_list = [16, 16, 16]
    heuslertype_list = ['Full Heusler', 'Full Heusler', 'Full Heusler']
    results = predict_heusler.remote(comps_list, numelec_list, heuslertype_list)
    print(results)

    comp_list = ['(BaO)0.1(SrO)0.9(CeO2)0.5(ZrO2)0.35(Y2O3)0.05(Sm2O3)0.025', '(BaO)0.1(SrO)0.9(CeO2)0.5(ZrO2)0.35(Y2O3)0.05(Sm2O3)0.025']
    temp_list = [700.716623, 649.022285]
    ph2o_list = [1.9, 1.9]
    po2_list = [0, 0]
    results = predict_perovskite_conductivity.remote(comp_list, temp_list, ph2o_list, po2_list)
    print(results)