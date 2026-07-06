import numpy as np
import simnibs
import pandas as pd
import pickle

# Read the simulation result
sim_120MT = simnibs.read_msh(r"C:\Users\risha\Projects\Derivatives\Kuramoto_model\variables\Efield_simulations\120MT\ernie_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh")
sim_MT = simnibs.read_msh(r"C:\Users\risha\Projects\Derivatives\Kuramoto_model\variables\Efield_simulations\MT\ernie_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh")
mt_magne =  sim_MT.field['magnE'][:]

#get element centres, volumes, and magnE
elm_centers = sim_120MT.elements_baricenters()[:]
elm_vols = sim_120MT.elements_volumes_and_areas()[:]
magnE = sim_120MT.field['magnE'][:]

# import roi coordinates
aal_coords = pd.read_excel(r"C:\Users\risha\Projects\Code\Kuramoto_model\dependencies\AAL_cords.xls",converters={'X':float})
aal_90 = aal_coords.drop(np.arange(90,len(aal_coords)))

r = 10
roi_magnE = {roi:None for roi in aal_90['Abbreviation']}
for i in range(len(aal_90)):
    roi = aal_90['Abbreviation'][i]
    roi_coords = simnibs.mni2subject_coords([aal_90['X'][i], aal_90['Y'][0], aal_90['Z'][0]], r"C:\Users\risha\Downloads\simnibs4_examples\m2m_ernie")
    
    # determine the elements in the ROI
    roi_elements = np.linalg.norm(elm_centers - roi_coords, axis=1) < r

    # calculate the mean
    mean_magnE = np.average(magnE[roi_elements])

    #store in dict
    roi_magnE[roi] = mean_magnE

cutoff = np.quantile(mt_magne,0.95)
affected_rois = [roi for roi in roi_magnE if roi_magnE[roi] >= cutoff]
affected_indices = [i for i, roi in enumerate(aal_90['Abbreviation']) if roi in affected_rois]

save_list = ['affected_indices']
for entry in save_list:
    with open(rf'C:\Users\risha\Projects\Derivatives\Mtsb_TMS\variables\Efield_simulations\indices_95.pkl', 'wb') as file:
        pickle.dump(eval(entry), file)
