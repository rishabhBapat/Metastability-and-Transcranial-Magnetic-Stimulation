# Metastability and Transcranial Magnetic Stimulation
This folder contains Python and MATLAB code associated with the manuscript "Using metastability to identify the global changes in dynamic working point of the brain following brain stimulation" (https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1336438/full).

# Overview

The data were preprocessed using the the MATLAB code in "preprocessing_TMS.mlx". The input for this code is the directory containing the raw EEG files. There are separate cells to be run for preprocessing resting state and TMS-EEG data.

"kuramoto_model.py" implements the kuramoto model as the class "Kuramoto".

"kuramoto_TMS.py" performs the modelling portion of the analysis and requires structural connectivity and cortical distance matrices as input. In the manuscript, these matrices are based on the AAL parcellation as provided in this repository. These files can be found in the "AAL_matrices" directory.

"Efield_TMS.py" defines a set of ROIs that will be instantly phase reset by a TMS pulse, given an electric field simulation of that TMS pulse and a list of ROIS with their MNI coordinates as input. In this study, SimNIBS was used to simulate the TMS electric field.

All other files take preprocessed EEG data as input and perform their respective analysis.

If there are any questions regarding this code or the manuscript that uses it please address them to Rishabh Bapat.
