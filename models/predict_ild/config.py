#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/05/25 09:43:05
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''


TARGET          = 'ILD'
TARGET_UNIT     = 'Ang.'
STACKTYPE       = 'AUB'
N_JOBS          = -1 

MODEL_VERSION   = '1.1'
MODEL_NAME      = f'hetml_{TARGET}_{MODEL_VERSION}.pkl'

NUM_FEATS = [
'avg_gap_nosoc', 'avg_evac', 'avg_hform', 'avg_emass1', 'avg_emass2', 'avg_efermi_hse_nosoc', 'avg_cbm', 
'avg_vbm', 'avg_excitonmass1', 'avg_excitonmass2', 'avg_alphax', 'avg_alphaz', 'avg_bse_binding', 'avg_c_11', 
'avg_c_12', 'avg_lattice_param', 'min_cbm', 'max_cbm', 'min_vbm', 'max_vbm', 'cbmvbm', 'avg_cbm_hybridization', 
'avg_cbm_score_1', 'avg_vbm_hybridization', 'avg_vbm_score_1', 'avg_cbm_s', 'avg_cbm_p', 'avg_cbm_d', 'avg_cbm_sp', 
'avg_cbm_sd', 'avg_cbm_pd', 'avg_vbm_s', 'avg_vbm_p', 'avg_vbm_d', 'avg_vbm_sp', 'avg_vbm_sd', 'avg_vbm_pd', 
'avg_cmb_spd_card', 'avg_vbm_spd_card', 'avg_cbm_char_d', 'avg_cbm_char_p', 'avg_cbm_char_s', 'avg_vbm_char_d', 
'avg_vbm_char_p', 'avg_vbm_char_s', 'avg_cbm_mend_no', 'avg_vbm_mend_no', 'avg_delta_mend_no', 'mean_cbmsite_mass', 
'mean_cbmsite_elecneg', 'mean_cbmsite_nvalence', 'mean_cbmsite_mp', 'mean_cbmsite_atomic_vol', 
'mean_cbmsite_atomic_rad', 'mean_cbmsite_vdw_radius', 'mean_vbmsite_mass', 'mean_vbmsite_elecneg', 
'mean_vbmsite_nvalence', 'mean_vbmsite_mp', 'mean_vbmsite_atomic_vol', 'mean_vbmsite_atomic_rad', 
'mean_vbmsite_vdw_radius', 'PymatgenData minimum X', 'PymatgenData maximum X', 'PymatgenData range X', 
'PymatgenData mean X', 'PymatgenData std_dev X', 'PymatgenData minimum atomic_mass', 
'PymatgenData maximum atomic_mass', 'PymatgenData range atomic_mass', 'PymatgenData mean atomic_mass', 
'PymatgenData std_dev atomic_mass', 'PymatgenData minimum atomic_radius', 'PymatgenData maximum atomic_radius', 
'PymatgenData range atomic_radius', 'PymatgenData mean atomic_radius', 'PymatgenData std_dev atomic_radius', 
'PymatgenData mean mendeleev_no', 'PymatgenData minimum thermal_conductivity', 
'PymatgenData maximum thermal_conductivity', 'PymatgenData range thermal_conductivity', 
'PymatgenData mean thermal_conductivity', 'PymatgenData std_dev thermal_conductivity', 
'PymatgenData minimum melting_point', 'PymatgenData maximum melting_point', 'PymatgenData range melting_point', 
'PymatgenData mean melting_point', 'PymatgenData std_dev melting_point', 'stacktype', 'band_alignment_1', 
'band_alignment_2', 'band_alignment_3'
]
