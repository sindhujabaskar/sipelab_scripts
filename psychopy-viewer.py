# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:11:01 2025

@author: sindhuja s. baskar
"""

import pandas as pd
import numpy as np

psychopy1 = pd.read_csv(r"E:\Projects\ACUTEVIS\data\sub-ACUTEVIS06\ses-02\beh\20250709_151139_sub-ACUTEVIS06_ses-02_task-movies_psychopy.csv")
psychopy2 = pd.read_csv(r"E:\Projects\ACUTEVIS\data\sub-ACUTEVIS09\ses-03\beh\20250710_163636_sub-ACUTEVIS09_ses-03_task-movies_psychopy.csv")
psychopy3 = pd.read_csv(r"E:\Projects\ACUTEVIS\data\sub-ACUTEVIS16\ses-03\beh\20251016_174459_sub-ACUTEVIS16_ses-03_task-movies_psychopy.csv")
psychopy4 = pd.read_csv(r"E:\Projects\ACUTEVIS\data\sub-ACUTEVIS14\ses-01\beh\20251014_160815_sub-ACUTEVIS14_ses-01_task-movies_psychopy.csv")
psychopy5 = pd.read_csv(r"E:\inbox\251215_GS27_ses-01\20251215_153312_sub-GS27_ses-01_task-movies_psychopy.csv")

result = np.subtract("thisRow.t","key_resp.rt"[2]) #trying to establish a 0 start time

column_dict = {
    "trial_start": ("thisRow.t",),
    "mov_start": ("display_mov.started",),
    "mov_end": ("display_mov.stopped",),
    "gray_start": ("grey.started",),
    "gray_end": ("grey.stopped",),
}

flat_cols = [(alias, col) for alias, cols in column_dict.items() for col in cols]
multi_cols = pd.MultiIndex.from_tuples(flat_cols, names=["alias", "original"])

stim1 = psychopy1.loc[:, [col for _, col in flat_cols]]

stim2 = psychopy2.loc[:, [col for _, col in flat_cols]]

stim3 = psychopy3.loc[:, [col for _, col in flat_cols]]

stim4 = psychopy4.loc[:, [col for _, col in flat_cols]]

stim5 = psychopy5.loc[:, [col for _, col in flat_cols]]

# subtract the 