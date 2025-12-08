from pathlib import Path
import numpy as np
from suite2p import run_s2p
from suite2p import default_ops
import cupy as cp
import tifffile

DATA_PATH = Path(r'E:\inbox')
SAVE_PATH = Path(r'E:\inbox\processed')
FRAME_SHAPE = (512, 512)
RAW_SUFFIX = '001.raw'
TIFF_SUFFIX = '.tif*'
SESSION_MAP = {
    251014: 'ses-01',
    251015: 'ses-02',
    251016: 'ses-03',
    251017: 'ses-04',
}
TASK_MAP = {
    'grat': 'task-gratings',
    'mov': 'task-movies',
}
CONDITION_MAP = {
    251014: 'baseline',
    251015: 'saline',
    251016: 'low',
    251017: 'high',
}

def discover_raw_files(root: Path):
    return sorted(root.glob(f'**/*{RAW_SUFFIX}'))


def discover_tiff_files(root: Path):
    return sorted(root.glob(f'**/*{TIFF_SUFFIX}'))


def load_video_gpu(raw_path: Path, frame_shape=FRAME_SHAPE, dtype='<u2'):
    raw_gpu = cp.fromfile(raw_path, dtype=dtype)
    frame_pixels = frame_shape[0] * frame_shape[1]
    frame_count = raw_gpu.size // frame_pixels
    return raw_gpu.reshape((frame_count, *frame_shape), order='C')


def write_ome_tiff(video_gpu, raw_path: Path):
    relative = raw_path.relative_to(DATA_PATH)
    folder_name = relative.parent.name or relative.stem
    output_path = SAVE_PATH / relative.parent / f'{folder_name}.tiff'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, cp.asnumpy(video_gpu), ome=True)
    return output_path


def build_save_folder_name(tiff_path: Path):
    stem_parts = tiff_path.stem.split('_')
    if len(stem_parts) < 3:
        return tiff_path.stem + '_suite2p'
    date_token, subject_token, task_token = stem_parts[:3]
    try:
        date_key = int(date_token)
    except ValueError:
        date_key = None
    session_label = SESSION_MAP.get(date_key, 'ses-unknown')
    condition_label = CONDITION_MAP.get(date_key, 'condition-unknown')
    task_label = TASK_MAP.get(task_token.lower(), task_token)
    subject_label = f'sub-{subject_token}'
    return f'{date_token}_{subject_label}_{session_label}_{task_label}_{condition_label}_suite2p'


raw_files = discover_raw_files(DATA_PATH)
print(f'Found {len(raw_files)} raw file(s) under {DATA_PATH}')

for raw_path in raw_files:
    video_gpu = load_video_gpu(raw_path)
    # if video_gpu.shape[0] > 9100:
    #     print(f"The raw file {raw_path} has more than 9100 frames.")
    saved_path = write_ome_tiff(video_gpu, raw_path)
    print(f'Converted {raw_path} -> {saved_path}')

tiff_files = discover_tiff_files(SAVE_PATH)
print(f'Running suite2p on {len(tiff_files)} TIFF file(s) under {SAVE_PATH}')

ops = default_ops()

# main settings
ops['tau'] = 2.0 # gcamp8s
ops['fs'] = 15

# IO
ops['save_path0'] = r"E:\Projects\ACUTEVIS\processed"
ops['save_folder'] = '251015_sub-ACUTEVIS14_ses-02_task-movies_low_suite2p'

# output settings
# ops['save_NWB'] = True
ops['combined'] = False

# registration
ops['do_registration']  = True
ops['maxregshift'] = 0.01
ops['keep_movie_raw'] = True

# nonrigid
ops['nonrigid'] = False

# functional detect
ops['denoise'] = True
ops['spatial_scale'] = 1
ops['threshold_scaling'] = 1.5
ops['max_overlap'] = 0.5

# neuropil extraction
ops['inner_neuropil_radius'] = 3
ops['min_neuropil_pixels'] = 300

# spike deconvolution
ops['spikedetect'] = False

np.save(r"E:\Projects\ACUTEVIS\processed\gcamp8s_ops.npy", ops)

for tiff_path in tiff_files:
    db = {'data_path': [str(tiff_path.parent)], 'tiff_list': [str(tiff_path)]}
    ops['save_folder'] = build_save_folder_name(tiff_path)
    print(f'Running suite2p for {tiff_path}')
    run_s2p(ops=ops, db=db)

ops = np.load(r"E:\Projects\ACUTEVIS\processed\gcamp8s_ops.npy", allow_pickle=True).item()

print(ops)