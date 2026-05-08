from pathlib import Path
import numpy as np
from suite2p import run_s2p
from suite2p import default_ops
import tifffile

DATA_PATH = Path(r'D:\Projects\spont')
SAVE_PATH = Path(r'D:\Projects\spont\processed\suite2p')
FRAME_SHAPE = (512, 512)
FRAMES_PER_CHUNK = 256
RAW_SUFFIX = '001.raw'
TIFF_SUFFIX = '.tif*'
SESSION_MAP = {
    260305: 'ses-01',
    260307: 'ses-02',
    260308: 'ses-03',
    260309: 'ses-04',
}
TASK_MAP = {
    'grat': 'task-gratings',
    'mov': 'task-movies',
    'spont': 'task-spontaneous',
}
CONDITION_MAP = {
    260305: 'baseline',
    260307: 'saline',
    260308: 'high',
    260309: 'low',
}

def discover_raw_files(root: Path):
    return sorted(root.glob(f'**/*{RAW_SUFFIX}'))


def discover_tiff_files(root: Path):
    return sorted(root.glob(f'**/*{TIFF_SUFFIX}'))


def load_video(raw_path: Path, frame_shape=FRAME_SHAPE, dtype='<u2'):
    raw_cpu = np.fromfile(raw_path, dtype=dtype)
    frame_pixels = frame_shape[0] * frame_shape[1]
    frame_count = raw_cpu.size // frame_pixels
    return raw_cpu.reshape((frame_count, *frame_shape), order='C')


def stream_raw_frames(raw_path: Path, frame_shape=FRAME_SHAPE, dtype='<u2', frames_per_chunk=FRAMES_PER_CHUNK):
    frame_pixels = frame_shape[0] * frame_shape[1]
    itemsize = np.dtype(dtype).itemsize
    bytes_per_frame = frame_pixels * itemsize
    file_size = raw_path.stat().st_size
    frame_count = file_size // bytes_per_frame
    if frame_count == 0:
        return
    raw_map = np.memmap(raw_path, dtype=dtype, mode='r', shape=(frame_count, *frame_shape), order='C')
    for start in range(0, frame_count, frames_per_chunk):
        end = min(start + frames_per_chunk, frame_count)
        # copy chunk so memmap slices are released quickly after writing
        yield np.asarray(raw_map[start:end]).copy()


def write_ome_tiff(raw_path: Path):
    relative = raw_path.relative_to(DATA_PATH)
    folder_name = relative.parent.name or relative.stem
    output_path = SAVE_PATH / relative.parent / f'{folder_name}.tiff'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_array = load_video(raw_path)
    if video_array.shape[0] > 36000:
        print(f'The raw file {raw_path} has more than 36000 frames.')
        video_array = video_array[::2]

    tifffile.imwrite(output_path, video_array, bigtiff=True, photometric='minisblack', metadata={'axes': 'TYX'})
    _ = tifffile.memmap(output_path)
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
    if date_key is None:
        session_label = 'ses-unknown'
        condition_label = 'condition-unknown'
    else:
        session_label = SESSION_MAP.get(date_key, 'ses-unknown')
        condition_label = CONDITION_MAP.get(date_key, 'condition-unknown')
    task_label = TASK_MAP.get(task_token.lower(), task_token)
    subject_label = f'sub-{subject_token}'
    return f'{date_token}_{subject_label}_{session_label}_{task_label}_{condition_label}_suite2p'


raw_files = discover_raw_files(DATA_PATH)
print(f'Found {len(raw_files)} raw file(s) under {DATA_PATH}')

for raw_path in raw_files:
    # if load_video(raw_path).shape[0] > 9100:
    #     print(f"The raw file {raw_path} has more than 9100 frames.")
    saved_path = write_ome_tiff(raw_path)
    print(f'Converted {raw_path} -> {saved_path}')

tiff_files = discover_tiff_files(SAVE_PATH)
print(f'Running suite2p on {len(tiff_files)} TIFF file(s) under {SAVE_PATH}')

ops = default_ops()

# main settings
ops['tau'] = 2.0 # gcamp8s
ops['fs'] = 30

# IO
ops['save_path0'] = r"D:\Projects\spont\processed\suite2p"
ops['save_folder'] = '260307_sub-SB18_ses-01_task-spontaneous_low_suite2p'

# output settings
# ops['save_NWB'] = True
ops['combined'] = False

# registration
ops['do_registration']  = True
ops['maxregshift'] = 0.01
ops['keep_movie_raw'] = True

# nonrigid
ops['nonrigid'] = True

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

np.save(r"D:\Projects\spont\processed\gcamp8s_ops.npy", ops)

for tiff_path in tiff_files:
    db = {'data_path': [str(tiff_path.parent)], 'tiff_list': [str(tiff_path)]}
    ops['save_folder'] = build_save_folder_name(tiff_path)
    print(f'Running suite2p for {tiff_path}')
    run_s2p(ops=ops, db=db)

ops = np.load(r"D:\Projects\spont\processed\gcamp8s_ops.npy", allow_pickle=True).item()

print(ops)