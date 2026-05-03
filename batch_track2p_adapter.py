"""
batch_track2p_adapter.py
========================
Adapter script for running track2p across BIDS-style suite2p output folders.

Problem:
    track2p expects each dataset path (ds_path) to contain a subfolder
    named exactly 'suite2p/', which then contains 'plane0/', 'plane1/', etc.

    Our batch suite2p script names output folders with BIDS-style prefixes:
        250708_sub-ACUTEVIS07_ses-01_task-gratings_baseline_suite2p/
    These folders contain plane0/ directly — they ARE the suite2p output.

Solution (no track2p modifications):
    This script does two things:

    1. Groups BIDS folders by (subject, task) and moves them into a parent
       folder named after the group.
    2. Inside each BIDS folder, creates a 'suite2p/' subdirectory and moves
       all plane folders (and loose files like data.bin) into it.

    The BIDS folder name is NEVER changed. Only its location and internal
    structure are adjusted.

    BEFORE (flat in processed/):
        processed/
        ├── 250708_sub-ACUTEVIS06_ses-01_task-gratings_baseline_suite2p/
        │   └── plane0/
        │       ├── F.npy, stat.npy, ops.npy, ...
        ├── 250709_sub-ACUTEVIS06_ses-02_task-gratings_saline_suite2p/
        │   └── plane0/
        │       ├── ...
        └── 250708_sub-ACUTEVIS06_ses-01_task-movies_baseline_suite2p/
            └── plane0/

    AFTER:
        processed/
        ├── sub-ACUTEVIS06_task-gratings/
        │   ├── 250708_sub-ACUTEVIS06_ses-01_task-gratings_baseline_suite2p/
        │   │   └── suite2p/              ← NEW: created by this script
        │   │       └── plane0/           ← MOVED from parent into suite2p/
        │   │           ├── F.npy, stat.npy, ops.npy, ...
        │   └── 250709_sub-ACUTEVIS06_ses-02_task-gratings_saline_suite2p/
        │       └── suite2p/
        │           └── plane0/
        └── sub-ACUTEVIS06_task-movies/
            └── 250708_sub-ACUTEVIS06_ses-01_task-movies_baseline_suite2p/
                └── suite2p/
                    └── plane0/

    track2p receives:
        all_ds_path = [
            'D:/.../sub-ACUTEVIS06_task-gratings/250708_..._suite2p',
            'D:/.../sub-ACUTEVIS06_task-gratings/250709_..._suite2p',
        ]
    ...and finds suite2p/plane0/F.npy exactly where it expects.

IMPORTANT:
    - This script MOVES folders, it does not copy. Since everything stays
      on the same drive, only directory entries change — it's instant
      regardless of data size.
    - A manifest file is saved so you can see (and undo) every move.
    - An undo function is provided to flatten everything back.

Usage:
    1. Set DATA_ROOT and OUTPUT_ROOT below.
    2. Run the script. It prints the grouping and planned moves, then
       pauses for your confirmation before touching anything.
    3. After reorganization, track2p runs automatically for each group.

    To undo: python batch_track2p_adapter.py --undo
"""

from pathlib import Path
import shutil
import sys
from datetime import datetime

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Parent folder containing all BIDS-style suite2p output folders.
DATA_ROOT = Path(r'D:\Projects\spont\processed\neuro')

# Where track2p should save its outputs. One subfolder per (subject, task)
# group will be created here, with track2p's own 'track2p/' folder inside.
OUTPUT_ROOT = Path(r'D:\Projects\spont\processed\neuro')
# ---- track2p algorithm parameters (same for all groups) -------------------
TRACK_OPS_PARAMS = {
    'reg_chan': 0,               # 0=functional, 1=anatomical
    'transform_type': 'affine',  # 'affine' or 'rigid'
    'iscell_thr': None,          # iscell probability threshold
    'matching_method': 'iou',    # 'iou', 'cent', or 'cent_int-filt'
    'iou_dist_thr': 16,          # centroid distance cutoff for IOU
    'thr_remove_zeros': False,
    'thr_method': 'otsu',        # 'otsu' or 'min'
    'show_roi_reg_output': False,
    'win_size': 48,
    'sat_perc': 99.9,
    'save_in_s2p_format': True,  # save matched cells in suite2p format
    'input_format': 'suite2p',
}

# Conditions to EXCLUDE from track2p runs. Baseline sessions will still be
# reorganized into the group folders, but they won't be passed to track2p.
# Use lowercase — matching is case-insensitive.
EXCLUDE_CONDITIONS = {'baseline'}

# Set to True if folders are still flat in DATA_ROOT and need reorganizing.
# Set to False if you've already run the reorganization and just want to
# run track2p on the existing grouped folder structure.
REORGANIZE = True


# ============================================================================
# STEP 1: DISCOVER AND PARSE BIDS FOLDERS
# ============================================================================

def discover_bids_folders(data_root: Path) -> list[Path]:
    """
    Find all directories in data_root whose name ends with '_suite2p'.
    Only looks at immediate children (not recursive).
    Returns a sorted list of full paths.
    """
    folders = sorted([
        p for p in data_root.iterdir()
        if p.is_dir() and p.name.endswith('_suite2p')
    ])
    return folders


def parse_bids_name(folder_name: str):
    """
    Extract BIDS tokens from a folder name like:
        '250708_sub-ACUTEVIS07_ses-01_task-gratings_baseline_suite2p'

    Returns a dict with keys: 'date', 'subject', 'session', 'task', 'condition'.
    Returns None if the name can't be parsed (prints a warning).

    Parsing logic:
        - Split on '_'
        - First token = date
        - Tokens starting with 'sub-', 'ses-', 'task-' = BIDS keys
        - Last token = 'suite2p' (already filtered by discover)
        - Everything else between known keys = condition
    """
    parts = folder_name.split('_')

    if len(parts) < 5:
        print(f"  [SKIP] Cannot parse (too few parts): {folder_name}")
        return None

    result = {'date': parts[0]}

    # Indices that aren't date (0), a known BIDS key, or 'suite2p' (last)
    remaining_indices = []

    for i, part in enumerate(parts):
        if part.startswith('sub-'):
            result['subject'] = part
        elif part.startswith('ses-'):
            result['session'] = part
        elif part.startswith('task-'):
            result['task'] = part
        elif part == 'suite2p':
            pass  # terminal marker
        elif i != 0:
            # Not the date, not a BIDS key, not 'suite2p' → condition token
            remaining_indices.append(i)

    # Assemble condition from leftover tokens (e.g. 'baseline', 'high_dose')
    condition_parts = [parts[i] for i in remaining_indices]
    result['condition'] = '_'.join(condition_parts) if condition_parts else 'unknown'

    # Validate required keys
    for key in ('subject', 'session', 'task'):
        if key not in result:
            print(f"  [SKIP] Missing '{key}' in: {folder_name}")
            return None

    return result


# ============================================================================
# STEP 2: GROUP BY (SUBJECT, TASK)
# ============================================================================

def group_by_subject_task(data_root: Path) -> dict:
    """
    Scan data_root for BIDS-style suite2p folders, parse their names, and
    group them by (subject, task).

    Returns:
        {
            ('sub-ACUTEVIS07', 'task-gratings'): [
                {
                    'path': Path('...processed/250708_sub-ACUTEVIS07_ses-01_...'),
                    'date': '250708',
                    'subject': 'sub-ACUTEVIS07',
                    'session': 'ses-01',
                    'task': 'task-gratings',
                    'condition': 'baseline',
                },
                ...
            ],
            ...
        }

    Each group is sorted by session label (ses-01, ses-02, ...).
    """
    folders = discover_bids_folders(data_root)
    print(f"Found {len(folders)} BIDS-style suite2p folder(s) in {data_root}\n")

    groups = {}

    for folder_path in folders:
        parsed = parse_bids_name(folder_path.name)
        if parsed is None:
            continue

        key = (parsed['subject'], parsed['task'])
        entry = {**parsed, 'path': folder_path}
        groups.setdefault(key, []).append(entry)

    # Sort each group by session label so track2p processes chronologically
    for key in groups:
        groups[key].sort(key=lambda e: e['session'])

    return groups


# ============================================================================
# STEP 3: REORGANIZE FOLDERS
# ============================================================================

def plan_moves(groups: dict, data_root: Path) -> list[dict]:
    """
    Build a list of planned filesystem operations WITHOUT executing them.
    Each entry describes what will happen to one BIDS folder.

    Returns a list of dicts:
        [
            {
                'bids_folder_name': '250708_sub-ACUTEVIS06_ses-01_..._suite2p',
                'original_path':    Path('D:/.../processed/250708_...'),
                'group_name':       'sub-ACUTEVIS06_task-gratings',
                'new_parent':       Path('D:/.../processed/sub-ACUTEVIS06_task-gratings'),
                'new_path':         Path('D:/.../processed/sub-ACUTEVIS06_task-gratings/250708_...'),
                'planes_to_wrap':   ['plane0'],
            },
            ...
        ]
    """
    moves = []

    for (subject, task), entries in groups.items():
        group_name = f"{subject}_{task}"
        new_parent = data_root / group_name

        for entry in entries:
            bids_path = entry['path']

            # Find all plane* folders inside the BIDS folder.
            # These are what we'll move into a suite2p/ wrapper.
            plane_folders = sorted([
                p.name for p in bids_path.iterdir()
                if p.is_dir() and p.name.startswith('plane')
            ])

            if not plane_folders:
                print(f"  [WARNING] No plane folders found in {bids_path.name}")

            moves.append({
                'bids_folder_name': bids_path.name,
                'original_path': bids_path,
                'group_name': group_name,
                'new_parent': new_parent,
                'new_path': new_parent / bids_path.name,
                'planes_to_wrap': plane_folders,
            })

    return moves


def execute_moves(moves: list[dict], data_root: Path):
    """
    Execute the planned reorganization in two stages:

    Stage A — Move each BIDS folder into its group parent:
        processed/250708_..._suite2p/
            → processed/sub-X_task-Y/250708_..._suite2p/

    Stage B — Inside each BIDS folder, create suite2p/ and move all
    plane* folders (and any loose files) into it:
        .../250708_..._suite2p/plane0/
            → .../250708_..._suite2p/suite2p/plane0/

    Saves a manifest file so the operation can be undone.
    """
    # ---- Save manifest BEFORE moving anything ----
    manifest_path = data_root / 'reorganize_manifest.txt'
    with open(manifest_path, 'w') as f:
        f.write(f"# Reorganization manifest — {datetime.now().isoformat()}\n")
        f.write(f"# DATA_ROOT: {data_root}\n")
        f.write(f"# Format: original_path -> new_path\n")
        f.write(f"# plane* folders were moved into suite2p/ inside each BIDS folder\n\n")
        for m in moves:
            f.write(f"{m['original_path']} -> {m['new_path']}\n")
    print(f"  Manifest saved: {manifest_path}\n")

    for m in moves:
        bids_name = m['bids_folder_name']
        original = m['original_path']
        new_parent = m['new_parent']
        new_path = m['new_path']
        planes = m['planes_to_wrap']

        # ---- Stage A: Move BIDS folder into group parent ----
        new_parent.mkdir(parents=True, exist_ok=True)
        print(f"  Moving: {bids_name}")
        print(f"    {original.parent.name}/ → {new_parent.name}/")
        shutil.move(str(original), str(new_path))

        # ---- Stage B: Create suite2p/ wrapper inside the BIDS folder ----
        suite2p_dir = new_path / 'suite2p'
        suite2p_dir.mkdir(exist_ok=True)

        # Move each plane* folder into suite2p/
        for plane_name in planes:
            plane_src = new_path / plane_name
            plane_dst = suite2p_dir / plane_name
            print(f"    {plane_name}/ → suite2p/{plane_name}/")
            shutil.move(str(plane_src), str(plane_dst))

        # Move any loose files (e.g. data.bin) into suite2p/ as well
        loose_files = [p for p in new_path.iterdir() if p.is_file()]
        for f in loose_files:
            print(f"    {f.name} → suite2p/{f.name}")
            shutil.move(str(f), str(suite2p_dir / f.name))

        print()


# ============================================================================
# STEP 3b: DISCOVER ALREADY-REORGANIZED GROUPS
# ============================================================================

def discover_reorganized_groups(data_root: Path) -> dict:
    """
    Scan data_root for group folders (sub-X_task-Y/) that already contain
    reorganized BIDS suite2p folders. Use this when REORGANIZE = False.

    Looks for:
        data_root/
        ├── sub-ACUTEVIS06_task-gratings/          ← group folder
        │   ├── 250708_..._suite2p/                ← BIDS folder
        │   │   └── suite2p/plane0/                ← already wrapped
        │   └── 250709_..._suite2p/
        │       └── suite2p/plane0/

    Returns the same dict shape as group_by_subject_task(), but with
    entry['path'] pointing to the actual current location of each folder.
    """
    groups = {}

    # Group folders are immediate children of data_root with '_task-' in name
    for group_dir in sorted(data_root.iterdir()):
        if not group_dir.is_dir() or '_task-' not in group_dir.name:
            continue

        # Inside each group folder, find BIDS suite2p folders
        for bids_dir in sorted(group_dir.iterdir()):
            if not bids_dir.is_dir() or not bids_dir.name.endswith('_suite2p'):
                continue

            parsed = parse_bids_name(bids_dir.name)
            if parsed is None:
                continue

            key = (parsed['subject'], parsed['task'])
            # path points to the BIDS folder's CURRENT location
            parsed['path'] = bids_dir
            groups.setdefault(key, []).append(parsed)

    # Sort each group by session label
    for key in groups:
        groups[key].sort(key=lambda e: e['session'])

    return groups


# ============================================================================
# STEP 4: BUILD DS_PATHS AND RUN TRACK2P
# ============================================================================

def build_ds_paths(groups: dict, exclude_conditions: set = None) -> dict:
    """
    Build the all_ds_path list for each group from the groups dict.
    Uses entry['path'] directly — works whether paths point to the
    original flat layout or the reorganized grouped layout.

    Parameters
    ----------
    groups : dict
        Grouping dict from group_by_subject_task() or
        discover_reorganized_groups().
    exclude_conditions : set, optional
        Condition labels to skip (case-insensitive). e.g. {'baseline'}

    Returns:
        {
            ('sub-ACUTEVIS07', 'task-gratings'): [
                'D:/.../sub-ACUTEVIS07_task-gratings/250708_..._suite2p',
                'D:/.../sub-ACUTEVIS07_task-gratings/250709_..._suite2p',
            ],
            ...
        }
    """
    if exclude_conditions is None:
        exclude_conditions = set()

    # Normalize to lowercase for case-insensitive matching
    exclude_lower = {c.lower() for c in exclude_conditions}

    ds_paths = {}
    for (subject, task), entries in groups.items():
        paths = []
        for entry in entries:
            # Check if this session's condition is excluded
            if entry['condition'].lower() in exclude_lower:
                print(f"  [EXCLUDE] {entry['path'].name}  (condition: {entry['condition']})")
                continue

            paths.append(str(entry['path']))

        ds_paths[(subject, task)] = paths

    return ds_paths


def run_track2p_for_group(
    group_key: tuple,
    ds_paths: list[str],
    output_root: Path,
    params: dict,
):
    """
    Configure and run track2p for a single (subject, task) group.

    Parameters
    ----------
    group_key : tuple
        (subject, task) e.g. ('sub-ACUTEVIS07', 'task-gratings')
    ds_paths : list of str
        Paths to the BIDS folders (now inside group parent), each containing
        suite2p/plane0/... as track2p expects.
    output_root : Path
        Root output directory. track2p saves to:
            output_root / <group_name> / track2p /
    params : dict
        Algorithm parameters applied to DefaultTrackOps.
    """
    # Lazy import so grouping/planning can be tested without track2p installed
    from track2p.ops.default import DefaultTrackOps
    from track2p.t2p import run_t2p

    subject, task = group_key
    group_name = f"{subject}_{task}"

    print(f"\n{'='*70}")
    print(f"  RUNNING TRACK2P: {group_name}")
    print(f"  Sessions: {len(ds_paths)}")
    for i, p in enumerate(ds_paths):
        print(f"    [{i}] {p}")
    print(f"{'='*70}\n")

    # --- Configure track_ops ---
    track_ops = DefaultTrackOps()

    # Dataset paths — track2p will look for suite2p/plane0/ inside each
    track_ops.all_ds_path = ds_paths

    # Output path: output_root/sub-ACUTEVIS06_task-gratings/
    # track2p appends 'track2p/' internally via init_save_paths()
    track_ops.save_path = str(output_root / group_name)

    # Apply algorithm parameters
    for key, value in params.items():
        setattr(track_ops, key, value)

    # --- Run ---
    run_t2p(track_ops)

    print(f"\n  [DONE] {group_name}")
    print(f"  Outputs: {track_ops.save_path}\n")


# ============================================================================
# UNDO — reverses the reorganization
# ============================================================================

def undo_reorganize(data_root: Path):
    """
    Reverse the reorganization:
        1. Move plane* folders from suite2p/ back up to the BIDS folder root
        2. Remove the empty suite2p/ directory
        3. Move each BIDS folder back to data_root (out of the group parent)
        4. Remove empty group parent folders

    Reads the manifest file to know what was moved.
    """
    manifest_path = data_root / 'reorganize_manifest.txt'
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}. Cannot undo.")
        return

    print(f"Reading manifest: {manifest_path}\n")

    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Parse: original_path -> new_path
        parts = line.split(' -> ')
        if len(parts) != 2:
            continue

        original_path = Path(parts[0].strip())
        new_path = Path(parts[1].strip())

        if not new_path.exists():
            print(f"  [SKIP] Not found (already undone?): {new_path}")
            continue

        # Step 1: Unwrap suite2p/ — move its contents back up
        suite2p_dir = new_path / 'suite2p'
        if suite2p_dir.exists():
            for item in suite2p_dir.iterdir():
                dest = new_path / item.name
                print(f"  Unwrap: suite2p/{item.name} → {item.name}")
                shutil.move(str(item), str(dest))
            suite2p_dir.rmdir()
            print(f"  Removed empty: suite2p/")

        # Step 2: Move BIDS folder back to original location
        print(f"  Moving: {new_path.name} → {original_path.parent.name}/")
        shutil.move(str(new_path), str(original_path))

        # Step 3: Remove group parent if now empty
        group_dir = new_path.parent
        if group_dir.exists() and not any(group_dir.iterdir()):
            group_dir.rmdir()
            print(f"  Removed empty group: {group_dir.name}/")

        print()

    print("Undo complete.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Handle --undo flag
    # ------------------------------------------------------------------
    if '--undo' in sys.argv:
        print("=" * 70)
        print("  UNDO MODE: Reversing reorganization")
        print("=" * 70 + "\n")
        undo_reorganize(DATA_ROOT)
        sys.exit(0)

    # ------------------------------------------------------------------
    # PHASE 1: Discover and group
    # ------------------------------------------------------------------
    if REORGANIZE:
        # Folders are flat in DATA_ROOT — discover, reorganize, then run
        print("=" * 70)
        print("  PHASE 1: Discovering BIDS folders (flat layout)")
        print("=" * 70)
        print(f"  Scanning: {DATA_ROOT}\n")

        groups = group_by_subject_task(DATA_ROOT)

        if not groups:
            print("No valid BIDS-style suite2p folders found. Exiting.")
            sys.exit(1)

        # Print grouping
        print("\n" + "=" * 70)
        print("  GROUPING RESULT")
        print("=" * 70)
        for (subject, task), entries in groups.items():
            print(f"\n  Group: {subject}_{task}")
            print(f"  Sessions: {len(entries)}")
            for e in entries:
                print(f"    {e['session']}  |  date={e['date']}  |  condition={e['condition']}")
                print(f"      path: {e['path']}")

        # Plan and show moves
        print("\n" + "=" * 70)
        print("  PHASE 2: Planned filesystem changes")
        print("=" * 70)

        moves = plan_moves(groups, DATA_ROOT)
        for m in moves:
            print(f"\n  {m['bids_folder_name']}")
            print(f"    → {m['group_name']}/{m['bids_folder_name']}/")
            print(f"    Wrap into suite2p/: {m['planes_to_wrap']}")

        # Confirm
        print("\n" + "-" * 70)
        print("  This will MOVE folders (not copy). No data is duplicated.")
        print("  A manifest file will be saved for undo capability.")
        print("  To reverse later: python batch_track2p_adapter.py --undo")
        response = input("\n  Proceed? (yes/no): ").strip().lower()
        if response not in ('yes', 'y'):
            print("  Exiting. Nothing was changed.")
            sys.exit(0)

        # Execute reorganization
        print("\n" + "=" * 70)
        print("  PHASE 3: Reorganizing folders")
        print("=" * 70 + "\n")
        execute_moves(moves, DATA_ROOT)
        print("  Reorganization complete.\n")

        # Re-discover from the now-reorganized layout
        groups = discover_reorganized_groups(DATA_ROOT)

    else:
        # Folders are already reorganized — just scan the grouped layout
        print("=" * 70)
        print("  PHASE 1: Discovering BIDS folders (already reorganized)")
        print("=" * 70)
        print(f"  Scanning: {DATA_ROOT}\n")

        groups = discover_reorganized_groups(DATA_ROOT)

        if not groups:
            print("No reorganized group folders found. Exiting.")
            print("  (Set REORGANIZE = True if folders are still flat.)")
            sys.exit(1)

        # Print grouping
        for (subject, task), entries in groups.items():
            print(f"\n  Group: {subject}_{task}")
            print(f"  Sessions: {len(entries)}")
            for e in entries:
                print(f"    {e['session']}  |  date={e['date']}  |  condition={e['condition']}")
                print(f"      path: {e['path']}")

    # ------------------------------------------------------------------
    # RUN TRACK2P
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Running track2p")
    print("=" * 70)

    ds_paths_per_group = build_ds_paths(groups, exclude_conditions=EXCLUDE_CONDITIONS)

    for group_key, ds_paths in ds_paths_per_group.items():
        if len(ds_paths) < 2:
            subject, task = group_key
            print(f"\n  [SKIP] {subject}_{task} — only {len(ds_paths)} session(s) "
                  f"(after exclusions). Track2p requires at least 2.")
            continue

        run_track2p_for_group(
            group_key=group_key,
            ds_paths=ds_paths,
            output_root=OUTPUT_ROOT,
            params=TRACK_OPS_PARAMS,
        )

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)
    print(f"\n  To undo reorganization: python batch_track2p_adapter.py --undo\n")