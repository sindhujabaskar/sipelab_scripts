import pandas as pd
import cv2
import math
import numpy as np
import statistics as st
from pathlib import Path

def deeplabcut_pickle(filepath: Path) -> pd.DataFrame:
    """
    Custom loader for DeepLabCut pickle output.

    Reads a pickled dict where keys are frame identifiers and values are dicts
    containing:
      - 'coordinates': array-like of shape (n_landmarks, 2) for each frame
      - 'confidence': array-like of length n_landmarks or single float per frame

    This function:
      1. Loads the pickle file.
      2. Skips the first entry (assumed metadata).
      3. Constructs a DataFrame indexed by frame keys (str).
      4. Provides exactly two columns:
         - 'coordinates': list/array of (x, y) coordinate pairs for that frame.
         - 'confidence' : list/array of confidence values (or single float) for that frame.

    Returned DataFrame shape: (F, 2)
      where F = number of actual frames (total keys minus metadata entry).

    Parameters
    ----------
    filepath : str
        Path to the pickled DeepLabCut output dict.

    Returns
    -------
    pd.DataFrame
        Index: frame identifiers (str), name='frame'
        Columns:
          - coordinates : object (array-like per row)
          - confidence  : object (array-like or float per row)
    """
    data = pd.read_pickle(filepath)

    # Build dictionaries for coordinates and confidence
    coordinates_dict = {}
    confidence_dict = {}
    for frame_key, frame_data in data.items():
        coordinates_dict[frame_key] = frame_data.get('coordinates')
        confidence_dict[frame_key] = frame_data.get('confidence')

    # Skip the first (metadata) entry by slicing off index 0
    coords_series = pd.Series(coordinates_dict).iloc[1:]
    conf_series  = pd.Series(confidence_dict).iloc[1:]

    # Create the DataFrame
    df = pd.DataFrame({
        'coordinates': coords_series,
        'confidence': conf_series,
    })
    df.index.name = 'frame'
    # Drop any leftover metadata column
    df = df.drop(columns=['metadata'], errors='ignore')

    # Debug statement
    # print(f"[load_deeplabcut_pickle][DEBUG] Loaded DeepLabCut pickle from: {filepath}")
    return df

def euclidean_distance(coord1, coord2):
    """Calculate the Euclidean distance between two points."""
    return math.dist(coord1, coord2)

def analyze_pupil_data(
    pickle_data: pd.DataFrame,
    confidence_threshold: float = 0.95,
    pixel_to_mm: float = 53.6,
    dpi: int = 300
) -> pd.DataFrame:
    """
    Analyze pupil data from DeepLabCut output.

    This function processes a pandas DataFrame containing per-frame DeepLabCut outputs
    with 'coordinates' and 'confidence' columns, skipping an initial metadata row,
    and computes interpolated pupil diameters in millimetres.

    Steps
    -----
    1. Skip the first (metadata) row.
    2. Extract and convert 'coordinates' and 'confidence' to NumPy arrays.
    3. For each frame:
       - Squeeze arrays and validate dimensions.
       - Mark landmarks with confidence ≥ threshold.
       - Compute Euclidean distances for predefined landmark pairs.
       - Average valid distances as pupil diameter or assign NaN.
    4. Build a pandas Series of diameters, interpolate missing values, convert from pixels to mm.
    5. Reindex to include the metadata index, then drop the initial NaN to align with valid frames.

    Parameters
    ----------
    pickle_data : pandas.DataFrame
        Input DataFrame with an initial metadata row. Must contain:
        - 'coordinates': array-like of shape (n_points, 2) per entry
        - 'confidence': array-like of shape (n_points,) per entry
    threshold : float, optional
        Minimum confidence to include a landmark in diameter computation.
        Default is 0.1.
    pixel_to_mm : float, optional
        Conversion factor from pixels to millimetres.
        Default is 53.6.
    dpi : int, optional
        Dots-per-inch resolution (not used directly).
        Default is 300.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame ('pupil_diameter_mm') indexed by the input labels
        (excluding the metadata row), containing linearly interpolated
        pupil diameter measurements in millimetres.

    Example
    -------
    Suppose the function returns a DataFrame `result_df`. Its structure would look like:

       frame | pupil_diameter_mm
       ------|------------------
         1   | 1.23
         2   | 1.25
         3   | 1.22
         4   | 1.27
        ...  | ...
    """

    # 1) pull lists, skip metadata row
    coords_list = pickle_data['coordinates'].tolist()[1:]
    conf_list   = pickle_data['confidence'].tolist()[1:]
    
    # Return a warning if no confidence values are above the threshold
    if not any(np.any(np.array(c) >= confidence_threshold) for c in conf_list):
        print(f"[WARNING] {pickle_data.index[0:3]} No confidence values above threshold {confidence_threshold}.")
        
    # 2) to numpy arrays
    coords_arrs = [np.array(c) for c in coords_list]
    conf_arrs   = [np.array(c) for c in conf_list]

    # DEBUG: print first 3 shapes
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords.shape={c.shape}, conf.shape={f.shape}")
        
    # Print the first few values of c and f
    # for idx, (c, f) in enumerate(zip(coords_arrs[:3], conf_arrs[:3])):
    #     print(f"[DEBUG] frame {idx} coords values:\n{c}")
    #     print(f"[DEBUG] frame {idx} conf values:\n{f}")
        
    # 3) compute mean diameters
    pairs     = [(0, 1), (2, 3), (4, 5), (6, 7)]
    diameters = []
    for i, (coords, conf) in enumerate(zip(coords_arrs, conf_arrs)):
        pts   = np.squeeze(coords)   # expect (n_points, 2)
        cvals = np.squeeze(conf)     # expect (n_points,)
        # DEBUG unexpected shapes
        if pts.ndim != 2 or cvals.ndim != 1:
            print(f"[WARNING] frame {i} unexpected pts.shape={pts.shape}, conf.shape={cvals.shape}")
            diameters.append(np.nan)
            continue
        #print(f"cval type ={type(cvals)}, with values of type {cvals.dtype}\n compared to {type(confidence_threshold)}")
        valid = cvals >= confidence_threshold
        # print("cvals:", cvals)
        # print("threshold:", confidence_threshold)
        # print("mask  :", valid)  
        ds = [
            euclidean_distance(pts[a], pts[b])
            for a, b in pairs
            if a < pts.shape[0] and b < pts.shape[0] and valid[a] and valid[b]
        ]
        diameters.append(st.mean(ds) if ds else np.nan)

    # 4) interpolate & convert to mm, align with original index
    pupil_series = (
        pd.Series(diameters, index=pickle_data.index[1:])
          .interpolate()
          .divide(pixel_to_mm)
    )
    pupil_full = pupil_series.reindex(pickle_data.index)

    # DEBUG
    # print(f"[DEBUG analyze_pupil_data] input index={pickle_data.index}")
    # print(f"[DEBUG analyze_pupil_data] output series head:\n{pupil_full.head()}")

    # 5) return DataFrame without the metadata NaN
    return pd.DataFrame({'pupil_diameter_mm': pupil_full.iloc[1:]})

def process_pupil_data(database: pd.DataFrame,
                       downsample_threshold: int = None,
                       downsample_factor: int = 5):
    """
    Downsample pupil diameter timecourses for each (subject, session).

    Adds new columns:
      ('analysis','pupil_timestamps_ds') : downsampled timestamps
      ('analysis','pupil_diameter_ds')  : downsampled pupil values

    Args:
        database            : main database DataFrame
        downsample_threshold: if length exceeds this, downsample
        downsample_factor   : factor by which to downsample
    """
    import numpy as np
    # initialize storage
    ts_ds = {}
    pup_ds = {}
    for (subj, sess), pup in database[('analysis','pupil_diameter_mm')].items():
        # get raw timestamp and pupil arrays
        ts = database[('toolkit','pupil_timestamps')].loc[(subj, sess)]
        ts_arr = np.array(ts)
        pup_arr = np.array(pup)
        # downsample if requested
        if downsample_threshold and len(pup_arr) > downsample_threshold:
            idx = np.arange(0, len(pup_arr), downsample_factor)
            ts2 = ts_arr[idx]
            pup2 = pup_arr[idx]
        else:
            ts2 = ts_arr
            pup2 = pup_arr
        ts_ds[(subj, sess)] = ts2
        pup_ds[(subj, sess)] = pup2
    database[('analysis','pupil_timestamps_ds')] = pd.Series(ts_ds)
    database[('analysis','pupil_diameter_ds')] = pd.Series(pup_ds)
    return database

filepath = r'C:\dev\ACUTEVIS\processed\pupil\ACUTEVIS-SSB-2025-07-25\videos'

# Additional helper functions for GUI extension

def apply_smoothing_filter(pupil_data: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """
    Apply moving average smoothing to pupil diameter data.
    
    Parameters
    ----------
    pupil_data : pd.DataFrame
        DataFrame with 'pupil_diameter_mm' column
    window_size : int
        Size of the moving average window
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'pupil_diameter_smoothed' column
    """
    result = pupil_data.copy()
    result['pupil_diameter_smoothed'] = (
        pupil_data['pupil_diameter_mm']
        .rolling(window=window_size, center=True)
        .mean()
    )
    return result

def detect_blinks(pupil_data: pd.DataFrame, 
                  velocity_threshold: float = 0.5,
                  duration_threshold: int = 3) -> pd.DataFrame:
    """
    Detect potential blink events based on rapid pupil size changes.
    
    Parameters
    ----------
    pupil_data : pd.DataFrame
        DataFrame with 'pupil_diameter_mm' column
    velocity_threshold : float
        Threshold for pupil diameter change rate (mm/frame)
    duration_threshold : int
        Minimum duration for blink detection (frames)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'blink_detected' boolean column
    """
    result = pupil_data.copy()
    
    # Calculate velocity (change in diameter)
    diameter = pupil_data['pupil_diameter_mm']
    velocity = diameter.diff().abs()
    
    # Mark rapid changes
    rapid_changes = velocity > velocity_threshold
    
    # Group consecutive rapid changes
    blink_mask = rapid_changes.rolling(window=duration_threshold).sum() >= duration_threshold
    
    result['blink_detected'] = blink_mask.fillna(False)
    return result

def calculate_statistics(pupil_data: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for pupil diameter data.
    
    Parameters
    ----------
    pupil_data : pd.DataFrame
        DataFrame with 'pupil_diameter_mm' column
        
    Returns
    -------
    dict
        Dictionary containing statistics
    """
    diameter = pupil_data['pupil_diameter_mm'].dropna()
    
    stats = {
        'mean': diameter.mean(),
        'std': diameter.std(),
        'min': diameter.min(),
        'max': diameter.max(),
        'median': diameter.median(),
        'range': diameter.max() - diameter.min(),
        'cv': diameter.std() / diameter.mean() if diameter.mean() != 0 else 0,
        'total_frames': len(pupil_data),
        'valid_frames': len(diameter),
        'missing_frames': len(pupil_data) - len(diameter)
    }
    
    return stats

def export_data(pupil_data: pd.DataFrame, output_path: str, format: str = 'csv'):
    """
    Export pupil data to various formats.
    
    Parameters
    ----------
    pupil_data : pd.DataFrame
        DataFrame to export
    output_path : str
        Path for output file
    format : str
        Output format ('csv', 'excel', 'pickle')
    """
    if format.lower() == 'csv':
        pupil_data.to_csv(output_path)
    elif format.lower() == 'excel':
        pupil_data.to_excel(output_path)
    elif format.lower() == 'pickle':
        pupil_data.to_pickle(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

if __name__ == "__main__":
    # Example usage for testing
    print("Pupil analysis functions loaded successfully!")
    print("Available functions:")
    print("- deeplabcut_pickle()")
    print("- analyze_pupil_data()")
    print("- apply_smoothing_filter()")
    print("- detect_blinks()")
    print("- calculate_statistics()")
    print("- export_data()")
    print("\nFor GUI interface, run: python pupil_gui_viewer.py")
