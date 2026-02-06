"""Interactive Plotly figure for mean deltaf/f trace."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

DATASET_PATH = r"C:\Projects\260205_ACUTEVIS.pkl"
SUBJECT_ID = "ACUTEVIS06"
TASK_NAME = "task-movies"
SESSION_IDS = ["ses-01", "ses-02", "ses-03", "ses-04"]
DFF_FRAME_RATE_HZ = 15.0
ENCODER_FRAME_RATE_HZ = 10.0
PUPIL_FRAME_RATE_HZ = 20.0


def _as_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _add_trace(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: str,
    legend: str,
    row: int,
) -> int:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=color),
            legend=legend,
        ),
        row=row,
        col=1,
    )
    return len(fig.data) - 1


def _annotation_texts(subject_id: str, session_id: str) -> list[dict]:
    return [
        {
            "text": f"Mean dF/F Trace ({subject_id}, {session_id}, {TASK_NAME})",
            "x": 0.5,
            "y": 1.0,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 14},
        },
        {
            "text": f"Encoder Speed (mm) ({subject_id}, {session_id}, {TASK_NAME})",
            "x": 0.5,
            "y": 0.64,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 14},
        },
        {
            "text": f"Pupil Diameter (mm) ({subject_id}, {session_id}, {TASK_NAME})",
            "x": 0.5,
            "y": 0.28,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 14},
        },
    ]


def main() -> None:
    dataset = pd.read_pickle(DATASET_PATH)
    mean_index = dataset.suite2p.mean_fluo_dff.index
    task_mask = mean_index.get_level_values(2) == TASK_NAME
    task_index = mean_index[task_mask]
    subjects = sorted(task_index.get_level_values(0).unique())
    sessions = sorted(task_index.get_level_values(1).unique())
    if SESSION_IDS:
        sessions = [session for session in SESSION_IDS if session in sessions]

    default_subject = SUBJECT_ID if SUBJECT_ID in subjects else subjects[0]
    default_session = sessions[0]
    if (default_subject, "ses-01", TASK_NAME) in task_index:
        default_session = "ses-01"

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Mean dF/F Trace ({default_subject}, {default_session}, {TASK_NAME})",
            f"Encoder Speed (mm) ({default_subject}, {default_session}, {TASK_NAME})",
            f"Pupil Diameter (mm) ({default_subject}, {default_session}, {TASK_NAME})",
        ),
    )

    colors = list(pc.qualitative.Plotly)
    color_map = {
        session_id: colors[idx % len(colors)] for idx, session_id in enumerate(SESSION_IDS)
    }

    trace_groups: dict[tuple[str, str], list[int]] = {}

    for subject_id in subjects:
        for session_id in sessions:
            key = (subject_id, session_id, TASK_NAME)
            if key not in task_index:
                continue

            try:
                mean_trace = _as_array(dataset.suite2p.mean_fluo_dff.loc[key])
                injection_label = dataset.loc[key, ("session_config", "injection")]
                speed_trace = _as_array(dataset.encoder.speed_mm.loc[key])
                pupil_trace = _as_array(dataset.pupil.pupil_diameter_mm.loc[key])
            except KeyError:
                continue

            label = str(injection_label) if pd.notna(injection_label) else session_id
            color = color_map.get(session_id, colors[0])
            dff_time_s = np.arange(mean_trace.size) / DFF_FRAME_RATE_HZ
            encoder_time_s = np.arange(speed_trace.size) / ENCODER_FRAME_RATE_HZ
            pupil_time_s = np.arange(pupil_trace.size) / PUPIL_FRAME_RATE_HZ

            group_key = (subject_id, session_id)
            trace_groups.setdefault(group_key, [])

            trace_groups[group_key].append(
                _add_trace(fig, dff_time_s, mean_trace, label, color, "legend", 1)
            )
            trace_groups[group_key].append(
                _add_trace(fig, encoder_time_s, speed_trace, label, color, "legend2", 2)
            )
            trace_groups[group_key].append(
                _add_trace(fig, pupil_time_s, pupil_trace, label, color, "legend3", 3)
            )

    buttons = []
    for subject_id, session_id in trace_groups.keys():
        visible = [False] * len(fig.data)
        for idx in trace_groups[(subject_id, session_id)]:
            visible[idx] = True
        buttons.append(
            {
                "label": f"{subject_id} | {session_id}",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Mean dF/F, Encoder Speed, and Pupil Diameter ({subject_id}, {session_id}, {TASK_NAME})",
                        "annotations": _annotation_texts(subject_id, session_id),
                    },
                ],
            }
        )

    if buttons:
        default_key = (default_subject, default_session)
        if default_key not in trace_groups:
            default_key = next(iter(trace_groups))
        visible = [False] * len(fig.data)
        for idx in trace_groups[default_key]:
            visible[idx] = True
        for trace, is_visible in zip(fig.data, visible):
            trace.visible = is_visible

    fig.update_layout(
        title=f"Mean dF/F, Encoder Speed, and Pupil Diameter ({default_subject}, {default_session}, {TASK_NAME})",
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            }
        ],
        legend=dict(
            title="dF/F",
            x=1.02,
            y=1.0,
            yanchor="top",
        ),
        legend2=dict(
            title="Encoder",
            x=1.02,
            y=0.62,
            yanchor="top",
        ),
        legend3=dict(
            title="Pupil",
            x=1.02,
            y=0.28,
            yanchor="top",
        ),
    )
    fig.update_yaxes(title_text="Mean dF/F", row=1, col=1)
    fig.update_yaxes(title_text="Speed (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Pupil (mm)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.show()


if __name__ == "__main__":
    main()
