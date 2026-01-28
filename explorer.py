"""Interactive Plotly figure for mean deltaf/f trace."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

DATASET_PATH = r"C:\Projects\260122_ACUTEVIS.pkl"
SUBJECT_ID = "ACUTEVIS14"
TASK_NAME = "task-movies"
SESSION_IDS = ["ses-01", "ses-02", "ses-03", "ses-04"]
DFF_FRAME_RATE_HZ = 10.0
ENCODER_FRAME_RATE_HZ = 10.0
PUPIL_FRAME_RATE_HZ = 20.0


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    mean_index = dataset.suite2p.mean_fluo_dff.index
    task_mask = mean_index.get_level_values(2) == TASK_NAME
    task_index = mean_index[task_mask]
    subjects = sorted(task_index.get_level_values(0).unique())
    sessions = sorted(task_index.get_level_values(1).unique())

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
                mean_trace = np.asarray(
                    dataset.suite2p.mean_fluo_dff.loc[key],
                    dtype=float,
                )
                injection_label = dataset.loc[
                    key,
                    ("session_config", "injection"),
                ]
                speed_trace = np.asarray(
                    dataset.encoder.speed_mm.loc[key],
                    dtype=float,
                )
                pupil_trace = np.asarray(
                    dataset.pupil.pupil_diameter_mm.loc[key],
                    dtype=float,
                )
            except KeyError:
                continue

            label = str(injection_label) if pd.notna(injection_label) else session_id
            color = color_map.get(session_id, colors[0])
            dff_time_s = np.arange(mean_trace.size) / DFF_FRAME_RATE_HZ
            encoder_time_s = np.arange(speed_trace.size) / ENCODER_FRAME_RATE_HZ
            pupil_time_s = np.arange(pupil_trace.size) / PUPIL_FRAME_RATE_HZ

            group_key = (subject_id, session_id)
            trace_groups[group_key] = []

            fig.add_trace(
                go.Scatter(
                    x=dff_time_s,
                    y=mean_trace,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                    legend="legend",
                ),
                row=1,
                col=1,
            )
            trace_groups[group_key].append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=encoder_time_s,
                    y=speed_trace,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                    legend="legend2",
                ),
                row=2,
                col=1,
            )
            trace_groups[group_key].append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=pupil_time_s,
                    y=pupil_trace,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                    legend="legend3",
                ),
                row=3,
                col=1,
            )
            trace_groups[group_key].append(len(fig.data) - 1)

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
                        "annotations": [
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
                        ],
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
