"""M8 — Data drift monitoring using Evidently."""

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    drift_threshold: int = 5,
) -> dict:
    """Compare reference and current data distributions for drift.

    Parameters
    ----------
    reference : DataFrame
        Training-time feature data (the "normal" baseline).
    current : DataFrame
        Incoming feature data to check against reference.
    columns : list[str]
        Which feature columns to monitor.
    drift_threshold : int
        Number of drifted columns required to flag drift_detected=True.

    Returns
    -------
    dict with keys: drift_detected, drifted_count, drifted_share, drifted_columns.
    """
    num_cols = [c for c in columns if c != "model"]
    cat_cols = [c for c in columns if c == "model"]
    data_def = DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )

    ref_ds = Dataset.from_pandas(reference[columns], data_definition=data_def)
    cur_ds = Dataset.from_pandas(current[columns], data_definition=data_def)

    report = Report([DataDriftPreset(columns=columns)])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)
    result_dict = result.dict()

    summary = result_dict["metrics"][0]
    drifted_count = int(summary["value"]["count"])
    drifted_share = summary["value"]["share"]

    drifted_columns = []
    for m in result_dict["metrics"][1:]:
        col = m["config"]["column"]
        threshold = m["config"]["threshold"]
        score = m["value"]
        if score >= threshold:
            drifted_columns.append(col)

    return {
        "drift_detected": drifted_count >= drift_threshold,
        "drifted_count": drifted_count,
        "drifted_share": drifted_share,
        "drifted_columns": drifted_columns,
    }
