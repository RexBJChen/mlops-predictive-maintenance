import pandas as pd


SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]


def _merge_telemetry_machines(tele : pd.DataFrame, machines : pd.DataFrame) -> pd.DataFrame:
    """把兩張表按照 machineID 合併"""
    df = pd.merge(tele, machines, on = ["machineID"], how = 'left', validate= "m:1")
    return df

def _sort_byID(df) -> pd.DataFrame:
    """把大表按照時間排正確後，之後才能set index 才會是照時序跟機器ID漸增的"""
    df = df.sort_values(["machineID", "datetime"])
    return df

def __compute_rolling(df : pd.DataFrame, window: str, op: str) -> pd.DataFrame:
    """計算大表的四個感測器數據的區間狀態轉換，轉換一次"""
    result = (
        df
        .set_index('datetime')
        .groupby('machineID')[SENSOR_COLS]
        .rolling(window, min_periods=1)
        .agg(op)
        .reset_index()
    )
    result.columns = ['machineID', 'datetime'] + [f'{c}_{op}_{window}' for c in SENSOR_COLS]
    return result

def _build_sensor_feature(df) -> pd.DataFrame:
    """迴圈呼叫四次__compute_rolling，建構3h 24h的四大感測數據的4種區間狀態"""
    
    windows = ['3h', '24h']
    ops = ['mean', 'std']

    for window in windows:
        for op in ops:
            rooled_df = __compute_rolling(df, window, op)
            df = df.merge(rooled_df, on=["machineID", "datetime"], how="left")
    
    return df

def _compute_event_rolling(event : pd.DataFrame, window : str, col: str) -> pd.DataFrame:
    event_dummies =pd.get_dummies(event, columns=[col]).sort_values(["machineID", "datetime"])
    event_cols = [c for c in event_dummies.columns if c.startswith(col)]

    event_rolling = (
        event_dummies
        .set_index('datetime')
        .groupby("machineID")[event_cols]
        .rolling( window,  min_periods=1)
        .sum()
        .reset_index()
    )

    return event_rolling


def _feature_merge_asof_event(feature : pd.DataFrame, event_rolled : pd.DataFrame, time_gap : str):
    
    cols_before = set(feature.columns)
    
    feature = pd.merge_asof(
        feature.sort_values('datetime'),
        event_rolled.sort_values('datetime'),
        on="datetime",
        by="machineID",
        direction="backward",
        tolerance=pd.Timedelta(time_gap)
    )

    new_cols = list(set(feature.columns) - cols_before)
    feature[new_cols] = feature[new_cols].fillna(0)

    return feature


def _build_label(feature : pd.DataFrame, failure : pd.DataFrame) -> pd.DataFrame:
    """比較failure跟feature的時間，如果 feature datetime 當下未來 24 小時有故障就要記 1"""
    failure_renamed = failure.rename(columns={"datetime" : "failuretime"}).sort_values(["machineID", "failuretime"])

    feature = pd.merge_asof(
        feature.sort_values("datetime"),
        failure_renamed[["machineID", "failuretime"]].sort_values("failuretime"),
        by="machineID",
        left_on="datetime",
        right_on="failuretime",
        direction='forward'
    )

    hour_to_failure = (feature["failuretime"]-feature["datetime"]).dt.total_seconds() / 3600

    feature["label"] = (hour_to_failure <= 24).astype(int)


    return feature


def build_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    telemetry   = raw["telemetry"]
    machines    = raw["machines"]
    errors      = raw["errors"]
    maintenance = raw["maintenance"]
    failures    = raw["failures"]

    df = _merge_telemetry_machines(telemetry, machines)
    df = _sort_byID(df)

    df = _build_sensor_feature(df)

    errors_rolling = _compute_event_rolling(errors, '24h', "errorID")
    maint_rolling = _compute_event_rolling(maintenance, '168h', "comp")

    df = _feature_merge_asof_event(df, errors_rolling, '24h')
    df = _feature_merge_asof_event(df, maint_rolling, '168h')

    df = _build_label(df, failures)

    return df