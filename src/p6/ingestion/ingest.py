import pandas as pd
from pathlib import Path


def load_raw_data(data_dir: Path | str) -> pd.DataFrame:
    """ 讀取原始資料，並合併成一個 DataFrame """
    data_dir = Path(data_dir)
    telemetry = pd.read_csv(data_dir/ 'PdM_telemetry.csv', parse_dates=['datetime'])
    machines = pd.read_csv(data_dir / 'PdM_machines.csv')

    result = pd.merge(telemetry, machines, on="machineID", how="left", validate = "m:1")

    assert result.shape[0] == telemetry.shape[0], "合併後的資料筆數不正確"
    assert result[['model', 'age']].isnull().sum().sum() == 0, "合併後的資料有缺值"
    
    return result

