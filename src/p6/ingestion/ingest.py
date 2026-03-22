import pandas as pd
from pathlib import Path


def load_raw_data(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """ 讀取原始資料，並輸出一個 DataFrame dict """
    data_dir = Path(data_dir)

    telemetry = pd.read_csv(data_dir/ 'PdM_telemetry.csv', parse_dates=['datetime'])
    machines = pd.read_csv(data_dir / 'PdM_machines.csv')
    errors = pd.read_csv(data_dir / 'PdM_errors.csv', parse_dates=['datetime'])
    maintenance = pd.read_csv(data_dir / 'PdM_maint.csv', parse_dates=['datetime'])
    failures = pd.read_csv(data_dir / 'PdM_failures.csv', parse_dates=['datetime'])

    result = {
    "telemetry" : telemetry,
    "machines" : machines,
    "errors" : errors,
    "maintenance" : maintenance,
    "failures" : failures}
    
   
    return result

