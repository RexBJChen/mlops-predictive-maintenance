from p6.orchestration import retrain_flow


def retrain_trigger(drift_report : dict, data_dir : str) -> bool:
    drifted_check = drift_report["drift_detected"]
    
    if drifted_check:
        retrain_flow(data_dir)
        return True
    else:
        return False
    
