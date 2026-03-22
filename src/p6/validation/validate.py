import pandas as pd


class ValidationError(Exception):
    pass    


SCHEMAS = {
    'telemetry': {
        'datetime':  'datetime64[ns]',
        'machineID': 'int64',
        'volt':      'float64',
        'rotate':    'float64',
        'pressure':  'float64',
        'vibration': 'float64',
    },
    'machines': {
        'machineID': 'int64',
        'model':     'object',
        'age':       'int64',
    },
    'errors': {
        'datetime':  'datetime64[ns]',
        'machineID': 'int64',
        'errorID':   'object',
    },
    'maintenance': {
        'datetime':  'datetime64[ns]',
        'machineID': 'int64',
        'comp':      'object',
    },
    'failures': {
        'datetime':  'datetime64[ns]',
        'machineID': 'int64',
        'failure':   'object',
    },
}

NOT_NULLABLE = {
    'telemetry':   ['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration'],
    'machines':    ['machineID', 'model', 'age'],
    'errors':      ['datetime', 'machineID', 'errorID'],
    'maintenance': ['datetime', 'machineID', 'comp'],
    'failures':    ['datetime', 'machineID', 'failure'],
}

RANGE_RULES = {
    'telemetry': {
        'volt':      (0, 300),
        'rotate':    (0, 700),
        'pressure':  (0, 200),
        'vibration': (0, 100),
    },
    'machines': {
        'age': (0, 50),
    },
    'errors':      {},
    'maintenance': {},
    'failures':    {},
}

CATEGORY_RULES = {
    'telemetry':   {},
    'machines':    {'model':   ['model1', 'model2', 'model3', 'model4']},
    'errors':      {'errorID': ['error1', 'error2', 'error3', 'error4', 'error5']},
    'maintenance': {'comp':    ['comp1', 'comp2', 'comp3', 'comp4']},
    'failures':    {'failure': ['comp1', 'comp2', 'comp3', 'comp4']},
}



def validate_schema(df: pd.DataFrame, expected_schema: dict) -> None:
    
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValidationError(f'缺少欄位：{missing_cols}')
    
    for col, expected_dtype in expected_schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            raise ValidationError(f'欄位 {col} 的資料型態不正確，期望 {expected_dtype}，實際 {actual_dtype}')


def validate_no_nulls(df: pd.DataFrame, not_nullable: list) -> None:
    errors = []
    
    for col in not_nullable:
        null_sum = df[col].isnull().sum()
        if null_sum > 0:
            errors.append(f'{col} has {null_sum} null values')

    if errors:
        raise ValidationError(f'發現空值：{", ".join(errors)}')


def validate_ranges(df : pd.DataFrame, range_rules:dict[str, tuple[float, float]]) -> None:
    
    errors = []
    
    for col, bounds in range_rules.items():
        min_val, max_val = bounds
        out_of_range_count = (~df[col].between(min_val, max_val)).sum()
        if out_of_range_count > 0:
            errors.append(f'{col} has {out_of_range_count} values out of range')

    if errors:
        raise ValidationError(f'發現超出範圍的值：{", ".join(errors)}')


def validate_category(df: pd.DataFrame, category_rule: dict[str, list]) -> None:
    errors = []
    for col, valid_values in category_rule.items():
        invalid_count = (~df[col].isin(valid_values)).sum()
        if invalid_count > 0:
            errors.append(f'{col} has {invalid_count} invalid category values')

    if errors:
        raise ValidationError(f'發現不合法的類別值：{", ".join(errors)}')



def validate(df_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for name, df in df_dict.items():
        validate_schema(df, SCHEMAS[name])
        validate_no_nulls(df, NOT_NULLABLE[name])
        validate_ranges(df, RANGE_RULES[name])
        validate_category(df, CATEGORY_RULES[name])

    return df_dict