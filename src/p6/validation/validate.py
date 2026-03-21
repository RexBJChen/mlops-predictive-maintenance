import pandas as pd


class ValidationError(Exception):
    pass    


EXPECTED_SCHEMA = {
    'datetime':  'datetime64[ns]',
    'machineID': 'int64',
    'volt':      'float64',
    'rotate':    'float64',
    'pressure':  'float64',
    'vibration': 'float64',
    'model':     'object',
    'age':       'int64',
}

NOT_NULLABLE = ['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration']


RANGE_RULES = {
    'volt':      (0, 300),
    'rotate':    (0, 700),
    'pressure':  (0, 200),
    'vibration': (0, 100),
    'age':       (0, 50),
}

CATEGORY_RULES = {
    'model': ['model1', 'model2', 'model3', 'model4'],
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
    errors =[]
    
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



def validate(df : pd.DataFrame) -> pd.DataFrame:
    validate_schema(df, EXPECTED_SCHEMA)
    validate_no_nulls(df, NOT_NULLABLE)
    validate_ranges(df, RANGE_RULES)
    validate_category(df, CATEGORY_RULES)
    
    return df