import pandas as pd

try:
    xl = pd.ExcelFile("data/nasa/spatialecon-lgii-measures-v1-xlsx/spatialecon-lgii-measures-v1-xlsx.xlsx")
    print("Sheet names:", xl.sheet_names)
    for sheet in xl.sheet_names:
        print(f"\n--- Sheet: {sheet} ---")
        df = pd.read_excel(xl, sheet_name=sheet)
        print("First 10 rows:\n", df.head(10))
except Exception as e:
    print(e)
