from pathlib import Path

base_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent
path = base_path / "ps_2_ex2_smoking_is_bad" "data" / "smoking_data.csv"
print(path)
