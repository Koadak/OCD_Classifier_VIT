import numpy as np
import pandas as pd
from scipy.io import loadmat

# ---------- Load Helper ----------
def load_and_print_mat(file_path, key='X'):
    data = loadmat(file_path)
    print(f"\nLoaded keys from {file_path}:\n", data.keys())
    if key in data:
        X = data[key]
        print("Type:", type(X))
        print("Shape:", X.shape)
        # print("Snippet:", X if X.shape[0] < 10 else X[:2])
        return X
    else:
        print(f"Key '{key}' not found in file.")
        return None

# ---------- Dataset Definitions ----------
datasets = {
    "ATR_OCD": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\ATR\OCD\BAL-ATR_2OCDPatients.mat', 'mat_1D'),
    "UTO_MDD": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\UTO\UTO_GEMR750W_30_BAL\UTO_GEMR750W_30_02_BAL_COR.mat', 'X'),
    "UTO_SCZ": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\UTO\UTO_GEMR750W_30_BAL\UTO_GEMR750W_30_04_BAL_COR.mat', 'X'),
    "KPUM_OCD": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\KPUM\BAL-KPUM_9OCDPatients.mat', 'X'),
    "KyotoU_DEP": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\KyotoU\DEP\BAL-KyotoU_UKY_DEP_16_BAL_BP39_COR.mat', 'X'),
    "KyotoU_HC": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\KyotoU\HC\BAL-KyotoU_UKY_HC_234_BAL_BP39_COR.mat', 'X'),
    "KyotoU_SCZ": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\KyotoU\SSD\BAL-KyotoU_UKY_SSD_92_BAL_BP39_COR.mat', 'X'),
    "UTO_HC": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\UTO\UTO_GEMR750W_30_BAL\UTO_GEMR750W_30_00_BAL_COR.mat', 'X'),
    "ATR_HC": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\ATR\HC\BAL-ATR_BP39_COR.mat', 'X'),
    "ShowaU": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\ShowaU\SWA_FC_Matrices_N235.mat', 'FCMat'),
    "HiroshimaU": (r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\SRPBS_FC\HiroshimaU\DEP_BAL-UHIBP39_COR.mat', 'X')
}

# ---------- Load All Standard .mat Data ----------
loaded_data = {name: load_and_print_mat(path, key) for name, (path, key) in datasets.items()}

# ---------- Load and Label Oblique OCD Data ----------
oblique_path = r'/SRPBS_FC/ATR/OCD-oblique/BAL-ATR_2OCDPatients_oblique.mat'
ATR_OCD_Oblique = np.loadtxt(oblique_path, comments='#')
print("\nATR_OCD_Oblique shape:", ATR_OCD_Oblique.shape)

oblique_df = pd.DataFrame(ATR_OCD_Oblique)
oblique_df['disorder'] = 'OCD'

# ---------- Process ShowaU Diagnosis ----------
showa_df = pd.read_csv(r'/SRPBS_FC/ShowaU/SUBINFO_ShowaU_for_SWA_N235.tsv', sep='\t')
assert loaded_data["ShowaU"].shape[0] == len(showa_df), "Mismatch in ShowaU"
showa_diag = showa_df['Diagnosis'].values

ShowaU_HC = pd.DataFrame(loaded_data["ShowaU"][showa_diag == 0])
ShowaU_HC['disorder'] = 'HC'

ShowaU_SCZ = pd.DataFrame(loaded_data["ShowaU"][showa_diag == 2])
ShowaU_SCZ['disorder'] = 'SCZ'

# ---------- Process HiroshimaU Diagnosis ----------
hiro_df = pd.read_csv(r'/SRPBS_FC/HiroshimaU/SUBINFO_HiroshimaU.tsv', sep='\t')
assert loaded_data["HiroshimaU"].shape[0] == len(hiro_df), "Mismatch in HiroshimaU"
hiro_diag = hiro_df['Diagnosis(Healthy=1, Depression=2)'].values

HiroshimaU_HC = pd.DataFrame(loaded_data["HiroshimaU"][hiro_diag == 1])
HiroshimaU_HC['disorder'] = 'HC'

Hiroshima_DEP = pd.DataFrame(loaded_data["HiroshimaU"][hiro_diag == 2])
Hiroshima_DEP['disorder'] = 'MDD/DEP'

# ---------- Helper: Infer disorder from dataset name ----------
def infer_disorder(name):
    name = name.lower()
    if 'hc' in name:
        return 'HC'
    elif 'scz' in name:
        return 'SCZ'
    elif 'mdd' in name or 'dep' in name:
        return 'MDD/DEP'
    elif 'ocd' in name:
        return 'OCD'
    else:
        return 'Unknown'

# ---------- Label and Convert Remaining Datasets ----------
other_dfs = []
exclude = ['ShowaU', 'HiroshimaU']  # already handled
for name, data in loaded_data.items():
    if name in exclude or data is None:
        continue
    df = pd.DataFrame(data)
    df['disorder'] = infer_disorder(name)
    other_dfs.append(df)

# ---------- Combine All ----------
final_df = pd.concat([
    oblique_df,
    ShowaU_HC,
    ShowaU_SCZ,
    HiroshimaU_HC,
    Hiroshima_DEP
] + other_dfs, ignore_index=True)

# Add a binary flag for whether the row is HC or not
final_df['is_HC'] = (final_df['disorder'] == 'HC').astype(int)

# ---------- Summary ----------
print("\nFinal dataset shape:", final_df.shape)
print("Disorder counts:\n", final_df['disorder'].value_counts())
print(final_df.head())
print(final_df.shape)


rows_with_nan = final_df.isna().any(axis=1).sum()
print(f"Number of rows with at least one NaN: {rows_with_nan}")

rows_with_nan_df = final_df[final_df.isna().any(axis=1)]
# print(rows_with_nan_df)
print(f"Original shape: {final_df.shape}")


#Drop nans
final_df.dropna(inplace=True)

print(f"New shape after dropping NaNs: {final_df.shape}")

print(len(final_df[final_df['disorder'] == 'OCD']))
# output_path = r'C:\Users\jgdga\PycharmProjects\OCD_Tester2\final_fc_dataset.csv'
# final_df.to_csv(output_path, index=False)
# print(f"Saved to: {output_path}")
