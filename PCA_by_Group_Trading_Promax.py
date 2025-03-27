import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = svd(np.dot(
            Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))
        ))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return np.dot(Phi, R)

def promax(loadings, power=4):
    # Step 1: Varimax rotation
    orthogonal = varimax(loadings)
    
    # Step 2: Create target matrix by raising to a power
    target = np.sign(orthogonal) * np.abs(orthogonal) ** power
    
    # Step 3: Regression to find oblique transformation
    U, _, _, _ = np.linalg.lstsq(orthogonal, target, rcond=None)
    
    # Step 4: Apply transformation
    promax_loadings = np.dot(orthogonal, U)
    
    return promax_loadings, U


# Define the groups based on provided sheet classifications
financial_sheets = [
    "Private Debt to GDP", "Money Supply M3", "Money Supply M1", "Money Supply M0",
    "Loans to Private Sector", "Interest Rate", "Interbank Rate", "Government Debt to GDP",
    "Foreign Exchange Reserves", "Deposit Interest Rate", "Current Account to GDP",
    "Central Bank Balance Sheet", "Banks Balance Sheet", "Unemployment Rate",
    "Households Debt to GDP", "Home Ownership Rate", "Government Debt", "Current Account",
    "Corporate Tax Rate", "Consumer Credit", "Bankruptcies", "Bank Lending Rate", "Balance of Trade"
]

nonfinancial_sheets = [
    "wb_historical_australia_stock_m", "Loan Growth", "Wage Growth", "Residential Property Prices",
    "Productivity", "Imports", "House Price Index YoY", "GDP", "Exports", "Business Confidence"
]

people_sheets = [
    "Wages", "Personal Savings", "New Home Sales", "Labour Costs", "Job Vacancies",
    "Employment Change", "Disposable Personal Income", "Consumer Confidence"
]

# Dictionary to map group names to sheets
groups_dict = {
    "financials": financial_sheets,
    "nonfinancials": nonfinancial_sheets,
    "people": people_sheets
}

input_dir = "/Users/xxn/Desktop/Jacklyn's research/data/Australia_Indicators_HistoricalData.xlsx"
output_dir = "/Users/xxn/Desktop/Jacklyn's research/PCA_by_Group_Trading"
os.makedirs(output_dir, exist_ok=True)

xls = pd.ExcelFile(input_dir)
sheets = xls.sheet_names

# Filter out sheets not in the predefined groups
valid_sheets = [sheet for sheet in sheets if sheet in sum(groups_dict.values(), [])]
if not valid_sheets:
    print("No valid sheets found. Exiting...")
    exit()

# Read and merge all valid sheets into one DataFrame
merged_data = None
for sheet in valid_sheets:
    try:
        df = pd.read_excel(input_dir, sheet_name=sheet)
        required_cols = ['DateTime', 'Value']
        if not all(col in df.columns for col in required_cols):
            print(f"Sheet '{sheet}' missing required columns. Skipping sheet.")
            continue

        df = df[['DateTime', 'Value']].rename(columns={'Value': sheet})
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.sort_values('DateTime', inplace=True)

        if merged_data is None:
            merged_data = df
        else:
            merged_data = pd.merge(merged_data, df, on='DateTime', how='outer')
    except Exception as e:
        print(f"Error in sheet '{sheet}': {e}")
        continue

if merged_data is None or merged_data.empty:
    print("No valid data. Exiting...")
    exit()

merged_data.set_index('DateTime', inplace=True)
numeric_cols = merged_data.select_dtypes(include=['number']).columns
if merged_data.isnull().values.any():
    print("Imputing missing values.")
    merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())

# ----------------------------------------
# Run PCA by Group
# ----------------------------------------
for group_name, sheets in groups_dict.items():
    group_cols = [col for col in numeric_cols if col in sheets]
    if not group_cols:
        print(f"No columns match {group_name}. Skipping this group.")
        continue

    group_df = merged_data[group_cols].copy()
    if group_df.shape[0] < 2:
        print(f"Insufficient rows for PCA - group: {group_name}. Skipping...")
        continue

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(group_df)

    pca = PCA()
    pca.fit(scaled_data)
    eigenvalues = pca.explained_variance_
    selected_components = [i + 1 for i, val in enumerate(eigenvalues) if val > 1]
    if not selected_components:
        print(f"No components retained, group={group_name} (Kaiser Rule). Skipping...")
        continue

    pca = PCA(n_components=len(selected_components))
    scores = pca.fit_transform(scaled_data)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotated_loadings, transformation_matrix = promax(loadings)

    # Prepare output folder
    group_output_dir = os.path.join(output_dir, f"{group_name}")
    os.makedirs(group_output_dir, exist_ok=True)

    # Save PCA results
    pd.DataFrame(rotated_loadings, columns=[f"PC{i}" for i in range(1, len(selected_components) + 1)],
                 index=group_cols).to_csv(os.path.join(group_output_dir, "pca_loadings.csv"))
    pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, len(selected_components) + 1)], index=group_df.index).to_csv(os.path.join(group_output_dir, "pca_scores.csv"))

    print(f"PCA completed for group={group_name}. Results saved in {group_output_dir}")

print("All files processed successfully.")