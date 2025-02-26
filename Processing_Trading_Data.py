import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define input and output directories
input_dir = "/Users/xxn/Desktop/Jacklyn's research/data2"
output_dir = "/Users/xxn/Desktop/Jacklyn's research/Trading_data_after_processing"
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    if file_name.endswith(".xlsx"):
        country_name = file_name.replace(".xlsx", "")
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {country_name}...")
        
        try:
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue
        
        # Only use sheets that do not start with "wb" (case-insensitive)
        valid_sheets = [sheet for sheet in sheets if not sheet.lower().startswith("wb")]
        if not valid_sheets:
            print(f"No valid sheets found for {country_name}. Skipping...")
            continue
        
        merged_data = None
        for sheet in valid_sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                required_cols = ['DateTime', 'Value']
                if not all(col in df.columns for col in required_cols):
                    print(f"Sheet '{sheet}' in {country_name} missing required columns. Skipping sheet.")
                    continue
                # Rename 'Value' column to sheet name for identification
                df = df[['DateTime', 'Value']].rename(columns={'Value': sheet})
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.sort_values('DateTime', inplace=True)
                
                if merged_data is None:
                    merged_data = df
                else:
                    merged_data = pd.merge(merged_data, df, on='DateTime', how='outer')
            except Exception as e:
                print(f"Error in sheet '{sheet}' for {country_name}: {e}")
                continue
        
        if merged_data is None or merged_data.empty:
            print(f"No valid data for {country_name}. Skipping...")
            continue
        
        merged_data.set_index('DateTime', inplace=True)
        numeric_cols = merged_data.select_dtypes(include=['number']).columns
        if merged_data.isnull().values.any():
            print(f"Imputing missing values for {country_name}.")
            merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())
        
        if merged_data.shape[0] < 2:
            print(f"Insufficient data for PCA in {country_name}. Skipping...")
            continue
        
        # Standardize the numeric data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(merged_data[numeric_cols])
        
        # Initial PCA based on Kaiser criterion (eigenvalue > 1)
        pca = PCA()
        pca.fit(scaled_data)
        eigenvalues = pca.explained_variance_
        selected_components = [i+1 for i, val in enumerate(eigenvalues) if val > 1]
        if not selected_components:
            print(f"No components retained for {country_name} (Kaiser Rule). Skipping...")
            continue
        
        pca = PCA(n_components=len(selected_components))
        scores = pca.fit_transform(scaled_data)
        loadings = pca.components_
        
        # Create a DataFrame for loadings
        loadings_df = pd.DataFrame(loadings, columns=numeric_cols, 
                                   index=[f"PC{i}" for i in range(1, len(selected_components)+1)])
        loadings_df = loadings_df.transpose()
        
        # Sort loadings in descending order and threshold values below 0.4 to 0
        sorted_loadings_dict = {}
        for pc in loadings_df.columns:
            pc_series = loadings_df[pc].copy()
            pc_series = pc_series.reindex(pc_series.abs().sort_values(ascending=False).index)
            pc_series = pc_series.apply(lambda x: x if abs(x) >= 0.4 else 0)
            sorted_loadings_dict[pc] = pc_series
        
        sorted_loadings_df = pd.DataFrame(sorted_loadings_dict)
        
        # Create output directory for current country
        pca_output_dir = os.path.join(output_dir, country_name)
        os.makedirs(pca_output_dir, exist_ok=True)
        
        # Save sorted and thresholded loadings
        sorted_loadings_df.to_csv(os.path.join(pca_output_dir, "pca_loadings_sorted_thresholded.csv"))
        
        # Save PCA scores from initial PCA
        scores_df = pd.DataFrame(scores, 
                                 columns=[f"PC{i}" for i in range(1, len(selected_components)+1)],
                                 index=merged_data.index)
        scores_df.to_csv(os.path.join(pca_output_dir, "pca_scores.csv"))
        
        # Generate a scatter plot for PC1 vs PC2 if at least 2 components are available
        if len(selected_components) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(scores_df["PC1"], scores_df["PC2"], color='blue', alpha=0.5)
            for idx, row in scores_df.iterrows():
                plt.text(row["PC1"], row["PC2"], str(idx.date()), fontsize=8)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"PCA Score Plot for {country_name}")
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(pca_output_dir, "pca_score_plot.png"))
            plt.close()
        
        # Write PCA summary into a text file
        with open(os.path.join(pca_output_dir, "PCA_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"PCA Summary for {country_name}\n")
            f.write("="*40 + "\n\n")
            f.write("Eigenvalues:\n")
            for i, val in enumerate(eigenvalues, start=1):
                f.write(f"PC{i}: {val:.4f}\n")
            f.write("\nComponents retained (Kaiser Rule):\n")
            f.write(f"{selected_components}\n")
            f.write(f"Number of components retained: {len(selected_components)}\n\n")
            f.write("Explained Variance Ratio:\n")
            for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
                f.write(f"PC{i}: {ratio*100:.2f}%\n")
        
        # --------------------- Additional Section for Composite Stress Index ---------------------
        # If there are enough numeric variables, perform an 8-dimensional PCA to construct a composite stress index.
        if len(numeric_cols) >= 8:
            # Force PCA to extract 8 principal components
            pca8 = PCA(n_components=8)
            scores8 = pca8.fit_transform(scaled_data)
            scores8_df = pd.DataFrame(scores8, 
                                      columns=[f"PC{i}" for i in range(1, 9)],
                                      index=merged_data.index)
            
            # Apply a Cumulative Distribution Function (CDF) transformation to each principal component score,
            # mapping the values to a 0-100 scale. Higher original values receive higher percentile ranks.
            cdf_scores_df = scores8_df.copy()
            for col in scores8_df.columns:
                cdf_scores_df[col] = scores8_df[col].rank(pct=True) * 100
            
            # Compute a composite weighted score using the explained variance ratio of each component as weights
            weights8 = pca8.explained_variance_ratio_
            cdf_scores_df["Composite_Weighted_Score"] = sum(
                cdf_scores_df[f"PC{i}"] * weights8[i-1] for i in range(1, 9)
            )
            
            # Final Stress Index = 100 - Composite Weighted Score (ensuring 0 means lower stress, lower risk)
            cdf_scores_df["Final_Stress_Index"] = 100 - cdf_scores_df["Composite_Weighted_Score"]
            
            # Save the composite stress index results to a CSV file
            final_csv_path = os.path.join(pca_output_dir, "final_stress_index.csv")
            cdf_scores_df.to_csv(final_csv_path)
            
            # Generate a line plot for the Final Stress Index over time
            plt.figure(figsize=(10, 6))
            plt.plot(cdf_scores_df.index, cdf_scores_df["Final_Stress_Index"], marker='o', linestyle='-')
            plt.title("Final Stress Index Over Time")
            plt.xlabel("DateTime")
            plt.ylabel("Final Stress Index")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(pca_output_dir, "final_stress_index_plot.png"))
            plt.close()
            
            # Write an explanation file describing the calculation process and field definitions
            explanation_text = (
                "Final Stress Index Explanation:\n"
                "--------------------------------\n"
                "This file is generated based on an 8-dimensional PCA to construct a composite stress index.\n"
                "1. The original numeric data is standardized and an 8-component PCA is performed.\n"
                "2. Each principal component score is transformed using a Cumulative Distribution Function (CDF),\n"
                "   mapping the values to a 0-100 scale. Higher CDF values indicate a higher level of the indicator\n"
                "   (implying higher profitability potential for the bank).\n"
                "3. The explained variance ratio of each principal component is used as a weight to compute a\n"
                "   composite weighted score from the CDF-transformed scores.\n"
                "4. To ensure that a lower stress index represents lower risk (i.e., 0 indicates low stress),\n"
                "   the final stress index is calculated as 100 - composite weighted score.\n"
                "Note: The composite stress index is the cumulative result across 8 dimensions; when ranking,\n"
                "a lower value indicates lower risk."
            )
            explanation_file_path = os.path.join(pca_output_dir, "final_stress_index_explanation.txt")
            with open(explanation_file_path, "w", encoding="utf-8") as f:
                f.write(explanation_text)
                
            print(f"Final stress index computed and saved for {country_name}.\n")
        else:
            print(f"Not enough numeric variables to run 8-dimension PCA for {country_name}.\n")
        # --------------------- End Additional Section ---------------------
        
        print(f"PCA completed for {country_name}. Results saved in {pca_output_dir}\n")

print("All files processed successfully.")
