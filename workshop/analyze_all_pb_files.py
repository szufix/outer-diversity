import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import re
import os
import glob

def load_poland_krakow_data(filepath):
    """Load the Poland Krakow 2023 PB data from .pb file"""

    # Read the file and find where PROJECTS section starts
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the start of PROJECTS section
    projects_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'PROJECTS':
            projects_start = i + 1  # Skip the header line
            break

    if projects_start is None:
        raise ValueError("PROJECTS section not found in file")

    # Get the header line and data lines
    header_line = lines[projects_start].strip()
    data_lines = lines[projects_start + 1:]

    # Parse header
    columns = [col.strip() for col in header_line.split(';')]

    # Parse data
    data = []
    for line in data_lines:
        line = line.strip()
        if line and not line.startswith('VOTES'):  # Stop at VOTES section if it exists
            row = [col.strip() for col in line.split(';')]
            if len(row) == len(columns):
                data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Convert numeric columns
    df['cost'] = pd.to_numeric(df['cost'])
    df['votes'] = pd.to_numeric(df['votes'])
    df['score'] = pd.to_numeric(df['score'])
    df['selected'] = pd.to_numeric(df['selected'])

    return df

def extract_project_number(project_id):
    """Extract numeric part from project ID (e.g., 'BO.OM.195/23' -> 195)"""
    match = re.search(r'\.(\d+)/', project_id)
    if match:
        return int(match.group(1))
    return None

def analyze_single_file(filepath):
    """Analyze a single .pb file and return correlation results"""
    try:
        df = load_poland_krakow_data(filepath)

        # Extract numeric project IDs
        df['project_number'] = df['project_id'].apply(extract_project_number)

        # Remove rows where we couldn't extract a project number
        valid_df = df[df['project_number'].notna()].copy()

        if len(valid_df) < 3:  # Need at least 3 points for correlation
            return None

        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(valid_df['project_number'], valid_df['votes'])
        spearman_corr, spearman_p = spearmanr(valid_df['project_number'], valid_df['votes'])

        return {
            'filename': os.path.basename(filepath),
            'total_projects': len(df),
            'valid_projects': len(valid_df),
            'total_votes': df['votes'].sum(),
            'avg_votes': df['votes'].mean(),
            'median_votes': df['votes'].median(),
            'selected_projects': df['selected'].sum(),
            'id_range_min': valid_df['project_number'].min(),
            'id_range_max': valid_df['project_number'].max(),
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'dataframe': valid_df
        }

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def analyze_all_pb_files(pb_directory):
    """Analyze all .pb files in the given directory"""

    # Find all .pb files
    pb_files = glob.glob(os.path.join(pb_directory, "*.pb"))

    if not pb_files:
        print(f"No .pb files found in {pb_directory}")
        return

    print(f"Found {len(pb_files)} .pb files:")
    for file in pb_files:
        print(f"  - {os.path.basename(file)}")
    print("\n" + "=" * 80)

    results = []

    for filepath in pb_files:
        print(f"\nAnalyzing: {os.path.basename(filepath)}")
        print("-" * 60)

        result = analyze_single_file(filepath)

        if result is None:
            print("Failed to analyze this file")
            continue

        results.append(result)

        # Print detailed results for this file
        print(f"Dataset: {result['filename']}")
        print(f"Total projects: {result['total_projects']}")
        print(f"Projects with extractable IDs: {result['valid_projects']}")
        print(f"Total votes: {result['total_votes']:,}")
        print(f"Average votes per project: {result['avg_votes']:.1f}")
        print(f"Median votes per project: {result['median_votes']:.1f}")
        print(f"Selected projects: {result['selected_projects']}")
        print(f"Project ID range: {result['id_range_min']} - {result['id_range_max']}")

        print(f"\nCorrelation Analysis:")
        print(f"PCC (Pearson Correlation Coefficient): {result['pearson_corr']:.4f} (p-value: {result['pearson_p']:.4f})")
        print(f"Spearman correlation: {result['spearman_corr']:.4f} (p-value: {result['spearman_p']:.4f})")

        if result['pearson_corr'] < 0:
            print("→ Negative correlation: projects with smaller IDs tend to receive MORE votes")
        else:
            print("→ Positive correlation: projects with smaller IDs tend to receive FEWER votes")

        # Show top and bottom projects for this dataset
        valid_df = result['dataframe']
        print(f"\nTop 5 projects by votes:")
        top_by_votes = valid_df.nlargest(5, 'votes')[['project_id', 'project_number', 'votes', 'name']]
        for _, row in top_by_votes.iterrows():
            print(f"  ID: {row['project_number']:3d} | Votes: {row['votes']:5d} | {row['name'][:40]}...")

        print(f"\nBottom 5 projects by votes:")
        bottom_by_votes = valid_df.nsmallest(5, 'votes')[['project_id', 'project_number', 'votes', 'name']]
        for _, row in bottom_by_votes.iterrows():
            print(f"  ID: {row['project_number']:3d} | Votes: {row['votes']:5d} | {row['name'][:40]}...")

    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON ACROSS ALL DATASETS")
        print("=" * 80)

        print(f"{'Dataset':<40} {'Projects':<10} {'PCC':<8} {'p-value':<10} {'Interpretation'}")
        print("-" * 90)

        for result in results:
            significance = "***" if result['pearson_p'] < 0.001 else "**" if result['pearson_p'] < 0.01 else "*" if result['pearson_p'] < 0.05 else ""
            interpretation = "Smaller IDs → MORE votes" if result['pearson_corr'] < 0 else "Smaller IDs → FEWER votes"

            print(f"{result['filename']:<40} {result['valid_projects']:<10} {result['pearson_corr']:<8.4f} {result['pearson_p']:<10.4f} {interpretation} {significance}")

        # Overall statistics
        all_pcc = [r['pearson_corr'] for r in results]
        significant_negative = sum(1 for r in results if r['pearson_corr'] < 0 and r['pearson_p'] < 0.05)
        significant_positive = sum(1 for r in results if r['pearson_corr'] > 0 and r['pearson_p'] < 0.05)

        print(f"\nOverall Summary:")
        print(f"  Mean PCC across datasets: {np.mean(all_pcc):.4f}")
        print(f"  Datasets with significant negative correlation (smaller ID → more votes): {significant_negative}/{len(results)}")
        print(f"  Datasets with significant positive correlation (smaller ID → fewer votes): {significant_positive}/{len(results)}")

        if significant_negative > significant_positive:
            print(f"  → CONCLUSION: Projects with smaller IDs tend to receive MORE votes across datasets")
        elif significant_positive > significant_negative:
            print(f"  → CONCLUSION: Projects with smaller IDs tend to receive FEWER votes across datasets")
        else:
            print(f"  → CONCLUSION: Mixed results - no clear pattern across datasets")

if __name__ == "__main__":
    pb_directory = "/Users/szufa/PyCharmProjects/outer-diversity/workshop/pb"
    analyze_all_pb_files(pb_directory)
