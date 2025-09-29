import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'threshold_ic')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    domain_sizes = []
    outer_diversities = []

    for file in csv_files:
        df = pd.read_csv(file)
        # Try to extract domain size and outer diversity from columns
        # Assume columns: 'domain_size', 'outer_diversity' or similar
        if 'domain_size' in df.columns and 'total_cost' in df.columns:
            domain_sizes.extend(df['domain_size'].values)
            outer_diversities.extend(df['total_cost'].values)
        else:
            # Try to infer from filename or other columns
            for col in df.columns:
                if 'domain' in col:
                    domain_sizes.extend(df[col].values)
                if 'diversity' in col:
                    outer_diversities.extend(df[col].values)

    print(f"Loaded {len(domain_sizes)} data points.")
    print(outer_diversities)

    plt.figure(figsize=(10, 6))
    plt.scatter(domain_sizes, outer_diversities, alpha=0.7)
    plt.xlabel('Domain Size', fontsize=32)
    plt.ylabel('Outer Diversity', fontsize=32)
    # plt.title('Scatter Plot of Outer Diversity vs Domain Size', fontsize=20)

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/threshold_ic_scatter.png', dpi=300)
    plt.show()

