import csv
import math
import sys
import os


def normalize_total_cost(filename, num_candidates):
    # Construct input and output paths
    input_dir = os.path.join(os.path.dirname(__file__), 'data', 'optimal_nodes')
    csv_path = os.path.join(input_dir, filename)
    base, ext = os.path.splitext(filename)
    out_path = os.path.join(input_dir, f"{base}{ext}")

    # Read all rows
    with open(csv_path, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    n = math.factorial(num_candidates)
    comb = math.comb(num_candidates, 2)

    # Normalize total_cost for each row
    for row in rows:
        try:
            cost = float(row['total_cost'])
            norm_cost = 1 - cost / n * 2 / comb
            row['total_cost'] = norm_cost
        except Exception as e:
            print(f"Warning: Could not normalize row {row}: {e}")

    # Write to output file
    with open(out_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Normalized file written to {out_path}")

if __name__ == "__main__":

    num_candidates = 6
    method = 'smpl_sa'
    filename = f'{method}_m{num_candidates}.csv'
    normalize_total_cost(filename, num_candidates)
