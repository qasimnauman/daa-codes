import pandas as pd
import time
from itertools import product
from multiprocessing import Pool

# Function to compute LCS length using dynamic programming
def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# Function to read and load sequences from a CSV file
def load_data(file_path, column_index=1):
    """Loads sequences from a CSV file."""
    try:
        df = pd.read_csv(file_path, header=None)
        sequences = df.iloc[:, column_index].tolist()
        return sequences
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

# Compute LCS for a pair of sequences
def compute_lcs_pair(args):
    i, j, seq1, seq2 = args
    print(f"Computing LCS for Sequence {i+1} and Sequence {j+1}...")
    start_time = time.time()
    lcs_value = lcs_length(seq1, seq2)
    elapsed_time = round(time.time() - start_time, 5)
    return [f"Seq{i+1}", f"Seq{j+1}", lcs_value, elapsed_time]

# Compute the combined LCS table using multiprocessing
def compute_combined_table_multiprocessing(sequences, num_workers):
    """Computes LCS for all sequence pairs using multiprocessing."""
    tasks = [(i, j, sequences[i], sequences[j]) for i, j in product(range(len(sequences)), repeat=2)]
    combined_data = []
    total_tasks = len(tasks)

    with Pool(processes=num_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(compute_lcs_pair, tasks)):
            combined_data.append(result)
            if idx % (total_tasks // 10) == 0:  # Log every 10% of progress
                print(f"Progress: {round((idx / total_tasks) * 100, 2)}%")

    return combined_data

# Save results to a CSV file
def save_results_to_csv(data, output_file):
    """Saves LCS results to a CSV file."""
    df = pd.DataFrame(data, columns=["Sequence 1", "Sequence 2", "LCS Value", "Time (s)"])
    try:
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving results: {e}")
        exit(1)

# Driver function to execute the pipeline
def main(input_file, output_file, num_workers=8):
    print("Loading data...")
    sequences = load_data(input_file)

    print("Computing LCS table using multiprocessing...")
    total_start_time = time.time()

    combined_data = compute_combined_table_multiprocessing(sequences, num_workers)

    print("Saving results to CSV...")
    save_results_to_csv(combined_data, output_file)

    total_time = round(time.time() - total_start_time, 2)
    print(f"Results successfully saved to '{output_file}'.")
    print(f"Total Execution Time: {total_time} seconds")

if __name__ == "__main__":
    INPUT_FILE = "Assignment_LCS_Data.csv"  # Replace with actual file path
    OUTPUT_FILE = "combined_lcs_time_table.csv"
    NUM_WORKERS = 16  # Adjust based on CPU availability

    main(INPUT_FILE, OUTPUT_FILE, NUM_WORKERS)
