import pandas as pd
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Read the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)  # No header in file
    sequences = df.iloc[:, 1].tolist()  # Load the second column (index 1)
    return sequences

# Function to compute LCS for a pair of sequences
def compute_lcs_pair(args):
    i, j, seq1, seq2 = args
    print(f"Computing Seq{i+1} and Seq{j+1}...")  # Log progress
    start_time = time.time()
    lcs_value = lcs_length(seq1, seq2)
    end_time = time.time()
    computation_time = round(end_time - start_time, 5)
    return [f"Seq{i+1}", f"Seq{j+1}", lcs_value, computation_time]

# Compute combined table with LCS and Time using parallel processing
def compute_combined_table_parallel(sequences, num_workers):
    combined_data = []
    tasks = []

    # Prepare tasks: pair all sequences
    n = len(sequences)
    for i, j in product(range(n), repeat=2):
        tasks.append((i, j, sequences[i], sequences[j]))

    # Process tasks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(compute_lcs_pair, task): task for task in tasks}
        for future in as_completed(future_to_task):
            result = future.result()
            combined_data.append(result)

    return combined_data

# Save results to a single CSV file
def save_combined_results_to_csv(combined_data, output_file):
    df = pd.DataFrame(combined_data, columns=["Sequence 1", "Sequence 2", "LCS Value", "Time (s)"])
    df.to_csv(output_file, index=False)

# Driver Code
if __name__ == "__main__":
    input_file = "Assignment_LCS_Data.csv"  # Replace with actual file path
    output_file = "combined_lcs_time_table.csv"

    total_start_time = time.time()  # Start logging total execution time

    print("Loading data...")
    sequences = load_data(input_file)

    print("Computing combined table using parallel processing...")
    num_workers = 16  # Number of CPUs to use
    combined_data = compute_combined_table_parallel(sequences, num_workers)

    print("Saving results to CSV...")
    save_combined_results_to_csv(combined_data, output_file)

    total_end_time = time.time()  # End logging total execution time
    total_time = round(total_end_time - total_start_time, 2)  # Total execution time in seconds

    print(f"Results saved successfully to {output_file}!")
    print(f"Total Execution Time: {total_time} seconds")
