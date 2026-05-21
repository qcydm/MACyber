
import pandas as pd
import argparse
from pathlib import Path

def sample_and_combine_csvs(input_dir: str, output_file: str, sample_size: int = 500):
    """
    Recursively finds all CSV files in a directory, samples a specified number of rows from each,
    and combines them into a single new CSV file.

    Args:
        input_dir (str): The path to the directory to search for CSV files.
        output_file (str): The path where the combined CSV file will be saved.
        sample_size (int): The number of rows to sample from each CSV file. 
                           If a file has fewer rows, all rows will be taken.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found or is not a directory.")
        return

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all CSV files recursively
    csv_files = list(input_path.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting sampling...")

    sampled_dfs = []
    for file in csv_files:
        try:
            print(f"Processing '{file}'...")
            df = pd.read_csv(file, low_memory=False)
            
            if df.empty:
                print(f"  -> Skipping empty file.")
                continue

            # Determine the number of rows to sample
            n_samples = min(len(df), sample_size)
            
            # Sample the dataframe
            sampled_df = df.sample(n=n_samples, random_state=42) # Using a fixed random state for reproducibility
            sampled_dfs.append(sampled_df)
            print(f"  -> Sampled {len(sampled_df)} rows.")

        except Exception as e:
            print(f"  -> Error processing file '{file}': {e}")

    if not sampled_dfs:
        print("No data was sampled. The output file will not be created.")
        return

    # Combine all sampled dataframes
    print("\nCombining all sampled data...")
    combined_df = pd.concat(sampled_dfs, ignore_index=True)

    # Save the combined dataframe to the output file
    combined_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully created combined CSV file with {len(combined_df)} rows at '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample rows from multiple CSV files within a directory and combine them into a single CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing the source CSV files."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        required=True,
        help="Path to the destination file for the combined CSV."
    )
    parser.add_argument(
        "-s", "--sample-size",
        type=int,
        default=500,
        help="Number of rows to sample from each CSV file (default: 500)."
    )

    args = parser.parse_args()

    sample_and_combine_csvs(args.input_dir, args.output_file, args.sample_size)

# --- HOW TO RUN ---
#
# Open your terminal and run the script with the following command:
#
# python sample_csv.py --input-dir /path/to/your/source/csvs --output-file /path/to/your/output/combined.csv
#
# Example:
# python sample_csv.py --input-dir ./raw_csv --output-file ./sampled_data/all_data_sampled.csv --sample-size 500
#
