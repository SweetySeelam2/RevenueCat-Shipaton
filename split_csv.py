import pandas as pd

# Set your file path
input_file = r"C:\Users\sweet\Desktop\DataScience\Github projects\Deployment files\Snowflake_Flight_Cancellation\data\flights_large.csv"  # <-- use raw string to fix slashes
output_prefix = 'flights_part_'   # Output files like flights_part_1.csv
rows_per_file = 250000            # Adjust this to control output file size

# Load data in chunks
chunk = pd.read_csv(input_file, chunksize=rows_per_file)

for i, df in enumerate(chunk):
    df.to_csv(f"{output_prefix}{i+1}.csv", index=False)
    print(f"✅ Created: {output_prefix}{i+1}.csv")