import pandas as pd

def merge_csv(input_file1, input_file2, output_file):
    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)
    
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    merged_df.to_csv(output_file, index=False)


input_file1 = 'data/train.csv'
input_file2 = 'data/dev.csv'
output_file = 'data/merged_data.csv'

merge_csv(input_file1, input_file2, output_file)
print("Merge complete.")