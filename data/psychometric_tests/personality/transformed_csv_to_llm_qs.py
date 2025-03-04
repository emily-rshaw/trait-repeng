import pandas as pd

def format_dataset(input_file='ipip_neo_120_formatted.csv', output_file='ipip_neo_120_formatted_new.csv'):
    """
    Reads the IPIP NEO dataset, adds a 'test' column with formatted instructions
    and an empty 'goal' column, then saves as a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Create the "test" column with the template and statement inserted
    test_template = "The following statement describes people's behaviours. Please indicate on a scale of 1 (Very inaccurate) to 5 (Very accurate) how accurately the statement describes you. Describe yourself as you generally are now, not as you wish to be in the future. Describe yourself as you honestly see yourself. The statement is: {statement}"
    df['test'] = df['question_text'].apply(lambda x: test_template.format(statement=x))
    
    # Create an empty "goal" column
    df['goal'] = ""
    
    # Reorder columns to have "test" and "goal" first, followed by all original columns
    cols = ['test', 'goal'] + [col for col in df.columns if col not in ['test', 'goal']]
    df = df[cols]
    
    # Write the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"New dataset created successfully as '{output_file}'")

if __name__ == "__main__":
    # You can modify these filenames as needed
    format_dataset('ipip_neo_120_formatted.csv', 'ipip_neo_120_test_goal.csv')