import pandas as pd

def correct_dataset_formatting(input_file='ipip_neo_120_formatted.csv', output_file='ipip_neo_120_corrected.csv'):
    """
    Corrects formatting issues in the IPIP NEO dataset:
    1. Fixes domain names (O facets should be under "Openness" domain)
    2. Standardizes special minus signs in facet_key
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Fix domain names based on facet keys
    # All O facets should be in Openness domain
    df['domain_name'] = df.apply(lambda row: 
                                "Openness" if row['facet_key'].strip('+-−').startswith('O') else 
                                row['domain_name'], 
                                axis=1)
    
    # Standardize minus signs in facet_key (replace special minus sign with standard dash)
    df['facet_key'] = df['facet_key'].str.replace('−', '-')
    
    # Write the corrected DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
    # Print summary of corrections
    o_facets_count = len(df[df['facet_key'].str.strip('+-−').str.startswith('O')])
    minus_signs_count = len(df[df['facet_key'].str.contains('−')])
    
    print(f"Formatting corrections completed:")
    print(f"  - Fixed domain for {o_facets_count} Openness facet items")
    print(f"  - Standardized minus signs in {minus_signs_count} facet keys")
    print(f"  - Corrected dataset saved to '{output_file}'")

if __name__ == "__main__":
    correct_dataset_formatting()