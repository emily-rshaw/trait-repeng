import pandas as pd
import os

def create_trait_specific_datasets(input_file='ipip_neo_120_test_goal.csv', output_dir='trait_specific'):
    """
    Creates trait-specific datasets for activation engineering in LLMs.
    
    For each domain and facet:
    - Creates a maximizing and minimizing dataset
    - Sets goal values to 1 or 5 based on item direction (+ or -) and max/min objective
    - Includes only relevant items for that trait
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save the output files
    """
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Fix domain names based on facet keys (O facets should be in Openness domain)
    df['domain_name'] = df.apply(lambda row: 
                                "Openness" if row['facet_key'].strip('+-').startswith('O') else 
                                row['domain_name'], 
                                axis=1)
    
    # Create test column with the template and statement inserted
    test_template = "The following statement describes people's behaviours. Please indicate on a scale of 1 (Very inaccurate) to 5 (Very accurate) how accurately the statement describes you. Describe yourself as you generally are now, not as you wish to be in the future. Describe yourself as you honestly see yourself. The statement is: {statement}"
    df['test'] = df['question_text'].apply(lambda x: test_template.format(statement=x))
    
    # Initialize goal column (will be set per dataset)
    df['goal'] = ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique domains
    domains = df['domain_name'].unique()
    
    # Get unique facets (from facet_name column, like "N1 Anxiety")
    facet_codes = set()
    for facet_name in df['facet_name']:
        # Extract facet code (e.g., "N1" from "N1 Anxiety")
        facet_code = facet_name.split()[0]
        facet_codes.add(facet_code)
    facet_codes = sorted(list(facet_codes))
    
    # Function to set goals based on direction
    def set_goals(subset_df, maximize):
        # Create a copy to avoid modifying the original
        result_df = subset_df.copy()
        
        # Set goal values based on facet_key and maximize/minimize
        for idx, row in result_df.iterrows():
            is_positive = row['facet_key'].startswith('+')
            
            if maximize:
                # Maximizing: + items get 5, - items get 1
                result_df.at[idx, 'goal'] = "5" if is_positive else "1"
            else:
                # Minimizing: + items get 1, - items get 5
                result_df.at[idx, 'goal'] = "1" if is_positive else "5"
        
        # Reorder columns to have "test" and "goal" first, followed by all original columns
        cols = ['test', 'goal'] + [col for col in result_df.columns if col not in ['test', 'goal']]
        result_df = result_df[cols]
        
        return result_df
    
    # Track created datasets
    created_datasets = []
    
    # Process domains
    for domain in domains:
        # Filter items for this domain
        domain_items = df[df['domain_name'] == domain]
        
        # Skip if no items found
        if len(domain_items) == 0:
            continue
        
        # Domain code (first letter of domain name)
        domain_code = domain[0]
        
        # Create maximizing dataset
        max_domain = set_goals(domain_items, maximize=True)
        max_filename = f"{output_dir}/max_{domain_code}_{domain.lower().replace(' ', '_')}.csv"
        max_domain.to_csv(max_filename, index=False)
        created_datasets.append((max_filename, len(max_domain)))
        
        # Create minimizing dataset
        min_domain = set_goals(domain_items, maximize=False)
        min_filename = f"{output_dir}/min_{domain_code}_{domain.lower().replace(' ', '_')}.csv"
        min_domain.to_csv(min_filename, index=False)
        created_datasets.append((min_filename, len(min_domain)))
    
    # Process facets
    for facet_code in facet_codes:
        # Filter items for this facet
        facet_items = df[df['facet_name'].str.startswith(facet_code + ' ')]
        
        # Skip if no items found
        if len(facet_items) == 0:
            continue
        
        # Get facet name (for the filename)
        facet_full_name = facet_items.iloc[0]['facet_name']
        facet_name_part = facet_full_name.split(' ', 1)[1] if ' ' in facet_full_name else ""
        facet_name_simplified = facet_name_part.lower().replace(' ', '_')
        
        # Create maximizing dataset
        max_facet = set_goals(facet_items, maximize=True)
        max_filename = f"{output_dir}/max_{facet_code.lower()}_{facet_name_simplified}.csv"
        max_facet.to_csv(max_filename, index=False)
        created_datasets.append((max_filename, len(max_facet)))
        
        # Create minimizing dataset
        min_facet = set_goals(facet_items, maximize=False)
        min_filename = f"{output_dir}/min_{facet_code.lower()}_{facet_name_simplified}.csv"
        min_facet.to_csv(min_filename, index=False)
        created_datasets.append((min_filename, len(min_facet)))
    
    # Print summary of created datasets
    print(f"\nSummary of created datasets in '{output_dir}' directory:")
    for filename, count in sorted(created_datasets):
        print(f"  - {os.path.basename(filename)}: {count} items")
    
    print(f"\nCompleted! {len(created_datasets)} trait-specific datasets created.")

if __name__ == "__main__":
    create_trait_specific_datasets()