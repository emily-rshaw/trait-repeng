import pandas as pd
import re
import os
from typing import List, Dict, Callable

def extract_ipip_traits(input_file: str) -> List[Dict]:
    """Extract personality traits from IPIP-NEO formatted file"""
    df = pd.read_csv(input_file, sep=',')
    results = []
    
    # Extract domains (Big Five)
    domains = df['domain_name'].unique()
    for domain in domains:
        results.append({
            'domain': 'Personality',
            'specificity': 'Broad',
            'trait_name': domain,
            'description': f"Big Five personality trait of {domain}",
            'parent_trait': ""
        })
    
    # Extract facets
    facets = df[['facet_name', 'domain_name']].drop_duplicates()
    for _, row in facets.iterrows():
        facet_name = row['facet_name']
        results.append({
            'domain': 'Personality',
            'specificity': 'Specific',
            'trait_name': facet_name,
            'description': f"A facet of {row['domain_name']}",
            'parent_trait': row['domain_name']
        })
    
    print(f"Extracted {len(domains)} domains and {len(facets)} facets from {os.path.basename(input_file)}")
    return results

def extract_political_traits(input_file: str) -> List[Dict]:
    """Extract political traits from a formatted file"""
    # This is a placeholder - implement your own logic for political traits
    df = pd.read_csv(input_file, sep=',')
    results = []
    
    # Example logic - adjust according to your actual political traits file structure
    # For broad traits
    if 'ideology' in df.columns:
        ideologies = df['ideology'].unique()
        for ideology in ideologies:
            results.append({
                'domain': 'Political',
                'specificity': 'Broad',
                'trait_name': ideology,
                'description': f"Political ideology of {ideology}",
                'parent_trait': ""
            })
    
    # For specific traits
    if 'specific_view' in df.columns and 'parent_ideology' in df.columns:
        views = df[['specific_view', 'parent_ideology']].drop_duplicates()
        for _, row in views.iterrows():
            results.append({
                'domain': 'Political',
                'specificity': 'Specific',
                'trait_name': row['specific_view'],
                'description': f"A specific political view within {row['parent_ideology']}",
                'parent_trait': row['parent_ideology']
            })
    
    print(f"Extracted political traits from {os.path.basename(input_file)}")
    return results

def extract_moral_traits(input_file: str) -> List[Dict]:
    """Extract moral values from a formatted file"""
    # This is a placeholder - implement your own logic for moral traits
    df = pd.read_csv(input_file, sep=',')
    results = []
    
    # Example logic - adjust according to your actual moral traits file structure
    # For broad values
    if 'value_category' in df.columns:
        categories = df['value_category'].unique()
        for category in categories:
            results.append({
                'domain': 'Moral',
                'specificity': 'Broad',
                'trait_name': category,
                'description': f"Moral value category: {category}",
                'parent_trait': ""
            })
    
    # For specific values
    if 'specific_value' in df.columns and 'parent_category' in df.columns:
        values = df[['specific_value', 'parent_category']].drop_duplicates()
        for _, row in values.iterrows():
            results.append({
                'domain': 'Moral',
                'specificity': 'Specific',
                'trait_name': row['specific_value'],
                'description': f"A specific moral value within {row['parent_category']}",
                'parent_trait': row['parent_category']
            })
    
    print(f"Extracted moral traits from {os.path.basename(input_file)}")
    return results

def extract_and_combine_traits(file_configs: List[Dict], output_file: str) -> pd.DataFrame:
    """
    Extract traits from multiple files and combine them into a single DataFrame
    
    Args:
        file_configs: List of dictionaries, each with:
            - 'file_path': Path to the input file
            - 'extractor': Function to use for extraction
        output_file: Path to save the combined CSV
    """
    all_traits = []
    
    for config in file_configs:
        file_path = config['file_path']
        extractor = config['extractor']
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Extract traits using the appropriate extractor function
        traits = extractor(file_path)
        all_traits.extend(traits)
    
    # Combine into DataFrame and save
    if all_traits:
        output_df = pd.DataFrame(all_traits)
        output_df.to_csv(output_file, index=False)
        print(f"Combined {len(all_traits)} traits saved to {output_file}")
        return output_df
    else:
        print("No traits were extracted!")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Define file configurations
    file_configs = [
        {
            'file_path': "data/psychometric_tests/personality/ipip_neo_120_formatted.csv",
            'extractor': extract_ipip_traits
        },
        # Add more files with their corresponding extractors as needed
        # {
        #     'file_path': "data/psychometric_tests/political/secs_items.csv", 
        #     'extractor': extract_political_traits
        # },
        # {
        #     'file_path': "data/psychometric_tests/moral/pvq_values.csv",
        #     'extractor': extract_moral_traits
        # }
    ]
    
    # Extract and combine all traits
    extract_and_combine_traits(
        file_configs,
        "scripts/database_management/trait_structure_management/all_traits.csv"
    )