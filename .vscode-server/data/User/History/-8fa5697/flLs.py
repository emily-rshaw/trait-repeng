import pandas as pd
import re

# Read the data - assuming it's saved as ipip_neo_120_formatted.csv
def extract_ipip_traits(input_file, output_file):
    # Read data (adjust sep if needed)
    df = pd.read_csv(input_file, sep=',')
    
    # Extract unique domains (Big Five)
    domains = df['domain_name'].unique()
    
    # Create output dataframe for trait hierarchy
    results = []
    
    # First add the broad domains
    for domain in domains:
        results.append({
            'domain': 'Personality',
            'specificity': 'Broad',
            'trait_name': domain,
            'description': f"Big Five personality trait of {domain}",
            'parent_trait': ""
        })
    
    # Extract unique facets and link to parent domains
    facets = df[['facet_name', 'domain_name']].drop_duplicates()
    
    for _, row in facets.iterrows():
        # Clean up the facet name (remove prefix like "N1 ")
        facet_name = re.sub(r'^[A-Z][0-9]\s+', '', row['facet_name'])
        
        results.append({
            'domain': 'Personality',
            'specificity': 'Specific',
            'trait_name': facet_name,
            'description': f"A facet of {row['domain_name']}", 
            'parent_trait': row['domain_name']
        })
    
    # Convert to DataFrame and save
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"Extracted {len(domains)} domains and {len(facets)} facets to {output_file}")
    
    return output_df

# Example usage
if __name__ == "__main__":
    extract_ipip_traits("ipip_neo_120_formatted.csv", "personality_traits.csv")