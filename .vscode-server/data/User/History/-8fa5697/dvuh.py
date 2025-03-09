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

def extract_moral_values_pvq() -> List[Dict]:
    """Extract Schwartz's moral values (PVQ-RR) into the trait hierarchy format"""
    results = []
    
    # Define higher order values (broad traits)
    higher_order_values = [
        {
            "name": "Self-transcendence",
            "description": "Concern for welfare and interests of others"
        },
        {
            "name": "Conservation",
            "description": "Preference for tradition, conformity and security"
        },
        {
            "name": "Self-enhancement",
            "description": "Pursuit of personal interests and relative success"
        },
        {
            "name": "Openness to change",
            "description": "Readiness for new ideas, actions and experiences"
        }
    ]
    
    # Define narrowly defined values (specific traits)
    specific_values = [
        # Self-transcendence values
        {
            "name": "Benevolence-Dependability", 
            "description": "Being a reliable and trustworthy member of the in-group",
            "parent": "Self-transcendence"
        },
        {
            "name": "Benevolence-Caring", 
            "description": "Devotion to the welfare of in-group members",
            "parent": "Self-transcendence"
        },
        {
            "name": "Universalism-Tolerance", 
            "description": "Acceptance and understanding of those who are different from oneself",
            "parent": "Self-transcendence"
        },
        {
            "name": "Universalism-Concern", 
            "description": "Commitment to equality, justice, and protection for all people",
            "parent": "Self-transcendence"
        },
        {
            "name": "Universalism-Nature", 
            "description": "Preservation of the natural environment",
            "parent": "Self-transcendence"
        },
        {
            "name": "Humility", 
            "description": "Recognizing one's insignificance in the larger scheme of things",
            "parent": "Self-transcendence"
        },
        
        # Conservation values
        {
            "name": "Conformity-Interpersonal", 
            "description": "Avoidance of upsetting or harming other people",
            "parent": "Conservation"
        },
        {
            "name": "Conformity-Rules", 
            "description": "Compliance with rules, laws, and formal obligations",
            "parent": "Conservation"
        },
        {
            "name": "Tradition", 
            "description": "Maintaining and preserving cultural, family, or religious traditions",
            "parent": "Conservation"
        },
        {
            "name": "Security-Societal", 
            "description": "Safety and stability in the wider society",
            "parent": "Conservation"
        },
        {
            "name": "Security-Personal", 
            "description": "Safety in one's immediate environment",
            "parent": "Conservation"
        },
        {
            "name": "Face", 
            "description": "Security and power through maintaining one's public image and avoiding humiliation",
            "parent": "Conservation"
        },
        
        # Self-enhancement values
        {
            "name": "Power-Resources", 
            "description": "Power through control of material and social resources",
            "parent": "Self-enhancement"
        },
        {
            "name": "Power-Dominance", 
            "description": "Power through exercising control over people",
            "parent": "Self-enhancement"
        },
        {
            "name": "Achievement", 
            "description": "Personal success through demonstrating competence according to social standards",
            "parent": "Self-enhancement"
        },
        {
            "name": "Hedonism", 
            "description": "Pleasure and sensuous gratification for oneself",
            "parent": "Self-enhancement"
        },
        
        # Openness to change values
        {
            "name": "Stimulation", 
            "description": "Excitement, novelty, and challenge in life",
            "parent": "Openness to change"
        },
        {
            "name": "Self-Direction-Action", 
            "description": "The freedom to determine one's own actions",
            "parent": "Openness to change"
        },
        {
            "name": "Self-Direction-Thought", 
            "description": "The freedom to cultivate one's own ideas and abilities",
            "parent": "Openness to change"
        }
    ]
    
    # Add higher order values (broad traits)
    for value in higher_order_values:
        results.append({
            'domain': 'Moral',
            'specificity': 'Broad',
            'trait_name': value['name'],
            'description': value['description'],
            'parent_trait': ""
        })
    
    # Add specific values
    for value in specific_values:
        results.append({
            'domain': 'Moral',
            'specificity': 'Specific',
            'trait_name': value['name'],
            'description': value['description'],
            'parent_trait': value['parent']
        })
    
    print(f"Extracted {len(higher_order_values)} broad moral values and {len(specific_values)} specific moral values")
    return results

def extract_and_combine_traits(file_configs: List[Dict], output_file: str) -> pd.DataFrame:
    all_traits = []
    
    for config in file_configs:
        file_path = config['file_path']
        extractor = config['extractor']
        
        # If file_path is None, just call the extractor directly
        if file_path is None:
            traits = extractor()
        # Otherwise check if file exists and call with file_path
        elif os.path.exists(file_path):
            traits = extractor(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
            continue
            
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

# In the main script:
if __name__ == "__main__":
    # Define file configurations
    file_configs = [
        {
            'file_path': "data/psychometric_tests/personality/ipip_neo_120_formatted.csv",
            'extractor': extract_ipip_traits
        },
        # This doesn't require a file path since the data is hardcoded
        {
            'file_path': None,
            'extractor': extract_moral_values_pvq
        },
        # Add political traits when available
        # ...
    ]
    
    # Extract and combine all traits
    extract_and_combine_traits(
        file_configs,
        "scripts/database_management/trait_structure_management/trait_master_db.csv"
    )