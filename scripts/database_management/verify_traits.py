# verify_traits.py
from db_helpers import get_connection

def verify_trait_hierarchy(db_path='results/database/experiments.db'):
    """Verify that trait hierarchies are properly established"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Get counts of different traits
    cursor.execute("SELECT COUNT(*) FROM traits WHERE specificity = 'Broad'")
    broad_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM traits WHERE specificity = 'Specific'")
    specific_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM traits WHERE parent_trait_id IS NOT NULL")
    with_parent_count = cursor.fetchone()[0]
    
    print(f"Database contains {broad_count} broad traits and {specific_count} specific traits")
    print(f"{with_parent_count} traits have a parent trait assigned")
    
    if specific_count != with_parent_count:
        print("WARNING: Some specific traits don't have a parent trait assigned!")
    
    # List broad traits and their specific traits
    cursor.execute("""
        SELECT domain, trait_name, trait_id 
        FROM traits 
        WHERE specificity = 'Broad' 
        ORDER BY domain, trait_name
    """)
    
    broad_traits = cursor.fetchall()
    
    for domain, trait_name, trait_id in broad_traits:
        # Get specific traits for this broad trait
        cursor.execute("""
            SELECT trait_name 
            FROM traits 
            WHERE parent_trait_id = ? 
            ORDER BY trait_name
        """, (trait_id,))
        
        specific_traits = [row[0] for row in cursor.fetchall()]
        
        print(f"\n{domain} - {trait_name} ({len(specific_traits)} specific traits):")
        for i, specific in enumerate(specific_traits, 1):
            print(f"  {i}. {specific}")
    
    conn.close()
    print("\nVerification complete!")

if __name__ == "__main__":
    verify_trait_hierarchy()