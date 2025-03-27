#!/usr/bin/env python3
"""
Preliminary script to analyze outputs for runs 29 and 30, 
using regression between actual vector scores and response ratings.

1) Still do refusal detection, phrase categorization.
2) Use actual vector_rank_score directly for grouping (is_high_score).
3) Perform regression analysis between vector scores and extracted ratings.
4) Compare refusal rate, extracted rating, etc.
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression

DB_PATH = "results/database/experiments.db"

# For comparison with old system (to assess improvements)
def legacy_detect_refusal(response_text):
    """The original refusal detection function for comparison"""
    refusal_keywords = [
        "i do not have personal feelings",
        "i cannot provide a personal rating",
        "as an ai language model, i don't"
    ]
    lower_resp = response_text.lower()
    for kw in refusal_keywords:
        if kw in lower_resp:
            return True
    return False

def legacy_categorize_response_phrase(response_text):
    """The original categorization function for comparison"""
    lower_resp = response_text.lower()
    if "i would rate myself" in lower_resp:
        return "i would rate myself"
    elif "i'd rate myself" in lower_resp:
        return "i'd rate myself" 
    elif "i would rate the statement" in lower_resp:
        return "i would rate the statement"
    else:
        pattern = re.compile(r"\b(\d(\.\d)?)\b")
        if pattern.search(lower_resp):
            return "numeric_only"
        return "unknown"

#########################################################################
# Helper functions (same as before, but repeated for completeness)
#########################################################################

def detect_refusal(response_text):
    """
    Detect if a response contains a refusal to provide a rating.
    Uses comprehensive pattern matching to identify various refusal patterns.
    """
    # Pre-process text
    lower_resp = response_text.lower().strip()
    
    # Common refusal phrases (direct matches)
    refusal_phrases = [
        # AI identity statements
        "i do not have personal feelings",
        "i cannot provide a personal rating",
        "as an ai", "as an assistant", "as a language model", "as an ai assistant",
        "i'm an ai", "i am an ai", "i'm just an", "i am just an",
        "i don't have personal", "i don't have subjective", "i don't have the ability to",
        "i cannot provide", "i'm not able to", "i am not able to",
        
        # Explicit refusals
        "i cannot rate", "i can't rate", "unable to rate", "not able to rate",
        "i should not rate", "i shouldn't rate", "i would not rate", "i wouldn't rate",
        "cannot assign", "can't assign", "won't assign",
        "i don't feel comfortable", "not comfortable", 
        
        # Policy references
        "my guidelines", "my programming", "my design", "my purpose",
        "not designed to", "not programmed to", "not appropriate for me to",
        
        # Alternative suggestions
        "would be better for a human", "human would be better",
        "recommend consulting", "suggest asking", "perhaps a human"
    ]
    
    # Check for direct phrase matches
    for phrase in refusal_phrases:
        if phrase in lower_resp:
            return True
    
    # Regex patterns for more complex refusals
    refusal_patterns = [
        r"i (don't|do not|cannot|can't) (have|provide|give|express|share) (personal|subjective|my own)",
        r"(not|isn't|is not) (within|in|part of) my (capabilities|ability|programming|design|function)",
        r"i (should|would|must|need to) (remain|stay|be) (neutral|objective|impartial)",
        r"(ethical|appropriate|proper) for (me|an ai|a language model) to",
        r"(instead of|rather than) (rating|providing|giving) (my own|personal|subjective)"
    ]
    
    # Check for regex pattern matches
    for pattern in refusal_patterns:
        if re.search(pattern, lower_resp):
            return True
            
    return False

def categorize_response_phrase(response_text):
    """
    Categorize the response format used to provide a rating.
    Identifies common patterns in how ratings are expressed.
    """
    lower_resp = response_text.lower().strip()
    
    # Self-rating patterns (first person)
    self_rating_patterns = {
        # Direct self-rating phrases
        r"i would rate (myself|me)": "i would rate myself",
        r"i'd rate (myself|me)": "i'd rate myself",
        r"i (would|'d) give (myself|me)": "i would give myself",
        r"i (would|'d) score (myself|me)": "i would score myself",
        r"i (would|'d) assess (myself|me)": "i would assess myself",
        r"i (would|'d) evaluate (myself|me)": "i would evaluate myself",
        r"i (would|'d) consider (myself|me)": "i would consider myself",
        r"i (would|'d) place (myself|me)": "i would place myself",
        r"i (would|'d) describe (myself|me)": "i would describe myself",
        r"i (am|feel|consider myself)": "i am/feel",
        r"i see myself as": "i see myself as",
        r"on this trait,? i": "on this trait, i",
        r"for this trait,? i": "for this trait, i",
        r"(regarding|concerning) this,? i": "regarding this, i",
    }
    
    # Statement/item rating patterns
    statement_rating_patterns = {
        r"i would rate (this|the) (statement|item|question|prompt|trait)": "i would rate the statement",
        r"i'd rate (this|the) (statement|item|question|prompt|trait)": "i'd rate the statement",
        r"i (would|'d) give (this|the) (statement|item|question|prompt|trait)": "i would give the statement",
        r"i (would|'d) score (this|the) (statement|item|question|prompt|trait)": "i would score the statement",
        r"i (would|'d) assess (this|the) (statement|item|question|prompt|trait)": "i would assess the statement",
        r"(this|the) (statement|item|question|prompt|trait) (is|seems|appears)": "the statement is",
        r"(this|the) (statement|item|question|prompt|trait) (deserves|rates|scores|merits)": "the statement deserves",
        r"(for|on) (this|the) (statement|item|question|prompt|trait)": "for the statement",
    }
    
    # General rating patterns
    general_rating_patterns = {
        r"(my|the) (rating|score|assessment|evaluation) (is|would be)": "my rating is",
        r"i would (rate|give|assign|choose|pick|select)": "i would rate",
        r"i'd (rate|give|assign|choose|pick|select)": "i'd rate",
        r"(rating|score|value|number)(:|is|=|would be)": "rating is",
        r"i (pick|choose|select|vote for|opt for)": "i pick",
    }
    
    # First check for refusal - if it's a refusal, that's the primary category
    if detect_refusal(response_text):
        return "refusal"
    
    # Check all pattern dictionaries in priority order
    for pattern_dict in [self_rating_patterns, statement_rating_patterns, general_rating_patterns]:
        for pattern, category in pattern_dict.items():
            if re.search(pattern, lower_resp):
                return category
    
    # Direct numeric check (with more context)
    # Look for numbers with rating context
    rating_number_patterns = [
        r"rating:?\s*\b(\d+(\.\d+)?)\b",
        r"score:?\s*\b(\d+(\.\d+)?)\b",
        r"(\d+(\.\d+)?)\s*out of\s*\b(\d+)\b",
        r"(\d+(\.\d+)?)\s*/\s*\b(\d+)\b",
        r"(\d+(\.\d+)?)\s*on a scale",
    ]
    
    for pattern in rating_number_patterns:
        if re.search(pattern, lower_resp):
            return "explicit_numeric_rating"
    
    # Check for just a number (with context checking)
    # First look for any number pattern
    basic_number = re.compile(r"\b(\d+(\.\d+)?)\b")
    if basic_number.search(lower_resp):
        # If the number appears to be the main content (short response with number)
        if len(lower_resp.split()) < 10 and basic_number.search(lower_resp):
            return "numeric_only"
        else:
            return "numeric_with_explanation"
    
    return "unknown"

def extract_rating(response_text):
    """
    Extract numeric rating from response text with improved pattern matching.
    Handles various rating formats and scales.
    """
    lower_resp = response_text.lower().strip()
    
    # First check for explicit rating formats
    explicit_patterns = [
        # Format: "rating: X" or "score: X"
        r"(?:rating|score|value)(?:\s+is|\s*:|\s*=)\s*(\d+(?:\.\d+)?)",
        # Format: "X out of Y" or "X/Y"
        r"(\d+(?:\.\d+)?)\s*(?:out of|\/|on a scale of)\s*(\d+)",
        # Format: "I rate myself/this a X"
        r"(?:rate|give|score|assign)(?:\s+\w+){0,3}\s+(?:a|an)\s+(\d+(?:\.\d+)?)",
        # Format: "My rating is X"
        r"my\s+(?:rating|score|evaluation)\s+(?:is|would be)\s+(?:a|an)?\s*(\d+(?:\.\d+)?)",
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, lower_resp)
        if match:
            # Handle X out of Y format to normalize to a 1-5 scale if needed
            if len(match.groups()) > 1 and match.group(2):
                try:
                    rating = float(match.group(1))
                    scale = float(match.group(2))
                    # If it's a different scale (e.g., 1-10), normalize to our target scale
                    if scale == 10:
                        # Convert 1-10 to 1-5
                        return (rating / 2) if rating > 0 else rating
                    elif scale == 100:
                        # Convert percentage to 1-5
                        return (rating / 20) if rating > 0 else rating
                    # Don't modify if already on the expected scale
                    return rating
                except (ValueError, IndexError):
                    pass
            else:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    pass
    
    # General pattern for finding numbers, prioritizing those in likely rating contexts
    # Look for numbers preceded or followed by rating keywords
    context_patterns = [
        r"(?:rate|rating|score|value|level|give|assign)(?:\s+\w+){0,3}\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+(?:on the scale|out of|points|\/)",
    ]
    
    for pattern in context_patterns:
        match = re.search(pattern, lower_resp)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
    
    # As a fallback, look for any numbers in the text, preferring the first one
    # which is often the actual rating in responses
    number_pattern = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    matches = number_pattern.findall(lower_resp)
    
    if matches:
        # If we have multiple matches, prioritize single-digit numbers from 1-5
        # which are more likely to be ratings on our expected scale
        for num_str in matches:
            try:
                num = float(num_str)
                # If it's in our expected rating range (1-5), return it immediately
                if 1 <= num <= 5:
                    return num
            except ValueError:
                continue
        
        # If no 1-5 ratings found, return the first number (typical case)
        try:
            first_num = float(matches[0])
            # If it's a percentage or 1-10 scale, normalize it
            if first_num > 10:
                # Assume it's a percentage and convert to 1-5 scale
                return (first_num / 20) if first_num <= 100 else 5.0
            elif first_num > 5:
                # Assume it's a 1-10 scale and convert to 1-5
                return first_num / 2
            return first_num
        except ValueError:
            pass
            
    return None

#########################################################################
# MAIN
#########################################################################

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # We'll do a join as before
    query = """
    SELECT 
        o.output_id,
        o.run_id,
        o.prompt_id,
        o.vector_id,
        o.output_text,
        sv.vector_rank_score,
        r.run_description
    FROM outputs o
    LEFT JOIN steering_vectors sv ON o.vector_id = sv.vector_id
    JOIN runs r ON o.run_id = r.run_id
    WHERE o.run_id IN (61);
    """
    rows = cursor.execute(query).fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    conn.close()

    # Add new columns
    df["is_refusal"] = df["output_text"].apply(detect_refusal)
    df["phrase_category"] = df["output_text"].apply(categorize_response_phrase)
    df["rating_extracted"] = df["output_text"].apply(extract_rating)

    # 1) Fill missing vector_rank_score with NaN if unsteered or not found
    df["vector_rank_score"] = df["vector_rank_score"].astype(float)
    df["vector_rank_score"] = df["vector_rank_score"].replace({None: np.nan})

    # 2) Use the actual vector scores directly instead of log scores
    # Decide high vs low vector score e.g. above/below median
    # ignoring NaN. (If user wanted a different threshold, adapt.)
    median_score = df["vector_rank_score"].median(skipna=True)
    df["is_high_score"] = df["vector_rank_score"] > median_score  # True/False/NaN

    # 4) Compare refusal rate by is_high_score
    # ignoring rows with NaN in is_high_score => e.g. unsteered
    sub_for_group = df.dropna(subset=["is_high_score"])
    refusal_by_group = sub_for_group.groupby("is_high_score")["is_refusal"].mean()
    print("Refusal Rate by is_high_score:")
    print(refusal_by_group)
    print()

    # 5) If we want to see rating for certain prompt_ids => e.g. 121 => want rating near 5
    # 124 => near 1 => same approach but grouping by is_high_score
    relevant_ids = [121, 124]
    sub2 = df[df["prompt_id"].isin(relevant_ids)].copy()
    # also drop rows if vector_rank_score is NaN => unsteered
    sub2 = sub2.dropna(subset=["vector_rank_score"])

    # Add regression analysis between vector scores and ratings
    for prompt_id in relevant_ids:
        prompt_data = sub2[sub2["prompt_id"] == prompt_id]
        if len(prompt_data) > 1 and not prompt_data["vector_rank_score"].isna().all() and not prompt_data["rating_extracted"].isna().all():
            # Calculate regression
            x = prompt_data["vector_rank_score"].values.reshape(-1, 1)
            y = prompt_data["rating_extracted"].dropna().values
            # Only perform regression if we have enough data points
            if len(x) > 1 and len(y) > 1 and len(x) == len(y):
                model = LinearRegression()
                model.fit(x, y)
                r_squared = model.score(x, y)
                print(f"Prompt ID {prompt_id} - Regression results:")
                print(f"Coefficient: {model.coef_[0]:.4f}")
                print(f"Intercept: {model.intercept_:.4f}")
                print(f"R-squared: {r_squared:.4f}")
                print()

    # Calculate and show mean ratings grouped by prompt_id and is_high_score
    rating_mean = sub2.groupby(["prompt_id", "is_high_score"])["rating_extracted"].mean()
    print("Mean rating_extracted by prompt_id and is_high_score:")
    print(rating_mean)
    print()

    # Phrase category distribution overall
    cat_counts = df["phrase_category"].value_counts()
    cat_percent = df["phrase_category"].value_counts(normalize=True) * 100
    print("Phrase category distribution overall:")
    print(cat_counts)
    print("\nPercentage distribution:")
    print(cat_percent.round(2))
    print()
    
    # Phrase category distribution by high vs low vector score
    sub_for_cat = df.dropna(subset=["is_high_score"])
    cat_by_score = pd.crosstab(
        sub_for_cat["phrase_category"], 
        sub_for_cat["is_high_score"],
        normalize="columns"
    ) * 100
    
    print("Phrase category distribution by high vs low vector score (%):")
    print(cat_by_score.round(2))
    print()
    
    # Rating value distribution by category
    rating_by_cat = df.groupby("phrase_category")["rating_extracted"].agg(
        ["mean", "median", "std", "count"]
    ).round(2)
    print("Rating stats by phrase category:")
    print(rating_by_cat)
    print()
    
    # Success rate of rating extraction by category
    df["has_rating"] = ~df["rating_extracted"].isna()
    extraction_success = df.groupby("phrase_category")["has_rating"].mean() * 100
    print("Rating extraction success rate by category (%):")
    print(extraction_success.round(2))
    print()
    
    # Check for ratings in refusal cases (should be minimal)
    refusal_with_rating = df[df["is_refusal"] & df["has_rating"]]
    if not refusal_with_rating.empty:
        print(f"WARNING: Found {len(refusal_with_rating)} refusals with ratings extracted!")
        print(refusal_with_rating[["output_id", "rating_extracted"]].head())
        print()
    
    print("====== COMPARISON WITH LEGACY SYSTEM ======")
    # Compare old and new refusal detection
    df["legacy_is_refusal"] = df["output_text"].apply(legacy_detect_refusal)
    df["legacy_phrase_category"] = df["output_text"].apply(legacy_categorize_response_phrase)
    
    # Refusal detection comparison
    refusal_comparison = pd.crosstab(
        df["legacy_is_refusal"], df["is_refusal"], 
        rownames=["Legacy"], colnames=["New System"],
        margins=True
    )
    print("Refusal detection comparison:")
    print(refusal_comparison)
    
    # Calculate detection metrics
    true_positives = sum((df["legacy_is_refusal"] == True) & (df["is_refusal"] == True))
    false_negatives = sum((df["legacy_is_refusal"] == True) & (df["is_refusal"] == False))
    false_positives = sum((df["legacy_is_refusal"] == False) & (df["is_refusal"] == True))
    true_negatives = sum((df["legacy_is_refusal"] == False) & (df["is_refusal"] == False))
    
    additional_refusals = sum((df["legacy_is_refusal"] == False) & (df["is_refusal"] == True))
    print(f"\nAdditional refusals caught by new system: {additional_refusals}")
    
    # If the dataframe is small enough, show examples of newly caught refusals
    if additional_refusals > 0 and additional_refusals <= 10:
        new_refusals = df[(df["legacy_is_refusal"] == False) & (df["is_refusal"] == True)]
        print("\nExamples of newly caught refusals:")
        for idx, row in new_refusals.iterrows():
            print(f"ID: {row['output_id']}: {row['output_text'][:100]}...")
    
    # Categorization comparison
    legacy_unknown = sum(df["legacy_phrase_category"] == "unknown")
    new_unknown = sum(df["phrase_category"] == "unknown")
    print(f"\nUnknown categories reduced from {legacy_unknown} to {new_unknown}")
    print(f"Improvement: {legacy_unknown - new_unknown} more responses categorized " 
          f"({((legacy_unknown - new_unknown) / len(df) * 100):.2f}% of total)")
    
    # Compare category distributions
    legacy_cat_counts = Counter(df["legacy_phrase_category"])
    new_cat_counts = Counter(df["phrase_category"])
    
    print("\nLegacy categories distribution:")
    for cat, count in legacy_cat_counts.most_common():
        print(f"{cat}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\nNew categories distribution:")
    for cat, count in new_cat_counts.most_common(10):  # Top 10 to keep it manageable
        print(f"{cat}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\nAnalysis complete (direct vector score regression).")
    
    # Save results to output file
    output_dir = "scripts/analysis/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prelim_analysis_results.txt")
    
    # Capture all printed output to the file
    from io import StringIO
    import sys
    
    # Create a backup of stdout
    original_stdout = sys.stdout
    
    # Create a string buffer to capture output
    output_buffer = StringIO()
    sys.stdout = output_buffer
    
    # Repeat the key outputs for the file
    print("Refusal Rate by is_high_score:")
    print(refusal_by_group)
    print()
    
    for prompt_id in relevant_ids:
        prompt_data = sub2[sub2["prompt_id"] == prompt_id]
        if len(prompt_data) > 1 and not prompt_data["vector_rank_score"].isna().all() and not prompt_data["rating_extracted"].isna().all():
            # Calculate regression
            valid_data = prompt_data.dropna(subset=["vector_rank_score", "rating_extracted"])
            if len(valid_data) > 1:
                x = valid_data["vector_rank_score"].values.reshape(-1, 1)
                y = valid_data["rating_extracted"].values
                if len(x) > 1 and len(y) > 1:
                    model = LinearRegression()
                    model.fit(x, y)
                    r_squared = model.score(x, y)
                    print(f"Prompt ID {prompt_id} - Regression results:")
                    print(f"Coefficient: {model.coef_[0]:.4f}")
                    print(f"Intercept: {model.intercept_:.4f}")
                    print(f"R-squared: {r_squared:.4f}")
                    print()
    
    print("Mean rating_extracted by prompt_id and is_high_score:")
    print(rating_mean)
    print()
    
    print("Phrase category distribution overall:")
    print(cat_counts)
    print("\nPercentage distribution:")
    print(cat_percent.round(2))
    print()
    
    print("Phrase category distribution by high vs low vector score (%):")
    print(cat_by_score.round(2))
    print()
    
    print("Rating stats by phrase category:")
    print(rating_by_cat)
    print()
    
    print("Rating extraction success rate by category (%):")
    print(extraction_success.round(2))
    print()
    
    print("Analysis complete (direct vector score regression).")
    
    # Get the captured output
    output_text = output_buffer.getvalue()
    
    # Restore stdout
    sys.stdout = original_stdout
    
    # Write the captured output to file
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nResults saved to {output_path}")
    
    # Optional: Generate visualization if matplotlib is available
    try:
        # Make sure the output directory exists
        output_dir = "scripts/analysis/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Category distribution comparison
        # Create a new figure explicitly for category comparison
        plt.figure(figsize=(15, 7))
        plt.clf()  # Clear any existing figures
        
        # Plot legacy categories vs new categories (top categories)
        legacy_top = dict(legacy_cat_counts.most_common(5))
        legacy_top["other"] = sum(count for cat, count in legacy_cat_counts.items() 
                                 if cat not in legacy_top)
        
        new_top = dict(new_cat_counts.most_common(5))
        new_top["other"] = sum(count for cat, count in new_cat_counts.items() 
                              if cat not in new_top)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Left subplot: Legacy categories
        ax1.pie(legacy_top.values(), labels=legacy_top.keys(), autopct='%1.1f%%')
        ax1.set_title('Legacy Category Distribution')
        
        # Right subplot: New categories
        ax2.pie(new_top.values(), labels=new_top.keys(), autopct='%1.1f%%')
        ax2.set_title('New Category Distribution')
        
        plt.tight_layout()
        category_plot_path = os.path.join(output_dir, "category_comparison.png")
        fig.savefig(category_plot_path)
        plt.close(fig)  # Close the figure to free memory
        print(f"Visualization saved to {category_plot_path}")
        
        # 2. Regression visualizations for each prompt ID
        for prompt_id in relevant_ids:
            prompt_data = sub2[sub2["prompt_id"] == prompt_id]
            # Clean data by removing NaN values
            valid_data = prompt_data.dropna(subset=["vector_rank_score", "rating_extracted"])
            
            if len(valid_data) > 1:
                # Get the data for plotting
                x_data = valid_data["vector_rank_score"].values
                y_data = valid_data["rating_extracted"].values
                
                if len(x_data) > 1 and len(y_data) > 1:
                    # Create a new figure for this regression plot
                    reg_fig = plt.figure(figsize=(10, 6))
                    plt.clf()  # Clear any existing plot
                    
                    # Reshape x data for regression
                    x_clean = x_data.reshape(-1, 1)
                    
                    # Plot the scatter points
                    plt.scatter(x_data, y_data, alpha=0.7)
                    
                    # Fit and plot regression line
                    model = LinearRegression()
                    model.fit(x_clean, y_data)
                    x_line = np.linspace(min(x_data), max(x_data), 100).reshape(-1, 1)
                    y_line = model.predict(x_line)
                    plt.plot(x_line, y_line, color='red')
                    
                    # Add regression equation and R-squared
                    r_squared = model.score(x_clean, y_data)
                    equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
                    plt.text(0.05, 0.95, f"{equation}\nR² = {r_squared:.4f}", 
                            transform=plt.gca().transAxes, 
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    
                    plt.title(f"Regression Analysis - Prompt ID {prompt_id}")
                    plt.xlabel("Vector Rank Score")
                    plt.ylabel("Extracted Rating")
                    plt.grid(True, alpha=0.3)
                    
                    # Save the plot
                    regression_plot_path = os.path.join(output_dir, f"regression_prompt_{prompt_id}.png")
                    plt.savefig(regression_plot_path)
                    plt.close(reg_fig)  # Close the figure
                    print(f"Regression visualization saved to {regression_plot_path}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    print("\nAnalysis complete (direct vector score regression).")

if __name__ == "__main__":
    main()
