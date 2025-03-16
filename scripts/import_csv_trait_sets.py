# A dictionary from the short-coded file name remainder
# to the EXACT DB trait name.

import os
import glob
import csv
from experiment_manager import ExperimentManager

# The dictionaries above go here...
BIG5_DOMAIN_MAP = {...}
BIG5_FACET_MAP = {...}

def import_trait_specific_csvs_no_new_trait(
    mgr,
    trait_specific_dir="data/psychometric_tests/personality/trait_specific",
    do_link_experiment_id=None
):
    """
    1) For each CSV in trait_specific_dir, parse filename => direction + raw_trait_part.
    2) Check if raw_trait_part is in the facet map or domain map => get DB trait name.
    3) Lookup that trait name in 'traits' table => get trait_id. If not found => raise error.
    4) Create prompt_set => set_name = "max_a1_trust" or "min_e6_cheerfulness", etc.
    5) Insert each row => prompt_text= row["test"], target_response= row["goal"].
    6) Link prompt_set to experiment if do_link_experiment_id is given.
    """
    BIG5_DOMAIN_MAP = {
        "A_agreeableness": "Agreeableness",
        "C_conscientiousness": "Conscientiousness",
        "E_extraversion": "Extraversion",
        "N_neuroticism": "Neuroticism",
        "O_openness": "Openness"
    }

    BIG5_FACET_MAP = {
        "a1_trust": "A1 Trust",
        "a2_morality": "A2 Morality",
        "a3_altruism": "A3 Altruism",
        "a4_cooperation": "A4 Cooperation",
        "a5_modesty": "A5 Modesty",
        "a6_sympathy": "A6 Sympathy",
        "c1_self-efficacy": "C1 Self-Efficacy",
        "c2_orderliness": "C2 Orderliness",
        "c3_dutifulness": "C3 Dutifulness",
        "c4_achievement-striving": "C4 Achievement-striving",
        "c5_self-discipline": "C5 Self-Discipline",
        "c6_cautiousness": "C6 Cautiousness",
        "e1_friendliness": "E1 Friendliness",
        "e2_gregariousness": "E2 Gregariousness",
        "e3_assertiveness": "E3 Assertiveness",
        "e4_activity_level": "E4 Activity level",
        "e5_excitement_seeking": "E5 Excitement Seeking",
        "e6_cheerfulness": "E6 Cheerfulness",
        "n1_anxiety": "N1 Anxiety",
        "n2_anger": "N2 Anger",
        "n3_depression": "N3 Depression",
        "n4_self-consciousness": "N4 Self-Consciousness",
        "n5_immoderation": "N5 Immoderation",
        "n6_vulnerability": "N6 Vulnerability",
        "o1_imagination": "O1 Imagination",
        "o2_artistic_interests": "O2 Artistic interests",
        "o3_emotionality": "O3 Emotionality",
        "o4_adventurousness": "O4 Adventurousness",
        "o5_intellect": "O5 Intellect",
        "o6_liberalism": "O6 Liberalism",
    }

    results = []
    csv_files = glob.glob(os.path.join(trait_specific_dir, "*.csv"))

    for csv_file in csv_files:
        base_name = os.path.basename(csv_file)  # e.g. max_a1_trust.csv
        if base_name.startswith("max_"):
            direction = "max"
            raw_part = base_name[len("max_") : -4]  # remove 'max_' and '.csv'
        elif base_name.startswith("min_"):
            direction = "min"
            raw_part = base_name[len("min_") : -4]
        else:
            # fallback if not "max_" or "min_"
            direction = None
            raw_part = base_name.rsplit(".",1)[0]

        # Now we see if 'raw_part' is in BIG5_FACET_MAP or BIG5_DOMAIN_MAP
        # e.g. raw_part="a1_trust" => "A1 Trust"
        # e.g. raw_part="A_agreeableness" => "Agreeableness"
        if raw_part in BIG5_FACET_MAP:
            db_trait_name = BIG5_FACET_MAP[raw_part]
        elif raw_part in BIG5_DOMAIN_MAP:
            db_trait_name = BIG5_DOMAIN_MAP[raw_part]
        else:
            # Not found => skip or raise error
            print(f"ERROR: {raw_part} not found in domain/facet maps. Skipping or raising.")
            continue  # or raise ValueError(f"No known trait name for {raw_part}")

        # Now we find trait_id in the DB
        row = mgr.cursor.execute("""
            SELECT trait_id
            FROM traits
            WHERE trait_name = ?
        """, (db_trait_name,)).fetchone()

        if not row:
            # We do NOT create a new trait => raise error or skip
            print(f"ERROR: trait_name='{db_trait_name}' not found in DB! Skipping {csv_file}")
            continue

        trait_id = row["trait_id"]

        # Create prompt_set
        set_name = base_name[:-4]  # remove ".csv"
        p_set_id = mgr.create_prompt_set(
            trait_id=trait_id,
            set_name=set_name,
            set_closed_or_open_ended="closed",
            set_description=f"Imported from {base_name}"
        )

        # Insert each row from CSV
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_in_csv in reader:
                prompt_text = row_in_csv["test"]
                target_resp = row_in_csv["goal"]
                mgr.add_prompt_to_set(
                    p_set_id,
                    prompt_text,
                    target_response=target_resp
                )

        # Optionally link to an experiment
        if do_link_experiment_id is not None:
            mgr.link_experiment_prompt_set(do_link_experiment_id, p_set_id)

        results.append((csv_file, p_set_id, db_trait_name))
        print(f"Imported {csv_file} => prompt_set_id={p_set_id}, trait='{db_trait_name}'")

    return results

def main():
    mgr = ExperimentManager("results/database/experiments.db")

    # If you want to link all new sets to a certain experiment:
    # e_id = mgr.create_experiment(...)
    e_id = None  # or your real experiment_id

    results = import_trait_specific_csvs_no_new_trait(
        mgr,
        trait_specific_dir="data/psychometric_tests/personality/trait_specific",
        do_link_experiment_id=e_id
    )

    mgr.close()

if __name__ == "__main__":
    main()