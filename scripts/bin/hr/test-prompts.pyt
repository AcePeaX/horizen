import sys
import os

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../lib")
)
# setting path
sys.path.append(dir_path)


from hr import generate_prompts_all_rows, pair_matches


job_file = "assets/hr/company_dataset_job_offers.csv"
profile_file = "assets/hr/company_dataset_candidate_profiles.csv"
match_file = "assets/hr/company_dataset_matching_scores.csv"
# Example matches
matches = pair_matches(job_file, profile_file, match_file)


# Generate prompts for all matches
all_prompts = generate_prompts_all_rows(matches)

# Process or save the prompts as needed
for prompt in all_prompts[:2]:
    print(prompt)
