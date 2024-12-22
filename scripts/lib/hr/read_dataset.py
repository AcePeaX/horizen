import csv

def load_csv(file_path):
    """
    Load a CSV file into a list of dictionaries.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        list[dict]: A list of dictionaries representing the rows in the CSV file.
    """
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def pair_matches(job_file, profile_file, match_file):
    """
    Pair job offers, profiles, and match scores into a structured matches list.
    
    Args:
        job_file (str): Path to the job offers CSV file.
        profile_file (str): Path to the profiles CSV file.
        match_file (str): Path to the match scores CSV file.
    
    Returns:
        list[dict]: A list of matches with job, profile, and score.
    """
    # Load data from the CSV files
    jobs = load_csv(job_file)
    profiles = load_csv(profile_file)
    matches = load_csv(match_file)

    # Create a lookup dictionary for jobs and profiles
    job_lookup = {job["Job ID"]: job for job in jobs}
    profile_lookup = {profile["Profile ID"]: profile for profile in profiles}

    # Construct the matches
    paired_matches = []
    for match in matches:
        job_id = match["Job ID"]
        profile_id = match["Profile ID"]
        score = float(match["Match Score"])
        
        # Pair job and profile based on IDs
        if job_id in job_lookup and profile_id in profile_lookup:
            paired_matches.append({
                "job": job_lookup[job_id],
                "profile": profile_lookup[profile_id],
                "score": score
            })


    return paired_matches

# Example usage
if __name__ == "__main__":
    job_file = "company_dataset_job_offers.csv"
    profile_file = "company_dataset_candidate_profiles.csv"
    match_file = "company_dataset_matching_scores.csv"

    matches = pair_matches(job_file, profile_file, match_file)
    for match in matches[:5]:  # Display the first 5 matches
        print(match)
