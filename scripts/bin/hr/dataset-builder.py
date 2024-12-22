import csv
import argparse
import os

# Utility function to get the next incremental ID
def get_next_id(file_name, id_prefix):
    if not os.path.exists(file_name) or os.stat(file_name).st_size == 0:
        return f"{id_prefix}001"  # Start with the first ID if file doesn't exist or is empty
    with open(file_name, mode="r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
        if len(rows) <= 1:  # Only headers present
            return f"{id_prefix}001"
        last_row = rows[-1]
        last_id = last_row[0]
        next_id = int(last_id[len(id_prefix):]) + 1
        return f"{id_prefix}{next_id:03d}"

# Function to write headers if the file is empty
def ensure_headers(file_name, headers):
    if not os.path.exists(file_name) or os.stat(file_name).st_size == 0:
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

# Function to add a job offer entry
def add_job_offer(file_prefix, args):
    file_name = f"{file_prefix}_job_offers.csv"
    headers = [
        "Job ID", "Job Title", "Company Name", "Location", "Employment Type",
        "Salary Range", "Required Experience", "Required Education", "Required Skills", "Job Description"
    ]
    ensure_headers(file_name, headers)
    
    job_id = get_next_id(file_name, "J")
    title = args.job_title or input("Enter Job Title: ")
    company = args.company_name or input("Enter Company Name: ")
    location = args.location or input("Enter Location: ")
    employment_type = args.employment_type or input("Enter Employment Type (Full-time, Part-time, Contract): ")
    salary = args.salary or input("Enter Salary Range (e.g., 70-90K): ")
    experience = args.experience or input("Enter Required Experience (e.g., 2+ years): ")
    education = args.education or input("Enter Required Education (e.g., Bachelor’s): ")
    skills = args.skills or input("Enter Required Skills (comma-separated): ")
    description = args.description or input("Enter Job Description: ")

    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([job_id, title, company, location, employment_type, salary, experience, education, skills, description])
    print(f"Job offer added successfully to {file_name} with Job ID: {job_id}")

# Function to add a candidate profile entry
def add_profile(file_prefix, args):
    file_name = f"{file_prefix}_candidate_profiles.csv"
    headers = [
        "Profile ID", "Candidate Name", "Location", "Desired Employment Type",
        "Expected Salary", "Experience", "Highest Education", "Skills", "Career Objective"
    ]
    ensure_headers(file_name, headers)

    profile_id = get_next_id(file_name, "P")
    name = args.name or input("Enter Candidate Name: ")
    location = args.location or input("Enter Location: ")
    employment_type = args.employment_type or input("Enter Desired Employment Type: ")
    salary = args.salary or input("Enter Expected Salary Range (e.g., 50-60K): ")
    experience = args.experience or input("Enter Total Experience (e.g., 3 years): ")
    education = args.education or input("Enter Highest Education Level: ")
    skills = args.skills or input("Enter Skills (comma-separated): ")
    objective = args.objective or input("Enter Career Objective: ")

    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([profile_id, name, location, employment_type, salary, experience, education, skills, objective])
    print(f"Profile added successfully to {file_name} with Profile ID: {profile_id}")

# Function to add a matching score entry
def add_match(file_prefix, args):
    file_name = f"{file_prefix}_matching_scores.csv"
    headers = ["Job ID", "Profile ID", "Match Score"]
    ensure_headers(file_name, headers)

    job_id = args.job_id or input("Enter Job ID: ")
    profile_id = args.profile_id or input("Enter Profile ID: ")
    score = args.score if args.score!=None else input("Enter Match Score (0 to 1): ")

    with open(file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([job_id, profile_id, score])
    print(f"Matching score added successfully to {file_name}!")

# Main function using argparse
def main():
    parser = argparse.ArgumentParser(description="Add entries to job offer, profile, or matching datasets.")
    parser.add_argument("--job", action="store_true", help="Add a job offer entry.")
    parser.add_argument("--profile", action="store_true", help="Add a candidate profile entry.")
    parser.add_argument("--matching", action="store_true", help="Add a matching score entry.")
    parser.add_argument("--dataset-name", type=str, help="Prefix for dataset file names.", required=True)
    
    # Optional arguments for additional input
    parser.add_argument("--job-title", type=str, help="Job Title.")
    parser.add_argument("--company-name", type=str, help="Company Name.")
    parser.add_argument("--location", type=str, help="Location.")
    parser.add_argument("--employment-type", type=str, help="Employment Type.")
    parser.add_argument("--salary", type=str, help="Salary Range.")
    parser.add_argument("--experience", type=str, help="Required Experience.")
    parser.add_argument("--education", type=str, help="Required Education.")
    parser.add_argument("--skills", type=str, help="Required Skills (comma-separated).")
    parser.add_argument("--description", type=str, help="Job Description.")
    parser.add_argument("--name", type=str, help="Candidate Name.")
    parser.add_argument("--objective", type=str, help="Career Objective.")
    parser.add_argument("--score", type=float, help="Match Score (0 to 1).")


    parser.add_argument("--job-id", type=str, help="Job id for the matching.")
    parser.add_argument("--profile-id", type=str, help="Profile id for the matching.")

    args = parser.parse_args()
    file_prefix = args.dataset_name

    if args.job:
        print("\nAdding a Job Offer:")
        add_job_offer(file_prefix, args)
    if args.profile:
        print("\nAdding a Candidate Profile:")
        add_profile(file_prefix, args)
    if args.matching:
        print("\nAdding a Matching Score:")
        add_match(file_prefix, args)

    if not any([args.job, args.profile, args.matching]):
        print("No type selected. Use --job, --profile, or --matching to add entries.")

if __name__ == "__main__":
    main()
