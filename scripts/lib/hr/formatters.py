
def format_job_basic(job):
    """
    Format job description in a basic informative style.
    """
    return (
        f"{job['Job Title']} at {job['Company Name']} located in {job['Location']} is a "
        f"{job['Employment Type']} position offering a salary of {job['Salary Range']} "
        f"for candidates with {job['Required Experience']} experience. "
        f"The role requires {job['Required Skills']} skills and a minimum education of {job['Required Education']}."
    )

def format_job_persuasive(job):
    """
    Format job description in a persuasive style to attract candidates.
    """
    return (
        f"Join {job['Company Name']} as a {job['Job Title']} in {job['Location']}! "
        f"We offer a competitive salary of {job['Salary Range']} for {job['Required Experience']} professionals. "
        f"In this {job['Employment Type']} role, you'll leverage your expertise in {job['Required Skills']} "
        f"and enjoy opportunities for growth and collaboration. Applicants must have at least a {job['Required Education']}."
    )

def format_job_mission_focused(job):
    """
    Format job description with a focus on the mission and responsibilities.
    """
    return (
        f"As a {job['Job Title']} at {job['Company Name']}, you'll play a key role in {job['Job Description']} "
        f"while collaborating with a dynamic team in {job['Location']}. "
        f"This {job['Employment Type']} position offers a salary range of {job['Salary Range']} "
        f"for candidates with {job['Required Experience']} experience and {job['Required Education']}."
    )

def format_job_minimalist(job):
    """
    Format job description in a minimalist style.
    """
    return (
        f"{job['Job Title']} ({job['Employment Type']}, {job['Location']}): "
        f"{job['Required Experience']} required, {job['Salary Range']}, {job['Required Education']} minimum."
    )

def format_job_storytelling(job):
    """
    Format job description in a storytelling style.
    """
    return (
        f"Imagine yourself as a {job['Job Title']} at {job['Company Name']}, located in the vibrant city of {job['Location']}. "
        f"In this {job['Employment Type']} role, you'll earn {job['Salary Range']} while applying your {job['Required Skills']} skills "
        f"to {job['Job Description']}. If you have {job['Required Experience']} experience and hold a {job['Required Education']}, "
        f"we'd love for you to join our team and make an impact!"
    )

def format_job_skills_first(job):
    """
    Format job description with a focus on skills and qualifications.
    """
    return (
        f"Are you skilled in {job['Required Skills']}? {job['Company Name']} is hiring a {job['Job Title']} in {job['Location']}. "
        f"This {job['Employment Type']} role offers a salary of {job['Salary Range']} for professionals with {job['Required Experience']} experience "
        f"and a {job['Required Education']}."
    )


job_formats = (format_job_basic, format_job_persuasive, format_job_mission_focused, format_job_minimalist, format_job_storytelling, format_job_skills_first)




def format_profile_basic(profile):
    """
    Format candidate profile in a basic informative style, excluding the candidate's name.
    """
    return (
        f"Based in {profile['Location']}, this candidate is seeking a {profile['Desired Employment Type']} "
        f"position with an expected salary of {profile['Expected Salary']}. "
        f"They have {profile['Experience']} experience and hold a {profile['Highest Education']}. "
        f"Key skills include {profile['Skills']}. Career Objective: {profile['Career Objective']}"
    )



def format_profile_persuasive(profile):
    """
    Format candidate profile in a persuasive style to highlight strengths, excluding the name.
    """
    return (
        f"This talented professional from {profile['Location']} is seeking a {profile['Desired Employment Type']} role. "
        f"With {profile['Experience']} of experience and a {profile['Highest Education']}, "
        f"they bring expertise in {profile['Skills']} to the table. Their goal? {profile['Career Objective']}."
    )


def format_profile_skills_focused(profile):
    """
    Format candidate profile with a focus on skills, excluding the name.
    """
    return (
        f"Looking for a candidate skilled in {profile['Skills']}? This professional from {profile['Location']} is "
        f"ready to contribute to your team as a {profile['Desired Employment Type']} professional. With {profile['Experience']} of "
        f"experience and a {profile['Highest Education']}, they aim to {profile['Career Objective']}."
    )



def format_profile_goal_oriented(profile):
    """
    Format candidate profile with a focus on their career goals, excluding the name.
    """
    return (
        f"This motivated professional based in {profile['Location']} is seeking a {profile['Desired Employment Type']} role. "
        f"Their career objective is to {profile['Career Objective']}. With {profile['Experience']} experience and a "
        f"{profile['Highest Education']}, they are proficient in {profile['Skills']}."
    )


def format_profile_minimalist(profile):
    """
    Format candidate profile in a minimalist style, excluding the name.
    """
    return (
        f"{profile['Location']}: {profile['Desired Employment Type']}, "
        f"{profile['Expected Salary']}, {profile['Experience']} experience, {profile['Highest Education']}."
    )


def format_profile_storytelling(profile):
    """
    Format candidate profile in a storytelling style, excluding the name.
    """
    return (
        f"With {profile['Experience']} of experience in {profile['Skills']}, this professional from {profile['Location']} "
        f"is eager to take on a new challenge as a {profile['Desired Employment Type']} professional. "
        f"With a {profile['Highest Education']} and a passion to {profile['Career Objective']}, they are ready to make a meaningful impact."
    )


profile_formats = (format_profile_basic, format_profile_persuasive, format_profile_skills_focused, format_profile_goal_oriented, format_profile_minimalist, format_profile_storytelling)




import random

def generate_prompts_all_rows(matches):
    """
    Generate prompts for all matches by applying random formats to each job and profile.
    Each match will use 3 random pairs of job and profile formats.
    
    Args:
        matches (list of dict): A list of matches, where each match contains job, profile, and score.
        job_formats (tuple): A tuple of job format functions.
        profile_formats (tuple): A tuple of profile format functions.
    
    Returns:
        list: A shuffled list of formatted prompts.
    """
    all_prompts = []

    for match in matches:
        job = match["job"]
        profile = match["profile"]
        score = match["score"]


        # Select 3 random pairs of job and profile formats
        selected_job_formats = random.sample(job_formats, 3)
        selected_profile_formats = random.sample(profile_formats, 3)

        for job_format, profile_format in zip(selected_job_formats, selected_profile_formats):
            job_text = job_format(job)
            profile_text = profile_format(profile)
            length = len(job_text) + len(profile_text)
            prompt = f"{job_text}: {profile_text}"
            all_prompts.append({'length': length, 'score': score, 'prompt': prompt})

    # Shuffle all prompts
    random.shuffle(all_prompts)
    
    return all_prompts
