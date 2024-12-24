import numpy as np
import os
import sys
import pickle

# Add the tokenizer library path
dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../lib")
)
sys.path.append(dir_path)

from hr import generate_prompts_all_rows, pair_matches
from utils.tokenizer import BPETokenizer, END_CHAR

# Initialize tokenizer
tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../../saves','tokenizers/fineweb-edu-1024.tok'))


# Define tokenizing function
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [
        tokenizer.eot_token
    ]  # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer.encode(doc["text"]))
    tokens.append(tokenizer.eot_token)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


# Paths to dataset files
job_file = "assets/hr/company_dataset_job_offers.csv"
profile_file = "assets/hr/company_dataset_candidate_profiles.csv"
match_file = "assets/hr/company_dataset_matching_scores.csv"

# Load matches
matches = pair_matches(job_file, profile_file, match_file)

# Generate prompts for all matches
all_prompts = generate_prompts_all_rows(matches)

# Convert prompts to tokenized format with length and scores
tokenized_prompts = []
for prompt_data in all_prompts:
    tokenized_prompt = tokenize({"text": prompt_data["prompt"]})
    length = len(tokenized_prompt)
    score = prompt_data["score"]
    tokenized_prompts.append((length, score, tokenized_prompt))

# Save as a binary file
output_path = "tokenized_prompts.bin"
with open(output_path, "wb") as f:
    for length, score, tokens in tokenized_prompts:
        # Save length, score, and tokens in binary format
        f.write(np.array([length], dtype=np.uint16).tobytes())
        f.write(np.array([score], dtype=np.float32).tobytes())
        f.write(tokens.tobytes())
print(f"Tokenized prompts saved to {output_path}")

# Dynamic loading function for training
def load_prompts(file_path):
    with open(file_path, "rb") as f:
        while True:
            # Read length
            length_data = f.read(2)  # uint16 is 2 bytes
            if not length_data:
                break
            length = np.frombuffer(length_data, dtype=np.uint16)[0]
            
            # Read score
            score_data = f.read(4)  # float32 is 4 bytes
            score = np.frombuffer(score_data, dtype=np.float32)[0]
            
            # Read tokens
            tokens_data = f.read(length * 2)  # uint16 tokens, each 2 bytes
            tokens = np.frombuffer(tokens_data, dtype=np.uint16)
            
            yield score, tokens

# Example: Load and print prompts dynamically
for score, tokens in load_prompts(output_path):
    print(f"Score: {score}, Tokens: {tokens[:10]}...")  # Print first 10 tokens for brevity
