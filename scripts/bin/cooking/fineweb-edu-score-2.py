import numpy as np
import os
import sys

dir_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../lib")
)
# setting path
sys.path.append(dir_path)


from utils.tokenizer import BPETokenizer


tokenizer = BPETokenizer(special_tokens={"<|endoftext|>": 1023})
TOK_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../../saves/tokenizers/fineweb-edu-1024.tok",
    )
)
if os.path.exists(TOK_PATH):
    tokenizer = BPETokenizer.load(TOK_PATH)
    if __name__ == "__main__":
        print("Loaded tokenizer of size :", len(tokenizer))


# Defining functions
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [
        tokenizer.eot_token
    ]  # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer.encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


if __name__ == "__main__":
    import multiprocessing as mp
    from datasets import load_dataset
    from tqdm import tqdm

    def write_datafile(filename, tokens_np):
        np.save(filename, tokens_np)

    print("Loading the dataset: `fineweb-edu-score-2` ...")
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
    )
    print("Loaded!\n")

    # Params
    shard_size = 100_000_000  # tokens
    max_tokens = 2_000_000_000 # max number of tokens

    DATA_CACHE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../assets/", "fineweb-edu-score-2"
        )
    )
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    print(f"Executing over {nprocs} cores\n")

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        total_token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit=" tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
                )
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[
                    :remainder
                ]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

            total_token_count += len(tokens)
            if total_token_count > max_tokens:
                token_count = 0
                break

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, all_tokens_np[:token_count])

    print("Done!")
