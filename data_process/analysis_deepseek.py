import os
import json
import tqdm
import sys
import argparse


def get_deepseek_score(response_dir, save_path):
    """Parse the DeepSeek API responses to extract scores."""
    result_dict = {}
    cnt_dict = {}

    for filename in tqdm.tqdm(os.listdir(response_dir), file=sys.stdout):
        with open(os.path.join(response_dir, filename), "r") as f:
            cur_dict = json.load(f)

        unique_idx = filename.split(".")[0]
        output = cur_dict["choices"][0]["message"]["content"]
        score_line = output.strip().split("\n")[0].split("/")[0].split(" ")[-1]

        try:
            cur_score = float(score_line)
        except:
            cur_score = None
        result_dict[unique_idx] = cur_score

        if cur_score not in cnt_dict:
            cnt_dict[cur_score] = 0
        cnt_dict[cur_score] += 1

    with open(save_path, "w") as f:
        json.dump(result_dict, f)


def norm_scores(score_dict: dict):
    """Normalize scores to range [-1, 1]."""
    min_score = min(score_dict.values())
    max_score = max(score_dict.values())

    normed_score_dict = {
        unique_idx: (score - min_score) / (max_score - min_score) * 2 - 1
        for unique_idx, score in score_dict.items()
    }

    return normed_score_dict


def process_none_scores(data_dir, raw_name, output_name):
    """Replace None scores with average score."""
    with open(os.path.join(data_dir, raw_name), "r") as f:
        raw_scores = json.load(f)

    tot_score = 0
    tot_valid_num = 0

    for unique_idx in raw_scores:
        cur_score = raw_scores[unique_idx]
        if cur_score is not None:
            tot_score += cur_score
            tot_valid_num += 1

    avg_score = tot_score / tot_valid_num

    new_result_dict = {}
    for unique_idx in raw_scores:
        cur_score = raw_scores[unique_idx]
        if cur_score is not None:
            new_result_dict[unique_idx] = cur_score
        else:
            new_result_dict[unique_idx] = avg_score

    with open(os.path.join(data_dir, output_name), "w") as f:
        json.dump(new_result_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DeepSeek API responses and extract scores")

    parser.add_argument(
        "--score_dir", 
        type=str, 
        default="./data/scores",
        help="Directory to save extracted scores"
    )
    parser.add_argument(
        "--response_dir", 
        type=str, 
        default="./data/deepseek_responses",
        help="Directory containing DeepSeek API responses"
    )

    parser.add_argument(
        "--model_name", 
        type=str, 
        default="deepseek-chat",
        help="Model name used for responses"
    )
    parser.add_argument(
        "--raw_score_filename", 
        type=str, 
        default="raw_score.json",
        help="Filename for raw scores"
    )
    parser.add_argument(
        "--processed_score_filename", 
        type=str, 
        default="processed_score.json",
        help="Filename for processed scores"
    )
    args = parser.parse_args()

    model_response_dir = os.path.join(args.response_dir, args.model_name)
    model_score_dir = os.path.join(args.score_dir, args.model_name)

    os.makedirs(model_score_dir, exist_ok=True)

    raw_score_path = os.path.join(model_score_dir, args.raw_score_filename)
    processed_score_path = os.path.join(model_score_dir, args.processed_score_filename)

    get_deepseek_score(model_response_dir, save_path=raw_score_path)
    process_none_scores(
        model_score_dir, args.raw_score_filename, args.processed_score_filename
    )
