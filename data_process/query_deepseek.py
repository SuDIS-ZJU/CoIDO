import os
import json
import requests
import argparse
import multiprocessing
import time
from tqdm import tqdm


def warp_inst(text, dimension="quality"):
    """Wrap instruction text for LLM evaluation."""
    inst = f"""We would like to request your feedback on the performance of AI assistant in response to the user's questions in the conversation displayed following.

Conversation: {text}

Please rate according to the {dimension} of the responses to the questions. The assistant should receive a score on a scale of 0 to 10, where a higher score indicates higher level of the {dimension}. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.

**Do not include any thinking process or intermediate steps. Directly provide the final output.**"""
    return inst


def query_deepseek(unique_idx):
    """Query DeepSeek API for a single data sample."""
    unique_idx = str(unique_idx)
    response_save_path = os.path.join(
        args.response_dir, args.deepseek_model_name, unique_idx + ".json"
    )
    success = 1

    if os.path.exists(response_save_path):
        return success
    
    text = text_data[unique_idx]["text"]
    prompt = warp_inst(text)
    
    try:
        headers = {
            "Authorization": f"Bearer {args.deepseek_api_key}",
            "Content-Type": "application/json"
        }

        # Split conversation into user and assistant messages
        parts = text.split("ASSISTANT:")
        messages = []
        for i in range(len(parts)):
            if i == 0:
                user_msg = parts[i].replace("USER:", "").strip()
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
            else:
                user_part, assistant_msg = parts[i].split("USER:", 1) if "USER:" in parts[i] else (parts[i], "")
                assistant_msg = assistant_msg.strip()
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
                user_msg = user_part.strip()
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})

        # Add evaluation prompt as the last user message
        evaluation_prompt = "Please rate according to the quality of the responses to the questions. The assistant should receive a score on a scale of 0 to 10, where a higher score indicates higher level of the quality. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias."
        messages.append({"role": "user", "content": evaluation_prompt})

        data = {
            "model": args.deepseek_model_name,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }

        response = requests.post(
            args.deepseek_api_url,
            headers=headers,
            json=data
        )
        response.raise_for_status()
        response_json = response.json()

        with open(response_save_path, "w") as f:
            json.dump(response_json, f)
            
    except requests.exceptions.HTTPError:
        success = 0
    except Exception:
        success = 0
    finally:
        time.sleep(0.1)

    return success


def main():
    """Main function for querying DeepSeek API."""
    global args
    global text_data

    parser = argparse.ArgumentParser(description="Query DeepSeek API to get quality ratings")

    # Index range: [start, end)
    parser.add_argument("--start", type=int, help="Start unique index")
    parser.add_argument("--end", type=int, help="End unique index")
    parser.add_argument(
        "--response_dir", 
        type=str, 
        default="./data/deepseek_responses",
        help="Directory to save API responses"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="./data/coco/train2017",
        help="Directory containing images"
    )
    parser.add_argument(
        "--text_data_path", 
        type=str, 
        default="./data/text_data.json",
        help="Path to the text data file"
    )

    # DeepSeek related parameters
    parser.add_argument(
        "--deepseek_api_key", 
        type=str, 
        default="your-deepseek-api-key-here",
        help="DeepSeek API Key"
    )
    parser.add_argument(
        "--deepseek_api_url", 
        type=str, 
        default="https://api.deepseek.com/v1/chat/completions",
        help="DeepSeek API URL"
    )
    parser.add_argument(
        "--deepseek_model_name", 
        type=str, 
        default="deepseek-chat",
        help="DeepSeek model name"
    )

    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.response_dir, args.deepseek_model_name), exist_ok=True)

    with open(args.text_data_path, "r") as f:
        text_data = json.load(f)

    if args.end == -1:
        args.end = len(text_data)

    pool = multiprocessing.Pool(processes=args.pool_size)
    tasks = range(args.start, args.end)
    results_generator = pool.imap_unordered(query_deepseek, tasks)

    tot_cnt = 0
    tot_success = 0
    with tqdm(total=len(tasks), desc="Processing data") as pbar:
        for res in results_generator:
            tot_cnt += 1
            tot_success += res
            pbar.update(1)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()