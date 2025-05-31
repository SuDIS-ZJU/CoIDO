import json
import os
import argparse
import tqdm


def add_idx(raw_annotation_path, new_annotation_save_path):
    """Add unique index to each data sample in the annotation file."""
    if os.path.exists(new_annotation_save_path):
        return
    
    with open(raw_annotation_path, "r") as f:
        raw_annotation = json.load(f)

    new_annotation = []
    for idx, sample in enumerate(raw_annotation):
        sample["unique_idx"] = idx
        new_annotation.append(sample)

    with open(new_annotation_save_path, "w") as f:
        json.dump(new_annotation, f)


def get_input_text(conv):
    """Extract input text from conversation."""
    prompt = (conv[0].strip().split("\n")[-1]).replace("</s>", " ")
    prompt = "USER: " + prompt
    return prompt.strip()


def get_text_instruction(conv):
    """Convert conversation format to text instruction format."""
    prompt = ""
    
    for idx, turn in enumerate(conv):
        value = turn["value"]
        if idx % 2 == 0:
            assert turn["from"] == "human"
            if "<image>" in value:
                value = value.replace("<image>", "").strip()
            prompt += "USER: "
            prompt += value + " "
        else:
            assert turn["from"] == "gpt"
            prompt += "ASSISTANT: "
            prompt += value + " "
    return prompt.strip()


def generate_text_data(annotation_path, text_data_save_path):
    """Generate text data from annotation file."""
    if os.path.exists(text_data_save_path):
        return

    with open(annotation_path, "r") as f:
        anno = json.load(f)

    text_data = {}
    for sample in tqdm.tqdm(anno):
        prompt = get_text_instruction(sample["conversations"])
        unique_idx = sample["unique_idx"]

        if "image" in sample:
            text_data[unique_idx] = {"image": sample["image"], "text": prompt}
        else:
            text_data[unique_idx] = {"text": prompt}

    with open(text_data_save_path, "w") as f:
        json.dump(text_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LLaVA dataset")
    parser.add_argument(
        "--raw_annotation_path", 
        type=str, 
        default="./data/llava_instruct_665k.json",
        help="Path to the raw annotation file"
    )
    parser.add_argument(
        "--new_annotation_save_path",
        type=str,
        default="./data/llava_instruct_665k_add_idx.json",
        help="Path to save the annotation file with unique indices"
    )
    parser.add_argument(
        "--text_data_save_path", 
        type=str, 
        default="./data/text_data.json",
        help="Path to save the extracted text data"
    )
    args = parser.parse_args()
    
    add_idx(args.raw_annotation_path, args.new_annotation_save_path)
    generate_text_data(args.new_annotation_save_path, args.text_data_save_path)
