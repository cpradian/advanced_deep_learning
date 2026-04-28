import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.jpg where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def draw_detections(
    image_path: str,
    info_path: str,
    font_scale: float = 0.5,
    thickness: int = 1,
    min_box_size: int = 5,
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        draw.rectangle(
            [(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)],
            outline=color,
            width=thickness,
        )

    return np.array(pil_image)


def extract_kart_objects(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
    min_box_size: int = 5,
) -> list:
    """
    Extract visible kart objects from the info.json file.

    Important:
        The actual ego kart is defined as the visible kart whose bounding-box center
        is closest to the image center after filtering tiny/out-of-bound boxes.

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image after resizing/default model size
        img_height: Height of the image after resizing/default model size
        min_box_size: Minimum box width/height after scaling

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - bbox: scaled bounding box
        - is_center_kart: True if this kart is the detected ego/center kart
    """
    with open(info_path) as f:
        info = json.load(f)

    if view_index >= len(info["detections"]):
        return []

    detections = info["detections"][view_index]
    kart_names = info["karts"]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    kart_objects = []

    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        # Class 1 = Kart
        if class_id != 1:
            continue

        if track_id < 0 or track_id >= len(kart_names):
            continue

        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y

        box_width = x2_scaled - x1_scaled
        box_height = y2_scaled - y1_scaled

        # Keep only karts that are meaningfully visible after scaling.
        if box_width < min_box_size or box_height < min_box_size:
            continue

        # Skip boxes fully outside the image.
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Clamp partially visible boxes to image boundaries.
        x1_clamped = max(0, min(img_width, x1_scaled))
        y1_clamped = max(0, min(img_height, y1_scaled))
        x2_clamped = max(0, min(img_width, x2_scaled))
        y2_clamped = max(0, min(img_height, y2_scaled))

        center_x = (x1_clamped + x2_clamped) / 2
        center_y = (y1_clamped + y2_clamped) / 2

        kart_objects.append(
            {
                "instance_id": track_id,
                "kart_name": kart_names[track_id],
                "center": (center_x, center_y),
                "bbox": (x1_clamped, y1_clamped, x2_clamped, y2_clamped),
                "is_center_kart": False,
            }
        )

    # Define the ego/center kart as the visible kart closest to the image center.
    if kart_objects:
        image_center_x = img_width / 2
        image_center_y = img_height / 2

        closest_kart = min(
            kart_objects,
            key=lambda kart: (
                (kart["center"][0] - image_center_x) ** 2
                + (kart["center"][1] - image_center_y) ** 2
            ),
        )

        closest_id = closest_kart["instance_id"]

        for kart in kart_objects:
            kart["is_center_kart"] = kart["instance_id"] == closest_id

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)

    return info["track"]


def get_ego_object(kart_objects: list) -> dict | None:
    """
    Return the detected ego/center kart object.
    """
    for kart in kart_objects:
        if kart.get("is_center_kart", False):
            return kart
    return None


def get_horizontal_relation(other_x: float, ego_x: float) -> str:
    """
    Determine whether another kart is left or right of the ego kart.
    """
    if other_x < ego_x:
        return "left"
    return "right"


def get_vertical_relation(other_y: float, ego_y: float) -> str:
    """
    Determine whether another kart is front or back of the ego kart.

    Image coordinates have y increasing downward, so a smaller y-value means the
    object is higher in the image and therefore in front.
    """
    if other_y < ego_y:
        return "front"
    return "back"


def generate_qa_pairs(
    info_path: str,
    view_index: int,
    img_width: int = 150,
    img_height: int = 100,
) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image
        img_height: Height of the image

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    ego_object = get_ego_object(kart_objects)

    # 1. Total visible karts question
    qa_pairs.append(
        {
            "question": "How many karts are there in the scenario?",
            "answer": str(len(kart_objects)),
        }
    )

    # 2. Track information question
    qa_pairs.append(
        {
            "question": "What track is this?",
            "answer": track_name,
        }
    )

    # If no visible ego/center kart exists, do not generate ego/relative questions.
    if ego_object is None:
        return qa_pairs

    ego_name = ego_object["kart_name"]
    ego_x, ego_y = ego_object["center"]

    # 3. Ego car question
    qa_pairs.append(
        {
            "question": "What kart is the ego car?",
            "answer": ego_name,
        }
    )

    left_count = 0
    right_count = 0
    front_count = 0
    back_count = 0

    other_karts = [kart for kart in kart_objects if kart["instance_id"] != ego_object["instance_id"]]

    for kart in other_karts:
        kart_name = kart["kart_name"]
        kart_x, kart_y = kart["center"]

        horizontal = get_horizontal_relation(kart_x, ego_x)
        vertical = get_vertical_relation(kart_y, ego_y)

        if horizontal == "left":
            left_count += 1
        else:
            right_count += 1

        if vertical == "front":
            front_count += 1
        else:
            back_count += 1

        # 4. Relative position questions
        qa_pairs.append(
            {
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": horizontal,
            }
        )

        qa_pairs.append(
            {
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": vertical,
            }
        )

        qa_pairs.append(
            {
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": f"{vertical} and {horizontal}",
            }
        )

    # 5. Counting questions
    qa_pairs.append(
        {
            "question": "How many karts are to the left of the ego car?",
            "answer": str(left_count),
        }
    )

    qa_pairs.append(
        {
            "question": "How many karts are to the right of the ego car?",
            "answer": str(right_count),
        }
    )

    qa_pairs.append(
        {
            "question": "How many karts are in front of the ego car?",
            "answer": str(front_count),
        }
    )

    qa_pairs.append(
        {
            "question": "How many karts are behind the ego car?",
            "answer": str(back_count),
        }
    )

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(info_file, view_index)

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_dataset(
    data_dir: str = "data/train",
    output_file: str = "data/train/generated_qa_pairs.json",
):
    """
    Generate QA pairs for every *_info.json file and every view in a dataset directory.

    Args:
        data_dir: Directory containing *_info.json files and corresponding images.
        output_file: Path where the generated QA-pairs JSON should be saved.
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)

    all_qa_pairs = []

    info_files = sorted(data_path.glob("*_info.json"))

    print(f"Found {len(info_files)} info files in {data_path}")

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")

        with open(info_file) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))

        for view_index in range(num_views):
            image_name = f"{base_name}_{view_index:02d}_im.jpg"
            image_path = data_path / image_name

            if not image_path.exists():
                continue

            qa_pairs = generate_qa_pairs(str(info_file), view_index)

            for qa in qa_pairs:
                all_qa_pairs.append(
                    {
                        # Important: image_file should be relative to the data/ directory.
                        # If data_dir is data/train, this becomes train/xxxxx_yy_im.jpg.
                        "image_file": f"{data_path.name}/{image_name}",
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Wrote {len(all_qa_pairs)} QA pairs to {output_path}")


def _info_path_from_image_file(image_file: str, valid_dir: str = "data/valid") -> tuple[Path, int]:
    """
    Convert an image_file string like 'valid/00048_09_im.jpg' into:
        data/valid/00048_info.json, 9
    """
    image_name = Path(image_file).name
    parts = image_name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected image_file format: {image_file}")

    base_name = parts[0]
    view_index = int(parts[1])

    info_path = Path(valid_dir) / f"{base_name}_info.json"

    return info_path, view_index


def validate_against_grader(
    valid_dir: str = "data/valid",
    grader_file: str = "data/valid_grader/balanced_qa_pairs.json",
    max_mismatches: int = 30,
):
    """
    Compare our generated answers against valid_grader/balanced_qa_pairs.json.

    This is for auditing the data pipeline only. Do not train on validation data.

    Args:
        valid_dir: Directory containing validation images and *_info.json files.
        grader_file: Path to the official valid_grader balanced QA file.
        max_mismatches: Maximum number of mismatches to print.
    """
    valid_dir = str(valid_dir)
    grader_path = Path(grader_file)

    with open(grader_path) as f:
        official_rows = json.load(f)

    total = 0
    correct = 0
    missing = 0
    mismatches = []

    cache = {}

    for row in official_rows:
        question = row["question"]
        expected_answer = str(row["answer"])
        image_file = row["image_file"]

        info_path, view_index = _info_path_from_image_file(image_file, valid_dir=valid_dir)

        if not info_path.exists():
            missing += 1
            mismatches.append(
                {
                    "image_file": image_file,
                    "question": question,
                    "expected": expected_answer,
                    "generated": "<missing info file>",
                }
            )
            total += 1
            continue

        cache_key = (str(info_path), view_index)

        if cache_key not in cache:
            generated_pairs = generate_qa_pairs(str(info_path), view_index)
            cache[cache_key] = {qa["question"]: str(qa["answer"]) for qa in generated_pairs}

        generated_answer = cache[cache_key].get(question)

        total += 1

        if generated_answer == expected_answer:
            correct += 1
        else:
            if generated_answer is None:
                generated_answer = "<question not generated>"

            mismatches.append(
                {
                    "image_file": image_file,
                    "question": question,
                    "expected": expected_answer,
                    "generated": generated_answer,
                }
            )

    accuracy = correct / total if total else 0.0

    print(f"Checked {total} official QA pairs")
    print(f"Correct: {correct}")
    print(f"Missing info files: {missing}")
    print(f"Alignment accuracy: {accuracy:.4f}")

    if mismatches:
        print()
        print(f"Showing up to {max_mismatches} mismatches:")
        print("-" * 80)

        for mismatch in mismatches[:max_mismatches]:
            print(f"Image: {mismatch['image_file']}")
            print(f"Q: {mismatch['question']}")
            print(f"Expected:  {mismatch['expected']}")
            print(f"Generated: {mismatch['generated']}")
            print("-" * 80)


"""
Usage examples:

Visualize QA pairs for a specific file and view:
    python -m homework.generate_qa check --info_file data/train/00000_info.json --view_index 0

Generate the full training QA file:
    python -m homework.generate_qa generate --data_dir data/train --output_file data/train/generated_qa_pairs.json

Validate generated logic against the provided valid_grader balanced QA file:
    python -m homework.generate_qa validate --valid_dir data/valid --grader_file data/valid_grader/balanced_qa_pairs.json
"""


def main():
    fire.Fire(
        {
            "check": check_qa_pairs,
            "generate": generate_dataset,
            "validate": validate_against_grader,
        }
    )


if __name__ == "__main__":
    main()