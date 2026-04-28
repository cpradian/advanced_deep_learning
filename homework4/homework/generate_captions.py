from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
    get_ego_object,
    get_horizontal_relation,
    get_vertical_relation,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.

    Caption types:
    1. Ego car
       "{kart_name} is the ego car."

    2. Counting
       "There are {num_karts} karts in the scene."

    3. Track name
       "The track is {track_name}."

    4. Relative position
       "{kart_name} is in front of the ego car."
       "{kart_name} is behind the ego car."
       "{kart_name} is left of the ego car."
       "{kart_name} is right of the ego car."
    """
    captions = []

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    ego_object = get_ego_object(kart_objects)

    # 1. Counting
    captions.append(f"There are {len(kart_objects)} karts in the scene.")

    # 2. Track name
    captions.append(f"The track is {track_name}.")

    # If no visible ego kart exists, stop here.
    if ego_object is None:
        return captions

    ego_name = ego_object["kart_name"]
    ego_x, ego_y = ego_object["center"]

    # 3. Ego car
    captions.append(f"{ego_name} is the ego car.")

    # 4. Relative position captions
    other_karts = [kart for kart in kart_objects if kart["instance_id"] != ego_object["instance_id"]]

    for kart in other_karts:
        kart_name = kart["kart_name"]
        kart_x, kart_y = kart["center"]

        horizontal = get_horizontal_relation(kart_x, ego_x)
        vertical = get_vertical_relation(kart_y, ego_y)

        # Vertical caption
        if vertical == "front":
            captions.append(f"{kart_name} is in front of the ego car.")
        else:
            captions.append(f"{kart_name} is behind the ego car.")

        # Horizontal caption
        captions.append(f"{kart_name} is {horizontal} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_dataset(
    data_dir: str = "data/train",
    output_file: str = "data/train/generated_captions.json",
):
    """
    Generate captions for every *_info.json file and every view in a dataset directory.

    Output format:
    [
      {
        "image_file": "train/00000_00_im.jpg",
        "caption": "nolok is the ego car."
      },
      ...
    ]
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)

    all_caption_rows = []

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

            captions = generate_caption(str(info_file), view_index)

            for caption in captions:
                all_caption_rows.append(
                    {
                        "image_file": f"{data_path.name}/{image_name}",
                        "caption": caption,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_caption_rows, f, indent=2)

    print(f"Wrote {len(all_caption_rows)} caption rows to {output_path}")


"""
Usage examples:

Visualize captions for a specific file and view:
    python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0

Generate the full training captions file:
    python -m homework.generate_captions generate --data_dir data/train --output_file data/train/generated_captions.json
"""


def main():
    fire.Fire(
        {
            "check": check_caption,
            "generate": generate_dataset,
        }
    )


if __name__ == "__main__":
    main()