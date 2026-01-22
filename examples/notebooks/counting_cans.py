"""
Counting Cans 示例（非 Jupyter 版本）

通过本地视觉 API 检测易拉罐数量，并计算库存状态。
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vision_agent_native.primitives import segment_by_text, vision_chat, write_json
from vision_agent_native.primitives.images import load_image, save_image
from vision_agent_native.workspace import Workspace

DEFAULT_IMAGE_URL = "https://drive.usercontent.google.com/u/0/uc?id=1OMt6GmUi-xBW0cDUi1lOEINXmgQVnpOM"


def _download_image(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        data = response.read()
    dest.write_bytes(data)
    return dest


def _resolve_image(image_path: str, image_url: str, workspace: Workspace) -> Path:
    if image_path and os.path.exists(image_path):
        return Path(image_path)

    url = image_url or DEFAULT_IMAGE_URL
    output_path = Path(workspace.input_dir) / "soda_cans.png"
    return _download_image(url, output_path)


def _count_by_llm(image_path: Path) -> Optional[int]:
    prompt = """请数出图像中可见的易拉罐数量，只输出整数。
如果无法判断，请输出 -1。"""

    try:
        image_format = Image.open(image_path).format
    except Exception:
        image_format = None

    if image_format != "PNG":
        image = load_image(str(image_path))
        temp_path = image_path.with_suffix(".png")
        save_image(image, str(temp_path))
        image_path = temp_path

    response = vision_chat([str(image_path)], prompt)
    response = response.strip()

    try:
        count = int(response)
        return count if count >= 0 else None
    except ValueError:
        return None


def _draw_detections(image_path: Path, detections: list, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if max(bbox) <= 1:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 255), width=3)
        score = det.get("score")
        if score is not None:
            draw.text((x1 + 4, y1 + 4), f"{score:.2f}", fill=(0, 180, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run(args: argparse.Namespace) -> dict:
    workspace = Workspace(args.workspace).initialize()

    image_path = _resolve_image(args.image, args.image_url, workspace)
    detections = segment_by_text(
        image=str(image_path),
        prompt=args.prompt,
        threshold=args.threshold,
        api_url=args.api_url or None,
    )

    detected_count = len(detections)
    llm_count = _count_by_llm(image_path)

    count = detected_count
    percentage = (count / args.max_capacity) * 100.0 if args.max_capacity else 0.0
    status = "Healthy" if percentage >= 50 else "Needs Restocking"

    annotated_path = workspace.get_output_path(f"counting_cans_{image_path.stem}.png")
    if detections:
        _draw_detections(image_path, detections, Path(annotated_path))
    else:
        save_image(load_image(str(image_path)), annotated_path)

    serialized_detections = []
    for det in detections:
        det_copy = dict(det)
        if "mask" in det_copy:
            det_copy.pop("mask")
        serialized_detections.append(det_copy)

    result = {
        "image": str(image_path),
        "count": count,
        "max_capacity": args.max_capacity,
        "percentage": round(percentage, 2),
        "status": status,
        "detections": serialized_detections,
        "llm_count": llm_count,
        "detected_count": detected_count,
        "annotated_image": annotated_path,
    }

    output_path = workspace.get_output_path(f"counting_cans_{image_path.stem}.json")
    write_json(output_path, result)

    print(f"SAM3 数量: {detected_count}")
    print(f"LLM 数量: {llm_count if llm_count is not None else 'N/A'}")
    print(f"最终数量: {count}")
    print(f"库存比例: {percentage:.2f}%")
    print(f"状态: {status}")
    print(f"标注图片: {annotated_path}")
    print(f"结果已保存: {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Counting Cans 示例（非 Jupyter）")
    parser.add_argument("--image", default=os.getenv("COUNTING_CANS_IMAGE", ""), help="本地图像路径")
    parser.add_argument(
        "--image-url",
        default=os.getenv("COUNTING_CANS_IMAGE_URL", ""),
        help="图像 URL（为空时使用默认示例）",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("COUNTING_CANS_WORKSPACE", "./counting_cans_workspace"),
        help="工作区目录",
    )
    parser.add_argument(
        "--max-capacity",
        type=int,
        default=int(os.getenv("COUNTING_CANS_MAX", "35")),
        help="最大库存数量",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("COUNTING_CANS_PROMPT", "soda can, beverage can"),
        help="检测提示词",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("COUNTING_CANS_THRESHOLD", "0.3")),
        help="置信度阈值",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("VISION_API_URL", ""),
        help="本地视觉 API 地址",
    )

    args = parser.parse_args()

    if args.image and not os.path.exists(args.image):
        print("指定的图像路径不存在。")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
