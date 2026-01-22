"""
热熔铜螺母检测器

基于 SAM3 + 颜色分析的检测方案
适用于位置和角度不固定的样品图像
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 添加 primitives 路径
sys.path.insert(0, str(Path(__file__).parent))
from primitives import segment_by_text, load_image, save_image


class CopperNutDetector:
    """热熔铜螺母检测器"""

    def __init__(
        self,
        expected_count: int = 18,
        min_area: int = 300,
        max_area: int = 4000,
        confidence_threshold: float = 0.15,
    ):
        """
        Args:
            expected_count: 预期铜螺母数量
            min_area: 最小面积阈值
            max_area: 最大面积阈值
            confidence_threshold: SAM3 置信度阈值
        """
        self.expected_count = expected_count
        self.min_area = min_area
        self.max_area = max_area
        self.confidence_threshold = confidence_threshold

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        检测图像中的铜螺母

        Args:
            image_path: 图像路径

        Returns:
            {
                "copper_nuts": List[Dict],  # 检测到的铜螺母
                "count": int,               # 检测数量
                "expected": int,            # 预期数量
                "missing": int,             # 缺漏数量
                "status": str,              # "OK" 或 "NG"
                "details": str,             # 详细说明
            }
        """
        # 加载图像
        img = load_image(image_path)
        height, width = img.shape[:2]

        # Step 1: 使用 SAM3 检测圆形区域
        # 尝试多个 prompt 获取最佳结果
        all_detections = []

        for prompt in ['small hole', 'circle', 'hole']:
            results = segment_by_text(image_path, prompt, self.confidence_threshold)
            all_detections.extend(results)

        # 去重 (基于中心点距离)
        unique_detections = self._deduplicate(all_detections, threshold=20)

        # Step 2: 按面积过滤
        size_filtered = []
        for det in unique_detections:
            bbox = det.get('bbox', [])
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if self.min_area <= area <= self.max_area:
                    det['area'] = area
                    det['center'] = ((x1 + x2) / 2, (y1 + y2) / 2)
                    size_filtered.append(det)

        # Step 3: 按颜色过滤 - 识别铜色区域
        copper_nuts = []
        for det in size_filtered:
            bbox = det.get('bbox', [])
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # 提取区域进行颜色分析
            region = img[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
            if region.size == 0:
                continue

            color_type = self._analyze_color(region)
            det['color_type'] = color_type

            # 只保留铜色或金属色区域
            if color_type in ['copper', 'metal', 'dark_hole']:
                copper_nuts.append(det)

        # 构建结果
        count = len(copper_nuts)
        missing = max(0, self.expected_count - count)
        status = "OK" if count >= self.expected_count else "NG"

        if count >= self.expected_count:
            details = f"检测到 {count} 个铜螺母，符合预期"
        elif count > 0:
            details = f"检测到 {count} 个铜螺母，缺漏 {missing} 个"
        else:
            details = "未检测到铜螺母，请检查图像质量"

        return {
            "copper_nuts": copper_nuts,
            "count": count,
            "expected": self.expected_count,
            "missing": missing,
            "status": status,
            "details": details,
        }

    def _analyze_color(self, region: np.ndarray) -> str:
        """分析区域颜色类型"""
        mean_color = region.mean(axis=(0, 1))
        r, g, b = mean_color

        # 计算亮度
        brightness = (r + g + b) / 3

        # 绿色标记
        if g > r * 1.3 and g > b * 1.3 and g > 80:
            return 'green_marker'

        # 深色孔洞 (螺纹孔中心)
        if brightness < 60:
            return 'dark_hole'

        # 铜色/金属色: 偏红偏黄
        if r > g and r > b and r > 100:
            # 检查是否过于偏橙色 (塑料)
            if r > 180 and g > 100 and b < 100:
                return 'orange_plastic'
            return 'copper'

        # 金属灰
        if abs(r - g) < 30 and abs(g - b) < 30 and brightness > 80:
            return 'metal'

        return 'other'

    def _deduplicate(self, detections: List[Dict], threshold: float = 20) -> List[Dict]:
        """去除重复检测"""
        unique = []

        for det in detections:
            bbox = det.get('bbox', [])
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 检查是否与已有检测重复
            is_duplicate = False
            for u in unique:
                ux, uy = u['center']
                dist = np.sqrt((cx - ux)**2 + (cy - uy)**2)
                if dist < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                det['center'] = (cx, cy)
                unique.append(det)

        return unique

    def visualize(
        self,
        image_path: str,
        result: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        可视化检测结果

        Args:
            image_path: 原图路径
            result: detect() 返回的结果
            output_path: 输出图像路径

        Returns:
            输出路径
        """
        img = load_image(image_path)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # 绘制检测框
        for i, nut in enumerate(result['copper_nuts']):
            bbox = nut.get('bbox', [])
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # 根据颜色类型选择框色
                color_type = nut.get('color_type', '')
                if color_type == 'copper':
                    color = '#00FF00'  # 绿色 - 铜色
                elif color_type == 'dark_hole':
                    color = '#00FFFF'  # 青色 - 孔洞
                else:
                    color = '#FFFF00'  # 黄色 - 其他金属

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, y1-15), f"#{i+1}", fill=color)

        # 添加统计信息
        status_color = '#00FF00' if result['status'] == 'OK' else '#FF0000'
        info_text = f"Status: {result['status']} | Count: {result['count']}/{result['expected']}"
        draw.text((10, 10), info_text, fill=status_color)

        pil_img.save(output_path)
        return output_path


def detect_copper_nuts(
    image_path: str,
    expected_count: int = 18,
    visualize: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    便捷函数：检测铜螺母

    Args:
        image_path: 图像路径
        expected_count: 预期数量
        visualize: 是否生成可视化
        output_path: 可视化输出路径

    Returns:
        检测结果字典
    """
    detector = CopperNutDetector(expected_count=expected_count)
    result = detector.detect(image_path)

    if visualize:
        if output_path is None:
            p = Path(image_path)
            output_path = str(p.parent / f"{p.stem}_detected{p.suffix}")
        detector.visualize(image_path, result, output_path)
        result['visualization'] = output_path

    return result


if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/Users/xc-tech/code/vision-agent/golden sample.jpg"

    print(f"检测图像: {image_path}")
    result = detect_copper_nuts(image_path)

    print(f"\n=== 检测结果 ===")
    print(f"状态: {result['status']}")
    print(f"检测数量: {result['count']}")
    print(f"预期数量: {result['expected']}")
    print(f"缺漏数量: {result['missing']}")
    print(f"详情: {result['details']}")

    if 'visualization' in result:
        print(f"\n可视化保存到: {result['visualization']}")
