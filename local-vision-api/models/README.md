# 模型权重存放目录

此目录用于存放本地模型权重（**不要提交到 GitHub**，已在 `local-vision-api/.gitignore` 中忽略）。

## 推荐文件名 & 下载链接（Ultralytics 官方）

- `yolov8x-worldv2.pt`: https://github.com/ultralytics/assets/releases/latest/download/yolov8x-worldv2.pt
- `sam2_l.pt`: https://github.com/ultralytics/assets/releases/latest/download/sam2_l.pt
- `yolo11x-pose.pt`: https://github.com/ultralytics/assets/releases/latest/download/yolo11x-pose.pt
- `yolo11x-cls.pt`: https://github.com/ultralytics/assets/releases/latest/download/yolo11x-cls.pt

SAM3 权重请参考：
- https://docs.ultralytics.com/models/sam-3/

## 一键下载

在仓库根目录执行：

```bash
bash local-vision-api/scripts/download_models.sh
```

