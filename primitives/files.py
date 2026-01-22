"""
文件操作原语

原子操作: 每个函数只做一件事
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def read_file(path: str, encoding: str = "utf-8") -> str:
    """读取文件内容

    Args:
        path: 文件路径
        encoding: 编码格式

    Returns:
        文件内容字符串
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """写入文件内容

    Args:
        path: 文件路径
        content: 要写入的内容
        encoding: 编码格式

    Returns:
        写入的文件路径
    """
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return path


def list_dir(path: str, pattern: str = "*") -> List[Dict[str, Any]]:
    """列出目录内容

    Args:
        path: 目录路径
        pattern: glob 匹配模式

    Returns:
        文件信息列表 [{"name": str, "path": str, "is_dir": bool, "size": int}]
    """
    p = Path(path)
    if not p.exists():
        return []

    results = []
    for item in p.glob(pattern):
        stat = item.stat()
        results.append({
            "name": item.name,
            "path": str(item.absolute()),
            "is_dir": item.is_dir(),
            "size": stat.st_size if item.is_file() else 0,
        })
    return sorted(results, key=lambda x: (not x["is_dir"], x["name"]))


def delete_file(path: str) -> bool:
    """删除文件

    Args:
        path: 文件路径

    Returns:
        是否删除成功
    """
    p = Path(path)
    if p.exists() and p.is_file():
        p.unlink()
        return True
    return False


def file_exists(path: str) -> bool:
    """检查文件是否存在

    Args:
        path: 文件路径

    Returns:
        是否存在
    """
    return Path(path).exists()


def read_json(path: str) -> Any:
    """读取 JSON 文件

    Args:
        path: 文件路径

    Returns:
        解析后的 JSON 对象
    """
    content = read_file(path)
    return json.loads(content)


def write_json(path: str, data: Any, indent: int = 2) -> str:
    """写入 JSON 文件

    Args:
        path: 文件路径
        data: 要写入的数据
        indent: 缩进空格数

    Returns:
        写入的文件路径
    """
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    return write_file(path, content)


def append_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """追加内容到文件

    Args:
        path: 文件路径
        content: 要追加的内容
        encoding: 编码格式

    Returns:
        文件路径
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding=encoding) as f:
        f.write(content)
    return path
