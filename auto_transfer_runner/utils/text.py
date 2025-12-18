import re


def sanitize_name(name: str) -> str:
    """
    将字符串转换为安全文件名片段：
    - 空白 -> 下划线
    - 非字母数字/._- -> 下划线
    """
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", name)
    return name
