from __future__ import annotations

import re

_PATTERNS = [
    (re.compile(r"\d{17}[\dXx]"), "[身份证]"),                        # 18位身份证
    (re.compile(r"1[3-9]\d{9}"), "[手机号]"),                         # 手机号
    (re.compile(r"\d{16,19}"), "[银行卡]"),                           # 银行卡号
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[邮箱]"),              # 邮箱
]


def desensitize(text: str) -> str:
    """对文本做脱敏,返回打码后的副本。"""
    if not text:
        return text
    out = text
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return out


def check_access(access_scope: list[str], requester_role: str) -> bool:
    """访问控制:requester_role 是否在允许范围内。
    access_scope 为空 = 公开(所有角色可读)。
    """
    if not access_scope:
        return True
    return requester_role in access_scope
