"""Minimal flat-color SVG avatars rendered as data URIs — deliberately
geometric/abstract (a diamond mark for the assistant, a generic person
glyph for the user) rather than robot/face emoji, so the UI reads as a
clean product rather than "look, an AI."
"""
import base64

_ASSISTANT_SVG = """
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="32" fill="#D97757"/>
  <path d="M32 14 L42 32 L32 50 L22 32 Z" fill="#FFFFFF" opacity="0.95"/>
</svg>
""".strip()

_USER_SVG = """
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <circle cx="32" cy="32" r="32" fill="#44403A"/>
  <circle cx="32" cy="25" r="10" fill="#F5F4EF"/>
  <path d="M12 54 C12 40 20 33 32 33 C44 33 52 40 52 54 Z" fill="#F5F4EF"/>
</svg>
""".strip()


def _data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


ASSISTANT_AVATAR = _data_uri(_ASSISTANT_SVG)
USER_AVATAR = _data_uri(_USER_SVG)
