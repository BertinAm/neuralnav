"""Minimal flat-color SVG avatars rendered as data URIs — deliberately
geometric/abstract (a diamond mark for the assistant, a generic person
glyph for the user) rather than robot/face emoji, so the UI reads as a
clean product rather than "look, an AI."
"""
import base64

_ASSISTANT_SVG = """
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="nnAssistantBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#E2876A"/>
      <stop offset="1" stop-color="#C65D3B"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="32" fill="url(#nnAssistantBg)"/>
  <path d="M32 15 L41.5 32 L32 49 L22.5 32 Z" fill="#FFFFFF"/>
  <path d="M32 23 L37 32 L32 41 L27 32 Z" fill="url(#nnAssistantBg)" opacity="0.4"/>
</svg>
""".strip()

_USER_SVG = """
<svg width="64" height="64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="nnUserBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#57534E"/>
      <stop offset="1" stop-color="#312D2A"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="32" fill="url(#nnUserBg)"/>
  <circle cx="32" cy="24" r="11" fill="#F5F4EF"/>
  <path d="M10 56 C10 39 19 31 32 31 C45 31 54 39 54 56 Z" fill="#F5F4EF"/>
</svg>
""".strip()


def _data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


ASSISTANT_AVATAR = _data_uri(_ASSISTANT_SVG)
USER_AVATAR = _data_uri(_USER_SVG)
