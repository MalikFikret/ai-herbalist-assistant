"""Reusable UI widgets and shared rendering helpers.

Contains the header bar, sidebar headers, copy/sources/feedback controls,
advanced settings popover, and navigation helpers used across pages.
"""

import html as _html
import json
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components

from .auth import (
    AVAILABLE_MODELS,
    AVAILABLE_WEB_SEARCH_PROVIDERS,
    DEFAULT_MODEL,
    DEFAULT_WEB_SEARCH_PROVIDER,
)
from .i18n import get_string
from .state import update_message_feedback
from .styles import _auth_hero_logo_data_uri


def _brand_logo_html(
    *,
    wrapper_class: str,
    lang: str,
    fallback_eyebrow_class: str = "",
) -> str:
    """Brand logo image (login + chat hero). Falls back to leaf/sparkle eyebrow when PNG is absent."""
    uri = _auth_hero_logo_data_uri()
    if uri:
        alt = _html.escape(str(get_string(lang, "app_title")))
        return (
            f'<div class="{wrapper_class}">'
            f'<img src="{uri}" alt="{alt}" loading="lazy" decoding="async" />'
            f"</div>"
        )
    if fallback_eyebrow_class:
        return (
            f'<p class="{fallback_eyebrow_class}" aria-hidden="true">'
            f'<span class="{fallback_eyebrow_class}__leaf">🌿</span>'
            f'<span class="{fallback_eyebrow_class}__sparkle">✦</span>'
            f'<span class="{fallback_eyebrow_class}__sparkle">✦</span>'
            f"</p>"
        )
    return ""


def _section_nav_label(lang: str, section_id: str) -> str:
    return {
        "Chat": get_string(lang, "nav_chat"),
        "Profile": get_string(lang, "profile"),
        "Login": get_string(lang, "sidebar_login"),
        "Admin Panel": get_string(lang, "admin_dashboard"),
    }.get(section_id, section_id)


def _on_guest_top_nav() -> None:
    """Sync ``active_page`` with Chat/Profile radio (Login is a separate control)."""
    pick = st.session_state.get("ha_nav_guest_top")
    if pick in ("Chat", "Profile"):
        st.session_state["active_page"] = pick


def _render_header(*, in_sidebar: bool = False) -> None:
    lang = st.session_state.get("language", "en")
    raw_title = get_string(lang, "app_title")
    title = raw_title if isinstance(raw_title, str) else "AI Herbalist Assistant"
    safe_title = (
        title.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .strip()
    )
    target = st.sidebar if in_sidebar else st
    target.markdown(
        f'''
        <style>
        [data-testid="stHeader"] {{
            background: var(--ha-primary) !important;
            border-bottom: 1px solid rgba(0, 0, 0, 0.2) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
            min-height: 3.45rem !important;
            position: relative !important;
            z-index: 1000010 !important;
        }}
        [data-testid="stHeader"]::after {{
            content: "{safe_title}";
            position: absolute;
            left: clamp(3.5rem, 12vw, 4.25rem);
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--ha-shell-2) !important;
            pointer-events: none;
            font-family: "Lora", serif !important;
            line-height: 1.2;
        }}
        /* Deploy / header actions: calmer chrome */
        [data-testid="stHeader"] a {{
            color: #ffffff !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            opacity: 1;
            background: #2F4F4F !important;
            padding: 0.4rem 1rem !important;
            border-radius: 8px !important;
            border: none !important;
            text-decoration: none !important;
        }}
        [data-testid="stHeader"] a:hover {{
            background: #243c3c !important;
            opacity: 1;
        }}
        /* Make hamburger menu white */
        [data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stHeader"] [data-testid="stIconMaterial"] {{
            color: var(--ha-shell-2) !important;
            fill: var(--ha-shell-2) !important;
        }}
        /* Main column: keep content snug under header */
        section.main .block-container {{
            padding-top: 0.5rem !important;
        }}
        section.main:has(.st-key-ha_chat_composer_row) .block-container {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}
        section.main:has(.st-key-ha_profile_page) .block-container {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}
        </style>
        ''',
        unsafe_allow_html=True,
    )


def _render_sidebar_user_header(lang: str, username: str) -> None:
    """Premium sidebar top: nav label + user card (markup/CSS only)."""
    uname = (username or "").strip()
    initial = _html.escape((uname[:1] or "?").upper())
    safe_user = _html.escape(uname or "—")
    nav_lbl = _html.escape(get_string(lang, "sidebar_nav"))
    hint = _html.escape(get_string(lang, "signed_in_as"))
    eyebrow_mod = " ha-sidebar-header__eyebrow--tr" if lang == "tr" else ""
    st.markdown(
        f"""
<div class="ha-sidebar-header">
  <div class="ha-sidebar-header__eyebrow{eyebrow_mod}" role="heading" aria-level="2">{nav_lbl}</div>
  <div class="ha-sidebar-header__user-card">
    <div class="ha-sidebar-header__avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    </div>
    <div class="ha-sidebar-header__user-meta">
      <span class="ha-sidebar-header__hint">{hint}</span>
      <span class="ha-sidebar-header__name">{safe_user}</span>
    </div>
  </div>
</div>
""".strip(),
        unsafe_allow_html=True,
    )


def _render_sidebar_guest_header(lang: str) -> None:
    """Sidebar top when not signed in."""
    guest = _html.escape(get_string(lang, "guest_display_name"))
    nav_lbl = _html.escape(get_string(lang, "sidebar_nav"))
    hint = _html.escape(get_string(lang, "signed_in_as"))
    eyebrow_mod = " ha-sidebar-header__eyebrow--tr" if lang == "tr" else ""
    
    # Check for English "Signed in as" and manually append a colon to match the mock
    hint_lower = hint.lower()
    if hint_lower == "signed in as":
        hint_display = "Signed In As:"
    else:
        hint_display = f"{hint}:" if not hint.endswith(":") else hint

    st.markdown(
        f"""
<style>
.ha-guest-profile-card-v2 {{
    background-color: #F7F4EB !important;
    background-image: url("data:image/svg+xml,%3C%3Fxml version='1.0' %3F%3E%3C!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN' 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'%3E%3C!-- Uploaded to: SVG Repo, www.svgrepo.com, Generator: SVG Repo Mixer Tools --%3E%3Csvg fill='%23000000' width='800px' height='800px' viewBox='0 0 100 100' style='fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2;' version='1.1' xml:space='preserve' xmlns='http://www.w3.org/2000/svg' xmlns:serif='http://www.serif.com/' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Cg id='Icon'%3E%3Cpath d='M59.16,30.907c0.312,1.413 0.69,2.832 0.878,4.251c0.002,0.021 0.006,0.042 0.01,0.063c-0.815,0.81 -1.601,1.643 -2.341,2.518c-3.282,3.876 -6.444,8.046 -9.381,12.342c-1.225,-1.759 -1.483,-3.644 -1.631,-5.729c0.024,-0.022 0.048,-0.046 0.07,-0.071c0.945,-1.048 1.334,-2.402 2.123,-3.538c1.126,-1.622 2.421,-3.077 3.384,-4.815c0.855,-1.542 1.189,-3.695 0.979,-5.79c-0.201,-2.003 -0.9,-3.94 -2.028,-5.248c-0.011,-0.02 -0.023,-0.04 -0.035,-0.059c-0.24,-0.386 -0.702,-0.552 -1.119,-0.434c-0.149,0.036 -0.292,0.107 -0.415,0.213c-2.568,1.66 -4.072,4.136 -5.583,6.724c-1.007,1.726 -1.948,3.627 -1.934,5.682c0.016,2.408 1.129,4.764 2.511,6.691c0.19,2.992 0.485,5.6 2.424,8.037c0.024,0.03 0.049,0.058 0.076,0.085l-0.003,0.005c-0.368,0.554 -0.733,1.111 -1.095,1.671c-0.793,1.228 -1.57,2.468 -2.329,3.719c-0.358,0.59 -0.712,1.182 -1.061,1.777c-0.087,0.148 -0.174,0.297 -0.261,0.445c-0.074,0.128 -0.149,0.256 -0.223,0.385c-0.11,0.193 -0.219,0.388 -0.325,0.584c-0.503,0.922 -0.963,1.866 -1.4,2.82l-0.013,0.029c-0.122,-0.587 -0.258,-1.172 -0.382,-1.756c-0.287,-1.354 -0.527,-2.727 -0.764,-4.094c1.242,-1.143 2.356,-2.051 3.183,-3.607c1.305,-2.454 1.664,-5.839 0.849,-8.748c-0.556,-1.984 -1.652,-3.743 -3.319,-4.879c-0.051,-0.064 -0.111,-0.122 -0.179,-0.173c-0.442,-0.331 -1.069,-0.241 -1.4,0.2c-0.747,0.997 -1.677,1.78 -2.246,2.915c-0.932,1.861 -1.168,4.021 -1.137,6.073c0.042,2.809 0.273,5.474 2.216,7.565c-0.045,0.146 -0.057,0.305 -0.029,0.465c0.27,1.566 0.541,3.144 0.87,4.698c0.363,1.716 0.899,3.44 0.666,5.219c-0.001,0.004 -0.001,0.009 -0.002,0.013c-0.502,1.2 -1.006,2.398 -1.547,3.578c-1.054,2.298 -2.138,4.627 -3.164,6.989c-0.045,-3.516 -0.375,-7.098 -0.305,-10.435c0.199,-9.439 1.196,-18.492 0.128,-27.936c0.003,-0.003 0.006,-0.005 0.009,-0.008c1.927,-1.639 2.576,-4.085 3.002,-6.46c0.528,-2.939 0.507,-6.749 -0.654,-9.911c-0.817,-2.228 -2.194,-4.136 -4.284,-5.279c-0.39,-0.213 -0.86,-0.139 -1.166,0.151c-0.057,0.043 -0.109,0.093 -0.156,0.15c-2.467,2.958 -5.144,7.415 -5.656,11.661c-0.364,3.021 0.332,5.941 2.81,8.238c1.158,1.073 2.736,1.445 4.146,2.061l0.003,0.001c0.604,5.613 0.477,11.09 0.241,16.597c-0.974,-0.483 -1.931,-1.052 -2.81,-1.685c-0.008,-0.037 -0.019,-0.074 -0.031,-0.111c-2.042,-5.964 -5.184,-12.862 -11.91,-14.643c-1.944,-0.514 -4.107,-0.507 -6.104,-0.382c-0.551,0.034 -0.97,0.509 -0.936,1.06c0.006,0.085 0.022,0.168 0.047,0.245c-0.156,2.446 -0.042,4.932 0.999,7.205c0.743,1.622 2.385,3.007 3.784,4.05c2.719,2.027 5.867,3.333 9.26,3.745c1.052,0.128 2.136,0.086 3.204,0.067c1.332,1.03 2.862,1.924 4.401,2.614c-0.13,2.822 -0.266,5.657 -0.326,8.528c-0.003,0.115 -0.005,0.231 -0.006,0.348c-0.768,-0.631 -1.576,-1.223 -2.419,-1.721c-0.887,-4.098 -3.982,-7.389 -7.995,-8.63c-2.816,-0.871 -6.577,-0.664 -9.437,-0.001c-0.23,0.053 -0.422,0.181 -0.558,0.353c-0.297,0.234 -0.447,0.628 -0.353,1.02c0.307,1.271 0.838,2.512 1.577,3.59c0.834,1.216 1.641,2.382 2.841,3.274c3.438,2.553 8.558,2.515 12.698,1.994c1.329,0.75 2.558,1.775 3.681,2.803c0.145,4.396 0.619,9.042 -0.069,13.271c-1.037,2.854 -1.897,5.757 -2.44,8.711c-0.1,0.543 0.26,1.064 0.803,1.164c0.542,0.1 1.064,-0.26 1.164,-0.802c0.522,-2.838 1.346,-5.626 2.339,-8.366c0.032,-0.066 0.058,-0.136 0.076,-0.21c0.23,-0.628 0.468,-1.254 0.714,-1.877c0.13,-0.071 0.246,-0.172 0.337,-0.301c2.473,-3.509 5.86,-6.151 9.394,-8.572c0.032,-0.013 0.065,-0.028 0.097,-0.045c2.191,-1.159 4.708,-1.694 7.187,-1.693c0.275,1.98 0.769,4.104 2.142,5.61c1.661,1.822 4.416,2.75 6.755,3.151c1.609,0.276 3.217,0.587 4.85,0.68c1.593,0.092 3.181,-0.039 4.765,0.021c0.552,0.02 1.016,-0.411 1.037,-0.962c0.005,-0.137 -0.017,-0.269 -0.063,-0.39c0.014,-0.123 0.005,-0.25 -0.03,-0.376c-0.525,-1.908 -0.785,-4.32 -2.004,-5.924c-1.05,-1.381 -2.992,-2.338 -4.55,-2.969c-3.776,-1.531 -8.042,-1.998 -11.968,-0.817c-0.911,-0.045 -1.832,-0.025 -2.748,0.063l0.001,-0.001c1.912,-1.298 3.742,-2.63 5.557,-3.955c2.325,-1.399 4.664,-2.564 7.399,-2.863c0.516,-0.056 1.043,-0.064 1.57,-0.051c0.8,2.323 2.508,4.345 4.338,5.921c0.584,0.504 1.136,1.143 1.825,1.503c2.111,1.105 4.9,1.687 7.268,1.469c3.031,-0.279 6.009,-1.841 8.449,-3.593c0.009,-0.006 0.018,-0.013 0.027,-0.02c0.096,-0.047 0.186,-0.112 0.266,-0.192c0.388,-0.392 0.385,-1.026 -0.007,-1.414c-1.623,-1.606 -3.467,-3.445 -5.543,-4.434c-1.372,-0.654 -3.002,-0.83 -4.485,-1.051c-3.342,-0.499 -6.846,-0.546 -10.212,-0.073c-0.071,-0.021 -0.146,-0.033 -0.223,-0.038c-1.15,-0.06 -2.343,-0.14 -3.491,-0.015c-0.129,0.014 -0.258,0.03 -0.386,0.048c1.372,-0.907 2.782,-1.784 4.256,-2.615c2.037,1.162 4.357,1.076 6.693,0.805c3.307,-0.384 6.6,-1.52 9.16,-3.708c1.284,-1.096 2.069,-2.477 3.008,-3.843c0.826,-1.204 1.765,-2.292 2.489,-3.566c0.054,-0.095 0.091,-0.196 0.111,-0.297c0.034,-0.07 0.061,-0.145 0.078,-0.225c0.118,-0.539 -0.224,-1.072 -0.764,-1.19c-6.253,-1.366 -12.4,2.826 -16.975,6.446c-1.473,1.166 -2.973,2.261 -4.41,3.468c-0.089,0.054 -0.171,0.122 -0.242,0.205c-0.064,0.055 -0.12,0.117 -0.166,0.183c-3.987,2.252 -7.516,4.835 -11.1,7.448c-0.019,0.011 -0.037,0.022 -0.056,0.034c-0.065,0.039 -0.125,0.085 -0.177,0.136c-1.758,1.281 -3.531,2.567 -5.381,3.823c-3.935,2.673 -8.161,5.235 -11.669,8.519c0.748,-1.662 1.513,-3.308 2.261,-4.939c1.185,-2.584 2.185,-5.26 3.386,-7.837c0.052,-0.047 0.1,-0.101 0.142,-0.16c1.26,-1.776 2.582,-2.655 4.502,-3.372c1.111,0.213 2.181,0.457 3.328,0.436c1.012,-0.019 2.016,-0.159 3.006,-0.37c0.663,-0.142 1.323,-0.282 1.973,-0.476c1.933,-0.575 4.532,-1.753 6.147,-3.505c1.207,-1.31 1.89,-2.925 1.549,-4.831c-0.051,-0.287 -0.22,-0.523 -0.449,-0.668c-0.115,-0.14 -0.269,-0.25 -0.453,-0.311c-4.541,-1.52 -8.505,0.896 -11.736,3.796c-1.352,1.213 -2.712,2.519 -3.825,3.967c-0.553,0.198 -1.063,0.406 -1.541,0.634c2.694,-4.489 5.695,-8.953 8.886,-13.206c0.057,-0.011 0.113,-0.027 0.169,-0.049c3.008,-1.167 6.033,-1.652 9.215,-1.854c0.074,0.075 0.162,0.138 0.261,0.188c4.752,2.376 11.358,2.853 15.856,-0.542c1.868,-1.41 2.956,-3.476 3.869,-5.563c0.221,-0.506 -0.01,-1.096 -0.516,-1.317c-0.024,-0.011 -0.048,-0.02 -0.072,-0.028c-1.501,-1.274 -4.046,-1.745 -5.894,-1.776c-5.581,-0.096 -9.988,2.894 -13.416,7.029c-2.513,0.15 -4.932,0.466 -7.318,1.111c1.028,-1.304 2.072,-2.584 3.129,-3.832c2.815,-3.325 6.321,-6.049 9.416,-9.122c0.097,0.014 0.198,0.014 0.3,-0.002c10.114,-1.616 19.49,-9.25 17.345,-20.529c-0.011,-0.06 -0.028,-0.117 -0.048,-0.171c0.078,-0.168 0.11,-0.359 0.083,-0.557c-0.074,-0.546 -0.579,-0.93 -1.126,-0.855c-10.808,1.479 -17.038,9.819 -17.557,20.287c-1.864,1.896 -3.895,3.648 -5.851,5.467c-0.232,-1.177 -0.537,-2.352 -0.779,-3.522c2.223,-3.171 3.742,-7.421 3.814,-11.49c0.076,-4.265 -1.422,-8.323 -5.234,-10.849c-0.101,-0.067 -0.209,-0.113 -0.321,-0.139c-0.399,-0.207 -0.904,-0.122 -1.212,0.232c-0.924,1.062 -1.653,2.296 -2.286,3.547c-1.447,2.862 -2.842,7.112 -2.81,10.894c0.035,4.049 1.646,7.566 6.193,8.685Zm9.551,48.097c-0.33,-1.517 -0.566,-3.237 -1.472,-4.429c-0.845,-1.113 -2.453,-1.817 -3.709,-2.326c-2.598,-1.053 -5.456,-1.557 -8.234,-1.245c2.846,0.982 5.302,2.344 7.877,4.205c0.224,0.161 0.275,0.474 0.113,0.698c-0.162,0.223 -0.475,0.274 -0.698,0.112c-2.957,-2.136 -5.75,-3.595 -9.206,-4.565c0.21,1.551 0.554,3.197 1.618,4.364c1.373,1.505 3.683,2.195 5.615,2.527c1.535,0.263 3.069,0.565 4.626,0.655c1.159,0.066 2.316,0.013 3.47,0.004Zm15.278,-11.991c-1.245,-1.221 -2.612,-2.471 -4.133,-3.195c-1.196,-0.57 -2.626,-0.686 -3.92,-0.879c-2.261,-0.338 -4.601,-0.457 -6.915,-0.333c2.919,0.808 5.717,2.122 8.21,3.798c0.229,0.154 0.29,0.465 0.136,0.694c-0.154,0.229 -0.465,0.29 -0.694,0.136c-3.079,-2.071 -6.64,-3.569 -10.312,-4.203c0.776,1.607 2.05,3.005 3.376,4.147c0.471,0.407 0.89,0.955 1.446,1.246c1.789,0.936 4.151,1.434 6.158,1.249c2.354,-0.216 4.659,-1.341 6.648,-2.66Zm-57.239,-2.592c-1.131,-2.503 -3.326,-4.437 -6.012,-5.268c-2.238,-0.692 -5.149,-0.589 -7.558,-0.139c0.252,0.728 0.595,1.428 1.028,2.059c0.707,1.032 1.366,2.043 2.384,2.799c2.777,2.063 6.808,2.097 10.281,1.73c-2.405,-1.231 -4.852,-2.254 -7.516,-2.8c-0.27,-0.056 -0.445,-0.32 -0.389,-0.59c0.055,-0.271 0.32,-0.445 0.59,-0.39c2.54,0.521 4.891,1.458 7.192,2.599Zm34.4,-12.168c-3.642,-1.007 -6.767,1.156 -9.384,3.505c-0.411,0.368 -0.822,0.745 -1.227,1.132c1.67,-0.684 3.34,-1.363 4.859,-2.346c0.232,-0.15 0.542,-0.084 0.692,0.148c0.15,0.231 0.083,0.541 -0.148,0.691c-2.107,1.364 -4.496,2.159 -6.784,3.17c0.392,0.055 0.789,0.089 1.197,0.081c0.885,-0.016 1.762,-0.142 2.627,-0.326c0.611,-0.131 1.22,-0.258 1.82,-0.436c1.382,-0.412 3.172,-1.164 4.532,-2.276c1.066,-0.872 1.878,-1.971 1.816,-3.343Zm23.982,-4.255c-2.283,-0.161 -4.524,0.474 -6.639,1.471c-2.675,1.262 -5.152,3.108 -7.277,4.79c-0.912,0.721 -1.833,1.414 -2.744,2.125c2.914,-0.847 5.835,-2.211 8.187,-4.017c0.219,-0.168 0.533,-0.127 0.701,0.092c0.168,0.218 0.127,0.533 -0.092,0.701c-2.318,1.781 -5.167,3.15 -8.038,4.04c1.1,0.139 2.255,0.02 3.408,-0.114c2.917,-0.339 5.832,-1.312 8.091,-3.241c1.146,-0.98 1.82,-2.235 2.658,-3.455c0.563,-0.821 1.184,-1.582 1.745,-2.392Zm-45.774,-5.798c-0.512,0.565 -1.035,1.11 -1.389,1.817c-0.789,1.576 -0.952,3.411 -0.926,5.148c0.025,1.712 0.062,3.382 0.737,4.83c0.068,-2.079 0.024,-4.205 0.496,-6.24c0.062,-0.268 0.331,-0.436 0.6,-0.374c0.269,0.063 0.436,0.332 0.374,0.6c-0.504,2.171 -0.393,4.451 -0.495,6.659c-0.009,0.199 -0.134,0.366 -0.307,0.438c0.062,0.078 0.126,0.155 0.194,0.23c0.795,-0.714 1.515,-1.401 2.067,-2.44c0.904,-1.699 1.247,-3.938 0.948,-6.036c-0.257,-1.798 -0.986,-3.503 -2.299,-4.632Zm-12.372,11.92c-0.862,-2.377 -1.919,-4.846 -3.368,-6.967c-1.629,-2.383 -3.757,-4.325 -6.71,-5.106c-1.436,-0.38 -3.005,-0.427 -4.515,-0.369c-0.093,1.952 0.023,3.921 0.852,5.731c0.61,1.333 2.011,2.421 3.16,3.278c2.44,1.819 5.263,2.994 8.306,3.364c0.287,0.035 0.576,0.055 0.866,0.067c-1.182,-1.386 -2.281,-2.843 -3.613,-4.096c-1.489,-1.403 -3.228,-2.39 -5.042,-3.314c-0.246,-0.126 -0.344,-0.427 -0.219,-0.673c0.126,-0.246 0.427,-0.344 0.673,-0.218c1.899,0.967 3.714,2.009 5.274,3.477c1.569,1.477 2.821,3.234 4.258,4.828l0.078,-0.002Zm54.162,-16.321c-1.188,-0.821 -3.012,-1.054 -4.344,-1.077c-4.677,-0.08 -8.384,2.355 -11.342,5.72c3.335,-0.791 6.546,-1.462 9.773,-2.743c0.256,-0.101 0.547,0.024 0.649,0.281c0.102,0.256 -0.024,0.547 -0.281,0.649c-2.88,1.142 -5.746,1.81 -8.698,2.501c3.753,1.17 8.157,1.083 11.341,-1.32c1.363,-1.029 2.201,-2.494 2.902,-4.011Zm-31.037,-11.066c-1.915,1.469 -3.09,3.516 -4.313,5.609c-0.83,1.423 -1.672,2.967 -1.661,4.661c0.01,1.516 0.571,2.997 1.344,4.322c0.102,-2.922 0.52,-5.981 1.965,-8.502c0.137,-0.239 0.443,-0.322 0.682,-0.185c0.239,0.137 0.322,0.443 0.185,0.682c-1.297,2.264 -1.703,4.987 -1.819,7.625c0.227,-0.461 0.457,-0.92 0.752,-1.343c1.087,-1.566 2.346,-2.965 3.277,-4.644c0.681,-1.228 0.905,-2.951 0.738,-4.62c-0.133,-1.321 -0.509,-2.618 -1.15,-3.605Zm-18.439,-6.853c-1.739,2.209 -3.515,5.142 -4.335,8.121c-0.835,3.034 -0.672,6.129 1.804,8.426c0.749,0.694 1.724,1.005 2.677,1.343c-1.328,-3.055 -2.447,-5.907 -2.203,-9.356c0.02,-0.276 0.259,-0.483 0.535,-0.464c0.275,0.02 0.482,0.259 0.463,0.534c-0.239,3.365 0.911,6.127 2.22,9.114c1.365,-1.328 1.758,-3.245 2.086,-5.068c0.472,-2.63 0.475,-6.04 -0.563,-8.869c-0.555,-1.511 -1.41,-2.856 -2.684,-3.781Zm27.374,-10.129c-0.573,0.775 -1.051,1.624 -1.484,2.48c-1.325,2.621 -2.624,6.509 -2.595,9.974c0.024,2.827 0.97,5.373 3.753,6.469c-0.197,-3.385 -0.37,-6.986 -0.172,-10.367c0.016,-0.275 0.252,-0.486 0.528,-0.47c0.275,0.016 0.486,0.253 0.47,0.529c-0.191,3.269 -0.03,6.746 0.159,10.03c1.83,-2.805 3.064,-6.422 3.125,-9.895c0.06,-3.369 -0.996,-6.61 -3.784,-8.75Zm25.321,0.207c1.649,9.325 -5.706,15.683 -14.013,17.635c3.169,-2.898 6.547,-5.528 9.111,-9.055c0.162,-0.223 0.113,-0.536 -0.111,-0.698c-0.223,-0.163 -0.536,-0.113 -0.698,0.11c-2.505,3.446 -5.804,6.018 -8.906,8.841c0.889,-8.461 5.991,-15.181 14.617,-16.833Z'/%3E%3C/g%3E%3C/svg%3E") !important;
    background-position: right -5px bottom -5px !important;
    background-repeat: no-repeat !important;
    background-size: 80px !important;
    opacity: 0.6 !important;
    border: 1px solid #D8D0C0 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04) !important;
    border-radius: 16px !important;
    display: flex !important;
    align-items: center !important;
    position: relative !important;
    overflow: hidden !important;
    padding: 15px !important;
}}
.ha-guest-profile-card-v2 .avatar {{
    border-radius: 50% !important;
    box-shadow: inset 0 3px 6px rgba(0,0,0,0.15) !important;
    margin-right: 15px !important;
    background-color: #e0d9cc !important;
    width: 48px !important;
    height: 48px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}}
.ha-guest-profile-card-v2 .avatar svg {{
    width: 28px !important;
    height: 28px !important;
    fill: #6E6659 !important;
    stroke: none !important;
}}
.ha-guest-profile-card-v2 .meta {{
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    z-index: 2 !important;
}}
.ha-guest-profile-card-v2 .hint {{
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    color: #5C5449 !important;
    margin-bottom: 2px !important;
    line-height: 1.2 !important;
}}
.ha-guest-profile-card-v2 .name {{
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #2C2B29 !important;
    line-height: 1.2 !important;
}}
</style>
<div class="ha-sidebar-header">
<<<<<<< HEAD
  <p class="ha-sidebar-header__eyebrow{eyebrow_mod}">{nav_lbl}</p>
  <div class="ha-guest-profile-card-v2">
    <div class="avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
        </svg>
    </div>
    <div class="meta">
      <span class="hint">{hint_display}</span>
      <span class="name">{guest}</span>
=======
  <div class="ha-sidebar-header__eyebrow{eyebrow_mod}" role="heading" aria-level="2">{nav_lbl}</div>
  <div class="ha-sidebar-header__user-card">
    <div class="ha-sidebar-header__avatar" aria-hidden="true">?</div>
    <div class="ha-sidebar-header__user-meta">
      <span class="ha-sidebar-header__hint">{hint}</span>
      <span class="ha-sidebar-header__name">{guest}</span>
>>>>>>> 6d4d6d8fe5a04c9702b14cf27cfd8747b2a7d812
    </div>
  </div>
</div>
""".strip(),
        unsafe_allow_html=True,
    )


def _render_auth_language_switch(lang: str) -> None:
    """Auth language dropdown (EN / TR); changes app language on selection."""
    options = ["en", "tr"]
    current = lang if lang in options else "en"

    def _lang_label(code: str) -> str:
        key = "auth_lang_option_english" if code == "en" else "auth_lang_option_turkish"
        return str(get_string(current, key))

    with st.container(key="ha_auth_lang_header"):
        picked = st.selectbox(
            get_string(current, "auth_lang_label"),
            options=options,
            index=options.index(current),
            format_func=_lang_label,
            key="auth_lang_select",
            label_visibility="collapsed",
        )
    if picked and picked != current:
        st.session_state.language = picked
        st.rerun()


def _render_advanced_settings_widgets(*, model_key: str, web_key: str) -> None:
    """Model + web search controls; syncs ``st.session_state`` each run."""
    selected_model = st.selectbox(
        "LLM Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model)
        if st.session_state.selected_model in AVAILABLE_MODELS
        else AVAILABLE_MODELS.index(DEFAULT_MODEL),
        key=model_key,
    )
    st.session_state.selected_model = selected_model
    web_search_provider = st.radio(
        "Web Search Provider",
        options=AVAILABLE_WEB_SEARCH_PROVIDERS,
        index=AVAILABLE_WEB_SEARCH_PROVIDERS.index(
            st.session_state.web_search_provider
        )
        if st.session_state.web_search_provider in AVAILABLE_WEB_SEARCH_PROVIDERS
        else AVAILABLE_WEB_SEARCH_PROVIDERS.index(DEFAULT_WEB_SEARCH_PROVIDER),
        key=web_key,
    )
    st.session_state.web_search_provider = web_search_provider


def _normalize_sources(sources: List[Any]) -> List[Dict[str, Any]]:
    """Accept both the legacy string shape and the new dict shape.

    Returns a list of dicts ready for the Sources popover. The schema is
    intentionally kept flexible so new ``kind`` values (e.g. ``"url"``) can
    be added without breaking existing chat history.
    """
    normalized: List[Dict[str, Any]] = []
    for src in sources or []:
        if isinstance(src, dict):
            if not src.get("kind"):
                src = {"kind": "pdf", **src}
            normalized.append(src)
        elif isinstance(src, str) and src.strip():
            normalized.append({"kind": "pdf", "file": src.strip(), "page": None})
    return normalized


def _source_entry_label(src: Dict[str, Any]) -> str:
    kind = str(src.get("kind") or "pdf").lower()
    if kind == "url":
        title = str(src.get("title") or src.get("url") or "Link").strip()
        return title
    file_name = str(src.get("file") or "unknown")
    page = src.get("page")
    return f"{file_name} (p. {page})" if page is not None else file_name


def _copy_confirm_text() -> str:
    lang = st.session_state.get("language", "en")
    return get_string(lang, "copy_done")


def _render_copy_button(
    *,
    text: str,
    key: str,
    label: str,
) -> None:
    """Render a small HTML+JS clipboard button.

    Streamlit has no native copy button, and ``st.code`` with its built-in
    copy icon would re-render the full answer as monospace. Instead we
    embed a tiny JS component sized to match the rest of the action row.
    """
    safe_text = json.dumps(text, ensure_ascii=False)
    safe_label = _html.escape(label)
    confirm_text = _html.escape(_copy_confirm_text())
    components.html(
        f"""
        <div class=\\"ha-assistant-copy-wrap\\" style=\\"display:flex;align-items:center;\\">
          <button id=\\"{key}\\" type=\\"button\\" class=\\"ha-assistant-copy-btn\\"
            title=\\"{safe_label}\\" aria-label=\\"{safe_label}\\">
            <svg class=\\"ha-assistant-copy-btn__icon\\" width=\\"18\\" height=\\"18\\"
              viewBox=\\"0 0 24 24\\" fill=\\"none\\" aria-hidden=\\"true\\">
              <rect width=\\"14\\" height=\\"14\\" x=\\"8\\" y=\\"8\\" rx=\\"2\\" ry=\\"2\\"></rect>
              <path d=\\"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2\\"></path>
            </svg>
          </button>
          <style>
            .ha-assistant-copy-btn {{
              border: 1px solid rgba(92, 111, 94, 0.16);
              background: rgba(255, 255, 255, 0.78);
              color: #4a5248;
              border-radius: 10px;
              padding: 0;
              width: 1.85rem;
              height: 1.85rem;
              min-width: 1.85rem;
              min-height: 1.85rem;
              cursor: pointer;
              display: inline-flex;
              align-items: center;
              justify-content: center;
              box-shadow: 0 1px 3px rgba(60, 78, 58, 0.04);
              transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
            }}
            .ha-assistant-copy-btn__icon {{
              stroke: #4a5248;
              stroke-width: 1.75;
              stroke-linecap: round;
              stroke-linejoin: round;
              fill: none;
              display: block;
            }}
            .ha-assistant-copy-btn:hover {{
              background: #ffffff;
              border-color: rgba(92, 111, 94, 0.26);
              box-shadow: 0 2px 8px rgba(60, 78, 58, 0.08);
            }}
            .ha-assistant-copy-btn:active {{
              transform: translateY(1px);
            }}
          </style>
          <script>
            (function() {{
              const btn = document.getElementById(\\"{key}\\");
              if (!btn) return;
              const defaultTitle = \\"{safe_label}\\";
              btn.addEventListener(\\"click\\", async function () {{
                try {{
                  await navigator.clipboard.writeText({safe_text});
                }} catch (err) {{
                  const ta = document.createElement(\\"textarea\\");
                  ta.value = {safe_text};
                  document.body.appendChild(ta);
                  ta.select();
                  try {{ document.execCommand(\\"copy\\"); }} catch (e) {{}}
                  document.body.removeChild(ta);
                }}
                btn.title = \\"{confirm_text}\\";
                setTimeout(function () {{ btn.title = defaultTitle; }}, 1200);
              }});
            }})();
          </script>
        </div>
        """,
        height=36,
        width=44,
    )


def _render_sources_popover(
    *,
    lang: str,
    sources: List[Any],
    message_index: int,
) -> None:
    """Render the Sources popover with one entry per source, keyed for uniqueness."""
    normalized = _normalize_sources(sources)
    if not normalized:
        return

    label_template = get_string(lang, "sources_btn")
    if not isinstance(label_template, str):
        label_template = "Sources ({count})"
    popover_label = label_template.format(count=len(normalized))

    try:
        container = st.popover(popover_label, use_container_width=False)
    except AttributeError:
        # Older Streamlit: fall back to an expander so nothing breaks.
        container = st.expander(popover_label, expanded=False)

    with container:
        for idx, src in enumerate(normalized, start=1):
            kind = str(src.get("kind") or "pdf").lower()
            if kind == "url" and src.get("url"):
                label = str(src.get("title") or src["url"]).strip() or str(src["url"])
                st.markdown(f"{idx}. [{label}]({src['url']})")
            else:
                st.markdown(f"{idx}. **{_source_entry_label(src)}**")


def _render_feedback_controls(
    *,
    lang: str,
    username: str,
    chat_id: str,
    message_index: int,
    current: str | None,
) -> None:
    """Render helpful / unhelpful pair. Clicking toggles (click-again clears)."""
    base_key = f"fb_{chat_id}_{message_index}"

    with st.container(key=f"ha_assistant_feedback_group_{message_index}"):
        col_up, col_down = st.columns(2, gap="small", vertical_alignment="center")
        with col_up:
            if st.button(
                "",
                key=f"{base_key}_up",
                icon=":material/thumb_up:",
                type="primary" if current == "up" else "secondary",
                help=get_string(lang, "feedback_up_help"),
                use_container_width=True,
            ):
                new_value: str | None = None if current == "up" else "up"
                if update_message_feedback(
                    username=username,
                    chat_id=chat_id,
                    message_index=message_index,
                    feedback=new_value,
                ):
                    st.toast(get_string(lang, "feedback_saved"))
                    st.rerun()
        with col_down:
            if st.button(
                "",
                key=f"{base_key}_down",
                icon=":material/thumb_down:",
                type="primary" if current == "down" else "secondary",
                help=get_string(lang, "feedback_down_help"),
                use_container_width=True,
            ):
                new_value = None if current == "down" else "down"
                if update_message_feedback(
                    username=username,
                    chat_id=chat_id,
                    message_index=message_index,
                    feedback=new_value,
                ):
                    st.toast(get_string(lang, "feedback_saved"))
                    st.rerun()


def _render_assistant_action_row(
    *,
    lang: str,
    username: str,
    chat_id: str,
    message_index: int,
    message: Dict[str, Any],
    with_feedback: bool = True,
) -> None:
    """Render the Copy / Sources / feedback row under an assistant message."""
    if not chat_id:
        return

    content = str(message.get("content", ""))
    sources = message.get("sources", []) or []
    current_feedback = message.get("feedback")

    has_sources = bool(_normalize_sources(sources))
    with st.container(key=f"ha_assistant_actions_row_{message_index}"):
        if with_feedback:
            if has_sources:
                col_copy, col_src, col_fb, spacer = st.columns(
                    [0.72, 1.65, 0.72, 6.6], vertical_alignment="center"
                )
            else:
                col_copy, col_fb, spacer = st.columns(
                    [0.72, 0.72, 8.5], vertical_alignment="center"
                )
                col_src = None
        else:
            if has_sources:
                col_copy, col_src, spacer = st.columns(
                    [0.72, 1.65, 7.3], vertical_alignment="center"
                )
            else:
                col_copy, spacer = st.columns([0.72, 9.2], vertical_alignment="center")
                col_src = None
            col_fb = None

        with col_copy:
            with st.container(key=f"ha_assistant_copy_cell_{message_index}"):
                _render_copy_button(
                    text=content,
                    key=f"copy_{chat_id}_{message_index}",
                    label=get_string(lang, "copy_btn"),
                )

        if col_src is not None and has_sources:
            with col_src:
                with st.container(key=f"ha_assistant_sources_cell_{message_index}"):
                    _render_sources_popover(
                        lang=lang,
                        sources=sources,
                        message_index=message_index,
                    )

        if with_feedback and col_fb is not None:
            with col_fb:
                _render_feedback_controls(
                    lang=lang,
                    username=username,
                    chat_id=chat_id,
                    message_index=message_index,
                    current=current_feedback,
                )

        del spacer  # reserved column, intentionally unused


def _extract_blocked_herbs(text: str | None) -> List[str]:
    """Extract blocked herbs from text based on known aliases."""
    if not text:
        return []
    text_lower = text.lower()
    blocked = []
    if "chamomile" in text_lower or "papatya" in text_lower:
        blocked.append("papatya/chamomile")
    return blocked

