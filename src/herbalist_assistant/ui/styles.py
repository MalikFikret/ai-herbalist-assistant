"""CSS injection and layout scripts for the Streamlit UI.

Contains all ``st.markdown`` style blocks and ``components.html`` scripts
that control the visual presentation of the application.
"""

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from herbalist_assistant.ui.botanical_assets import BOTANICAL_MAIN_B64, BOTANICAL_SIDEBAR_B64
except ImportError:
    BOTANICAL_MAIN_B64 = ""
    BOTANICAL_SIDEBAR_B64 = ""

# Login hero: bundled logo (replaces "Welcome" heading when present).
_AUTH_LOGO_PATH = Path(__file__).resolve().parent / "static" / "herbalist_logo.png"
# Login background: botanical photo only (no UI/mockup text baked in).
_AUTH_BG_PATH = Path(__file__).resolve().parent / "static" / "login_background.png"
# Login sol panel: dikey mortar / bitki illüstrasyonu.
_AUTH_HERO_LEFT_PATH = Path(__file__).resolve().parent / "static" / "login_hero_left.png"
# Login kartı: tam kutu arka plan (suluboya bitki).
_AUTH_CARD_BG_PATH = Path(__file__).resolve().parent / "static" / "login_card_bg.png"
# Chat shell: single cream paper plate (corner botanicals baked in; center clear).
_CHAT_SHELL_BG_PATH = Path(__file__).resolve().parent / "static" / "chat_shell_background.png"
_HA_SHELL_CREAM = "#F4F2EC"
_HA_AUTH_PAGE_BG = "#F5F1EA"
_HA_SHELL_CREAM_DEEP = "#EBE8E0"
_HA_SIDEBAR_BG = "#F2EBE3"
_HA_SIDEBAR_BEIGE_MID = "#EBE3D8"
_HA_SIDEBAR_EDGE = "rgba(44, 48, 42, 0.085)"
_HA_SIDEBAR_SHADOW = (
    "3px 0 16px rgba(38, 42, 36, 0.04), "
    "1px 0 0 rgba(255, 255, 255, 0.4) inset"
)
_SIDEBAR_TEXTURE_LEAF_PATH = (
    Path(__file__).resolve().parent / "static" / "shell_bg_leaf_tl.svg"
)
_HA_CREAM_SURFACE = "#FDFCF8"
_HA_CHAT_USER_BUBBLE = "#f0ebe4"
_HA_SAGE = "#5c6f5e"
_HA_SAGE_DEEP = "#3e4e35"
_HA_SAGE_SOFT = "#7a9170"
_HA_SAGE_MIST = "rgba(92, 111, 94, 0.14)"
_HA_GLASS = "rgba(255, 255, 255, 0.58)"
_HA_GLASS_STRONG = "rgba(255, 255, 255, 0.78)"
_HA_GLASS_BORDER = "rgba(92, 111, 94, 0.2)"
_HA_INPUT_GLOW = (
    "0 0 0 1px rgba(122, 145, 112, 0.28), "
    "0 0 22px rgba(122, 145, 112, 0.14), "
    "0 4px 20px rgba(60, 78, 58, 0.07)"
)

@st.cache_data(show_spinner=False)
def _auth_hero_logo_data_uri() -> str:
    if not _AUTH_LOGO_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_LOGO_PATH.read_bytes()
    ).decode("ascii")


@st.cache_data(show_spinner=False)
def _auth_background_data_uri(_file_mtime: float) -> str:
    if not _AUTH_BG_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_BG_PATH.read_bytes()
    ).decode("ascii")


def _auth_background_uri_for_css() -> str:
    if not _AUTH_BG_PATH.is_file():
        return ""
    return _auth_background_data_uri(_AUTH_BG_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _auth_hero_left_data_uri(_file_mtime: float) -> str:
    if not _AUTH_HERO_LEFT_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_HERO_LEFT_PATH.read_bytes()
    ).decode("ascii")


def _auth_hero_left_uri_for_css() -> str:
    """Sol login paneli görseli (dikey illüstrasyon)."""
    if not _AUTH_HERO_LEFT_PATH.is_file():
        return _auth_background_uri_for_css()
    return _auth_hero_left_data_uri(_AUTH_HERO_LEFT_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _auth_card_bg_data_uri(_file_mtime: float) -> str:
    if not _AUTH_CARD_BG_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_CARD_BG_PATH.read_bytes()
    ).decode("ascii")


def _auth_card_bg_uri_for_css() -> str:
    """Login kartının tamamı için arka plan görseli."""
    if not _AUTH_CARD_BG_PATH.is_file():
        return ""
    return _auth_card_bg_data_uri(_AUTH_CARD_BG_PATH.stat().st_mtime)


def _guest_auth_card_background_css() -> str:
    uri = _auth_card_bg_uri_for_css()
    if not uri:
        return ""
    return f"""
        .st-key-ha_auth_shell .st-key-ha_auth_card {{
            background-color: #f4f2ec !important;
            background-image: url("{uri}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
        }}
        .st-key-ha_auth_shell .ha-lux-welcome--hero-photo {{
            display: none !important;
        }}
        @media (min-width: 901px) {{
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1) {{
                background: transparent !important;
                border-right: none !important;
            }}
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--panel) {{
                flex: 0.9 1 0% !important;
            }}
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(2) {{
                flex: 1.15 1 0% !important;
                background: transparent !important;
            }}
        }}
    """


@st.cache_data(show_spinner=False)
def _shell_asset_data_uri(path_str: str, file_mtime: float) -> str:
    path = Path(path_str)
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if path.suffix.lower() == ".svg":
        return "data:image/svg+xml;base64," + encoded
    return "data:image/png;base64," + encoded


def _chat_shell_background_uri_for_css() -> str:
    if not _CHAT_SHELL_BG_PATH.is_file():
        return ""
    return _shell_asset_data_uri(
        str(_CHAT_SHELL_BG_PATH),
        _CHAT_SHELL_BG_PATH.stat().st_mtime,
    )


_GUEST_AUTH_CARD_SOLID_CSS = """
            .st-key-ha_auth_shell .st-key-ha_auth_card {
                background-color: transparent !important;
                min-height: min(28rem, 62vh) !important;
            }
"""

_GUEST_AUTH_FLATTEN_LAYERS_CSS = """
        /* Tek login kartı: görsel kabuk yalnızca ha_auth_card; iç sarmalayıcılar düz */
        .st-key-ha_auth_shell > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_auth_shell > [data-testid="stVerticalBlock"],
        .st-key-ha_auth_card > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_auth_card > [data-testid="stVerticalBlock"],
        .st-key-ha_auth_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"],
        .st-key-ha_auth_card [data-testid="column"] > [data-testid="stVerticalBlock"],
        .st-key-ha_auth_card [data-testid="stElementContainer"],
        .st-key-ha_auth_card [data-testid="element-container"],
        .st-key-ha_auth_card .ha-lux-welcome,
        .st-key-ha_auth_card .ha-lux-welcome__inner,
        .st-key-ha_auth_form_card,
        .st-key-ha_auth_form_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {
            background-color: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before,
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::after {
            display: none !important;
            content: none !important;
        }
        .st-key-ha_auth_card .ha-lux-welcome__vine,
        .st-key-ha_auth_card .ha-lux-welcome__ghost-leaf,
        .st-key-ha_auth_card .ha-lux-welcome__rule,
        .st-key-ha_auth_card .ha-lux-botanical,
        .st-key-ha_auth_card .ha-lux-botanical__accent,
        .st-key-ha_auth_card .ha-lux-botanical__photo,
        .st-key-ha_auth_card .ha-lux-pot {
            display: none !important;
        }
"""

_GUEST_AUTH_PAGE_BACKGROUND_CSS = f"""
        /* Login sayfası: düz krem arka plan (yaprak / görsel yok) */
        html:has(.st-key-ha_auth_shell),
        body:has(.st-key-ha_auth_shell),
        [data-testid="stApp"]:has(.st-key-ha_auth_shell),
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell),
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) [data-testid="stMain"],
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) section.main,
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) [data-testid="stMainBlockContainer"],
        [data-testid="stApp"]:has(.st-key-ha_auth_shell) footer,
        [data-testid="stApp"]:has(.st-key-ha_auth_shell) [data-testid="stFooter"] {{
            background-color: {_HA_AUTH_PAGE_BG} !important;
            background-image: none !important;
        }}
"""

_GUEST_AUTH_MOCKUP_CSS = """
        /* Login mockup polish (kutu boyutu aynı; görünüm referans tasarıma yakın) */
        .st-key-ha_auth_shell .st-key-ha_auth_card {
            border: 1px solid rgba(74, 93, 69, 0.1) !important;
            box-shadow: 0 10px 36px rgba(60, 78, 58, 0.07), 0 1px 4px rgba(60, 78, 58, 0.04) !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            padding-top: 2.05rem !important;
        }
        @media (min-width: 901px) {
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):not(:has(.ha-lux-welcome--hero-photo)) {
                border-right: 1px solid rgba(74, 93, 69, 0.1) !important;
                background: linear-gradient(180deg, #f8f6f1 0%, #f3f1eb 100%) !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--hero-photo) {
                border-right: 1px solid rgba(74, 93, 69, 0.1) !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(2)
                > div[data-testid="stVerticalBlock"] {
                padding: 0.25rem 1.55rem 1.65rem 1.55rem !important;
            }
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) .ha-lux-welcome__inner {
            position: relative !important;
            min-height: min(30vh, 15.5rem) !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) .ha-lux-welcome__inner::before {
            content: "" !important;
            position: absolute !important;
            left: 50% !important;
            top: 50% !important;
            width: min(92%, 17.5rem) !important;
            aspect-ratio: 1 !important;
            transform: translate(-50%, -52%) !important;
            border-radius: 50% !important;
            background: radial-gradient(
                circle at 42% 38%,
                rgba(198, 218, 186, 0.55) 0%,
                rgba(220, 232, 210, 0.28) 42%,
                rgba(253, 252, 249, 0) 72%
            ) !important;
            pointer-events: none !important;
            z-index: 0 !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) {
            padding: 1.15rem 1.25rem 1.35rem 1.25rem !important;
        }
        /* Sol panel: botanik arka plan görseli tüm sütunu kaplar */
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }
        /* Sol panel: Welcome + slogan */
        .st-key-ha_auth_shell .ha-lux-welcome--panel {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            min-height: min(27.5rem, 60vh) !important;
            margin: 0 !important;
            padding: 2.5rem 1.35rem 2rem 1.5rem !important;
            box-sizing: border-box !important;
            background: transparent !important;
            text-align: left !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--panel__inner {
            width: 100% !important;
            max-width: 17.5rem !important;
            margin: 0 auto !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--panel__title {
            font-family: "Lora", Georgia, "Times New Roman", serif !important;
            font-size: clamp(1.95rem, 3.6vw, 2.6rem) !important;
            font-weight: 700 !important;
            color: #3e4e35 !important;
            margin: 0 0 0.28rem 0 !important;
            line-height: 1.12 !important;
            letter-spacing: -0.02em !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--panel__deco {
            display: none !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--panel__tagline {
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: clamp(0.9rem, 1.45vw, 1.02rem) !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
            color: #4a5d45 !important;
            margin: 0 !important;
            max-width: 16rem !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub--login-only {
            text-align: left !important;
            margin: 0 0 0.65rem auto !important;
        }
        @media (max-width: 900px) {
            .st-key-ha_auth_shell .ha-lux-welcome--panel {
                min-height: 0 !important;
                padding: 1.35rem 1.25rem 0.85rem 1.25rem !important;
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome--panel__inner {
                max-width: 100% !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome--panel__title,
            .st-key-ha_auth_shell .ha-lux-welcome--panel__tagline {
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome--panel__tagline {
                margin-left: auto !important;
                margin-right: auto !important;
            }
        }
        .st-key-ha_auth_shell .ha-lux-welcome--hero-photo.ha-lux-welcome {
            position: relative !important;
            width: 100% !important;
            min-height: min(28rem, 62vh) !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            background: #f4f2ec !important;
            flex: 1 1 auto !important;
            align-self: stretch !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--hero-photo .ha-lux-welcome__inner {
            position: relative !important;
            width: 100% !important;
            min-height: min(28rem, 62vh) !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 1.15rem 0.85rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--hero-photo .ha-lux-welcome__inner::before {
            display: none !important;
            content: none !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--hero-photo .ha-lux-welcome__hero-photo {
            position: relative !important;
            inset: auto !important;
            display: block !important;
            width: auto !important;
            height: auto !important;
            max-width: min(78%, 13.5rem) !important;
            max-height: min(88%, 24rem) !important;
            margin: 0 auto !important;
            object-fit: contain !important;
            object-position: center center !important;
            border-radius: 0 !important;
        }
        @media (min-width: 901px) {
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--hero-photo) {
                display: flex !important;
                flex-direction: column !important;
                padding: 0 !important;
                overflow: hidden !important;
                min-height: min(28rem, 62vh) !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--hero-photo)
                > div[data-testid="stVerticalBlock"] {
                flex: 1 1 auto !important;
                min-height: min(28rem, 62vh) !important;
                height: auto !important;
                padding: 0 !important;
                margin: 0 !important;
                align-items: stretch !important;
                justify-content: stretch !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--hero-photo)
                [data-testid="stMarkdownContainer"],
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--hero-photo)
                [data-testid="stElementContainer"] {
                flex: 1 1 auto !important;
                width: 100% !important;
                min-height: min(28rem, 62vh) !important;
                height: 100% !important;
                margin: 0 !important;
                padding: 0 !important;
            }
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand .ha-lux-welcome__logo {
            position: relative !important;
            z-index: 1 !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand .ha-lux-welcome__logo img {
            max-width: min(100%, 15.5rem) !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-title,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"]:has(.ha-lux-form-title) {
            text-align: left !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-deco {
            justify-content: flex-start !important;
            gap: 0.5rem !important;
            margin: 0 0 0.35rem 0 !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-deco__line {
            display: block !important;
            width: 1.65rem !important;
            height: 1px !important;
            background: rgba(122, 145, 112, 0.45) !important;
            border-radius: 1px !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-deco__dot {
            display: none !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"]:has(.ha-lux-form-sub) {
            text-align: left !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            max-width: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row [data-testid="stRadio"] {
            width: 100% !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: stretch !important;
            gap: 0.5rem !important;
            width: 100% !important;
            margin: 0 0 1rem 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            overflow: visible !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label {
            flex: 1 1 0 !important;
            min-width: 0 !important;
            width: auto !important;
            min-height: 2.75rem !important;
            height: 2.75rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 9999px !important;
            padding: 0 0.7rem !important;
            margin: 0 !important;
            box-sizing: border-box !important;
            background: #F5F1EA !important;
            border: 1px solid rgba(62, 78, 53, 0.12) !important;
            cursor: pointer !important;
            overflow: visible !important;
            transition: background-color 0.18s ease, border-color 0.18s ease !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-align: center !important;
            margin: 0 !important;
            color: #3e4e35 !important;
            white-space: nowrap !important;
            line-height: 1.2 !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) {
            background: #3e4e35 !important;
            border-color: #3e4e35 !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:not(:has(input:checked)) {
            background: #F5F1EA !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside div[data-baseweb="input"],
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"] {
            border-radius: 14px !important;
            min-height: 50px !important;
            background-color: #ffffff !important;
            background-repeat: no-repeat !important;
            background-size: 1.05rem auto !important;
            border: 1px solid rgba(74, 93, 69, 0.18) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside div[data-baseweb="input"] {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%237a8578' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='4' width='20' height='16' rx='2'/%3E%3Cpath d='m2 7 10 7 10-7'/%3E%3C/svg%3E") !important;
            background-position: 1rem center !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"] {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%237a8578' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='5' y='11' width='14' height='10' rx='2'/%3E%3Cpath d='M8 11V8a4 4 0 0 1 8 0v3'/%3E%3C/svg%3E") !important;
            background-position: 1rem center !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside [data-baseweb="input"] input,
        .st-key-ha_auth_shell [data-testid="stForm"] [data-baseweb="input"] input {
            padding-left: 2.65rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] {
            position: relative !important;
            border-radius: 14px !important;
            min-height: 50px !important;
            font-family: "Inter", system-ui, sans-serif !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            background-color: #3e4e35 !important;
            border: 1px solid #354433 !important;
            box-shadow: 0 4px 14px rgba(62, 78, 53, 0.22) !important;
            padding-right: 2.85rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]::after,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]::after {
            content: "" !important;
            position: absolute !important;
            right: 1.1rem !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 1.05rem !important;
            height: 1.05rem !important;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3c-1.5 2.2-4.5 4.2-4.5 8a4.5 4.5 0 0 0 9 0c0-3.8-3-5.8-4.5-8z'/%3E%3Cpath d='M12 14v7'/%3E%3C/svg%3E") center / contain no-repeat !important;
            opacity: 0.92 !important;
            pointer-events: none !important;
        }
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap button::before {
            content: "🌿" !important;
            margin-right: 0.35rem !important;
            font-size: 0.82rem !important;
            opacity: 0.85 !important;
        }
"""

def _herbal_shell_background_css(
    main_bg_images: str,
    main_bg_positions: str,
    main_bg_sizes: str,
    main_bg_repeats: str,
    main_bg_attachments: str,
) -> str:
    """Sidebar: inner panes flat; shell depth in premium CSS. Main: botanical plate."""
    shell_base = """
        html:not(:has(.st-key-ha_auth_shell)),
        body:not(:has(.st-key-ha_auth_shell)),
        #root:not(:has(.st-key-ha_auth_shell)),
        [data-testid="stApp"]:not(:has(.st-key-ha_auth_shell)),
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
    """
    shell_sidebar = """
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) section[data-testid="stSidebar"] > div,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stSidebar"] > div,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stSidebarHeader"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stSidebarContent"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stSidebarUserContent"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stSidebar"] .block-container
    """
    shell_main = """
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stMain"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) section.main
    """
    return f"""
        {shell_base} {{
            background-color: {_HA_SHELL_CREAM} !important;
            background-image: none !important;
        }}
        {shell_sidebar} {{
            background-color: {_HA_SIDEBAR_BG} !important;
            background-image: none !important;
        }}
        {shell_main} {{
            background-color: {_HA_SHELL_CREAM} !important;
            background-image: {main_bg_images} !important;
            background-position: {main_bg_positions} !important;
            background-size: {main_bg_sizes} !important;
            background-repeat: {main_bg_repeats} !important;
            background-attachment: {main_bg_attachments} !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(244, 242, 236, 0.82) !important;
            backdrop-filter: blur(14px) saturate(1.05) !important;
            -webkit-backdrop-filter: blur(14px) saturate(1.05) !important;
            border-bottom: 1px solid rgba(92, 111, 94, 0.08) !important;
        }}
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) footer,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stFooter"] {{
            background-color: transparent !important;
            background-image: none !important;
        }}
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stMain"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) section.main {{
            position: relative !important;
            overflow: hidden !important;
        }}
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) .main .block-container {{
            background-color: transparent !important;
            background-image: none !important;
            position: relative !important;
            z-index: 1 !important;
        }}
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) [data-testid="stMainBlockContainer"]::before {{
            content: "" !important;
            display: block !important;
            position: absolute !important;
            inset: 0 !important;
            pointer-events: none !important;
            z-index: 0 !important;
            background: radial-gradient(
                ellipse 62% 56% at 50% 44%,
                rgba(255, 253, 248, 0.72) 0%,
                rgba(255, 253, 248, 0.28) 48%,
                transparent 72%
            ) !important;
        }}
    """


def _sidebar_texture_uri_for_css() -> str:
    path = _SIDEBAR_TEXTURE_LEAF_PATH
    if not path.is_file():
        return ""
    return _shell_asset_data_uri(str(path), path.stat().st_mtime)


def _sidebar_shell_surface_layers() -> str:
    """Glass wash + neutral paper grain (no strong green cast)."""
    return f"""linear-gradient(
            168deg,
            rgba(255, 255, 255, 0.36) 0%,
            rgba(255, 255, 255, 0.1) 42%,
            rgba(255, 255, 255, 0.02) 100%
        ),
        radial-gradient(
            ellipse 90% 65% at 108% -8%,
            rgba(205, 200, 188, 0.13) 0%,
            transparent 56%
        ),
        radial-gradient(
            ellipse 75% 58% at -14% 102%,
            rgba(196, 191, 178, 0.1) 0%,
            transparent 60%
        ),
        linear-gradient(180deg, rgba(250, 245, 238, 0.4) 0%, transparent 32%),
        {_HA_SIDEBAR_BG}"""


def _sidebar_botanical_texture_css(sb: str) -> str:
    """Faint corner leaf line-art on the sidebar shell only."""
    texture_uri = _sidebar_texture_uri_for_css()
    if texture_uri:
        return f"""
        {sb}::before {{
            content: "" !important;
            position: absolute !important;
            inset: 0 !important;
            pointer-events: none !important;
            z-index: 0 !important;
            background-image: url("{texture_uri}") !important;
            background-size: min(200px, 50%) auto !important;
            background-position: -14% -4% !important;
            background-repeat: no-repeat !important;
            opacity: 0.2 !important;
        }}
        {sb}::after {{
            content: "" !important;
            position: absolute !important;
            inset: 0 !important;
            pointer-events: none !important;
            z-index: 0 !important;
            background-image: url("{texture_uri}") !important;
            background-size: min(168px, 44%) auto !important;
            background-position: 110% 106% !important;
            background-repeat: no-repeat !important;
            transform: scaleX(-1) rotate(6deg) !important;
            opacity: 0.12 !important;
        }}
        """
    return f"""
        {sb}::before {{
            content: "" !important;
            position: absolute !important;
            inset: 0 !important;
            pointer-events: none !important;
            z-index: 0 !important;
            background-image:
                radial-gradient(circle at 16% 10%, rgba(186, 181, 168, 0.07) 0%, transparent 42%),
                radial-gradient(circle at 90% 92%, rgba(178, 173, 160, 0.06) 0%, transparent 38%) !important;
            background-repeat: no-repeat !important;
        }}
        """


def _chat_shell_main_background_layers() -> tuple[str, str, str, str, str]:
    """Single plate image only — scale to pane so corner botanicals are not cropped."""
    shell_uri = _chat_shell_background_uri_for_css()
    if shell_uri:
        return (
            f'url("{shell_uri}")',
            "center center",
            "100% 100%",
            "no-repeat",
            "scroll",
        )
    cream = _HA_SHELL_CREAM
    deep = _HA_SHELL_CREAM_DEEP
    return (
        f"linear-gradient(180deg, {cream} 0%, {deep} 100%)",
        "0 0",
        "100% 100%",
        "no-repeat",
        "fixed",
    )


def _sidebar_modern_herbal_css() -> str:
    """Sidebar: glass shell + soft depth; sage accents on widgets only."""
    # Single scope selector — never comma-join `section[stSidebar]` with descendant
    # selectors (e.g. ::before); ::before/::after are scoped to the shell only.
    sb = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        '[data-testid="stSidebar"]'
    )
    surface = _sidebar_shell_surface_layers()
    botanical = _sidebar_botanical_texture_css(sb)
    return f"""
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            section[data-testid="stSidebar"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stSidebar"] {{
            flex: 0 0 var(--ha-sidebar-target-width, 17rem) !important;
            width: var(--ha-sidebar-target-width, 17rem) !important;
            min-width: var(--ha-sidebar-target-width, 17rem) !important;
            max-width: var(--ha-sidebar-target-width, 17rem) !important;
        }}
        {sb} {{
            position: relative !important;
            isolation: isolate !important;
            overflow-x: hidden !important;
            background: {surface} !important;
            background-color: {_HA_SIDEBAR_BG} !important;
            border-right: 1px solid {_HA_SIDEBAR_EDGE} !important;
            box-shadow: {_HA_SIDEBAR_SHADOW} !important;
            backdrop-filter: blur(10px) saturate(1.04) !important;
            -webkit-backdrop-filter: blur(10px) saturate(1.04) !important;
            flex: 0 0 var(--ha-sidebar-target-width, 17rem) !important;
            width: var(--ha-sidebar-target-width, 17rem) !important;
            min-width: var(--ha-sidebar-target-width, 17rem) !important;
            max-width: var(--ha-sidebar-target-width, 17rem) !important;
        }}
        {botanical}
        {sb} > * {{
            position: relative !important;
            z-index: 1 !important;
        }}
        {sb} .block-container {{
            padding-top: 1.25rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-bottom: 1rem !important;
            background: transparent !important;
            background-image: none !important;
        }}
        {sb} [data-testid="stSidebarContent"] {{
            padding-top: 0.5rem !important;
            background: transparent !important;
            background-image: none !important;
        }}
        {sb} .ha-sidebar-header {{
            margin: 0 0 1rem 0 !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) {{
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) > div {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) [data-testid="stSidebarHeader"] {{
            flex: 0 0 auto !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: 1.65rem !important;
            padding: 0 0.45rem 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .block-container {{
            padding-top: 0.2rem !important;
            padding-bottom: 0 !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) [data-testid="stSidebarContent"],
        {sb}:has(.st-key-ha_sidebar_login_footer) [data-testid="stSidebarUserContent"] {{
            padding-top: 0 !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            .block-container > [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            .block-container > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:has(.ha-sidebar-header) {{
            flex-shrink: 0 !important;
            overflow: visible !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.ha-sidebar-header),
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="element-container"]:has(.ha-sidebar-header) {{
            flex-shrink: 0 !important;
            overflow: visible !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .ha-sidebar-header {{
            margin-top: 0 !important;
            margin-bottom: 0.65rem !important;
            flex-shrink: 0 !important;
            overflow: visible !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .ha-sidebar-header__eyebrow {{
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            min-height: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0.45rem !important;
            flex-shrink: 0 !important;
            color: #6a7568 !important;
            -webkit-text-fill-color: #6a7568 !important;
        }}
        {sb} .ha-sidebar-header__eyebrow::before {{
            width: 1.25rem !important;
            height: 3px !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, {_HA_SAGE_SOFT}, rgba(122, 145, 112, 0.12)) !important;
        }}
        {sb} .ha-sidebar-header__eyebrow::after {{
            content: "✦";
            margin-left: auto;
            font-size: 0.5rem;
            color: {_HA_SAGE_SOFT};
            opacity: 0.65;
            letter-spacing: 0;
        }}
        {sb} .ha-sidebar-header__user-card {{
            background: {_HA_GLASS_STRONG} !important;
            border: 1px solid {_HA_GLASS_BORDER} !important;
            border-radius: 16px !important;
            backdrop-filter: blur(14px) saturate(1.1) !important;
            -webkit-backdrop-filter: blur(14px) saturate(1.1) !important;
            box-shadow:
                0 4px 18px rgba(60, 78, 58, 0.07),
                inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
            padding: 0.62rem 0.75rem !important;
            transition: box-shadow 0.22s ease, border-color 0.22s ease, transform 0.2s ease !important;
        }}
        {sb} .ha-sidebar-header__user-card:hover {{
            border-color: rgba(122, 145, 112, 0.28) !important;
            box-shadow:
                0 6px 22px rgba(60, 78, 58, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
            transform: translateY(-1px);
        }}
        {sb} .ha-sidebar-header__avatar {{
            border-radius: 12px !important;
            background: linear-gradient(145deg, {_HA_SAGE} 0%, {_HA_SAGE_DEEP} 92%) !important;
            box-shadow:
                0 2px 8px rgba(60, 78, 58, 0.18),
                inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        }}
        {sb} div[data-testid="stExpander"] {{
            background: {_HA_GLASS_STRONG} !important;
            border: 1px solid {_HA_GLASS_BORDER} !important;
            border-radius: 16px !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            box-shadow: 0 3px 16px rgba(60, 78, 58, 0.06) !important;
            margin-bottom: 0.85rem !important;
        }}
        {sb} hr {{
            border-top-color: rgba(92, 111, 94, 0.12) !important;
            margin: 1.35rem 0 1rem 0 !important;
        }}
        {sb} .ha-sidebar-title {{
            color: {_HA_SAGE} !important;
            letter-spacing: 0.08em !important;
            margin: 0.85rem 0 0.35rem 0 !important;
        }}
        {sb} .ha-sidebar-subtitle {{
            color: #6f7d6c !important;
            margin-bottom: 0.5rem !important;
        }}
        {sb} .ha-sidebar-selected {{
            background: rgba(255, 255, 255, 0.55) !important;
            border: 1px solid {_HA_GLASS_BORDER} !important;
            border-radius: 12px !important;
            padding: 0.45rem 0.62rem !important;
            box-shadow: 0 2px 10px rgba(60, 78, 58, 0.05) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
        }}
        {sb} button[kind="primary"],
        {sb} button[kind="secondary"],
        {sb} [data-testid="stButton"] button {{
            border-radius: 12px !important;
            transition:
                background-color 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease,
                transform 0.18s ease,
                color 0.2s ease !important;
        }}
        {sb} button[kind="primary"]:hover,
        {sb} button[kind="secondary"]:hover,
        {sb} [data-testid="stButton"] button:hover {{
            background: {_HA_SAGE_MIST} !important;
            border-color: rgba(122, 145, 112, 0.22) !important;
            box-shadow: 0 2px 12px rgba(60, 78, 58, 0.08) !important;
            transform: translateY(-1px);
        }}
        {sb} [data-testid="stButton"] button:hover [data-testid="stIconMaterial"],
        {sb} [data-testid="stButton"] button:hover .material-symbols-rounded {{
            color: {_HA_SAGE_DEEP} !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            .st-key-ha_sidebar_guest_main [data-testid="stSelectbox"] > div {{
            background: {_HA_GLASS_STRONG} !important;
            border: 1px solid {_HA_GLASS_BORDER} !important;
            border-radius: 14px !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            box-shadow: 0 3px 14px rgba(60, 78, 58, 0.06) !important;
            transition: box-shadow 0.22s ease, border-color 0.22s ease !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            .st-key-ha_sidebar_guest_main [data-testid="stSelectbox"] > div:hover {{
            border-color: rgba(122, 145, 112, 0.32) !important;
            box-shadow: {_HA_INPUT_GLOW} !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.ha-sidebar-login-push),
        {sb}:has(.st-key-ha_sidebar_login_footer) .ha-sidebar-login-push {{
            display: none !important;
            flex: 0 0 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .st-key-ha_sidebar_guest_main {{
            flex: 0 0 auto !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.35rem !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            .st-key-ha_sidebar_guest_main [data-testid="stSelectbox"] {{
            margin-bottom: 0.65rem !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .ha-sidebar-flex-gap,
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.ha-sidebar-flex-gap),
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stMarkdownContainer"]:has(.ha-sidebar-flex-gap) {{
            flex: 0 0 0.65rem !important;
            min-height: 0.65rem !important;
            max-height: 0.65rem !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_login_footer),
        {sb}:has(.st-key-ha_sidebar_login_footer)
            [data-testid="element-container"]:has(.st-key-ha_sidebar_login_footer) {{
            margin-top: 0.85rem !important;
            flex-shrink: 0 !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .st-key-ha_sidebar_login_footer
            [data-testid="stVerticalBlock"] {{
            display: flex !important;
            flex-direction: column !important;
            gap: 0.85rem !important;
            align-items: stretch !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .st-key-ha_sidebar_login_footer
            [data-testid="stElementContainer"] {{
            margin: 0 !important;
            padding: 0 !important;
            flex-shrink: 0 !important;
            position: relative !important;
        }}
        {sb}:has(.st-key-ha_sidebar_login_footer) .st-key-ha_sidebar_login_btn {{
            margin-top: 0.15rem !important;
            padding-top: 0 !important;
        }}
        .st-key-ha_sidebar_login_footer {{
            position: relative !important;
            z-index: 12 !important;
            margin-top: 0 !important;
            flex-shrink: 0 !important;
            background: linear-gradient(
                180deg,
                rgba(242, 235, 227, 0.72) 0%,
                {_HA_SIDEBAR_BG} 28%,
                {_HA_SIDEBAR_BG} 100%
            ) !important;
            border-top: 1px solid rgba(44, 48, 42, 0.1) !important;
            border-radius: 18px 18px 0 0 !important;
            backdrop-filter: blur(14px) saturate(1.08) !important;
            -webkit-backdrop-filter: blur(14px) saturate(1.08) !important;
            box-shadow: 0 -6px 24px rgba(38, 42, 36, 0.07) !important;
            padding: 1rem 1rem 1rem !important;
            gap: 0.85rem !important;
        }}
        .ha-sidebar-login-hint {{
            margin: 0 0 0.15rem 0 !important;
            padding-bottom: 0.35rem !important;
        }}
        .st-key-ha_sidebar_login_btn,
        .st-key-ha_sidebar_login_btn [data-testid="stElementContainer"],
        .st-key-ha_sidebar_login_btn [data-testid="element-container"] {{
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            flex-shrink: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow: visible !important;
        }}
        .ha-sidebar-login-title {{
            color: {_HA_SAGE_DEEP} !important;
            -webkit-text-fill-color: {_HA_SAGE_DEEP} !important;
        }}
        .ha-sidebar-login-hint {{
            color: #6f7d6c !important;
            -webkit-text-fill-color: #6f7d6c !important;
        }}
    """


def _sidebar_ensure_visible_css() -> str:
    """Force sidebar widgets/text visible (Streamlit theme + global button rules)."""
    return f"""
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stSidebar"] .ha-sidebar-header,
        [data-testid="stSidebar"] .ha-sidebar-header *:not(.ha-sidebar-header__eyebrow) {{
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}
        [data-testid="stSidebar"] .ha-sidebar-header__eyebrow {{
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            color: #6a7568 !important;
            -webkit-text-fill-color: #6a7568 !important;
        }}
        [data-testid="stSidebar"] [data-testid="element-container"]:has(button),
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(button) {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        [data-testid="stSidebar"] [data-testid="stSelectbox"],
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div,
        [data-testid="stSidebar"] .st-key-ha_sidebar_guest_main,
        [data-testid="stSidebar"] .st-key-ha_sidebar_login_footer {{
            opacity: 1 !important;
            visibility: visible !important;
        }}
    """


def _nav_key_selectors(suffix: str = "") -> str:
    """Comma-safe ha_nav_* selectors — suffix applies to each key (avoids hiding whole widgets)."""
    keys = ("ha_nav_guest_top", "ha_nav_user", "ha_nav_admin")
    tail = f" {suffix}" if suffix else ""
    return ",\n        ".join(
        f'[data-testid="stSidebar"] .st-key-{key}{tail}' for key in keys
    )


def _sidebar_guest_main_shell_css() -> str:
    """Guest sidebar body only (do not style logged-in shell here)."""
    panel = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        '[data-testid="stSidebar"]'
    )
    gm = f"{panel} .st-key-ha_sidebar_guest_main"
    return f"""
        {gm} {{
            flex: 0 0 auto !important;
            margin-top: 0.1rem !important;
            margin-bottom: 0.75rem !important;
            padding: 0 0 0.2rem 0 !important;
        }}
        {panel} .st-key-ha_sidebar_guest_nav {{
            margin-bottom: 0.15rem !important;
            padding-bottom: 0.1rem !important;
        }}
        {gm} > [data-testid="stMarkdownContainer"]:has(hr),
        {gm} hr {{
            border-top-color: rgba(44, 48, 42, 0.1) !important;
            margin: 0.75rem 0 !important;
        }}
        {gm} [data-testid="stSelectbox"] {{
            margin-bottom: 0.65rem !important;
        }}
    """


def _sidebar_user_logged_in_css() -> str:
    """Logged-in sidebar shell: compact body + pinned footer."""
    panel = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        '[data-testid="stSidebar"]'
    )
    um = f"{panel} .st-key-ha_sidebar_user_main"
    body = f"{um} .st-key-ha_sidebar_user_body"
    footer = f"{um} .st-key-ha_sidebar_user_footer"
    chat = f"{body} .st-key-ha_sidebar_chat_panel"
    profile = f"{body} .st-key-ha_sidebar_profile_panel"
    return f"""
        {um} {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        {panel} .st-key-ha_sidebar_user_nav {{
            margin-bottom: 0.35rem !important;
            padding-bottom: 0 !important;
        }}
        {um} .st-key-ha_nav_user,
        {um} .st-key-ha_nav_admin {{
            margin-bottom: 0.35rem !important;
        }}
        {footer} .st-key-ha_sidebar_user_footer_lang [data-testid="stSelectbox"] {{
            margin-bottom: 0 !important;
        }}
        {um} [data-testid="stVerticalBlockBorderWrapper"],
        {um} > div[data-testid="stVerticalBlock"],
        {um} [data-testid="stVerticalBlock"],
        {chat} [data-testid="stVerticalBlockBorderWrapper"],
        {chat} > div[data-testid="stVerticalBlock"],
        {chat} [data-testid="stVerticalBlock"],
        {profile} [data-testid="stVerticalBlockBorderWrapper"],
        {profile} > div[data-testid="stVerticalBlock"],
        {profile} [data-testid="stVerticalBlock"] {{
            background: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }}
        {chat}, {profile} {{
            margin-top: 0.25rem !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}
        {chat} .ha-sidebar-title {{
            margin: 0.45rem 0 0.1rem 0 !important;
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
            opacity: 0.72 !important;
        }}
        {chat} .ha-sidebar-subtitle {{
            margin-bottom: 0.18rem !important;
            font-size: 0.66rem !important;
            opacity: 0.65 !important;
        }}
        {body} .st-key-ha_sidebar_user_nav [data-testid="stVerticalBlockBorderWrapper"],
        {body} .st-key-ha_sidebar_user_nav [data-testid="stVerticalBlock"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
    """


def _sidebar_user_compact_top_css() -> str:
    """Logged-in sidebar: trim Streamlit top chrome so NAVIGATION sits higher."""
    sb = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        ':is(section[data-testid="stSidebar"], [data-testid="stSidebar"])'
        ':has(.st-key-ha_sidebar_user_main)'
    )
    return f"""
        {sb} [data-testid="stSidebarHeader"] {{
            flex: 0 0 auto !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: 1.65rem !important;
            padding: 0 0.45rem 0 !important;
            margin: 0 !important;
        }}
        {sb} [data-testid="stSidebarCollapseButton"],
        {sb} [data-testid="stSidebarNavViewButton"] {{
            min-height: 0 !important;
            padding: 0.18rem !important;
        }}
        {sb} .block-container {{
            padding-top: 0.35rem !important;
            padding-bottom: 0.6rem !important;
        }}
        {sb} [data-testid="stSidebarContent"],
        {sb} [data-testid="stSidebarUserContent"] {{
            padding-top: 0 !important;
        }}
        {sb} .block-container > [data-testid="stVerticalBlock"] {{
            gap: 0.2rem !important;
        }}
        {sb} [data-testid="stElementContainer"]:has(.ha-sidebar-header),
        {sb} [data-testid="element-container"]:has(.ha-sidebar-header),
        {sb} [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        {sb} .ha-sidebar-header {{
            margin: 0 0 0.45rem 0 !important;
        }}
        {sb} .ha-sidebar-header__eyebrow {{
            margin: 0 0 0.38rem 0 !important;
            min-height: 0.9rem !important;
        }}
        {sb} .ha-sidebar-header__user-card {{
            padding: 0.5rem 0.6rem !important;
        }}
        {sb} .st-key-ha_sidebar_user_main {{
            margin-top: 0 !important;
        }}
    """


def _sidebar_user_flex_layout_css() -> str:
    """Logged-in sidebar: body scrolls, footer pinned; single scrollbar."""
    sb = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        ':is(section[data-testid="stSidebar"], [data-testid="stSidebar"])'
        ':has(.st-key-ha_sidebar_user_main)'
    )
    um = f"{sb} .st-key-ha_sidebar_user_main"
    body = f"{um} .st-key-ha_sidebar_user_body"
    footer = f"{um} .st-key-ha_sidebar_user_footer"
    shell_chain = (
        f"{sb} > div",
        f"{sb} [data-testid='stSidebarContent']",
        f"{sb} [data-testid='stSidebarUserContent']",
        f"{sb} .block-container",
        f"{sb} .block-container > [data-testid='stVerticalBlock']",
    )
    shell_sel = ",\n        ".join(shell_chain)
    return f"""
        {sb} {{
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
        }}
        {shell_sel} {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }}
        {um} {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        {um} [data-testid="stVerticalBlockBorderWrapper"],
        {um} > div[data-testid="stVerticalBlock"],
        {um} [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        {body} {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            margin: 0 !important;
            padding: 0 0 0.35rem 0 !important;
        }}
        {footer} {{
            flex: 0 0 auto !important;
            margin-top: auto !important;
            padding: 0.5rem 0 max(0.7rem, env(safe-area-inset-bottom, 0px)) 0 !important;
            border-top: 1px solid rgba(44, 48, 42, 0.1) !important;
            background: linear-gradient(
                180deg,
                rgba(242, 235, 227, 0.5) 0%,
                {_HA_SIDEBAR_BG} 55%
            ) !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            overflow: hidden !important;
        }}
        {footer} [data-testid="stVerticalBlockBorderWrapper"],
        {footer} > div[data-testid="stVerticalBlock"],
        {footer} [data-testid="stVerticalBlock"] {{
            flex: 0 0 auto !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            gap: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        {footer} .st-key-ha_sidebar_user_footer_lang,
        {footer} [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_user_footer_lang) {{
            flex: 0 0 auto !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        {footer} .ha-sidebar-user-footer-spacer,
        {footer} [data-testid="stMarkdownContainer"]:has(.ha-sidebar-user-footer-spacer),
        {footer} [data-testid="stElementContainer"]:has(.ha-sidebar-user-footer-spacer) {{
            flex: 0 0 auto !important;
            min-height: 1.65rem !important;
            height: 1.65rem !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }}
        {footer} .ha-sidebar-user-footer-spacer {{
            display: block !important;
            width: 100% !important;
        }}
        {footer} .st-key-ha_sidebar_user_footer_logout,
        {footer} [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_user_footer_logout),
        {footer} .st-key-ha_sidebar_logout {{
            flex: 0 0 auto !important;
            margin-top: 0 !important;
            padding: 0 !important;
        }}
        {sb} [data-testid="stSidebarContent"]::-webkit-scrollbar,
        {sb} [data-testid="stSidebarUserContent"]::-webkit-scrollbar,
        {sb} .block-container::-webkit-scrollbar,
        {um}::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }}
        {body}::-webkit-scrollbar {{
            width: 6px !important;
        }}
        {body}::-webkit-scrollbar-thumb {{
            background: rgba(44, 48, 42, 0.18) !important;
            border-radius: 999px !important;
        }}
        {body}::-webkit-scrollbar-track {{
            background: transparent !important;
        }}
    """


def _sidebar_user_pill_chrome_css() -> str:
    """Logged-in sidebar only — same soft glass pills as guest Chat/Profile nav."""
    panel = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        '[data-testid="stSidebar"]'
    )
    um = f"{panel} .st-key-ha_sidebar_user_main"
    footer = f"{um} .st-key-ha_sidebar_user_footer"
    chat = f"{um} .st-key-ha_sidebar_chat_panel"
    delete_btn = f"{um} .st-key-ha_sidebar_delete_chat [data-testid='stButton'] button"
    # Match _sidebar_section_nav_css (guest) tokens — translucent, not flat #fff.
    soft_inactive_bg = "rgba(255, 255, 255, 0.34)"
    soft_active_bg = "rgba(255, 253, 248, 0.95)"
    soft_hover_bg = "rgba(255, 253, 248, 0.72)"
    soft_border = "rgba(44, 48, 42, 0.09)"
    soft_border_hover = "rgba(44, 48, 42, 0.12)"
    soft_border_active = "rgba(44, 48, 42, 0.11)"
    soft_shadow = "0 1px 5px rgba(38, 42, 36, 0.04)"
    soft_shadow_hover = "0 2px 8px rgba(38, 42, 36, 0.06)"
    soft_shadow_active = "0 2px 10px rgba(38, 42, 36, 0.07)"
    pill_radius = "14px"
    user_nav_keys = (".st-key-ha_nav_user", ".st-key-ha_nav_admin")
    user_nav_rg = ",\n        ".join(f"{um} {key} div[role='radiogroup']" for key in user_nav_keys)
    user_nav_label = ",\n        ".join(
        f"{um} {key} div[role='radiogroup'] label" for key in user_nav_keys
    )
    user_nav_checked = ",\n        ".join(
        f"{um} {key} div[role='radiogroup'] label:has(input:checked)" for key in user_nav_keys
    )
    user_nav_label_p = ",\n        ".join(
        f"{um} {key} div[role='radiogroup'] label p" for key in user_nav_keys
    )
    um_btn = (
        f"{um} .st-key-ha_nav_user_chat [data-testid='stButton'] button, "
        f"{um} .st-key-ha_nav_user_chat [data-testid='stButton'] button[kind='primary'], "
        f"{um} .st-key-ha_nav_user_chat [data-testid='stButton'] button[kind='secondary'], "
        f"{um} .st-key-ha_nav_user_profile [data-testid='stButton'] button, "
        f"{um} .st-key-ha_nav_user_profile [data-testid='stButton'] button[kind='primary'], "
        f"{um} .st-key-ha_nav_user_profile [data-testid='stButton'] button[kind='secondary'], "
        f"{footer} .st-key-ha_sidebar_logout [data-testid='stButton'] button, "
        f"{footer} .st-key-ha_sidebar_logout [data-testid='stButton'] button[kind='primary'], "
        f"{footer} .st-key-ha_sidebar_logout [data-testid='stButton'] button[kind='secondary']"
    )
    um_btn_hover = (
        f"{um} .st-key-ha_nav_user_chat [data-testid='stButton'] button:hover, "
        f"{um} .st-key-ha_nav_user_profile [data-testid='stButton'] button:hover, "
        f"{footer} .st-key-ha_sidebar_logout [data-testid='stButton'] button:hover"
    )
    um_btn_text = (
        f"{um} .st-key-ha_nav_user_chat [data-testid='stButton'] button p, "
        f"{um} .st-key-ha_nav_user_chat [data-testid='stButton'] button span, "
        f"{um} .st-key-ha_nav_user_profile [data-testid='stButton'] button p, "
        f"{um} .st-key-ha_nav_user_profile [data-testid='stButton'] button span, "
        f"{footer} .st-key-ha_sidebar_logout [data-testid='stButton'] button p, "
        f"{footer} .st-key-ha_sidebar_logout [data-testid='stButton'] button span"
    )
    return f"""
        {user_nav_rg} {{
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0.45rem !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        {user_nav_label} {{
            flex: 0 0 auto !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0.52rem 0.7rem !important;
            min-height: 0 !important;
            border-radius: {pill_radius} !important;
            border: 1px solid {soft_border} !important;
            background: {soft_inactive_bg} !important;
            background-image: none !important;
            box-shadow: {soft_shadow} !important;
            box-sizing: border-box !important;
            transition:
                background-color 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease,
                transform 0.18s ease,
                color 0.2s ease !important;
        }}
        {user_nav_checked} {{
            background: {soft_active_bg} !important;
            border: 1px solid {soft_border_active} !important;
            box-shadow: {soft_shadow_active} !important;
            transform: translateY(0) !important;
        }}
        {user_nav_label}:hover {{
            background: {soft_hover_bg} !important;
            border-color: {soft_border_hover} !important;
            box-shadow: {soft_shadow_hover} !important;
            transform: translateY(-1px) !important;
        }}
        {user_nav_label_p} {{
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            text-align: center !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
        }}
        {user_nav_checked} p {{
            font-weight: 600 !important;
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
        }}
        {user_nav_label}:hover p {{
            color: #4a4540 !important;
            -webkit-text-fill-color: #4a4540 !important;
        }}
        {um} .st-key-ha_nav_user div[role="radiogroup"] label > div:first-child,
        {um} .st-key-ha_nav_admin div[role="radiogroup"] label > div:first-child {{
            display: none !important;
        }}
        /* Chat history: compact list rows — less visual weight than nav pills */
        {chat} div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0.22rem !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}
        {chat} div[role="radiogroup"] label {{
            flex: 0 0 auto !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            margin: 0 !important;
            padding: 0.3rem 0.48rem !important;
            min-height: 0 !important;
            border-radius: 8px !important;
            border: 1px solid rgba(44, 48, 42, 0.05) !important;
            background: rgba(255, 255, 255, 0.2) !important;
            background-image: none !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
            transition: background-color 0.15s ease, border-color 0.15s ease !important;
        }}
        {chat} div[role="radiogroup"] label:has(input:checked) {{
            background: rgba(255, 255, 255, 0.48) !important;
            border-color: rgba(44, 48, 42, 0.09) !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        {chat} div[role="radiogroup"] label:hover {{
            background: rgba(255, 255, 255, 0.36) !important;
            border-color: rgba(44, 48, 42, 0.08) !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        {chat} div[role="radiogroup"] label p {{
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            text-align: left !important;
            font-size: 0.76rem !important;
            font-weight: 450 !important;
            line-height: 1.2 !important;
            color: #8a847c !important;
            -webkit-text-fill-color: #8a847c !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        {chat} div[role="radiogroup"] label:has(input:checked) p {{
            font-weight: 520 !important;
            color: #5c574f !important;
            -webkit-text-fill-color: #5c574f !important;
        }}
        {chat} div[role="radiogroup"] label:hover p {{
            color: #6e6860 !important;
            -webkit-text-fill-color: #6e6860 !important;
        }}
        {chat} div[role="radiogroup"] label > div:first-child {{
            display: none !important;
        }}
        {chat} [data-testid="stElementContainer"]:has(div[role="radiogroup"]),
        {chat} [data-testid="element-container"]:has(div[role="radiogroup"]) {{
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list [data-testid="stHorizontalBlock"] {{
            gap: 0.28rem !important;
            margin: 0 0 0.22rem 0 !important;
            align-items: center !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:first-child [data-testid="stButton"] button {{
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: 100% !important;
            min-height: 0 !important;
            height: auto !important;
            padding: 0.3rem 0.48rem !important;
            margin: 0 !important;
            border-radius: 8px !important;
            border: 1px solid rgba(44, 48, 42, 0.05) !important;
            background: rgba(255, 255, 255, 0.2) !important;
            box-shadow: none !important;
            font-size: 0.76rem !important;
            font-weight: 450 !important;
            line-height: 1.2 !important;
            color: #8a847c !important;
            -webkit-text-fill-color: #8a847c !important;
            text-align: left !important;
            transform: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:first-child
            [data-testid="stButton"] button[kind="primary"] {{
            background: rgba(255, 255, 255, 0.48) !important;
            border-color: rgba(44, 48, 42, 0.09) !important;
            color: #5c574f !important;
            -webkit-text-fill-color: #5c574f !important;
            font-weight: 520 !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:first-child
            [data-testid="stButton"] button:hover {{
            background: rgba(255, 255, 255, 0.36) !important;
            border-color: rgba(44, 48, 42, 0.08) !important;
            transform: none !important;
            box-shadow: none !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:first-child [data-testid="stButton"] button p,
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:first-child [data-testid="stButton"] button span {{
            font-size: 0.76rem !important;
            font-weight: inherit !important;
            text-align: left !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:last-child [data-testid="stButton"] {{
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:last-child [data-testid="stButton"] button {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 1.65rem !important;
            min-width: 1.65rem !important;
            height: 1.65rem !important;
            min-height: 1.65rem !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            color: #b85c5c !important;
            -webkit-text-fill-color: #b85c5c !important;
            transform: none !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:last-child [data-testid="stButton"] button:hover {{
            background: rgba(184, 92, 92, 0.1) !important;
            color: #a04848 !important;
            -webkit-text-fill-color: #a04848 !important;
            box-shadow: none !important;
        }}
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:last-child
            [data-testid="stButton"] [data-testid="stIconMaterial"],
        {chat} .st-key-ha_sidebar_chat_list
            [data-testid="column"]:last-child
            [data-testid="stButton"] .material-symbols-rounded {{
            font-size: 0.88rem !important;
            width: 0.88rem !important;
            height: 0.88rem !important;
            color: #c07070 !important;
        }}
        {um} [data-testid="stElementContainer"]:has([data-testid="stButton"]):not(
            :has(.st-key-ha_sidebar_delete_chat)
        ):not(:has(.st-key-ha_sidebar_chat_list)),
        {um} [data-testid="element-container"]:has([data-testid="stButton"]):not(
            :has(.st-key-ha_sidebar_delete_chat)
        ):not(:has(.st-key-ha_sidebar_chat_list)) {{
            margin: 0 0 0.45rem 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }}
        {um} .st-key-ha_sidebar_user_nav [data-testid="stHorizontalBlock"] {{
            gap: 0.45rem !important;
            margin: 0 0 0.35rem 0 !important;
        }}
        {um} .st-key-ha_nav_user_chat [data-testid="stButton"] button[kind="primary"],
        {um} .st-key-ha_nav_user_profile [data-testid="stButton"] button[kind="primary"] {{
            background: {soft_active_bg} !important;
            border-color: {soft_border_active} !important;
            box-shadow: {soft_shadow_active} !important;
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
            font-weight: 600 !important;
        }}
        {um} .st-key-ha_nav_user_chat [data-testid="stButton"],
        {um} .st-key-ha_nav_user_profile [data-testid="stButton"] {{
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }}
        {um_btn} {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: 100% !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            padding: 0.52rem 0.7rem !important;
            margin: 0 !important;
            line-height: 1.25 !important;
            border-radius: {pill_radius} !important;
            border: 1px solid {soft_border} !important;
            background: {soft_inactive_bg} !important;
            background-image: none !important;
            box-shadow: {soft_shadow} !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            text-align: left !important;
            transform: none !important;
            filter: none !important;
            backdrop-filter: blur(8px) saturate(1.02) !important;
            -webkit-backdrop-filter: blur(8px) saturate(1.02) !important;
            transition:
                background-color 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease,
                transform 0.18s ease,
                color 0.2s ease !important;
        }}
        {um_btn_hover} {{
            background: {soft_hover_bg} !important;
            background-image: none !important;
            border-color: {soft_border_hover} !important;
            box-shadow: {soft_shadow_hover} !important;
            transform: translateY(-1px) !important;
            filter: none !important;
            color: #4a4540 !important;
            -webkit-text-fill-color: #4a4540 !important;
        }}
        {um_btn_text} {{
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            font-weight: 500 !important;
            font-size: 0.86rem !important;
            line-height: 1.25 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        {um} .st-key-ha_nav_user_chat [data-testid="stButton"] button,
        {um} .st-key-ha_nav_user_profile [data-testid="stButton"] button {{
            justify-content: center !important;
            text-align: center !important;
        }}
        {footer} .st-key-ha_sidebar_logout [data-testid="stButton"] button [data-testid="stIconMaterial"],
        {footer} .st-key-ha_sidebar_logout [data-testid="stButton"] button .material-symbols-rounded {{
            color: #6a7568 !important;
            margin-right: 0.35rem !important;
            font-size: 0.95rem !important;
            width: 0.95rem !important;
            height: 0.95rem !important;
            line-height: 1 !important;
        }}
        {delete_btn} {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            width: auto !important;
            min-height: 0 !important;
            height: auto !important;
            padding: 0.28rem 0.45rem !important;
            margin: 0 !important;
            border-radius: 10px !important;
            border: none !important;
            background: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
            color: #b85c5c !important;
            -webkit-text-fill-color: #b85c5c !important;
            font-size: 0.76rem !important;
            font-weight: 500 !important;
            line-height: 1.2 !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            transform: none !important;
        }}
        {delete_btn}:hover {{
            background: rgba(184, 92, 92, 0.1) !important;
            color: #a04848 !important;
            -webkit-text-fill-color: #a04848 !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        {delete_btn} p, {delete_btn} span {{
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            font-size: 0.76rem !important;
            font-weight: 500 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        {delete_btn} [data-testid="stIconMaterial"],
        {delete_btn} .material-symbols-rounded {{
            color: #c07070 !important;
            font-size: 0.88rem !important;
            width: 0.88rem !important;
            height: 0.88rem !important;
            margin-right: 0.28rem !important;
        }}
        {footer} [data-testid="stSelectbox"] > div {{
            background: {_HA_GLASS_STRONG} !important;
            border: 1px solid {_HA_GLASS_BORDER} !important;
            border-radius: {pill_radius} !important;
            box-shadow: 0 3px 14px rgba(60, 78, 58, 0.06) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            min-height: 0 !important;
            padding-top: 0.45rem !important;
            padding-bottom: 0.45rem !important;
            transition: box-shadow 0.22s ease, border-color 0.22s ease !important;
        }}
        {footer} [data-testid="stSelectbox"] > div:hover {{
            background: {_HA_GLASS_STRONG} !important;
            border-color: rgba(122, 145, 112, 0.32) !important;
            box-shadow: {_HA_INPUT_GLOW} !important;
        }}
        {um} .st-key-ha_nav_user_chat [data-testid="stButton"] button,
        {um} .st-key-ha_nav_user_profile [data-testid="stButton"] button,
        {footer} .st-key-ha_sidebar_logout [data-testid="stButton"] button {{
            background: {soft_inactive_bg} !important;
            background-image: none !important;
            border: 1px solid {soft_border} !important;
            box-shadow: {soft_shadow} !important;
            min-height: 0 !important;
            height: auto !important;
            padding: 0.52rem 0.7rem !important;
        }}
    """


def _sidebar_section_panels_css() -> str:
    """Legacy hook — chat/profile panels styled in _sidebar_user_logged_in_css."""
    return ""


def _sidebar_section_nav_css() -> str:
    """Chat / Profile segmented control — targets ha_nav_* widget keys only."""
    nav = _nav_key_selectors()
    nav_rg = _nav_key_selectors('div[role="radiogroup"]')
    nav_rg_label = _nav_key_selectors('div[role="radiogroup"] label')
    nav_hide_dot = _nav_key_selectors('div[role="radiogroup"] label > div:first-child')
    nav_el = _nav_key_selectors('[data-testid="stElementContainer"]')
    nav_el_legacy = _nav_key_selectors('[data-testid="element-container"]')
    nav_btn_group = _nav_key_selectors('[data-testid="stButtonGroup"]')
    nav_baseweb = _nav_key_selectors('[data-baseweb="button-group"]')
    nav_label_p = _nav_key_selectors('div[role="radiogroup"] label p')
    nav_label_span = _nav_key_selectors('div[role="radiogroup"] label span')
    nav_label_md = _nav_key_selectors('div[role="radiogroup"] label [data-testid="stMarkdownContainer"]')
    nav_label_md_p = _nav_key_selectors(
        'div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p'
    )
    nav_label_hover = _nav_key_selectors('div[role="radiogroup"] label:hover')
    nav_label_hover_p = _nav_key_selectors('div[role="radiogroup"] label:hover p')
    nav_label_hover_span = _nav_key_selectors('div[role="radiogroup"] label:hover span')
    nav_radio_btn = _nav_key_selectors('[data-testid="stButtonGroup"] button[role="radio"]')
    nav_radio_baseweb = _nav_key_selectors('[data-baseweb="button-group"] button[role="radio"]')
    nav_radio_checked = _nav_key_selectors(
        '[data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"]'
    )
    nav_radio_checked_bw = _nav_key_selectors(
        '[data-baseweb="button-group"] button[role="radio"][aria-checked="true"]'
    )
    nav_radio_p = _nav_key_selectors('[data-testid="stButtonGroup"] button[role="radio"] p')
    nav_radio_span = _nav_key_selectors('[data-testid="stButtonGroup"] button[role="radio"] span')
    nav_radio_p_bw = _nav_key_selectors('[data-baseweb="button-group"] button[role="radio"] p')
    nav_radio_span_bw = _nav_key_selectors('[data-baseweb="button-group"] button[role="radio"] span')
    nav_checked_label = _nav_key_selectors('div[role="radiogroup"] label:has(input:checked)')
    nav_checked_label_p = _nav_key_selectors('div[role="radiogroup"] label:has(input:checked) p')
    return f"""
        {nav} {{
            display: block !important;
            width: 100% !important;
            margin: 0 0 0.9rem 0 !important;
            padding: 0 !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}
        {nav_el},
        {nav_el_legacy} {{
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        {nav_btn_group},
        {nav_baseweb},
        {nav_rg} {{
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            justify-content: flex-start !important;
            gap: 0.45rem !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }}
        {nav_rg_label} {{
            flex: 0 0 auto !important;
            width: 100% !important;
            min-width: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0.52rem 0.7rem !important;
            border-radius: 14px !important;
            border: 1px solid rgba(44, 48, 42, 0.09) !important;
            background: rgba(255, 255, 255, 0.34) !important;
            box-shadow: 0 1px 5px rgba(38, 42, 36, 0.04) !important;
            box-sizing: border-box !important;
            transition:
                background-color 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease,
                transform 0.18s ease,
                color 0.2s ease !important;
        }}
        {nav_label_p},
        {nav_label_span},
        {nav_label_md},
        {nav_label_md_p} {{
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            text-align: center !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
            opacity: 1 !important;
            visibility: visible !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        {nav_label_hover} {{
            background: rgba(255, 253, 248, 0.72) !important;
            border-color: rgba(44, 48, 42, 0.12) !important;
            box-shadow: 0 2px 8px rgba(38, 42, 36, 0.06) !important;
            transform: translateY(-1px);
        }}
        {nav_label_hover_p},
        {nav_label_hover_span} {{
            color: #4a4540 !important;
            -webkit-text-fill-color: #4a4540 !important;
        }}
        {nav_radio_btn},
        {nav_radio_baseweb} {{
            flex: 0 0 auto !important;
            width: 100% !important;
            min-width: 0 !important;
            margin: 0 !important;
            border-radius: 14px !important;
            border: 1px solid rgba(44, 48, 42, 0.09) !important;
            background: rgba(255, 255, 255, 0.34) !important;
            box-shadow: 0 1px 5px rgba(38, 42, 36, 0.04) !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
        }}
        {nav_radio_checked},
        {nav_radio_checked_bw} {{
            background: rgba(255, 253, 248, 0.95) !important;
            border: 1px solid rgba(44, 48, 42, 0.11) !important;
            box-shadow: 0 2px 10px rgba(38, 42, 36, 0.07) !important;
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
        }}
        {nav_radio_p},
        {nav_radio_span},
        {nav_radio_p_bw},
        {nav_radio_span_bw} {{
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            opacity: 1 !important;
        }}
        {nav_checked_label} {{
            background: rgba(255, 253, 248, 0.95) !important;
            border: 1px solid rgba(44, 48, 42, 0.11) !important;
            box-shadow: 0 2px 10px rgba(38, 42, 36, 0.07) !important;
            transform: translateY(0);
        }}
        {nav_checked_label_p} {{
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
            font-weight: 600 !important;
        }}
        {nav_hide_dot} {{
            display: none !important;
        }}
    """


def _sidebar_login_btn_as_nav_css() -> str:
    """Login CTA — same pill chrome as Chat / Profile section nav."""
    btn = (
        "[data-testid='stSidebar'] .st-key-ha_sidebar_login_btn [data-testid='stButton'] button, "
        "[data-testid='stSidebar'] .st-key-ha_sidebar_login_btn button, "
        ".st-key-ha_sidebar_login_btn button"
    )
    btn_hover = (
        "[data-testid='stSidebar'] .st-key-ha_sidebar_login_btn [data-testid='stButton'] button:hover, "
        ".st-key-ha_sidebar_login_btn button:hover"
    )
    btn_active = (
        "[data-testid='stSidebar'] .st-key-ha_sidebar_login_btn [data-testid='stButton'] button[kind='primary'], "
        ".st-key-ha_sidebar_login_btn button[kind='primary']"
    )
    btn_text = (
        "[data-testid='stSidebar'] .st-key-ha_sidebar_login_btn [data-testid='stButton'] button p, "
        "[data-testid='stSidebar'] .st-key-ha_sidebar_login_btn [data-testid='stButton'] button span, "
        ".st-key-ha_sidebar_login_btn button p, "
        ".st-key-ha_sidebar_login_btn button span"
    )
    return f"""
        {btn} {{
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 100% !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0.52rem 0.7rem !important;
            min-height: 2.35rem !important;
            border-radius: 14px !important;
            border: 1px solid rgba(44, 48, 42, 0.09) !important;
            background: rgba(255, 255, 255, 0.34) !important;
            background-image: none !important;
            box-shadow: 0 1px 5px rgba(38, 42, 36, 0.04) !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
            text-align: center !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            filter: none !important;
            transition:
                background-color 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease,
                transform 0.18s ease,
                color 0.2s ease !important;
        }}
        {btn_active} {{
            background: rgba(255, 253, 248, 0.95) !important;
            border: 1px solid rgba(44, 48, 42, 0.11) !important;
            box-shadow: 0 2px 10px rgba(38, 42, 36, 0.07) !important;
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
            font-weight: 600 !important;
        }}
        {btn_text} {{
            color: inherit !important;
            -webkit-text-fill-color: inherit !important;
            font-weight: inherit !important;
            font-size: 0.86rem !important;
            text-align: center !important;
        }}
        {btn_hover} {{
            background: rgba(255, 253, 248, 0.72) !important;
            border-color: rgba(44, 48, 42, 0.12) !important;
            box-shadow: 0 2px 8px rgba(38, 42, 36, 0.06) !important;
            color: #4a4540 !important;
            -webkit-text-fill-color: #4a4540 !important;
            transform: translateY(-1px);
            filter: none !important;
        }}
        {btn_active}:hover {{
            background: rgba(255, 253, 248, 0.95) !important;
            border-color: rgba(44, 48, 42, 0.14) !important;
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
        }}
    """


def _sidebar_guest_no_scroll_css() -> str:
    """Guest sidebar: no scrollbars; flex gap pushes login block without overlap."""
    sb = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        '[data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)'
    )
    scroll_targets = (
        "",
        " [data-testid='stSidebarContent']",
        " [data-testid='stSidebarUserContent']",
        " .block-container",
        " .block-container > [data-testid='stVerticalBlock']",
        " > div",
    )
    no_scroll = ",\n        ".join(f"{sb}{t}" for t in scroll_targets)
    return f"""
        {no_scroll} {{
            overflow: hidden !important;
            overflow-y: hidden !important;
            overflow-x: hidden !important;
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }}
        {sb}::-webkit-scrollbar,
        {sb} [data-testid="stSidebarContent"]::-webkit-scrollbar,
        {sb} [data-testid="stSidebarUserContent"]::-webkit-scrollbar,
        {sb} .block-container::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }}
        {sb} [data-testid="stSidebarHeader"] {{
            flex: 0 0 auto !important;
            max-height: 2.5rem !important;
            padding: 0.15rem 0.5rem 0 !important;
        }}
        {sb} .block-container {{
            padding-top: 0.55rem !important;
        }}
        {sb} .st-key-ha_sidebar_login_footer {{
            margin-top: 0 !important;
            overflow: visible !important;
        }}
        {sb} .st-key-ha_sidebar_login_btn button {{
            position: relative !important;
            top: auto !important;
            margin-top: 0 !important;
        }}
    """


def _chat_composer_herbal_css() -> str:
    """Chat input bar — same beige pill chrome as sidebar nav."""
    comp = ".st-key-ha_chat_composer_row"
    return f"""
        {comp} {{
            background: linear-gradient(
                180deg,
                rgba(242, 235, 227, 0) 0%,
                rgba(242, 235, 227, 0.9) 38%,
                {_HA_SIDEBAR_BG} 100%
            ) !important;
        }}
        {comp} [data-testid="stHorizontalBlock"] {{
            background: rgba(255, 253, 248, 0.94) !important;
            border: 1px solid rgba(44, 48, 42, 0.1) !important;
            backdrop-filter: blur(12px) saturate(1.04) !important;
            -webkit-backdrop-filter: blur(12px) saturate(1.04) !important;
            box-shadow:
                0 2px 12px rgba(38, 42, 36, 0.06),
                inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
            transition:
                box-shadow 0.22s ease,
                border-color 0.22s ease,
                background-color 0.22s ease !important;
        }}
        {comp} [data-testid="stHorizontalBlock"]:focus-within {{
            border-color: rgba(44, 48, 42, 0.14) !important;
            box-shadow:
                0 2px 14px rgba(38, 42, 36, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        }}
        {comp} [data-testid="stChatInput"] > div {{
            background: rgba(255, 255, 255, 0.38) !important;
            border: 1px solid rgba(44, 48, 42, 0.07) !important;
            border-radius: 999px !important;
            box-shadow: inset 0 1px 2px rgba(38, 42, 36, 0.03) !important;
        }}
        {comp} [data-testid="stChatInput"] textarea,
        {comp} [data-testid="stChatInput"] input {{
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
            font-size: 0.9rem !important;
            font-weight: 450 !important;
        }}
        {comp} [data-testid="stChatInput"] textarea::placeholder,
        {comp} [data-testid="stChatInput"] input::placeholder {{
            color: #8a8278 !important;
            -webkit-text-fill-color: #8a8278 !important;
            opacity: 1 !important;
        }}
        {comp} [data-testid="stPopover"] button {{
            background: rgba(255, 255, 255, 0.42) !important;
            border: 1px solid rgba(44, 48, 42, 0.09) !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
            box-shadow: 0 1px 5px rgba(38, 42, 36, 0.04) !important;
        }}
        {comp} [data-testid="stPopover"] button:hover {{
            background: rgba(255, 253, 248, 0.78) !important;
            border-color: rgba(44, 48, 42, 0.12) !important;
            color: #4a4540 !important;
            -webkit-text-fill-color: #4a4540 !important;
        }}
        {comp} [data-testid="stChatInput"] button,
        {comp} [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] button,
        {comp} [data-testid="stChatInput"] button[kind="secondary"],
        {comp} [data-testid="stChatInput"] button[kind="primary"] {{
            background: rgba(255, 255, 255, 0.5) !important;
            border: 1px solid rgba(44, 48, 42, 0.08) !important;
            color: {_HA_SAGE} !important;
            -webkit-text-fill-color: {_HA_SAGE} !important;
            box-shadow: 0 1px 4px rgba(38, 42, 36, 0.05) !important;
        }}
        {comp} [data-testid="stChatInput"] button:hover,
        {comp} [data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] button:hover {{
            background: rgba(255, 253, 248, 0.92) !important;
            border-color: rgba(44, 48, 42, 0.12) !important;
            color: {_HA_SAGE_DEEP} !important;
            -webkit-text-fill-color: {_HA_SAGE_DEEP} !important;
        }}
        {comp} [data-testid="stChatInput"] button svg,
        {comp} [data-testid="stChatInput"] [data-testid="stIconMaterial"] {{
            color: {_HA_SAGE} !important;
            fill: {_HA_SAGE} !important;
        }}
        .st-key-ha_guest_empty_shell {comp} [data-testid="stChatInput"] > div,
        .st-key-ha_user_empty_shell {comp} [data-testid="stChatInput"] > div {{
            background: rgba(255, 255, 255, 0.38) !important;
            border: 1px solid rgba(44, 48, 42, 0.07) !important;
            box-shadow: inset 0 1px 2px rgba(38, 42, 36, 0.03) !important;
        }}
    """


def _inject_premium_herbal_ui(*, in_sidebar: bool = False) -> None:
    """Sage + cream premium tokens: glass composer, unified sidebar/chat chrome."""
    _inject_html(
        f"""
        <style>
        {_sidebar_modern_herbal_css()}
        {_sidebar_ensure_visible_css()}
        {_sidebar_guest_main_shell_css()}
        {_sidebar_user_logged_in_css()}
        {_sidebar_user_flex_layout_css()}
        {_sidebar_user_compact_top_css()}
        {_sidebar_section_nav_css()}
        {_sidebar_user_pill_chrome_css()}
        {_sidebar_login_btn_as_nav_css()}
        {_sidebar_section_panels_css()}
        {_chat_composer_herbal_css()}
        .st-key-ha_guest_suggested_cards button[data-testid^="baseButton"],
        .st-key-ha_user_suggested_cards button[data-testid^="baseButton"] {{
            background: {_HA_GLASS_STRONG} !important;
            border: 1px solid {_HA_GLASS_BORDER} !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            box-shadow: 0 2px 14px rgba(60, 78, 58, 0.06) !important;
        }}
        .st-key-ha_guest_suggested_cards button[data-testid^="baseButton"]:hover,
        .st-key-ha_user_suggested_cards button[data-testid^="baseButton"]:hover {{
            border-color: rgba(122, 145, 112, 0.32) !important;
            box-shadow: {_HA_INPUT_GLOW} !important;
        }}
        .ha-chat-guest-empty__title {{
            color: {_HA_SAGE_DEEP} !important;
            -webkit-text-fill-color: {_HA_SAGE_DEEP} !important;
        }}
        .ha-chat-guest-empty__subtitle {{
            color: #5a6654 !important;
            -webkit-text-fill-color: #5a6654 !important;
        }}
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"]:not(:has(
            [data-testid="stChatMessageAvatarUser"]
        )):not(:has([data-testid="chatAvatarIcon-user"])):not(.st-emotion-cache-user) {{
            background: {_HA_GLASS_STRONG} !important;
            border: 1px solid rgba(92, 111, 94, 0.08) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            box-shadow: 0 4px 18px rgba(60, 78, 58, 0.07) !important;
        }}
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ),
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"].st-emotion-cache-user {{
            background: {_HA_CHAT_USER_BUBBLE} !important;
            border: 1px solid rgba(92, 111, 94, 0.06) !important;
        }}
        {_sidebar_guest_no_scroll_css()}
        {_sidebar_collapse_injected_chrome_css()}
        {_layout_logged_in_sidebar_final_css()}
        {_layout_profile_main_final_css()}
        </style>
        """,
        in_sidebar=in_sidebar,
    )


def _sidebar_collapse_injected_chrome_css() -> str:
    """Logged-in only: style/script nodes in sidebar must not steal vertical space."""
    sb = (
        ':is(section[data-testid="stSidebar"], [data-testid="stSidebar"])'
        ':has(.st-key-ha_sidebar_user_main)'
    )
    return f"""
        {sb} [data-testid="stElementContainer"]:has(iframe),
        {sb} [data-testid="element-container"]:has(iframe),
        {sb} [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style),
        {sb} [data-testid="element-container"]:has([data-testid="stMarkdownContainer"] style) {{
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            border: none !important;
        }}
    """


def _layout_logged_in_sidebar_final_css() -> str:
    """Last-injected sidebar rules: full-height column, nav up, logout at bottom."""
    sb = (
        '[data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell)) '
        ':is(section[data-testid="stSidebar"], [data-testid="stSidebar"])'
        ':has(.st-key-ha_sidebar_user_main)'
    )
    um = f"{sb} .st-key-ha_sidebar_user_main"
    body = f"{um} .st-key-ha_sidebar_user_body"
    footer = f"{um} .st-key-ha_sidebar_user_footer"
    return f"""
        /* Injected style/script rows sit above NAVIGATION — same top as guest sidebar */
        {sb} .block-container > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:not(:has(.ha-sidebar-header)):not(
                :has(.st-key-ha_sidebar_user_main)
            ),
        {sb} .block-container > [data-testid="stVerticalBlock"]
            > [data-testid="element-container"]:not(:has(.ha-sidebar-header)):not(
                :has(.st-key-ha_sidebar_user_main)
            ) {{
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            visibility: hidden !important;
        }}
        {sb} [data-testid="stSidebarHeader"] {{
            flex: 0 0 auto !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: 1.65rem !important;
            padding: 0 0.45rem 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }}
        {sb} [data-testid="stSidebarContent"],
        {sb} [data-testid="stSidebarUserContent"] {{
            padding-top: 0 !important;
        }}
        {sb} .block-container {{
            padding-top: 0.2rem !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }}
        {sb} .block-container > [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 0.15rem !important;
        }}
        {sb} [data-testid="stElementContainer"]:has(.ha-sidebar-header),
        {sb} [data-testid="element-container"]:has(.ha-sidebar-header) {{
            flex: 0 0 auto !important;
        }}
        {sb} .ha-sidebar-header {{
            margin: 0 0 0.65rem 0 !important;
        }}
        {sb} .ha-sidebar-header__eyebrow {{
            margin: 0 0 0.45rem 0 !important;
            min-height: 0 !important;
        }}
        {um} {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
        }}
        {body} {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }}
        {footer} {{
            flex: 0 0 auto !important;
            margin-top: auto !important;
            width: 100% !important;
            padding-top: 0.65rem !important;
            padding-bottom: max(0.55rem, env(safe-area-inset-bottom, 0px)) !important;
        }}
        {footer} .ha-sidebar-user-footer-spacer,
        {footer} [data-testid="stElementContainer"]:has(.ha-sidebar-user-footer-spacer) {{
            flex: 0 0 auto !important;
            min-height: 1.65rem !important;
            height: 1.65rem !important;
        }}
        {footer} .st-key-ha_sidebar_user_footer_logout,
        {footer} [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_user_footer_logout) {{
            flex: 0 0 auto !important;
            margin-top: 0 !important;
        }}
    """


def _inject_html(html: str, *, in_sidebar: bool = False) -> None:
    """Inject markup; prefer sidebar so main column stays free of empty style nodes."""
    if in_sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def _layout_profile_main_final_css() -> str:
    """Profile page: content starts under header; main area scrolls."""
    layout = """
        /* Style/script injections in main used to reserve ~150px each — hide leftovers */
        html:has(.st-key-ha_profile_center) section.main
            [data-testid="stElementContainer"]:not(:has(.st-key-ha_profile_center)),
        html:has(.st-key-ha_profile_center) section.main
            [data-testid="element-container"]:not(:has(.st-key-ha_profile_center)) {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            visibility: hidden !important;
        }
        html:has(.st-key-ha_profile_page),
        html:has(.st-key-ha_profile_page) body {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
        }
        html:has(.st-key-ha_profile_page) [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
        }
        html:has(.st-key-ha_profile_page) [data-testid="stMain"],
        html:has(.st-key-ha_profile_page) section.main {
            justify-content: flex-start !important;
            align-items: stretch !important;
            overflow: hidden !important;
        }
        html:has(.st-key-ha_profile_page) [data-testid="stMainBlockContainer"],
        html:has(.st-key-ha_profile_page) section.main [data-testid="stMainBlockContainer"] {
            justify-content: flex-start !important;
            align-items: stretch !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
            padding-top: 0 !important;
            padding-bottom: 1.75rem !important;
        }
        html:has(.st-key-ha_profile_page) section.main .block-container,
        html:has(.st-key-ha_profile_page) [data-testid="stMainBlockContainer"] .block-container {
            flex: 0 1 auto !important;
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            overflow: visible !important;
            padding-top: 0 !important;
            margin-top: 0 !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }
        html:has(.st-key-ha_profile_page) .st-key-ha_profile_page,
        html:has(.st-key-ha_profile_page) .st-key-ha_profile_page > div,
        html:has(.st-key-ha_profile_page) .st-key-ha_profile_page [data-testid="stVerticalBlock"] {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        html:has(.st-key-ha_profile_page) .ha-section-title {
            margin-top: 0 !important;
            margin-bottom: 0.1rem !important;
        }
        html:has(.st-key-ha_profile_page) .ha-section-subtitle {
            margin-top: 0 !important;
            margin-bottom: 0.3rem !important;
        }
        html:has(.st-key-ha_profile_page)
            [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]:first-child,
        html:has(.st-key-ha_profile_page)
            section.main .block-container > [data-testid="stVerticalBlock"]:first-child {
            padding-top: 0 !important;
            margin-top: 0 !important;
            gap: 0 !important;
        }
        html:has(.st-key-ha_profile_page) .st-key-ha_profile_page {
            padding-top: 0.25rem !important;
        }
    """
    return layout + _profile_page_premium_card_css()


def _profile_page_label_override_css() -> str:
    """Profile labels — injected last so global ``label`` rules cannot win."""
    card = "html:has(.st-key-ha_profile_card) .st-key-ha_profile_card"
    return f"""
        {card} label[data-testid="stWidgetLabel"],
        {card} label[data-testid="stWidgetLabel"] p {{
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            color: #5c6658 !important;
            -webkit-text-fill-color: #5c6658 !important;
            letter-spacing: 0.01em !important;
            line-height: 1.25 !important;
            margin: 0 0 0.32rem 0 !important;
            padding: 0 !important;
            opacity: 1 !important;
        }}
        {card} label[data-testid="stWidgetLabel"] {{
            gap: 0.35rem !important;
            margin-bottom: 0.32rem !important;
        }}
        {card} label[data-testid="stWidgetLabel"] [data-testid="stIconMaterial"],
        {card} label[data-testid="stWidgetLabel"] .material-symbols-rounded {{
            font-size: 0.9rem !important;
            width: 0.9rem !important;
            height: 0.9rem !important;
            color: #6d8270 !important;
            opacity: 0.9 !important;
            font-variation-settings: "FILL" 0, "wght" 360, "GRAD" 0, "opsz" 20 !important;
        }}
    """


def _profile_page_premium_card_css() -> str:
    """Profile mockup: left header, white centered card, icon labels, sage save CTA."""
    center = ".st-key-ha_profile_center"
    card = ".st-key-ha_profile_card"
    page = ".st-key-ha_profile_page"
    root = "html:has(.st-key-ha_profile_center)"
    profile_w = "min(34rem, calc(100% - 2rem))"
    profile_input_bg = "#f3f4f2"
    profile_input_border = "#dce1d8"
    return f"""
        {root} section.main .block-container,
        {root} [data-testid="stMainBlockContainer"] .block-container {{
            align-items: center !important;
        }}
        {root} [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
        {root} section.main .block-container > [data-testid="stVerticalBlock"] {{
            align-items: center !important;
            width: 100% !important;
        }}
        {root} [data-testid="stElementContainer"]:has(.st-key-ha_profile_center),
        {root} [data-testid="element-container"]:has(.st-key-ha_profile_center) {{
            width: 100% !important;
            max-width: {profile_w} !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            box-sizing: border-box !important;
        }}
        {center},
        {center} > div,
        {center} [data-testid="stVerticalBlock"],
        {center} [data-testid="stVerticalBlockBorderWrapper"],
        {page},
        {page} > div,
        {page} [data-testid="stVerticalBlock"],
        {page} [data-testid="stVerticalBlockBorderWrapper"] {{
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: auto !important;
            margin-right: auto !important;
            box-sizing: border-box !important;
        }}
        {card},
        {card} > div,
        {card} [data-testid="stVerticalBlock"],
        {card} [data-testid="stVerticalBlockBorderWrapper"] {{
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
        }}
        {page} {{
            padding: 0.15rem 0 1.25rem 0 !important;
        }}
        {page} .ha-profile-header {{
            text-align: left !important;
            margin: 0 0 0.85rem 0 !important;
            padding: 0 !important;
        }}
        {page} .ha-profile-header__title {{
            font-family: "Lora", Georgia, serif !important;
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            color: {_HA_SAGE_DEEP} !important;
            margin: 0 0 0.35rem 0 !important;
            line-height: 1.15 !important;
        }}
        {page} .ha-profile-header__desc {{
            display: flex !important;
            align-items: flex-start !important;
            gap: 0.4rem !important;
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
            color: #5c6658 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }}
        {page} .ha-profile-header__leaf {{
            flex: 0 0 auto !important;
            font-size: 0.82rem !important;
            line-height: 1.45 !important;
            opacity: 0.85 !important;
        }}
        {page} .st-key-ha_profile_adv {{
            margin: 0 0 1rem 0 !important;
            width: 100% !important;
        }}
        {page} .st-key-ha_profile_adv [data-testid="stPopover"] {{
            width: 100% !important;
        }}
        {page} .st-key-ha_profile_adv [data-testid="stPopover"] > button {{
            width: 100% !important;
            justify-content: space-between !important;
            padding: 0.62rem 0.85rem !important;
            border-radius: 12px !important;
            border: 1px solid #e5e7eb !important;
            background: #ffffff !important;
            color: #3d4540 !important;
            -webkit-text-fill-color: #3d4540 !important;
            font-weight: 550 !important;
            font-size: 0.88rem !important;
            box-shadow: 0 1px 3px rgba(40, 55, 45, 0.04) !important;
        }}
        {page} .st-key-ha_profile_adv [data-testid="stPopover"] > button:hover {{
            border-color: rgba(92, 111, 94, 0.28) !important;
            background: #fafaf8 !important;
        }}
        {page} [data-testid="stElementContainer"]:has({card}),
        {page} [data-testid="element-container"]:has({card}) {{
            width: 100% !important;
            max-width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }}
        {card} {{
            background: #ffffff !important;
            border: 1px solid rgba(92, 111, 94, 0.12) !important;
            border-radius: 16px !important;
            box-shadow:
                0 8px 24px rgba(60, 78, 58, 0.07),
                0 2px 6px rgba(60, 78, 58, 0.04) !important;
            padding: 1.2rem 1.4rem 1.15rem !important;
            margin: 0 auto !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }}
        {card} [data-testid="stElementContainer"],
        {card} [data-testid="element-container"] {{
            margin-bottom: 0.35rem !important;
            padding-bottom: 0 !important;
        }}
        {card} [data-testid="stTextInput"],
        {card} [data-testid="stTextArea"],
        {card} [data-testid="stSelectbox"] {{
            gap: 0.25rem !important;
        }}
        {card} label[data-testid="stWidgetLabel"] {{
            width: fit-content !important;
            max-width: 100% !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.35rem !important;
            margin: 0 0 0.32rem 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }}
        {card} label[data-testid="stWidgetLabel"],
        {card} label[data-testid="stWidgetLabel"] p {{
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            color: #5c6658 !important;
            -webkit-text-fill-color: #5c6658 !important;
            letter-spacing: 0.01em !important;
            text-transform: none !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.25 !important;
            white-space: nowrap !important;
            opacity: 1 !important;
        }}
        {card} label[data-testid="stWidgetLabel"] [data-testid="stIconMaterial"],
        {card} label[data-testid="stWidgetLabel"] .material-symbols-rounded {{
            font-size: 0.9rem !important;
            width: 0.9rem !important;
            height: 0.9rem !important;
            color: #6d8270 !important;
            opacity: 0.9 !important;
            font-variation-settings: "FILL" 0, "wght" 360, "GRAD" 0, "opsz" 20 !important;
        }}
        {card} [data-testid="stForm"] div[data-baseweb="input"],
        {card} [data-testid="stForm"] div[data-baseweb="select"] {{
            border-radius: 10px !important;
            border: 1px solid {profile_input_border} !important;
            background: {profile_input_bg} !important;
            min-height: 2.35rem !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }}
        {card} [data-testid="stTextInput"] input,
        {card} [data-testid="stSelectbox"] [data-baseweb="select"] {{
            min-height: 2.35rem !important;
            padding: 0.45rem 0.65rem !important;
            font-size: 0.875rem !important;
            line-height: 1.35 !important;
        }}
        {card} [data-testid="stTextArea"] textarea {{
            border-radius: 10px !important;
            border: 1px solid {profile_input_border} !important;
            background: {profile_input_bg} !important;
        }}
        {card} [data-testid="stMarkdownContainer"] svg {{
            max-width: 1.25rem !important;
            max-height: 1.25rem !important;
            width: auto !important;
            height: auto !important;
        }}
        {card} [data-testid="stVerticalBlockBorderWrapper"],
        {card} > div[data-testid="stVerticalBlock"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            gap: 0.5rem !important;
        }}
        {card} [data-testid="stForm"] {{
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
        }}
        {card} [data-testid="stForm"] [data-testid="stVerticalBlock"] {{
            gap: 0.85rem !important;
        }}
        {card} input,
        {card} textarea,
        {card} [data-testid="stTextInput"] input,
        {card} [data-testid="stTextArea"] textarea,
        {card} [data-testid="stSelectbox"] > div {{
            border-radius: 10px !important;
            border: 1px solid {profile_input_border} !important;
            background: {profile_input_bg} !important;
            font-size: 0.875rem !important;
            color: #2f3830 !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}
        {card} input:focus,
        {card} textarea:focus,
        {card} [data-testid="stTextInput"] input:focus,
        {card} [data-testid="stTextArea"] textarea:focus {{
            border-color: rgba(122, 145, 112, 0.45) !important;
            box-shadow: {_HA_INPUT_GLOW} !important;
            outline: none !important;
        }}
        {card} [data-testid="stTextArea"] textarea {{
            min-height: 4.75rem !important;
            max-height: 9rem !important;
            line-height: 1.4 !important;
            resize: vertical !important;
            width: 100% !important;
            padding: 0.5rem 0.65rem !important;
            font-size: 0.875rem !important;
        }}
        {card} [data-testid="stTextArea"] {{
            margin-bottom: 0 !important;
        }}
        {card} [data-testid="stForm"] [data-testid="stTextInput"],
        {card} [data-testid="stForm"] [data-testid="stTextArea"],
        {card} [data-testid="stForm"] [data-testid="stSelectbox"] {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        {card} .st-key-ha_profile_row_name_age [data-testid="stHorizontalBlock"] {{
            gap: 0.85rem !important;
            width: 100% !important;
        }}
        {card} .st-key-ha_profile_row_name_age [data-testid="column"] {{
            min-width: 0 !important;
        }}
        {card} .st-key-ha_profile_row_gender [data-testid="stHorizontalBlock"] {{
            width: 100% !important;
        }}
        {card} .st-key-ha_profile_row_gender [data-testid="column"]:first-child {{
            flex: 0 0 50% !important;
            max-width: 50% !important;
        }}
        {card} .st-key-ha_profile_row_gender [data-testid="stSelectbox"],
        {card} .st-key-ha_profile_row_gender [data-testid="stSelectbox"] > div {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        {card} .st-key-ha_profile_row_allergies [data-testid="stTextArea"],
        {card} .st-key-ha_profile_row_conditions [data-testid="stTextArea"] {{
            width: 100% !important;
            max-width: 100% !important;
        }}
        {card} .st-key-ha_profile_save_row {{
            margin-top: 0.55rem !important;
            padding-top: 0 !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            width: 100% !important;
        }}
        {card} .st-key-ha_profile_save_row [data-testid="stVerticalBlock"],
        {card} .st-key-ha_profile_save_row [data-testid="stHorizontalBlock"] {{
            width: 100% !important;
            justify-content: flex-end !important;
        }}
        {card} .st-key-ha_profile_save_row [data-testid="stFormSubmitButton"],
        {card} .st-key-ha_profile_save_row [data-testid="stElementContainer"]:has(
            [data-testid="stFormSubmitButton"]
        ),
        {card} .st-key-ha_profile_save_row [data-testid="element-container"]:has(
            [data-testid="stFormSubmitButton"]
        ) {{
            display: flex !important;
            justify-content: flex-end !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        {card} [data-testid="stFormSubmitButton"] {{
            display: flex !important;
            justify-content: flex-end !important;
            width: auto !important;
            margin-left: auto !important;
        }}
        {card} [data-testid="stFormSubmitButton"] button,
        {card} [data-testid="stFormSubmitButton"] button[kind="primary"] {{
            width: auto !important;
            min-width: 10.5rem !important;
            max-width: none !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            padding: 0.55rem 1.15rem !important;
            border-radius: 10px !important;
            min-height: 2.35rem !important;
            font-size: 0.875rem !important;
            border: 1px solid rgba(62, 78, 53, 0.2) !important;
            background: linear-gradient(
                180deg,
                #5f7359 0%,
                {_HA_SAGE_DEEP} 100%
            ) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 3px 10px rgba(60, 78, 58, 0.14) !important;
            transition: transform 0.15s ease, box-shadow 0.2s ease !important;
        }}
        {card} [data-testid="stFormSubmitButton"] button:hover {{
            border-color: rgba(62, 78, 53, 0.28) !important;
            box-shadow: 0 6px 18px rgba(60, 78, 58, 0.22) !important;
            transform: translateY(-1px) !important;
        }}
        {card} [data-testid="stFormSubmitButton"] button:active {{
            transform: translateY(0) !important;
        }}
        {card} [data-testid="stFormSubmitButton"] button p,
        {card} [data-testid="stFormSubmitButton"] button span {{
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }}
        {page} [data-testid="stAlert"] {{
            border-radius: 12px !important;
            margin-top: 0.65rem !important;
        }}
        @media (min-width: 900px) {{
            {root} [data-testid="stElementContainer"]:has(.st-key-ha_profile_center),
            {root} [data-testid="element-container"]:has(.st-key-ha_profile_center) {{
                max-width: min(34rem, calc(100% - 3rem)) !important;
            }}
            {page} .ha-profile-header__title {{
                font-size: 2rem !important;
            }}
            {card} {{
                padding: 1.25rem 1.45rem 1.2rem !important;
            }}
        }}
        @media (max-width: 640px) {{
            {root} [data-testid="stElementContainer"]:has(.st-key-ha_profile_center),
            {root} [data-testid="element-container"]:has(.st-key-ha_profile_center) {{
                max-width: min(34rem, calc(100% - 1.25rem)) !important;
            }}
            {card} .st-key-ha_profile_row_gender [data-testid="column"]:first-child {{
                flex: 0 0 100% !important;
                max-width: 100% !important;
            }}
            {page} .ha-profile-hero .ha-section-title {{
                font-size: 1.45rem !important;
            }}
            {card} {{
                padding: 0.75rem 0.85rem 0.65rem !important;
                border-radius: 12px !important;
            }}
            {card} .st-key-ha_profile_row_name_age [data-testid="stHorizontalBlock"] {{
                gap: 0.65rem !important;
            }}
        }}
    """


def _inject_herbal_shell_background(*, in_sidebar: bool = False) -> None:
    """Chat shell: user's cream paper background image on the main pane only."""
    (
        main_bg_images,
        main_bg_positions,
        main_bg_sizes,
        main_bg_repeats,
        main_bg_attachments,
    ) = _chat_shell_main_background_layers()
    shell_css = _herbal_shell_background_css(
        main_bg_images,
        main_bg_positions,
        main_bg_sizes,
        main_bg_repeats,
        main_bg_attachments,
    )
    _inject_html(f"<style>{shell_css}</style>", in_sidebar=in_sidebar)


def _inject_guest_login_fullbleed_styles() -> None:
    """Guest Login only: remove sidebar column and expand main to full width (Streamlit 1.38+)."""
    card_bg_css = _GUEST_AUTH_CARD_SOLID_CSS
    st.markdown(
        """
        <style>
        """
        + _GUEST_AUTH_PAGE_BACKGROUND_CSS
        + """
        /* Sidebar column + all chrome (1.51: stExpandSidebarButton, stSidebarCollapseButton) */
        [data-testid="stSidebar"],
        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavItems"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            flex: 0 0 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            border: none !important;
            transform: none !important;
            pointer-events: none !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarNavViewButton"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        /* Flex row: main consumes full width */
        [data-testid="stAppViewContainer"] {
            display: flex !important;
            flex-direction: row !important;
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100% !important;
            max-width: 100vw !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stMain"],
        [data-testid="stAppViewContainer"] section.main {
            flex: 1 1 100% !important;
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
        }
        /* Header brand: no gap for hidden sidebar menu */
        [data-testid="stHeader"]::after {
            left: 1rem !important;
        }
        [data-testid="stHeader"]{
          height:55px !important;
          padding-top:0 !important;
          margin-bottom:0 !important;
        }

        [data-testid="stAppViewContainer"]{
          padding-top:0 !important;
             }

             [data-testid="stMainBlockContainer"]{
           padding-top:0 !important;
              padding-bottom:clamp(2rem, 7vh, 4.5rem) !important;
              justify-content:flex-start !important;
              align-items:stretch !important;
              height:calc(100dvh - 3.5rem) !important;
              max-height:calc(100dvh - 3.5rem) !important;
              min-height:0 !important;
              overflow-x:hidden !important;
              overflow-y:auto !important;
            }

             section.main .block-container{
              padding-top:0 !important;
              margin-top:0 !important;
               }

               section.main *{
                 margin-top:0 !important;
                 }

        /* Guest login: kaydırma kapalı, tek ekran */
        html, body, #root {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
        }
        [data-testid="stApp"] {
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"] {
            height: calc(100dvh - 3.5rem) !important;
            max-height: calc(100dvh - 3.5rem) !important;
            overflow: hidden !important;
            align-items: stretch !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stMain"],
        [data-testid="stAppViewContainer"] section.main {
            overflow: hidden !important;
            height: calc(100dvh - 3.5rem) !important;
            max-height: calc(100dvh - 3.5rem) !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        [data-testid="stMainBlockContainer"] {
            flex: 1 1 auto !important;
            height: calc(100dvh - 3.5rem) !important;
            max-height: calc(100dvh - 3.5rem) !important;
            min-height: 0 !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
            padding: 0.5rem 1rem clamp(2rem, 7vh, 4.5rem) 1rem !important;
        }
        /* Profile / Admin: top-aligned, scrollable — full form visible */
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stMain"]:not(:has(.st-key-ha_chat_composer_row)) {
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            min-height: 0 !important;
        }
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stMainBlockContainer"]:not(:has(.st-key-ha_chat_composer_row)) {
            flex: 1 1 auto !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
            padding-top: 0.2rem !important;
            padding-bottom: 2rem !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stMainBlockContainer"]:not(:has(.st-key-ha_chat_composer_row))
            .block-container,
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stMain"]:not(:has(.st-key-ha_chat_composer_row))
            .block-container {
            flex: 0 1 auto !important;
            height: auto !important;
            max-height: none !important;
            min-height: 0 !important;
            overflow: visible !important;
            padding-top: 0.15rem !important;
            margin-top: 0 !important;
            padding-bottom: 1.5rem !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stMainBlockContainer"]:not(:has(.st-key-ha_chat_composer_row))
            [data-testid="stVerticalBlock"],
        [data-testid="stAppViewContainer"]:not(:has(.st-key-ha_auth_shell))
            [data-testid="stMainBlockContainer"]:not(:has(.st-key-ha_chat_composer_row))
            [data-testid="stVerticalBlockBorderWrapper"] {{
            overflow: visible !important;
            max-height: none !important;
        }}
        [data-testid="stMainBlockContainer"]:has(.st-key-ha_profile_page) .ha-section-title,
        [data-testid="stMainBlockContainer"]:not(:has(.st-key-ha_chat_composer_row)) .ha-section-title {{
            margin-top: 0 !important;
            margin-bottom: 0.12rem !important;
        }}
        [data-testid="stMainBlockContainer"]:has(.st-key-ha_profile_page) .ha-section-subtitle,
        [data-testid="stMainBlockContainer"]:not(:has(.st-key-ha_chat_composer_row)) .ha-section-subtitle {{
            margin-top: 0 !important;
            margin-bottom: 0.35rem !important;
        }}
        [data-testid="stMainBlockContainer"]:has(.st-key-ha_profile_page) .st-key-ha_profile_adv {{
            margin-top: 0 !important;
            margin-bottom: 0.25rem !important;
        }}
        .st-key-ha_profile_page,
        .st-key-ha_profile_page [data-testid="stVerticalBlock"],
        .st-key-ha_profile_page [data-testid="stVerticalBlockBorderWrapper"] {{
            overflow: visible !important;
            max-height: none !important;
        }}

        section.main .block-container:has(.st-key-ha_chat_composer_row),
        [data-testid="stMainBlockContainer"]:has(.st-key-ha_chat_composer_row) .block-container {{
            flex: 0 0 auto !important;
            height: auto !important;
            max-height: 100% !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }}
        section.main .block-container:not(:has(.st-key-ha_chat_composer_row)) {{
            flex: 0 1 auto !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }}
        .main .block-container:has(.st-key-ha_auth_shell) {
            min-height: 0 !important;
            height: auto !important;
            max-height: 100% !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            background: transparent !important;
            overflow: hidden !important;
        }
        .st-key-ha_auth_shell {
            --ha-lux-card-h: min(29.5rem, 58vh) !important;
            --ha-auth-card-fixed-h: min(37.5rem, 68vh) !important;
            width: 100% !important;
            max-width: 900px !important;
            flex: 0 0 auto !important;
            margin: 0 auto !important;
            transform: translateY(clamp(-2.5rem, -4vh, -0.5rem)) !important;
            overflow: visible !important;
            max-height: none !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            justify-content: flex-start !important;
            gap: 0.1rem !important;
        }
        .st-key-ha_auth_shell [data-testid="element-container"]:has(button[data-testid="baseButton-tertiary"]) {
            margin-bottom: 0 !important;
        }
        /* Misafir linki: mockup — kartın sol üstü */
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap {
            position: absolute !important;
            top: 0.85rem !important;
            left: 1.15rem !important;
            right: auto !important;
            z-index: 6 !important;
            display: flex !important;
            justify-content: flex-start !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap button {
            background: #ffffff !important;
            border: 1px solid rgba(74, 93, 69, 0.22) !important;
            border-radius: 999px !important;
            padding: 0.4rem 1rem !important;
            min-height: 2.1rem !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            font-family: "Inter", system-ui, sans-serif !important;
            color: var(--ha-lux-ink, #4a5d45) !important;
            box-shadow: none !important;
            text-decoration: none !important;
        }
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap button:hover {
            background: #ffffff !important;
            border-color: rgba(112, 130, 96, 0.38) !important;
            box-shadow: 0 2px 8px rgba(75, 89, 64, 0.12) !important;
        }
        /* Login: Continue as Guest ile Welcome arası ince yatay çizgi (kart yan border korunur) */
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap,
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap [data-testid="stVerticalBlock"],
        .st-key-ha_auth_card .st-key-ha_auth_back_chat_wrap [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_auth_card [data-testid="stElementContainer"]:has(.st-key-ha_auth_back_chat_wrap),
        .st-key-ha_auth_card [data-testid="element-container"]:has(.st-key-ha_auth_back_chat_wrap) {
            border-bottom: none !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_card [data-testid="stElementContainer"]:has(.st-key-ha_auth_back_chat_wrap) + [data-testid="stElementContainer"],
        .st-key-ha_auth_card [data-testid="element-container"]:has(.st-key-ha_auth_back_chat_wrap) + [data-testid="element-container"],
        .st-key-ha_auth_card [data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"]),
        .st-key-ha_auth_card [data-testid="element-container"]:has([data-testid="stHorizontalBlock"]) {
            border-top: none !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            border-top: none !important;
        }
        .st-key-ha_auth_card [data-testid="stVerticalBlock"] > hr,
        .st-key-ha_auth_card [data-testid="stElementContainer"]:has(.st-key-ha_auth_back_chat_wrap) + [data-testid="stElementContainer"] hr,
        .st-key-ha_auth_card [data-testid="element-container"]:has(.st-key-ha_auth_back_chat_wrap) + [data-testid="element-container"] hr {
            display: none !important;
            border: none !important;
            background: none !important;
            height: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_card {
            flex: 0 1 auto !important;
            height: auto !important;
            min-height: var(--ha-auth-card-fixed-h) !important;
            max-height: none !important;
            overflow: hidden !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            flex: 1 1 auto !important;
            height: auto !important;
            min-height: calc(var(--ha-auth-card-fixed-h) - 2.05rem) !important;
            max-height: none !important;
            overflow: hidden !important;
            align-items: stretch !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2) {
            height: auto !important;
            min-height: 100% !important;
            align-self: stretch !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            > div[data-testid="stVerticalBlock"] {
            height: auto !important;
            min-height: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand):not(.ha-lux-welcome--panel) {
            min-height: 0 !important;
            max-height: none !important;
            margin-top: 0 !important;
            padding: 0.15rem 1rem 0.55rem 1rem !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
            padding: 0.5rem 1.65rem 2.15rem 1.65rem !important;
            min-height: 100% !important;
            max-height: none !important;
            overflow: hidden !important;
            justify-content: flex-start !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_form_card {
            flex: 0 1 auto !important;
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
            padding-bottom: 0.35rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_form_body {
            flex: 0 1 auto !important;
            min-height: 0 !important;
            height: auto !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--panel {
            min-height: 100% !important;
            height: auto !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="element-container"] {
            margin-bottom: 0.4rem !important;
        }
        @media (min-width: 901px) {
            .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_lang_header {
                margin-bottom: 0.08rem !important;
            }
            .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row div[role="radiogroup"] {
                margin-bottom: 0.15rem !important;
            }
            .st-key-ha_auth_shell .ha-lux-form-title {
                font-size: clamp(1.75rem, 3vw, 2.35rem) !important;
            }
            .st-key-ha_auth_shell .ha-lux-form-sub {
                font-size: 0.88rem !important;
            }
            .st-key-ha_lux_remember_forgot_bar .st-key-ha_lux_forgot_row {
                bottom: 3.65rem !important;
            }
            .st-key-ha_auth_shell .st-key-ha_auth_primary_submit {
                margin-bottom: 0.15rem !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) > div[data-testid="stVerticalBlock"] {
                height: auto !important;
                min-height: 0 !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: flex-start !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand) {
                flex: 0 1 auto !important;
                min-height: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                justify-content: flex-start !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand) .ha-lux-welcome__inner {
                align-items: center !important;
                text-align: center !important;
                justify-content: flex-start !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand):has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
                min-height: min(32vh, 18rem) !important;
                justify-content: flex-start !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1 {
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center !important;
            }
        }
        @media (max-width: 900px) {
            .st-key-ha_auth_shell {
                --ha-lux-card-h: auto !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1)
                > div[data-testid="stVerticalBlock"] {
                justify-content: flex-start !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                min-height: 0 !important;
                max-height: none !important;
                margin-top: -0.58rem !important;
                padding: 0.5rem 1rem 0.65rem 1rem !important;
                justify-content: flex-start !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__inner {
                align-items: center !important;
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1,
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center !important;
            }
            .st-key-ha_auth_shell .st-key-ha_auth_card {
                height: auto !important;
                min-height: min(37.5rem, 68vh) !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
                height: auto !important;
                min-height: min(35rem, 65vh) !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                min-height: min(26rem, 48vh) !important;
                max-height: none !important;
                height: auto !important;
                padding: calc(1.15rem + 0.75rem) 1.25rem 1.5rem 1.25rem !important;
            }
            .st-key-ha_auth_shell .st-key-ha_lux_username_outside {
                margin-top: 0.85rem !important;
            }
            .st-key-ha_lux_remember_forgot_bar .st-key-ha_lux_forgot_row {
                bottom: 3.55rem !important;
                max-width: 55% !important;
            }
            .st-key-ha_lux_remember_forgot_bar .st-key-ha_lux_remember_row {
                padding-right: 48% !important;
            }
        }
        """
        + _GUEST_AUTH_FLATTEN_LAYERS_CSS
        + card_bg_css
        + _GUEST_AUTH_MOCKUP_CSS
        + _guest_auth_card_background_css()
        + """
        </style>
        """,
        unsafe_allow_html=True,
    )

def _inject_global_styles(*, in_sidebar: bool = False) -> None:
    _inject_html(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Inter:wght@400;500;600&display=swap");
        :root {
            /* Premium herbal: cream shell + sage accents */
            --ha-bg: #F4F2EC;
            --ha-bg-2: #EBE8E0;
            --ha-sidebar-bg: #F2EBE3;
            --ha-sidebar-glass: rgba(242, 235, 227, 0.88);
            --ha-sidebar-edge: rgba(44, 48, 42, 0.085);
            --ha-sidebar-shadow: 3px 0 16px rgba(38, 42, 36, 0.04);
            --ha-sidebar-radius: 16px;
            --ha-sidebar-target-width: 17rem;
            --ha-text: #2a3228;
            --ha-text-soft: #6a7568;
            --ha-border: rgba(92, 111, 94, 0.12);
            --ha-surface: rgba(255, 255, 255, 0.72);
            --ha-chat-ink: #2a3228;
            --ha-sage: #5c6f5e;
            --ha-sage-deep: #3e4e35;
            --ha-sage-soft: #7a9170;
            --ha-sage-mist: rgba(92, 111, 94, 0.14);
            --ha-glass: rgba(255, 255, 255, 0.58);
            --ha-glass-border: rgba(92, 111, 94, 0.2);
            --ha-input-glow: 0 0 0 1px rgba(122, 145, 112, 0.28),
                0 0 22px rgba(122, 145, 112, 0.14),
                0 4px 20px rgba(60, 78, 58, 0.07);
            --ha-accent-soft: #5c6f5e;
            --ha-shell-1: #ffffff;
            --ha-shell-2: #ffffff;
            --ha-shell-3: #ffffff;
            --ha-shell-4: #ffffff;
            --ha-card-edge: rgba(92, 111, 94, 0.14);
            --ha-card-shadow: 0 3px 14px rgba(60, 78, 58, 0.06);
            --ha-btn-max-w: min(100%, 15rem);
            /* Noise dokusu sade görünüm için kapatıldı (boş url, opaklık 0) */
            --ha-noise-tile: none;
        }
        /* App-wide: compact buttons; width follows label (not full column) */
        button[data-testid^="baseButton"] {
            min-height: 2.05rem !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            font-size: 0.84rem !important;
            width: auto !important;
            max-width: var(--ha-btn-max-w) !important;
            box-sizing: border-box !important;
        }
        [data-testid="element-container"]:has(button[data-testid^="baseButton"]) {
            width: fit-content !important;
            max-width: 100% !important;
        }
        /* Logged-in UI: nötr beyaz buton dolguları (ChatGPT-vari sade) */
        button[data-testid^="baseButton"] {
            background-color: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
        }
        button[data-testid^="baseButton"] p,
        button[data-testid^="baseButton"] span {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        button[data-testid^="baseButton"]:hover {
            background: #f5f5f5 !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
        }
        div[role="radiogroup"] label {
            padding-top: 0.28rem !important;
            padding-bottom: 0.28rem !important;
            line-height: 1.2 !important;
        }
        [data-testid="stCheckbox"] label {
            line-height: 1.2 !important;
            padding-top: 0.18rem !important;
            padding-bottom: 0.18rem !important;
        }
        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
            font-size: 16px !important;
            zoom: 1 !important;
            transform: none !important;
        }
        .main .block-container {
            width: 100%;
            max-width: 1560px;
            padding-top: 0rem !important;
            padding-bottom: 1.35rem !important;
            position: relative;
            z-index: 1;
        }
        html:not(:has(.st-key-ha_auth_shell)) [data-testid="stApp"] {
            background: transparent !important;
        }
        html:has(.st-key-ha_auth_shell) [data-testid="stAppViewContainer"] {
            background: #F5F1EA !important;
        }
        html:has(.st-key-ha_auth_shell),
        html:has(.st-key-ha_auth_shell) body,
        html:has(.st-key-ha_auth_shell) #root,
        html:has(.st-key-ha_auth_shell) [data-testid="stApp"] {
            background: #F5F1EA !important;
        }
        [data-testid="stApp"]:has(.st-key-ha_auth_shell) footer,
        [data-testid="stApp"]:has(.st-key-ha_auth_shell) [data-testid="stFooter"] {
            background: #F5F1EA !important;
        }
        [data-testid="stMainBlockContainer"],
        .main .block-container {
            background-color: transparent !important;
            background-image: none !important;
        }
        footer,
        [data-testid="stFooter"] {
            background: var(--ha-bg) !important;
        }
        section.main {
            position: relative;
            z-index: 0;
        }
        .main .block-container:has(.st-key-ha_auth_shell) {
            max-width: 940px;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: transparent !important;
        }
        html:has(.st-key-ha_auth_shell),
        html:has(.st-key-ha_auth_shell) body,
        html:has(.st-key-ha_auth_shell) #root {
            overflow: hidden !important;
            height: 100dvh !important;
            max-height: 100dvh !important;
        }
        html:has(.st-key-ha_auth_shell) [data-testid="stApp"] {
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) {
            height: calc(100dvh - 3.5rem) !important;
            max-height: calc(100dvh - 3.5rem) !important;
            overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) [data-testid="stMain"],
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) section.main {
            height: calc(100dvh - 3.5rem) !important;
            max-height: calc(100dvh - 3.5rem) !important;
            min-height: 0 !important;
            overflow: hidden !important;
        }
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) [data-testid="stMainBlockContainer"] {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            height: calc(100dvh - 3.5rem) !important;
            max-height: calc(100dvh - 3.5rem) !important;
            min-height: 0 !important;
            overflow: hidden !important;
            padding-top: 0.5rem !important;
            padding-bottom: clamp(2rem, 7vh, 4.5rem) !important;
        }
        .main .block-container:has(.st-key-ha_auth_shell) {
            overflow: hidden !important;
            max-height: 100% !important;
        }
        html:has(.st-key-ha_auth_shell) footer,
        html:has(.st-key-ha_auth_shell) [data-testid="stFooter"] {
            display: none !important;
        }
        html:has(.st-key-ha_auth_shell) {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        html:has(.st-key-ha_auth_shell)::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        .st-key-ha_auth_shell {
            transform: translateY(clamp(-2.5rem, -4vh, -0.5rem));
            overflow: hidden !important;
            max-height: calc(100dvh - 4.5rem) !important;
            --ha-lux-ink: #3e4e35;
            --ha-lux-moss: #4a5d45;
            --ha-lux-cream: #f7f8f3;
            --ha-auth-field-inset: 1.15rem;
            --ha-auth-field-width: calc(100% - (2 * var(--ha-auth-field-inset)));
            --ha-lux-card-radius: 20px;
            --ha-lux-card-shadow: 0 12px 40px rgba(60, 78, 58, 0.08), 0 2px 8px rgba(60, 78, 58, 0.04);
            --ha-lux-card-h: min(29.5rem, 58vh);
            --ha-auth-card-fixed-h: min(37.5rem, 68vh);
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
            font-family: "Inter", system-ui, -apple-system, sans-serif;
            color: var(--ha-lux-ink);
        }
        .st-key-ha_auth_shell .st-key-ha_auth_card {
            width: 100%;
            margin-top: -0.35rem !important;
            position: relative !important;
            border-radius: var(--ha-lux-card-radius) !important;
            overflow: hidden !important;
            border: 1px solid rgba(74, 93, 69, 0.12) !important;
            box-shadow: var(--ha-lux-card-shadow) !important;
            height: auto !important;
            min-height: var(--ha-auth-card-fixed-h) !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            height: auto !important;
            min-height: calc(var(--ha-auth-card-fixed-h) - 2.05rem) !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            > div[data-testid="stVerticalBlock"],
        .st-key-ha_auth_shell .st-key-ha_auth_form_card,
        .st-key-ha_auth_shell .ha-lux-welcome--panel {
            height: auto !important;
            min-height: 100% !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            position: relative;
            align-items: stretch !important;
            gap: 0 !important;
            border-radius: 0 !important;
            overflow: hidden;
            border-top: none !important;
            padding-top: 2.05rem !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            .st-key-ha_auth_top_bar {
            padding-top: 1.65rem !important;
        }
        /* Login Welcome hero (sol sütun): üst ayırıcı / border-top yok */
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stElementContainer"]:has(.ha-lux-welcome),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="element-container"]:has(.ha-lux-welcome),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"]:has(.ha-lux-welcome),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdown"]:has(.ha-lux-welcome),
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome__inner,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome h1,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome__title {
            border-top: none !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome__rule,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"]:has(.ha-lux-welcome) hr {
            display: none !important;
            border: none !important;
            background: none !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome h1::before,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome h1::after,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome__title::before,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome__title::after,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"]:has(.ha-lux-welcome)::before,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"]:has(.ha-lux-welcome)::after,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdown"]:has(.ha-lux-welcome)::before,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdown"]:has(.ha-lux-welcome)::after,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome::before,
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            .ha-lux-welcome__inner::before {
            display: none !important;
            content: none !important;
            border: none !important;
            background: none !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: 0 !important;
            position: relative;
            z-index: 1;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1) {
            flex: 0.9 1 0% !important;
            border-right: 1px solid rgba(74, 93, 69, 0.2) !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2) {
            flex: 1.1 1 0% !important;
            box-sizing: border-box !important;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) {
            border-right: 1px solid rgba(74, 93, 69, 0.2) !important;
            border-bottom: none !important;
        }
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"] {
            position: relative;
            padding: 0 !important;
            margin: 0 !important;
            gap: 0 !important;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"] {
            height: 100% !important;
            min-height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: center !important;
            padding-top: 0.45rem !important;
        }
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            > div[data-testid="stVerticalBlock"] {
            position: relative;
            padding: 0.35rem 2rem 2.15rem 2rem !important;
            min-height: var(--ha-auth-card-fixed-h, var(--ha-lux-card-h));
            height: auto !important;
            overflow: hidden !important;
            box-sizing: border-box;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            > div[data-testid="stVerticalBlock"] {
            padding-top: 0.35rem !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome {
            margin-top: 0 !important;
            padding-top: 0.15rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"]:has(.ha-lux-form-title) {
            margin-top: 0 !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-title {
            margin-top: 0 !important;
        }
        }
        /* —— Welcome column (markdown) —— */
        .st-key-ha_auth_shell .ha-lux-welcome {
            position: relative;
            min-height: var(--ha-lux-card-h);
            box-sizing: border-box;
            margin-top: -0.72rem;
            padding: 0.75rem 2.05rem 2rem 2.05rem;
            display: flex;
            flex-direction: column;
            background: transparent;
            flex: 1 1 auto;
            width: 100%;
            overflow: hidden;
            justify-content: flex-start;
            align-items: center;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand):not(.ha-lux-welcome--panel) {
            min-height: 0;
            flex: 0 1 auto;
            align-self: flex-start;
            margin-top: -0.78rem;
            padding-top: 0.65rem;
        }
        .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand):has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
            min-height: min(42vh, 24rem) !important;
            justify-content: center !important;
        }
        }
        .st-key-ha_auth_shell .ha-lux-welcome__inner {
            position: relative;
            z-index: 2;
            flex: 0 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            width: 100%;
            max-width: 100%;
        }
        .st-key-ha_auth_shell .ha-lux-welcome:not(.ha-lux-welcome--brand) .ha-lux-welcome__logo {
            margin: 2.75rem auto;
            max-width: min(100%, 560px);
            width: 100%;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo img {
            display: block;
            width: 100%;
            max-width: 560px;
            height: auto;
            margin: 0 auto;
            object-fit: contain;
            object-position: center center;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__vine {
            position: absolute;
            top: -24px;
            right: -36px;
            width: 200px;
            height: 200px;
            border: 2px solid rgba(112, 130, 96, 0.14);
            border-radius: 52% 38% 62% 48%;
            pointer-events: none;
            z-index: 0;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__ghost-leaf {
            position: absolute;
            bottom: 12%;
            left: -28px;
            width: 110px;
            height: 110px;
            background: rgba(112, 130, 96, 0.09);
            border-radius: 10% 85% 35% 80%;
            transform: rotate(-18deg);
            pointer-events: none;
            z-index: 0;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1,
        .st-key-ha_auth_shell .ha-lux-welcome__title {
            font-family: "Lora", Georgia, "Times New Roman", serif;
            font-size: clamp(1.85rem, 3.2vw, 2.45rem);
            font-weight: 700;
            color: #354530;
            margin: 0 0 0.45rem 0;
            line-height: 1.12;
            letter-spacing: -0.02em;
            text-align: center;
            width: 100%;
            text-shadow: 0 1px 14px rgba(255, 255, 255, 0.92), 0 0 1px rgba(255, 255, 255, 0.75);
            -webkit-font-smoothing: antialiased;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__lead {
            margin: 0;
            max-width: 40ch;
            font-size: 1.05rem;
            line-height: 1.5;
            color: #3e4e35;
            font-weight: 600;
            text-align: center;
            text-shadow: 0 1px 10px rgba(255, 255, 255, 0.9), 0 0 1px rgba(255, 255, 255, 0.65);
            -webkit-font-smoothing: antialiased;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdownContainer"] {
            text-align: center !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stMarkdown"] {
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"]
            > [data-testid="element-container"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="element-container"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            [data-testid="stElementContainer"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1,
        .st-key-ha_auth_shell p.ha-lux-welcome__lead {
            text-align: center !important;
        }
        /* Streamlit [data-testid="stMarkdown"] wrapper can force start alignment; override on welcome only */
        [data-testid="stMarkdown"] .ha-lux-welcome,
        [data-testid="stMarkdown"] .ha-lux-welcome__inner,
        [data-testid="stMarkdown"] .ha-lux-welcome h1,
        [data-testid="stMarkdown"] .ha-lux-welcome p.ha-lux-welcome__lead,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            text-align: center !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome .ha-lux-welcome__title {
            font-size: clamp(1.85rem, 3.2vw, 2.45rem) !important;
            font-weight: 700 !important;
            color: #354530 !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            color: #3e4e35 !important;
        }
        .st-key-ha_auth_shell .ha-lux-botanical {
            display: none !important;
        }
        .st-key-ha_auth_shell .ha-lux-botanical__accent {
            position: absolute;
            right: -0.5rem;
            bottom: -0.25rem;
            width: 140px;
            height: 140px;
            background: radial-gradient(
                circle at 35% 35%,
                rgba(112, 130, 96, 0.22),
                transparent 68%
            );
            border-radius: 50%;
            pointer-events: none;
        }
        .st-key-ha_auth_shell .ha-lux-botanical__photo,
        .st-key-ha_auth_shell .ha-lux-pot {
            display: none;
        }
        /* —— Auth language switcher (kart sağ üst; tek konum) —— */
        .st-key-ha_auth_card .st-key-ha_auth_lang_header,
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_lang_header {
            position: absolute !important;
            top: 0.85rem !important;
            right: 1.15rem !important;
            left: auto !important;
            z-index: 6 !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header {
            width: 100%;
            margin: 0 0 0.1rem 0;
            padding: 0;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: flex-start !important;
            opacity: 1;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-end !important;
            flex-wrap: wrap !important;
            gap: 0.25rem !important;
            width: auto !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stElementContainer"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="element-container"] {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 9.75rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"] > div {
            width: 100% !important;
            min-width: 9.75rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"] [data-baseweb="select"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header div[data-baseweb="select"] {
            background-color: #F5F1EA !important;
            border: 1px solid rgba(74, 93, 69, 0.16) !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 10px rgba(60, 78, 58, 0.1) !important;
            min-height: 2.35rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header div[data-baseweb="select"] > div {
            background-color: #F5F1EA !important;
            border: none !important;
            min-height: 2.35rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"] [data-baseweb="select"] span,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header div[data-baseweb="select"] span,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"] [data-baseweb="select"] input {
            color: #3e4e35 !important;
            -webkit-text-fill-color: #3e4e35 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            font-family: "Inter", system-ui, sans-serif !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"] svg {
            color: #5f6b56 !important;
            fill: #5f6b56 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"]:focus-within [data-baseweb="select"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stSelectbox"]:focus-within div[data-baseweb="select"] {
            border-color: rgba(74, 93, 69, 0.28) !important;
            box-shadow: 0 2px 12px rgba(60, 78, 58, 0.14) !important;
        }
        /* Top: EN/TR sağda (absolute → ha_auth_card), altında Login | Create Account */
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar {
            width: 100%;
            margin: 0 0 0.05rem 0;
            min-height: 0 !important;
            padding-top: 1.65rem !important;
            padding-right: 0.15rem !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row {
            width: 100%;
            max-width: 100%;
            margin-left: auto;
            margin-right: auto;
            box-sizing: border-box;
            padding-right: 0.1rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row div[role="radiogroup"] {
            margin: 0 0 0.15rem 0 !important;
        }
        /* Login / Create Account — iki eşit pill; dil seçici bu kapsamda değil */
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row [data-testid="stRadio"] {
            width: 100% !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: stretch !important;
            justify-content: stretch !important;
            gap: 0.5rem !important;
            margin: 0 0 0.55rem 0 !important;
            width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            overflow: visible !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label {
            flex: 1 1 0 !important;
            min-width: 0 !important;
            width: auto !important;
            min-height: 2.75rem !important;
            height: 2.75rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 9999px !important;
            padding: 0 0.7rem !important;
            margin: 0 !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            box-sizing: border-box !important;
            background: #F5F1EA !important;
            border: 1px solid rgba(62, 78, 53, 0.12) !important;
            cursor: pointer !important;
            overflow: visible !important;
            transition: background-color 0.18s ease, border-color 0.18s ease !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label p {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-align: center !important;
            margin: 0 !important;
            color: #3e4e35 !important;
            white-space: nowrap !important;
            line-height: 1.2 !important;
            letter-spacing: -0.01em !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) {
            background: #3e4e35 !important;
            border-color: #3e4e35 !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:not(:has(input:checked)) {
            background: #F5F1EA !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        /* —— Form stack —— */
        .st-key-ha_auth_shell .ha-lux-form-title {
            font-family: "Lora", Georgia, "Times New Roman", serif;
            font-size: clamp(1.85rem, 3.2vw, 2.45rem);
            font-weight: 700;
            color: var(--ha-lux-ink);
            text-align: center;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
            line-height: 1.12;
        }
        .st-key-ha_auth_shell .ha-lux-form-deco {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            margin: 0 0 0.2rem 0;
            padding: 0;
            line-height: 1;
        }
        .st-key-ha_auth_shell .ha-lux-form-deco__leaf {
            font-size: 0.72rem;
            opacity: 0.88;
            filter: grayscale(0.15);
        }
        .st-key-ha_auth_shell .ha-lux-form-deco__dot {
            font-size: 0.42rem;
            color: #7a9170;
            opacity: 0.75;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub {
            text-align: center;
            font-size: 0.92rem;
            line-height: 1.45;
            color: #5f6b56;
            margin: 0 auto 0.65rem auto;
            max-width: 38ch;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--brand)
            > div[data-testid="stVerticalBlock"] {
            justify-content: center !important;
            align-items: center !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--brand)
            [data-testid="stMarkdownContainer"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--brand)
            [data-testid="stMarkdown"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--brand)
            [data-testid="stElementContainer"],
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--brand)
            [data-testid="element-container"] {
            width: 100% !important;
            max-width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) {
            width: 100% !important;
            max-width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            align-self: center !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            padding: 1.5rem 1.75rem 2rem 1.75rem !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) .ha-lux-welcome__inner {
            width: 100% !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            min-height: min(38vh, 22rem) !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand .ha-lux-welcome__logo {
            width: fit-content !important;
            max-width: min(100%, 300px) !important;
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome--brand .ha-lux-welcome__logo img {
            display: block !important;
            width: 100% !important;
            max-width: 280px !important;
            margin: 0 auto !important;
            object-fit: contain !important;
            object-position: center center !important;
        }
        @media (min-width: 901px) {
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--brand)
                > div[data-testid="stVerticalBlock"] {
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) {
                align-self: center !important;
                justify-content: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo) .ha-lux-welcome__inner,
            .st-key-ha_auth_shell .ha-lux-welcome--brand:not(.ha-lux-welcome--hero-photo):has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1):has(.ha-lux-welcome--hero-photo)
                > div[data-testid="stVerticalBlock"] {
                justify-content: stretch !important;
                align-items: stretch !important;
            }
        }
        @media (max-width: 900px) {
            .st-key-ha_auth_shell .ha-lux-welcome--hero-photo.ha-lux-welcome,
            .st-key-ha_auth_shell .ha-lux-welcome--hero-photo .ha-lux-welcome__inner {
                min-height: min(34vh, 14rem) !important;
            }
        }
        .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        .st-key-ha_auth_form_card {
            padding: 0.09rem 0.35rem 0.5rem 0.35rem !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_form_body {
            flex: 0 1 auto !important;
            min-height: 0 !important;
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
        }
        /* E-posta, şifre, Login: aynı genişlik ve hizalama */
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside,
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="element-container"]:has(div[data-baseweb="input"]),
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="stElementContainer"]:has(div[data-baseweb="input"]),
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit {
            flex: 0 0 auto !important;
            max-width: var(--ha-auth-field-width) !important;
            width: var(--ha-auth-field-width) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside [data-testid="stVerticalBlock"],
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside [data-testid="element-container"],
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside [data-testid="stElementContainer"],
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside [data-testid="stTextInput"],
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="element-container"],
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="stElementContainer"],
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="stFormSubmitButton"] {
            width: 100% !important;
            max-width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            box-sizing: border-box !important;
        }
        /* Tüm auth inputları aynı görünüm (form içi + e-posta dışarıda) */
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside div[data-baseweb="input"],
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"],
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="select"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            border-radius: 10px !important;
            min-height: 52px !important;
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(74, 93, 69, 0.22) !important;
            box-shadow: none !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"] > div,
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside div[data-baseweb="input"] > div {
            background-color: transparent !important;
            border: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside [data-baseweb="input"] input,
        .st-key-ha_auth_shell [data-testid="stForm"] [data-baseweb="input"] input {
            padding: 0.62rem 1rem !important;
            line-height: 1.45 !important;
            font-size: 0.94rem !important;
            color: var(--ha-lux-ink) !important;
            background: transparent !important;
        }
        .st-key-ha_auth_shell input::placeholder,
        .st-key-ha_auth_shell input::-webkit-input-placeholder {
            color: #7a8578 !important;
            opacity: 1 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside div[data-baseweb="input"]:focus-within,
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"]:focus-within {
            border-color: var(--ha-lux-moss) !important;
            box-shadow: 0 0 0 3px rgba(112, 130, 96, 0.18) !important;
        }
        .st-key-ha_auth_shell input:-webkit-autofill,
        .st-key-ha_auth_shell input:-webkit-autofill:hover,
        .st-key-ha_auth_shell input:-webkit-autofill:focus {
            -webkit-box-shadow: 0 0 0 1000px #ffffff inset !important;
            box-shadow: 0 0 0 1000px #ffffff inset !important;
            -webkit-text-fill-color: var(--ha-lux-ink) !important;
            caret-color: var(--ha-lux-ink) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_lux_username_outside {
            margin-bottom: 0.05rem !important;
            margin-top: 0.65rem !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_lux_remember_row [data-testid="stVerticalBlock"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            margin-top: 0 !important;
            margin-bottom: 0.2rem !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            gap: 0.35rem !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            display: flex !important;
            align-items: center !important;
        }
        .st-key-ha_lux_remember_row [data-testid="column"]:first-child [data-testid="stCheckbox"] {
            display: flex !important;
            align-items: center !important;
        }
        .st-key-ha_lux_remember_row [data-testid="column"]:first-child label {
            margin-bottom: 0 !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stCheckbox"] label p {
            font-size: 0.88rem !important;
            color: #5f6b56 !important;
        }
        /* Remember me + Forgot password: aynı satır (forgot form dışında, absolute) */
        .st-key-ha_lux_remember_forgot_bar {
            position: relative !important;
            width: 100% !important;
            margin-bottom: 0.25rem !important;
            padding-bottom: 0.15rem !important;
        }
        .st-key-ha_lux_remember_forgot_bar [data-testid="stForm"] {
            margin-bottom: 0 !important;
        }
        .st-key-ha_lux_remember_forgot_bar .st-key-ha_lux_remember_row {
            min-height: 2.1rem !important;
            padding-right: 46% !important;
            box-sizing: border-box !important;
        }
        .st-key-ha_lux_remember_forgot_bar .st-key-ha_lux_forgot_row {
            position: absolute !important;
            bottom: 3.65rem !important;
            right: 0 !important;
            left: auto !important;
            top: auto !important;
            width: auto !important;
            max-width: 50% !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
            z-index: 3 !important;
            pointer-events: auto !important;
        }
        .st-key-ha_lux_remember_forgot_bar .st-key-ha_auth_primary_submit {
            position: relative !important;
            z-index: 4 !important;
            margin-top: 0.35rem !important;
            flex-shrink: 0 !important;
            visibility: visible !important;
        }
        .st-key-ha_lux_forgot_row [data-testid="stElementContainer"] {
            width: auto !important;
            margin: 0 !important;
        }
        .st-key-ha_lux_forgot_row [data-testid="stButton"] {
            display: inline-flex !important;
            justify-content: flex-end !important;
        }
        .st-key-ha_lux_remember_forgot_bar .st-key-ha_lux_forgot_row [data-testid="stButton"] {
            width: 100% !important;
            justify-content: flex-end !important;
        }
        .st-key-ha_lux_forgot_row button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.25rem 0.5rem !important;
            min-height: 1.8rem !important;
            font-size: 0.88rem !important;
            color: var(--ha-text-soft, #6b7280) !important;
            text-decoration: underline;
            text-underline-offset: 3px;
            text-decoration-color: rgba(107, 114, 128, 0.35);
            white-space: nowrap !important;
            text-align: right !important;
        }
        .st-key-ha_lux_forgot_row button:hover {
            background: transparent !important;
            color: var(--ha-text, #1f1f1f) !important;
            text-decoration-color: currentColor;
        }
        /* Primary submit — inputlarla aynı genişlik */
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit {
            display: flex !important;
            justify-content: stretch !important;
            margin-top: 0.35rem !important;
            margin-bottom: 0.15rem !important;
            padding: 0 !important;
            position: relative !important;
            z-index: 4 !important;
            visibility: visible !important;
            flex-shrink: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="element-container"],
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="stElementContainer"] {
            padding: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="stFormSubmitButton"] {
            margin: 0 !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] {
            width: 100% !important;
            border-radius: 10px !important;
            min-height: 50px !important;
            font-family: "Lora", Georgia, "Times New Roman", serif !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.01em !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background-color: #5c6f5e !important;
            background-image: none !important;
            border: 1px solid #4d5f50 !important;
            box-shadow: 0 4px 12px rgba(92, 111, 94, 0.2) !important;
            transition:
                box-shadow 0.2s ease,
                transform 0.2s ease,
                filter 0.2s ease !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] *,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-family: inherit !important;
            font-weight: inherit !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            filter: brightness(1.05) saturate(1.03) !important;
            box-shadow:
                0 6px 16px rgba(75, 89, 64, 0.2),
                0 2px 5px rgba(75, 89, 64, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
            transform: translateY(-1px) !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:active,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]:active {
            transform: translateY(0) !important;
            filter: brightness(0.98) !important;
            box-shadow:
                0 2px 8px rgba(75, 89, 64, 0.14),
                inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"]:has(.st-key-ha_auth_primary_submit) [data-testid="stVerticalBlock"] > div:last-of-type [data-testid="element-container"] {
            margin-bottom: 0 !important;
        }
        /* Forgot password — link style (tertiary + secondary fallback) */
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="secondary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-tertiary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="tertiary"] {
            border-radius: 0 !important;
            min-height: auto !important;
            min-width: 0 !important;
            padding: 0.2rem 0 !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            color: var(--ha-lux-moss) !important;
            background: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
            text-decoration: underline !important;
            text-underline-offset: 3px;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-secondary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-tertiary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="tertiary"]:hover {
            color: var(--ha-lux-ink) !important;
            background: transparent !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="element-container"] {
            margin-bottom: 0.5rem;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"]:has(.ha-lux-form-title),
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"]:has(.ha-lux-form-sub) {
            margin-bottom: 0 !important;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"]:has(.ha-lux-form-title) {
            margin-top: -0.85rem !important;
        }
        }
        .st-key-ha_auth_shell [data-testid="stAlert"],
        .st-key-ha_auth_shell [data-baseweb="notification"] {
            width: 100% !important;
            max-width: 420px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border-radius: 14px !important;
        }
        .st-key-ha_auth_shell [data-testid="column"]:nth-of-type(2) [data-testid="stVerticalBlock"] [data-testid="stAlert"] {
            margin-top: 0.45rem !important;
            margin-bottom: 0.35rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-ft-strong {
            color: var(--ha-lux-moss);
            font-weight: 600;
        }
        .ha-auth-hero-foot {
            font-size: 0.85rem;
            color: #a8ceb8;
        }
        .ha-auth-wrap {
            max-width: 100%;
            margin: 0;
        }
        .ha-auth-card {
            display: contents;
        }
        section[data-testid="stSidebar"],
        [data-testid="stSidebar"] {
            position: relative;
            z-index: 0;
            overflow-x: hidden;
            overflow-y: auto;
            align-self: stretch !important;
            height: auto !important;
            min-height: 100% !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"],
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            background: transparent !important;
        }
        [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) [data-testid="stSidebar"] {
            background: var(--ha-sidebar-bg) !important;
        }
        [data-testid="stSidebar"] > * {
            position: relative;
            z-index: 1;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            background: transparent !important;
            background-image: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0.5rem !important;
            background: transparent !important;
            background-image: none !important;
        }
        /* Guest sidebar: tek kolon flex, scroll yok; login altta sabit. */
        section[data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer),
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer) {
            height: 100dvh !important;
            max-height: 100dvh !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer) [data-testid="stSidebarHeader"] {
            flex: 0 0 auto !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: 1.65rem !important;
            padding: 0 0.45rem 0 !important;
            margin: 0 !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer) [data-testid="stSidebarContent"],
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer) [data-testid="stSidebarUserContent"] {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
            padding-top: 0 !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer) .block-container {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
            padding-top: 0.2rem !important;
            padding-bottom: 0 !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
            background: transparent !important;
            background-image: none !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer) .ha-sidebar-flex-gap,
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.ha-sidebar-flex-gap),
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stMarkdownContainer"]:has(.ha-sidebar-flex-gap) {
            flex: 0 0 0.65rem !important;
            min-height: 0.65rem !important;
            max-height: 0.65rem !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_login_footer),
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="element-container"]:has(.st-key-ha_sidebar_login_footer) {
            margin-top: 0.85rem !important;
            flex-shrink: 0 !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            .block-container > [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
            min-height: 0 !important;
            overflow: hidden !important;
            gap: 0.35rem !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            .block-container > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:has(.ha-sidebar-header) {
            flex-shrink: 0 !important;
            overflow: visible !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            .st-key-ha_sidebar_guest_main {
            flex: 0 0 auto !important;
            margin-bottom: 1.5rem !important;
            padding-bottom: 0.35rem !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            .st-key-ha_sidebar_guest_main [data-testid="stSelectbox"] {
            margin-bottom: 0.65rem !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stMarkdownContainer"]:has(.ha-sidebar-login-push),
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.ha-sidebar-login-push),
        .ha-sidebar-login-push {
            display: none !important;
            flex: 0 0 0 !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_login_footer),
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-ha_sidebar_login_footer) {
            flex-shrink: 0 !important;
            margin-top: 0 !important;
            width: 100% !important;
            overflow: visible !important;
        }
        .st-key-ha_sidebar_login_footer [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.85rem !important;
            align-items: stretch !important;
        }
        .st-key-ha_sidebar_login_footer [data-testid="stElementContainer"] {
            margin: 0 !important;
            padding: 0 !important;
            flex-shrink: 0 !important;
            position: relative !important;
        }
        .st-key-ha_sidebar_login_btn {
            margin-top: 0.15rem !important;
            padding-top: 0 !important;
        }
        .st-key-ha_sidebar_login_footer {
            position: relative !important;
            z-index: 12 !important;
            margin-top: 0 !important;
            flex-shrink: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 0.65rem !important;
            width: calc(100% + 1.7rem) !important;
            max-width: none !important;
            margin-left: -0.85rem !important;
            margin-right: -0.85rem !important;
            box-sizing: border-box !important;
            padding: 1.15rem 0.85rem 1rem !important;
            margin-bottom: 0 !important;
            overflow: visible !important;
            background: linear-gradient(
                180deg,
                rgba(242, 235, 227, 0.72) 0%,
                var(--ha-sidebar-bg) 28%,
                var(--ha-sidebar-bg) 100%
            ) !important;
            border-top: 1px solid rgba(44, 48, 42, 0.1) !important;
            border-radius: 18px 18px 0 0 !important;
            box-shadow: 0 -6px 24px rgba(38, 42, 36, 0.07) !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            .st-key-ha_sidebar_login_btn {
            flex-shrink: 0 !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            overflow: visible !important;
            min-height: 2.75rem !important;
        }
        /* Login kartı: başlık + açıklama (sade, kutusuz) */
        .ha-sidebar-login-card {
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .ha-sidebar-login-title {
            margin: 0 0 0.35rem 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.84rem !important;
            font-weight: 650 !important;
            line-height: 1.3 !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        /* Açıklama metni: küçük, açık gri, sade (kutusuz) */
        .ha-sidebar-login-hint {
            margin: 0 0 0.15rem 0 !important;
            padding: 0 0 0.35rem 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 400 !important;
            line-height: 1.45 !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
        }
        /* Style the login button inside the row */
        .st-key-ha_sidebar_login_row button {
            background: var(--ha-primary) !important;
            color: var(--ha-bg) !important;
            -webkit-text-fill-color: var(--ha-bg) !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }
        .st-key-ha_sidebar_login_row button:hover {
            opacity: 0.9 !important;
        }
        /* Premium header: nav label + user card */
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) {
            margin-top: 0 !important;
            margin-bottom: 0.75rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) {
            margin-top: 0.2rem !important;
            margin-bottom: 0.85rem !important;
            flex-shrink: 0 !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: relative;
            z-index: 2;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) p {
            margin: 0 !important;
            color: inherit !important;
        }
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="stElementContainer"]:has(.ha-sidebar-header),
        [data-testid="stSidebar"]:has(.st-key-ha_sidebar_login_footer)
            [data-testid="element-container"]:has(.ha-sidebar-header) {
            flex-shrink: 0 !important;
            overflow: visible !important;
        }
        .ha-sidebar-header {
            margin: 0.35rem 0 1rem 0;
            flex-shrink: 0 !important;
            overflow: visible !important;
        }
        .ha-sidebar-header__eyebrow {
            display: flex !important;
            align-items: center;
            gap: 0.45rem;
            visibility: visible !important;
            opacity: 1 !important;
            min-height: 1.1rem !important;
            flex-shrink: 0 !important;
            margin: 0 0 0.65rem 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.62rem !important;
            font-weight: 650 !important;
            letter-spacing: 0.2em !important;
            text-transform: uppercase !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
        }
        .ha-sidebar-header__eyebrow--tr {
            text-transform: none !important;
            letter-spacing: 0.14em !important;
            font-size: 0.68rem !important;
        }
        .ha-sidebar-header__eyebrow::before {
            content: "";
            flex-shrink: 0;
            width: 1.1rem;
            height: 2px;
            border-radius: 2px;
            background: linear-gradient(90deg, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0.05));
        }
        .ha-sidebar-header__user-card {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.62rem 0.75rem;
            background: var(--ha-surface);
            border: 1px solid var(--ha-border);
            border-radius: var(--ha-sidebar-radius);
            box-shadow: var(--ha-card-shadow);
        }
        .ha-sidebar-header__avatar {
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 12px;
            flex-shrink: 0;
            display: flex !important;
            align-items: center;
            justify-content: center;
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background: linear-gradient(145deg, var(--ha-sage) 0%, var(--ha-sage-deep) 92%);
            box-shadow:
                0 2px 8px rgba(60, 78, 58, 0.18),
                inset 0 1px 0 rgba(255, 255, 255, 0.25);
        }
        .ha-sidebar-header__user-meta {
            min-width: 0;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.12rem;
        }
        .ha-sidebar-header__hint {
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.62rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
        }
        .ha-sidebar-header__name {
            font-family: "Inter", system-ui, sans-serif !important;
            font-size: 0.94rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.25 !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            word-break: break-word;
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] {
            border: 1px solid var(--ha-card-edge) !important;
            border-radius: var(--ha-sidebar-radius) !important;
            overflow: hidden;
            background: var(--ha-surface) !important;
            margin-bottom: 0.85rem !important;
            box-shadow: var(--ha-card-shadow);
        }
        [data-testid="stSidebar"] div[data-testid="stExpander"] details {
            border: none !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 1.15rem 0 0.75rem 0 !important;
            border: none !important;
            border-top: 1px solid var(--ha-border) !important;
        }
        /* Sidebar butonları: ChatGPT tarzı minimal — kutu/border yok,
           sola hizalı icon + label, sadece hover'da yumuşak yuvarlatılmış
           pill arkaplan. Bu kural Login CTA butonu HARİÇ tüm sidebar
           butonlarına uygulanır (Login kendi style'ını aşağıda override eder). */
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:not(.st-key-ha_sidebar_login_btn) button[kind="primary"],
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:not(.st-key-ha_sidebar_login_btn) button[kind="secondary"],
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:not(.st-key-ha_sidebar_login_btn) [data-testid="stButton"] button {
            background: transparent !important;
            background-image: none !important;
            border: 1px solid transparent !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            border-radius: 12px !important;
            min-height: 2.1rem !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            padding: 0.45rem 0.7rem !important;
            box-shadow: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
            transition:
                background-color 0.22s ease,
                border-color 0.22s ease,
                box-shadow 0.22s ease,
                transform 0.18s ease !important;
        }
        [data-testid="stSidebar"] button[kind="primary"] p,
        [data-testid="stSidebar"] button[kind="primary"] span,
        [data-testid="stSidebar"] button[kind="secondary"] p,
        [data-testid="stSidebar"] button[kind="secondary"] span {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            font-weight: 500 !important;
        }
        /* Material icon görünümü */
        [data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stIconMaterial"],
        [data-testid="stSidebar"] [data-testid="stButton"] button .material-symbols-rounded {
            color: var(--ha-text-soft) !important;
            margin-right: 0.45rem !important;
            font-size: 1.05rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:not(.st-key-ha_sidebar_login_btn) button[kind="primary"]:hover,
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:not(.st-key-ha_sidebar_login_btn) button[kind="secondary"]:hover,
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:not(.st-key-ha_sidebar_login_btn) [data-testid="stButton"] button:hover {
            background: var(--ha-sage-mist) !important;
            filter: none !important;
            border-color: rgba(122, 145, 112, 0.2) !important;
            box-shadow: 0 2px 12px rgba(60, 78, 58, 0.08) !important;
            transform: translateY(-1px);
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover [data-testid="stIconMaterial"] {
            color: var(--ha-text) !important;
        }

        [data-testid="stSidebar"] .ha-sidebar-title {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--ha-text-soft);
            margin: 0.45rem 0 0.1rem 0;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            opacity: 0.72;
        }
        [data-testid="stSidebar"] .ha-sidebar-subtitle {
            font-size: 0.66rem;
            color: var(--ha-text-soft);
            opacity: 0.65;
            margin-bottom: 0.18rem;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .st-key-ha_sidebar_chat_panel div[role="radiogroup"] label {
            padding: 0.3rem 0.48rem !important;
            min-height: 0 !important;
            border-radius: 8px !important;
            border: 1px solid rgba(44, 48, 42, 0.05) !important;
            background: rgba(255, 255, 255, 0.2) !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] .st-key-ha_sidebar_chat_panel div[role="radiogroup"] label p {
            font-size: 0.76rem !important;
            font-weight: 450 !important;
            text-align: left !important;
            color: #8a847c !important;
        }
        [data-testid="stSidebar"] .ha-sidebar-selected {
            background: rgba(0, 0, 0, 0.04);
            border: 1px solid transparent;
            border-radius: 9px;
            padding: 0.38rem 0.52rem;
            margin-bottom: 0.32rem;
            font-size: 0.86rem;
            font-weight: 520;
            color: var(--ha-text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: none;
        }
        /* Chat history radios only — section nav uses premium ha_nav_* CSS */
        /* ====== ChatGPT tarzı sabit chat düzeni ======
           - body/html: 100vh, overflow hidden (sayfa scroll kapalı)
           - stApp / stAppViewContainer: Streamlit'in kendi flex/height hesabına
             güveniriz; sadece overflow'u kapatırız (yoksa bizim verdiğimiz 100vh
             header'ın altında taşmaya yol açıyordu).
           - Tek scrollable alan: stMainBlockContainer (mesajlar)
           - Composer: position: fixed; viewport'a göre konumlanır, sidebar açıkken
             ana içerik alanına hizalı gözükür. */
        html:has(.st-key-ha_chat_composer_row),
        body:has(.st-key-ha_chat_composer_row) {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }
        [data-testid="stApp"]:has(.st-key-ha_chat_composer_row),
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) {
            overflow: hidden !important;
        }
        /* stMain: taşma kesilir, içerik scroll alanı içinde kalsın. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMain"] {
            overflow: hidden !important;
            position: relative !important;
            padding-top: 0 !important;
        }
        /* stMainBlockContainer: TEK scroll alanı (mesajlar). Composer'ın altta
           kapladığı alan kadar bottom padding ekleriz, son mesaj gizlenmesin. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMainBlockContainer"] {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-top: 0.75rem !important;
            padding-left: clamp(0.75rem, 3vw, 1.5rem) !important;
            padding-right: clamp(0.75rem, 3vw, 1.5rem) !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
            padding-bottom: calc(8.75rem + env(safe-area-inset-bottom, 0px)) !important;
            scroll-behavior: smooth !important;
            position: relative !important;
            z-index: 1 !important;
            background: transparent !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            box-sizing: border-box !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) .ha-chat-welcome-line,
        section.main:has(.st-key-ha_chat_composer_row) .ha-section-subtitle,
        section.main:has(.st-key-ha_chat_composer_row) .ha-section-title.ha-chat-page-title {
            background: #f2f0ed !important;
            border: none !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            padding: 0.55rem 0.85rem !important;
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) .block-container,
        .main .block-container:has(.st-key-ha_chat_composer_row) {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row)
            [data-testid="stVerticalBlock"]:first-of-type
            > [data-testid="stElementContainer"]:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) .ha-chat-welcome-line {
            margin-top: 0 !important;
            margin-bottom: 0.2rem !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) .ha-section-subtitle {
            margin-top: 0 !important;
            margin-bottom: 0.4rem !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) .ha-section-title.ha-chat-page-title {
            margin-top: 0 !important;
            margin-bottom: 0.4rem !important;
            padding-bottom: 0.35rem !important;
        }
        /* Guest empty state: centered hero (max 760px) */
        [data-testid="stMain"]:has(.st-key-ha_guest_empty_shell) [data-testid="stMainBlockContainer"] {
            min-height: calc(100dvh - 3.25rem) !important;
            padding-top: clamp(3.5rem, 11vh, 6.5rem) !important;
            padding-bottom: clamp(1.5rem, 4vh, 2.75rem) !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            position: relative !important;
        }
        /* Logged-in user empty: welcome flush to top (no logo gap) */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_user_empty_shell) [data-testid="stMainBlockContainer"],
        [data-testid="stMain"]:has(.st-key-ha_user_empty_shell) [data-testid="stMainBlockContainer"] {
            min-height: calc(100dvh - 3.25rem) !important;
            padding-top: 0 !important;
            padding-bottom: clamp(1.5rem, 4vh, 2.75rem) !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            position: relative !important;
        }
        [data-testid="stMain"]:has(.st-key-ha_user_empty_shell)
            [data-testid="stVerticalBlock"]:first-of-type
            > [data-testid="stElementContainer"]:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        [data-testid="stMain"]:has(.st-key-ha_guest_empty_shell) [data-testid="stMainBlockContainer"] > *,
        [data-testid="stMain"]:has(.st-key-ha_user_empty_shell) [data-testid="stMainBlockContainer"] > * {
            position: relative !important;
            z-index: 1 !important;
        }
        .st-key-ha_guest_empty_shell,
        .st-key-ha_user_empty_shell {
            width: 100% !important;
            max-width: 760px !important;
            margin: 0 auto !important;
            padding: 0 1.25rem !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            justify-content: flex-start !important;
            gap: 0.5rem !important;
        }
        .st-key-ha_user_empty_shell {
            gap: 0.85rem !important;
        }
        .st-key-ha_guest_empty_shell [data-testid="stVerticalBlock"] {
            gap: 0.4rem !important;
            width: 100% !important;
        }
        .st-key-ha_user_empty_shell [data-testid="stVerticalBlock"] {
            gap: 1.1rem !important;
            width: 100% !important;
        }
        /* Logged-in empty chat hero (reference mockup) */
        .ha-chat-user-empty__hero {
            margin: 0 0 0.25rem 0 !important;
            padding: 0 !important;
            width: 100% !important;
            text-align: center !important;
        }
        .ha-chat-user-empty__title {
            margin: 0 0 0.55rem 0 !important;
            padding-top: 0 !important;
            font-family: "Lora", Georgia, serif !important;
            font-size: clamp(1.65rem, 3vw, 2.2rem) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.18 !important;
            color: #3e4e35 !important;
            -webkit-text-fill-color: #3e4e35 !important;
        }
        .ha-chat-user-empty__leaf-deco {
            margin: 0 0 0.65rem 0 !important;
            font-size: 0.82rem !important;
            line-height: 1 !important;
            color: #7a9170 !important;
            opacity: 0.9 !important;
        }
        .ha-chat-user-empty__tagline {
            margin: 0 0 0.5rem 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: clamp(0.95rem, 1.5vw, 1.08rem) !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            color: #3d4540 !important;
            -webkit-text-fill-color: #3d4540 !important;
        }
        .ha-chat-user-empty__lead {
            margin: 0 auto 0 auto !important;
            max-width: 34rem !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: clamp(0.86rem, 1.4vw, 0.96rem) !important;
            font-weight: 400 !important;
            line-height: 1.45 !important;
            color: #5a6654 !important;
            -webkit-text-fill-color: #5a6654 !important;
        }
        .ha-chat-empty__suggested-label {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.65rem !important;
            width: 100% !important;
            margin: 0.75rem 0 0.45rem 0 !important;
            padding: 0 !important;
        }
        .st-key-ha_user_empty_shell .ha-chat-empty__suggested-label {
            margin: 0.25rem 0 0.65rem 0 !important;
        }
        .ha-chat-empty__suggested-rule {
            flex: 1 1 auto !important;
            max-width: 5.5rem !important;
            height: 1px !important;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(122, 145, 112, 0.35) 50%,
                transparent 100%
            ) !important;
        }
        .ha-chat-empty__suggested-text {
            flex: 0 0 auto !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
            color: #5a6654 !important;
            -webkit-text-fill-color: #5a6654 !important;
            white-space: nowrap !important;
        }
        .ha-chat-empty__suggested-leaf {
            margin: 0 0.2rem !important;
            font-size: 0.72rem !important;
            opacity: 0.85 !important;
        }
        .ha-chat-guest-empty__hero {
            margin: 0 0 0.35rem 0 !important;
            padding: 0 !important;
            width: 100% !important;
            text-align: center !important;
        }
        .ha-chat-guest-empty__logo {
            margin: 0 auto 0.55rem auto !important;
            max-width: min(100%, 11rem) !important;
            width: 100% !important;
        }
        .ha-chat-guest-empty__logo img {
            display: block !important;
            width: 100% !important;
            max-width: 11rem !important;
            height: auto !important;
            margin: 0 auto !important;
            object-fit: contain !important;
            object-position: center center !important;
        }
        .ha-chat-guest-empty__eyebrow {
            margin: 0 0 0.45rem 0 !important;
            font-size: 1rem !important;
            line-height: 1 !important;
            color: #7a9170 !important;
            opacity: 0.85 !important;
        }
        .ha-chat-guest-empty__leaf {
            margin-right: 0.2rem !important;
        }
        .ha-chat-guest-empty__sparkle {
            margin: 0 0.12rem !important;
            color: #c9b46a !important;
            font-size: 0.72rem !important;
            vertical-align: middle !important;
        }
        .ha-chat-guest-empty__title {
            margin: 0 !important;
            font-family: "Lora", Georgia, serif !important;
            font-size: clamp(1.75rem, 3.2vw, 2.35rem) !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            line-height: 1.15 !important;
            color: #3e4e35 !important;
            -webkit-text-fill-color: #3e4e35 !important;
        }
        .ha-chat-guest-empty__subtitle {
            margin: 0.55rem 0 0 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: clamp(0.92rem, 1.6vw, 1.05rem) !important;
            font-weight: 400 !important;
            line-height: 1.45 !important;
            color: #5a6654 !important;
            -webkit-text-fill-color: #5a6654 !important;
            max-width: 36rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .ha-chat-guest-empty__footer {
            margin: 1.25rem auto 0 auto !important;
            padding: 0 !important;
            text-align: center !important;
            font-size: 0.78rem !important;
            line-height: 1.35 !important;
            color: #8a9288 !important;
            -webkit-text-fill-color: #8a9288 !important;
        }
        .ha-chat-guest-empty__footer-icon {
            margin-right: 0.35rem !important;
            opacity: 0.75 !important;
        }
        .st-key-ha_guest_empty_shell .st-key-ha_chat_composer_row,
        [data-testid="stMain"]:has(.st-key-ha_guest_empty_shell) .st-key-ha_chat_composer_row {
            position: relative !important;
            bottom: auto !important;
            left: auto !important;
            right: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 1.1rem 0 0 0 !important;
            padding: 0 !important;
            z-index: 1 !important;
        }
        .st-key-ha_user_empty_shell .st-key-ha_chat_composer_row,
        [data-testid="stMain"]:has(.st-key-ha_user_empty_shell) .st-key-ha_chat_composer_row {
            position: relative !important;
            bottom: auto !important;
            left: auto !important;
            right: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            margin: 1.35rem 0 0 0 !important;
            padding: 0 !important;
            z-index: 1 !important;
        }
        .st-key-ha_guest_empty_shell .st-key-ha_chat_composer_row [data-testid="stChatInput"] > div,
        .st-key-ha_user_empty_shell .st-key-ha_chat_composer_row [data-testid="stChatInput"] > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .st-key-ha_guest_empty_shell [data-testid="stElementContainer"]:has(.ha-chat-guest-empty__hero),
        .st-key-ha_guest_empty_shell [data-testid="stElementContainer"]:has(.ha-chat-empty__suggested-label),
        .st-key-ha_guest_empty_shell [data-testid="stElementContainer"]:has(.st-key-ha_chat_composer_row),
        .st-key-ha_guest_empty_shell [data-testid="stElementContainer"]:has(.st-key-ha_guest_suggested_cards),
        .st-key-ha_user_empty_shell [data-testid="stElementContainer"]:has(.ha-chat-user-empty__hero),
        .st-key-ha_user_empty_shell [data-testid="stElementContainer"]:has(.ha-chat-empty__suggested-label),
        .st-key-ha_user_empty_shell [data-testid="stElementContainer"]:has(.st-key-ha_chat_composer_row),
        .st-key-ha_user_empty_shell [data-testid="stElementContainer"]:has(.st-key-ha_user_suggested_cards) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        .st-key-ha_guest_suggested_cards,
        .st-key-ha_user_suggested_cards {
            width: 100% !important;
            margin-top: 0.35rem !important;
            margin-bottom: 0 !important;
        }
        .st-key-ha_user_empty_shell .st-key-ha_user_suggested_cards {
            margin-top: 0.5rem !important;
            margin-bottom: 0.25rem !important;
        }
        .st-key-ha_guest_suggested_cards [data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        .st-key-ha_user_suggested_cards [data-testid="stVerticalBlock"] {
            gap: 1rem !important;
        }
        .st-key-ha_guest_suggested_cards [data-testid="stHorizontalBlock"] {
            gap: 0.8rem !important;
        }
        .st-key-ha_user_suggested_cards [data-testid="stHorizontalBlock"] {
            gap: 1rem !important;
        }
        .st-key-ha_guest_suggested_cards button[data-testid^="baseButton"],
        .st-key-ha_user_suggested_cards button[data-testid^="baseButton"] {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important;
            min-height: 5.75rem !important;
            height: auto !important;
            padding: 0.95rem 1.1rem !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            line-height: 1.35 !important;
            text-align: left !important;
            white-space: normal !important;
            border-radius: 14px !important;
            border: 1px solid rgba(0, 0, 0, 0.06) !important;
            background: rgba(255, 255, 255, 0.94) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
            max-width: none !important;
            width: 100% !important;
        }
        .st-key-ha_guest_suggested_cards button[data-testid^="baseButton"]:hover,
        .st-key-ha_user_suggested_cards button[data-testid^="baseButton"]:hover {
            background: #ffffff !important;
            border-color: rgba(107, 138, 90, 0.18) !important;
            box-shadow: 0 2px 5px rgba(16, 24, 40, 0.06) !important;
        }
        .st-key-ha_guest_suggested_cards button [data-testid="stIconMaterial"],
        .st-key-ha_guest_suggested_cards button .material-symbols-rounded,
        .st-key-ha_user_suggested_cards button [data-testid="stIconMaterial"],
        .st-key-ha_user_suggested_cards button .material-symbols-rounded {
            color: #6b8a5a !important;
            font-size: 1.2rem !important;
            margin-right: 0.55rem !important;
            margin-left: 0 !important;
            flex-shrink: 0 !important;
            align-self: center !important;
        }
        .st-key-ha_guest_suggested_cards button p,
        .st-key-ha_guest_suggested_cards button span,
        .st-key-ha_user_suggested_cards button p,
        .st-key-ha_user_suggested_cards button span {
            text-align: left !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            flex: 1 1 auto !important;
        }
        .st-key-ha_guest_suggested_cards button[data-testid^="baseButton"]::after,
        .st-key-ha_user_suggested_cards button[data-testid^="baseButton"]::after {
            content: "›" !important;
            flex: 0 0 auto !important;
            margin-left: 0.35rem !important;
            font-size: 1.1rem !important;
            font-weight: 400 !important;
            color: #b8bfb4 !important;
            -webkit-text-fill-color: #b8bfb4 !important;
            align-self: center !important;
        }
        .st-key-ha_guest_suggested_cards [data-testid="element-container"]:has(button),
        .st-key-ha_user_suggested_cards [data-testid="element-container"]:has(button) {
            width: 100% !important;
            max-width: 100% !important;
        }
        /* Guest hero: tighter top spacing and gap before "New Chat" */
        section.main:has(.ha-chat-guest-hero) [data-testid="stVerticalBlock"]:first-of-type {
            gap: 0.2rem !important;
        }
        section.main:has(.ha-chat-guest-hero)
            [data-testid="stElementContainer"]:has(.ha-chat-guest-hero) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        section.main:has(.ha-chat-guest-hero) .ha-chat-guest-hero {
            margin-top: 0 !important;
        }
        section.main:has(.ha-chat-guest-hero) .ha-section-title.ha-chat-page-title {
            margin-top: 0 !important;
            margin-bottom: 0.32rem !important;
            padding-bottom: 0.28rem !important;
        }
        /* İçerideki vertical block'ların yüksekliği serbest kalsın. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMain"] [data-testid="stVerticalBlock"] {
            overflow: visible !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"],
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"] {
            margin-bottom: 0.3rem !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) [class*="st-key-ha_assistant_actions_row"],
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [class*="st-key-ha_assistant_actions_row"] {
            margin-bottom: 0.05rem !important;
        }
        /* Agent thinking + welcome empty slot'ları composer'ı yukarı itmesin */
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(
            + [data-testid="stElementContainer"] .st-key-ha_chat_composer_row
        ),
        [data-testid="stMain"]:has(.st-key-ha_chat_composer_row) [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(
            + [data-testid="stElementContainer"]:has(
                + [data-testid="stElementContainer"] .st-key-ha_chat_composer_row
            )
        ) {
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
        }
        /* Chat composer: HER ZAMAN ekranın altında sabit (viewport bottom).
           Sidebar açıkken ana içerik alanına hizalanması için aşağıdaki JS,
           stMain'in soluna ve genişliğine göre --ha-main-left ve --ha-main-width
           CSS değişkenlerini günceller. Değişkenler yoksa viewport tamamına yayılır. */
        .st-key-ha_chat_composer_row {
            position: fixed !important;
            bottom: 1rem !important;
            top: auto !important;
            left: var(--ha-main-left, 0) !important;
            right: auto !important;
            width: var(--ha-main-width, 100vw) !important;
            max-width: none !important;
            z-index: 999 !important;
            margin: 0 !important;
            padding: 0.5rem 0 0.35rem 0 !important;
            background: linear-gradient(
                180deg,
                rgba(242, 235, 227, 0) 0%,
                rgba(242, 235, 227, 0.9) 38%,
                #F2EBE3 100%
            ) !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            pointer-events: none; /* boş kenarlar tıklanabilir olmasın */
        }
        /* Composer'ın iç sütun bloğunu ortalayıp kart görünümü ona veriyoruz.
           Streamlit ara sarmalayıcıları (stVerticalBlockBorderWrapper,
           stVerticalBlock, stElementContainer) varsayılan olarak şeffaf bırakılır. */
        .st-key-ha_chat_composer_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_chat_composer_row > div,
        .st-key-ha_chat_composer_row [data-testid="stVerticalBlock"],
        .st-key-ha_chat_composer_row [data-testid="stElementContainer"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        /* THE UNIFIED PILL CONTAINER */
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
            pointer-events: auto;
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: max(0.55rem, env(safe-area-inset-bottom, 0px)) !important;
            padding: 0.38rem 0.55rem !important;
            border-radius: 999px !important;
            background: rgba(255, 253, 248, 0.94) !important;
            border: 1px solid rgba(44, 48, 42, 0.1) !important;
            box-shadow:
                0 2px 12px rgba(38, 42, 36, 0.06),
                inset 0 1px 0 rgba(255, 255, 255, 0.72) !important;
            backdrop-filter: blur(12px) saturate(1.04) !important;
            -webkit-backdrop-filter: blur(12px) saturate(1.04) !important;
        }
        @media (max-width: 720px) {
            .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
                margin-left: 0.5rem !important;
                margin-right: 0.5rem !important;
            }
        }
        
        /* THE GEAR BUTTON (LEFT) */
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            flex: 0 0 3rem !important;
            width: 3rem !important;
            min-width: 3rem !important;
            max-width: 3rem !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button {
            min-height: 2.35rem !important;
            width: 100% !important;
            max-width: none !important;
            padding: 0.15rem !important;
            border-radius: 12px !important;
            border: 1px solid rgba(44, 48, 42, 0.09) !important;
            background: rgba(255, 255, 255, 0.42) !important;
            color: #6a6258 !important;
            -webkit-text-fill-color: #6a6258 !important;
            box-shadow: 0 1px 5px rgba(38, 42, 36, 0.04) !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button:hover {
            background: rgba(255, 253, 248, 0.78) !important;
            border-color: rgba(44, 48, 42, 0.12) !important;
            color: #4a4540 !important;
            -webkit-text-fill-color: #4a4540 !important;
        }
        /* Hide the figure space and chevron */
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button [data-testid="stMarkdownContainer"],
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button svg:not([data-testid="stIconMaterial"]) {
            display: none !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
            margin: 0 !important;
            font-size: 1.4rem !important;
        }

        /* THE CHAT INPUT (RIGHT) */
        /* Make chat input outer wrapper transparent */
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] > div {
            border-radius: 999px !important;
            border: 1px solid rgba(44, 48, 42, 0.07) !important;
            background: rgba(255, 255, 255, 0.38) !important;
            box-shadow: inset 0 1px 2px rgba(38, 42, 36, 0.03) !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] textarea,
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] input {
            color: #3d3834 !important;
            -webkit-text-fill-color: #3d3834 !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] textarea::placeholder,
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] input::placeholder {
            color: #8a8278 !important;
            -webkit-text-fill-color: #8a8278 !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] button,
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] button[kind="secondary"],
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] button[kind="primary"] {
            background: rgba(255, 255, 255, 0.5) !important;
            border: 1px solid rgba(44, 48, 42, 0.08) !important;
            color: #5c6f5e !important;
            -webkit-text-fill-color: #5c6f5e !important;
            box-shadow: 0 1px 4px rgba(38, 42, 36, 0.05) !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] button:hover {
            background: rgba(255, 253, 248, 0.92) !important;
            border-color: rgba(44, 48, 42, 0.12) !important;
            color: #3e4e35 !important;
            -webkit-text-fill-color: #3e4e35 !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"]:focus-within {
            border-color: rgba(44, 48, 42, 0.14) !important;
            box-shadow:
                0 2px 14px rgba(38, 42, 36, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        }
        /* Konuşma kutuları: kullanıcı bej, asistan beyaz kart (mockup). */
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"] {
            border-radius: 16px !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.85rem !important;
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border: none !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }
        /* Kullanıcı mesajı: düz bej kutu */
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ),
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"].st-emotion-cache-user,
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) {
            background: #f2f0ed !important;
            box-shadow: none !important;
        }
        /* Asistan cevabı: beyaz kart + hafif gölge */
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stChatMessage"]:not(:has(
            [data-testid="stChatMessageAvatarUser"]
        )):not(:has([data-testid="chatAvatarIcon-user"])):not(.st-emotion-cache-user) {
            background: #ffffff !important;
            box-shadow:
                0 4px 18px rgba(40, 55, 45, 0.07),
                0 1px 4px rgba(16, 24, 40, 0.04) !important;
        }
        /* Agent thinking / status */
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stStatus"],
        section.main:has(.st-key-ha_chat_composer_row) [data-testid="stStatusWidget"],
        section.main:has(.st-key-ha_chat_composer_row) div[data-testid="stStatus"] {
            background: #ffffff !important;
            border: none !important;
            border-radius: 16px !important;
            box-shadow:
                0 4px 18px rgba(40, 55, 45, 0.07),
                0 1px 4px rgba(16, 24, 40, 0.04) !important;
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        section.main:has(.st-key-ha_chat_composer_row) [class*="st-key-ha_assistant_actions_row"] {
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 0.28rem 0.55rem !important;
            border-radius: 11px !important;
            background: rgba(255, 255, 255, 0.42) !important;
            border: 1px solid rgba(92, 111, 94, 0.09) !important;
            box-shadow: 0 1px 4px rgba(60, 78, 58, 0.04) !important;
        }
        section.main:has(.st-key-ha_chat_composer_row)
            [class*="st-key-ha_assistant_actions_row"]
            [data-testid="stVerticalBlockBorderWrapper"],
        section.main:has(.st-key-ha_chat_composer_row)
            [class*="st-key-ha_assistant_actions_row"]
            > div[data-testid="stVerticalBlock"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        /* Misafir / giriş yapmış boş ekran: tam sayfa dekor görünsün */
        [data-testid="stMain"]:has(.st-key-ha_guest_empty_shell) [data-testid="stMainBlockContainer"],
        [data-testid="stMain"]:has(.st-key-ha_user_empty_shell) [data-testid="stMainBlockContainer"] {
            background: transparent !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
        }
        /* Admin: modern control panel (scoped to ha_admin_panel / ha_admin_feedback) */
        .st-key-ha_admin_panel .ha-admin-hero,
        .st-key-ha_admin_feedback .ha-admin-feedback-head {
            margin-bottom: 0.35rem;
        }
        .ha-admin-hero__title {
            font-size: 1.42rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #2a3028;
            margin: 0 0 0.35rem 0;
            line-height: 1.2;
        }
        .ha-admin-hero__sub {
            font-size: 0.92rem;
            color: #5a6456;
            margin: 0;
            line-height: 1.45;
        }
        .ha-admin-metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.35rem 0 1.1rem 0;
        }
        .ha-admin-metric-row--tight {
            margin-top: 0.5rem;
            margin-bottom: 0.85rem;
        }
        @media (max-width: 900px) {
            .ha-admin-metric-row {
                grid-template-columns: 1fr;
            }
        }
        .ha-admin-metric-card {
            background: #ffffff;
            border: 1px solid var(--ha-card-edge);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow: var(--ha-card-shadow);
        }
        .ha-admin-metric-card__label {
            font-size: 0.68rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6a7562;
            margin-bottom: 0.35rem;
        }
        .ha-admin-metric-card__value {
            font-size: 1.28rem;
            font-weight: 700;
            color: #2d352b;
            line-height: 1.25;
            word-break: break-word;
        }
        .ha-admin-metric-card__value--sm {
            font-size: 0.95rem;
            font-weight: 600;
        }
        .ha-admin-section-h {
            font-size: 0.85rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #5a6456;
            margin: 0.25rem 0 0.65rem 0;
        }
        .st-key-ha_admin_panel .ha-admin-op-title {
            font-size: 0.95rem;
            font-weight: 650;
            color: #2d352b;
            margin: 0 0 0.5rem 0;
        }
        .st-key-ha_admin_panel [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            background: #ffffff !important;
            box-shadow: var(--ha-card-shadow) !important;
            border: 1px solid var(--ha-card-edge) !important;
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-head__title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #2a3028;
            margin: 0 0 0.2rem 0;
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-head__sub {
            font-size: 0.88rem;
            color: #5a6456;
            margin: 0 0 0.75rem 0;
            line-height: 1.45;
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-toolbar {
            background: #ffffff;
            border: 1px solid var(--ha-card-edge);
            border-radius: 12px;
            padding: 0.55rem 0.75rem 0.65rem 0.75rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--ha-card-shadow);
        }
        .st-key-ha_admin_feedback .ha-admin-feedback-toolbar__label {
            font-size: 0.72rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6a7562;
            margin: 0 0 0.35rem 0;
        }
        .st-key-ha_admin_feedback .ha-admin-glance-label {
            font-size: 0.72rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6a7562;
            margin: 0.15rem 0 0.45rem 0;
        }
        .st-key-ha_admin_feedback div[data-testid="stExpander"] {
            margin-bottom: 0.55rem !important;
        }
        .st-key-ha_admin_feedback [data-testid="stExpander"] summary {
            font-weight: 500 !important;
            font-size: 0.88rem !important;
        }
        .ha-admin-entry-meta {
            font-size: 0.78rem;
            color: #6a7562;
            margin: 0 0 0.5rem 0;
        }
        /* Main: expanders (e.g. admin, profile blocks) as distinct sections */
        section.main div[data-testid="stExpander"] {
            border: 1px solid var(--ha-card-edge) !important;
            border-radius: 12px !important;
            background: var(--ha-surface) !important;
            box-shadow: var(--ha-card-shadow) !important;
            margin-bottom: 0.85rem !important;
        }
        section.main [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        section.main [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: var(--ha-chat-ink) !important;
            line-height: 1.55 !important;
        }
        /* Suggested prompts: ChatGPT-vari sade, hafif kenarlı chip butonlar */
        .st-key-ha_suggested_prompts {
            margin-top: 0.5rem !important;
            margin-bottom: 0.15rem !important;
            max-width: 48rem;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .st-key-ha_suggested_prompts [data-testid="stHorizontalBlock"] {
            gap: 0.55rem !important;
        }
        .st-key-ha_suggested_prompts button[data-testid^="baseButton"] {
            min-height: 2.25rem !important;
            font-size: 0.85rem !important;
            font-weight: 450 !important;
            line-height: 1.4 !important;
            border-radius: 12px !important;
            border: 1px solid var(--ha-border) !important;
            background: #ffffff !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            box-shadow: none !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .st-key-ha_suggested_prompts button[data-testid^="baseButton"]:hover {
            background: #f7f7f8 !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
            color: var(--ha-text) !important;
        }
        /* Assistant action toolbar: copy, sources, feedback */
        [class*="st-key-ha_assistant_actions_row"] [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            flex-wrap: nowrap !important;
            gap: 0.35rem !important;
        }
        [class*="st-key-ha_assistant_actions_row"] [data-testid="stElementContainer"],
        [class*="st-key-ha_assistant_actions_row"] [data-testid="element-container"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        [class*="st-key-ha_assistant_copy_cell"] {
            overflow: visible !important;
        }
        [class*="st-key-ha_assistant_copy_cell"] iframe {
            height: 36px !important;
            min-height: 36px !important;
            max-height: 36px !important;
            width: 44px !important;
            min-width: 44px !important;
            max-width: 44px !important;
            border: none !important;
            overflow: visible !important;
        }
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stPopover"] button,
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stPopover"] > button {
            min-height: 1.85rem !important;
            padding: 0.28rem 0.65rem !important;
            font-size: 0.76rem !important;
            font-weight: 550 !important;
            line-height: 1.25 !important;
            white-space: nowrap !important;
            border-radius: 10px !important;
            border: 1px solid rgba(92, 111, 94, 0.18) !important;
            background: rgba(255, 255, 255, 0.82) !important;
            color: #4f6352 !important;
            -webkit-text-fill-color: #4f6352 !important;
            box-shadow: 0 1px 3px rgba(60, 78, 58, 0.05) !important;
        }
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stPopover"] button:hover {
            background: #ffffff !important;
            border-color: rgba(92, 111, 94, 0.28) !important;
            box-shadow: 0 2px 8px rgba(60, 78, 58, 0.08) !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] {
            display: flex !important;
            justify-content: flex-end !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stHorizontalBlock"] {
            gap: 0.28rem !important;
            align-items: center !important;
            width: auto !important;
            margin-left: auto !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 0 0 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stButton"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"] {
            min-height: 1.85rem !important;
            min-width: 1.85rem !important;
            width: 1.85rem !important;
            max-width: 1.85rem !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 10px !important;
            border: 1px solid rgba(92, 111, 94, 0.16) !important;
            background: rgba(255, 255, 255, 0.78) !important;
            box-shadow: 0 1px 3px rgba(60, 78, 58, 0.04) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[kind="primary"][data-testid^="baseButton"] {
            background: rgba(92, 111, 94, 0.14) !important;
            border-color: rgba(92, 111, 94, 0.32) !important;
            box-shadow: 0 1px 6px rgba(60, 78, 58, 0.1) !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"]:hover {
            background: #ffffff !important;
            border-color: rgba(92, 111, 94, 0.26) !important;
            box-shadow: 0 2px 8px rgba(60, 78, 58, 0.08) !important;
            transform: none !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"] [data-testid="stIconMaterial"],
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"] .material-symbols-rounded {
            font-size: 0.95rem !important;
            width: 0.95rem !important;
            height: 0.95rem !important;
            color: #6d8270 !important;
            margin: 0 !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[kind="primary"][data-testid^="baseButton"] [data-testid="stIconMaterial"],
        [class*="st-key-ha_assistant_feedback_group"] button[kind="primary"][data-testid^="baseButton"] .material-symbols-rounded {
            color: #4f6352 !important;
            font-variation-settings: "FILL" 1, "wght" 500, "GRAD" 0, "opsz" 20 !important;
        }
        /* Profile: full-width settings control */
        .st-key-ha_profile_adv {
            margin-bottom: 0.35rem;
        }
        .st-key-ha_profile_adv [data-testid="stPopover"] button {
            border-radius: 10px !important;
            border: 1px solid rgba(72, 92, 78, 0.2) !important;
            background: #ffffff !important;
            font-weight: 550 !important;
            color: #3a4534 !important;
            -webkit-text-fill-color: #3a4534 !important;
            box-shadow: 0 1px 2px rgba(40, 55, 45, 0.05) !important;
        }
        .st-key-ha_profile_adv [data-testid="stPopover"] button:hover {
            background: #f9fbf9 !important;
            border-color: rgba(72, 92, 78, 0.28) !important;
        }
        .ha-auth-title {
            text-align: center;
            font-size: 1.7rem;
            margin-bottom: 0.2rem;
            font-weight: 700;
            color: var(--ha-text);
        }
        .ha-auth-subtitle {
            text-align: center;
            margin-bottom: 1rem;
            color: var(--ha-text);
            opacity: 0.9;
        }
        .ha-auth-switch {
            margin-top: 0.45rem;
            text-align: center;
            font-size: 0.86rem;
            color: var(--ha-text);
        }
        .ha-auth-switch strong {
            color: #5a7d6e;
            font-weight: 600;
        }
        .ha-auth-right-head {
            text-align: center;
            font-size: 1.9rem;
            font-weight: 700;
            color: var(--ha-text);
            margin-top: 0.1rem;
            margin-bottom: 0.65rem;
        }
        [data-testid="stRadio"] label p {
            font-size: 1.05rem !important;
        }
        /* Make labels visible against white card bg */
        label, [data-testid="stMarkdownContainer"] p {
            color: var(--ha-text) !important;
        }
        /* Login primary: Streamlit puts label in stMarkdownContainer; global rule above must not win. */
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] span,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"]:has(button[data-testid="baseButton-primary"]) [data-testid="stMarkdownContainer"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"]:has(button[kind="primary"]) [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"],
        [data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 12px !important;
            border: 1px solid var(--ha-border) !important;
            background: var(--ha-bg-2) !important;
            min-height: 56px;
            transition: all 0.15s ease;
            overflow: hidden;
            color: var(--ha-text) !important;
        }
        /* Login kartı: global form input stili auth alanlarını ezmesin */
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"],
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 10px !important;
            min-height: 52px !important;
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(74, 93, 69, 0.22) !important;
        }
        /* Profile card: mockup — light grey fields, sage save (after global form rules) */
        .st-key-ha_profile_card [data-testid="stForm"] div[data-baseweb="input"],
        .st-key-ha_profile_card [data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 10px !important;
            min-height: 2.35rem !important;
            background: #f3f4f2 !important;
            border: 1px solid #dce1d8 !important;
            box-shadow: none !important;
        }
        .st-key-ha_profile_card [data-testid="stTextInput"] input {
            min-height: 2.35rem !important;
            padding: 0.45rem 0.65rem !important;
            font-size: 0.875rem !important;
        }
        .st-key-ha_profile_card [data-testid="stForm"] div[data-baseweb="input"]:focus-within,
        .st-key-ha_profile_card [data-testid="stForm"] div[data-baseweb="select"]:focus-within {
            border-color: rgba(122, 145, 112, 0.45) !important;
            box-shadow: var(--ha-input-glow) !important;
        }
        .st-key-ha_profile_card [data-testid="stTextArea"] textarea {
            border-radius: 10px !important;
            border: 1px solid #dce1d8 !important;
            background: #f3f4f2 !important;
            min-height: 4.75rem !important;
            max-height: 9rem !important;
            padding: 0.5rem 0.65rem !important;
            font-size: 0.875rem !important;
            line-height: 1.4 !important;
        }
        .st-key-ha_profile_card [data-testid="stTextArea"] textarea:focus {
            border-color: rgba(122, 145, 112, 0.45) !important;
            box-shadow: var(--ha-input-glow) !important;
            outline: none !important;
        }
        .st-key-ha_profile_card .st-key-ha_profile_save_row
            [data-testid="element-container"]:has([data-testid="stFormSubmitButton"]),
        .st-key-ha_profile_card .st-key-ha_profile_save_row
            [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) {
            width: 100% !important;
            max-width: 100% !important;
            display: flex !important;
            justify-content: flex-end !important;
        }
        [data-testid="stForm"] input,
        [data-testid="stForm"] textarea,
        [data-testid="stForm"] [contenteditable="true"] {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            caret-color: var(--ha-text) !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"]:focus-within,
        [data-testid="stForm"] div[data-baseweb="select"]:focus-within {
            border-color: rgba(109, 139, 124, 0.55) !important;
            box-shadow: 0 0 0 2px rgba(109, 139, 124, 0.12) !important;
        }
        [data-testid="stForm"] div[data-baseweb="input"] > div,
        [data-testid="stForm"] div[data-baseweb="select"] > div {
            background-color: transparent !important;
            border: none !important;
        }
        /* Forms (logged-in): primary = nötr beyaz buton sade çerçevesi */
        [data-testid="stForm"] button[kind="primary"] {
            border-radius: 10px !important;
            min-height: 2.2rem !important;
            font-size: 0.85rem !important;
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            transition: transform 0.05s ease, filter 0.15s ease;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] {
            border-radius: 10px !important;
            min-height: 2.35rem !important;
            font-size: 0.85rem !important;
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"] p,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        [data-testid="stFormSubmitButton"] button[kind="primary"] span,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span {
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover {
            background: #f5f5f5 !important;
            filter: none !important;
            border-color: rgba(0, 0, 0, 0.12) !important;
        }
        /* Login kartı: global submit stillerinden sonra — beyaz yazı, yeşil buton */
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background-color: #5c6f5e !important;
            background-image: none !important;
            border: 1px solid #4d5f50 !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] *,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] *,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"] span,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] p,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] span {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background-color: #5c6f5e !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        /* Profile card: sage Save Profile (after global form submit rules) */
        .st-key-ha_profile_card .st-key-ha_profile_save_row
            [data-testid="element-container"]:has([data-testid="stFormSubmitButton"]),
        .st-key-ha_profile_card .st-key-ha_profile_save_row
            [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) {
            width: 100% !important;
            max-width: 100% !important;
            display: flex !important;
            justify-content: flex-end !important;
        }
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[kind="primary"],
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] {
            background: linear-gradient(180deg, #5f7359 0%, var(--ha-sage-deep) 100%) !important;
            background-image: none !important;
            border: 1px solid rgba(62, 78, 53, 0.2) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            min-width: 10.5rem !important;
            min-height: 2.35rem !important;
            padding: 0.55rem 1.15rem !important;
            font-size: 0.875rem !important;
            border-radius: 10px !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            box-shadow: 0 4px 14px rgba(60, 78, 58, 0.18) !important;
        }
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[kind="primary"] p,
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[kind="primary"] span,
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] span,
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"] p,
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"] span {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
        .st-key-ha_profile_card [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover {
            background: linear-gradient(180deg, #5f7359 0%, var(--ha-sage-deep) 100%) !important;
            border-color: rgba(62, 78, 53, 0.28) !important;
            box-shadow: 0 6px 18px rgba(60, 78, 58, 0.22) !important;
            filter: none !important;
        }
        button[kind="primary"] {
            background: #ffffff !important;
            background-image: none !important;
            border: 1px solid var(--ha-border) !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        a {
            color: #2563eb !important;
        }
        .ha-forgot-link > div > button {
            background: transparent !important;
            border: none !important;
            color: var(--ha-text-soft) !important;
            padding: 0 !important;
            min-height: auto !important;
            font-size: 0.9rem !important;
            justify-content: flex-start !important;
            box-shadow: none !important;
        }
        .ha-forgot-link > div > button:hover {
            color: var(--ha-text) !important;
            text-decoration: underline !important;
        }
        [data-testid="stForm"] button[kind="primary"]:hover {
            background: #f9fbf9 !important;
            filter: none !important;
        }
        [data-testid="stForm"] button[kind="primary"]:active {
            transform: translateY(1px);
        }
        @media (max-width: 900px) {
            .main .block-container:has(.st-key-ha_auth_shell) {
                min-height: auto;
                justify-content: center;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
            html:has(.st-key-ha_auth_shell),
            html:has(.st-key-ha_auth_shell) body {
                overflow: hidden !important;
            }
            [data-testid="stAppViewContainer"]:has(.st-key-ha_auth_shell) [data-testid="stMainBlockContainer"] {
                justify-content: center !important;
                align-items: center !important;
                height: calc(100dvh - 3.5rem) !important;
                max-height: calc(100dvh - 3.5rem) !important;
                min-height: 0 !important;
                overflow: hidden !important;
                padding-top: 0.35rem !important;
                padding-bottom: clamp(1.75rem, 6vh, 3.5rem) !important;
            }
            .st-key-ha_auth_shell {
                transform: translateY(clamp(-1.75rem, -3vh, -0.35rem)) !important;
            }
            .st-key-ha_auth_shell {
                --ha-lux-card-h: auto;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1),
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(2) {
                flex: 1 1 auto !important;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1) {
                border-right: none !important;
                border-bottom: 1px solid rgba(74, 93, 69, 0.2) !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                min-height: 240px;
                padding: 1.75rem 1.35rem 1.25rem 1.35rem;
                justify-content: center;
                align-items: center;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__inner {
                align-items: center;
                text-align: center;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1,
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center;
            }
            .ha-lux-botanical {
                padding-top: 1.25rem;
            }
            .st-key-ha_auth_shell .st-key-ha_auth_card {
                height: auto !important;
                min-height: min(37.5rem, 68vh) !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                min-height: min(32rem, 58vh) !important;
                height: auto !important;
                padding: 1.05rem 1.5rem 2.5rem 1.5rem !important;
            }
            .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {
                padding: 0.5rem 0.4rem 1.1rem 0.4rem !important;
            }
            .st-key-ha_auth_shell .st-key-ha_auth_lang_header,
            .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stVerticalBlock"] {
                justify-content: flex-end !important;
            }
        }
        .ha-section-title {
            margin-top: 0.05rem;
            margin-bottom: 0.15rem;
            font-size: 1.18rem;
            font-weight: 650;
            color: var(--ha-chat-ink);
            letter-spacing: -0.02em;
        }
        /* Chat page: space between conversation title and composer (gear + input) */
        .ha-section-title.ha-chat-page-title {
            margin-top: 0.05rem !important;
            margin-bottom: 0.55rem !important;
            padding-bottom: 0.5rem !important;
            line-height: 1.25 !important;
            border-bottom: 1px solid rgba(72, 92, 78, 0.11) !important;
        }
        .ha-chat-welcome-line {
            margin: 0 0 0.2rem 0;
            font-size: 1.08rem;
            font-weight: 650;
            color: #3d4540 !important;
            letter-spacing: -0.01em;
            line-height: 1.3;
        }
        /* Section #### headings in main (incl. suggested questions): calmer scale */
        section.main [data-testid="stMarkdownContainer"] h4 {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #4d544d !important;
            margin: 0.45rem 0 0.32rem 0 !important;
            letter-spacing: 0.02em;
        }
        .ha-section-subtitle {
            margin-bottom: 1rem;
            color: var(--text-color);
            opacity: 0.68;
            font-size: 0.95rem;
        }
        #MainMenu, [data-testid="stToolbarActions"] {
            display: none !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] p.ha-lux-footer {
            color: #6a7168 !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-form-title,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-form-sub {
            color: inherit;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover *,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"]:hover * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        """
        + _GUEST_AUTH_MOCKUP_CSS
        + _guest_auth_card_background_css()
        + """
        </style>
        """,
        in_sidebar=in_sidebar,
    )
    _inject_herbal_shell_background(in_sidebar=in_sidebar)
    _inject_premium_herbal_ui(in_sidebar=in_sidebar)
    _inject_html(
        f"<style>{_profile_page_label_override_css()}</style>",
        in_sidebar=in_sidebar,
    )
    
    if BOTANICAL_MAIN_B64 or BOTANICAL_SIDEBAR_B64:
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("{BOTANICAL_MAIN_B64}") !important;
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-blend-mode: multiply;
            }}
            .ha-sidebar-header__user-card {{
                background-image: url("{BOTANICAL_SIDEBAR_B64}") !important;
                background-size: cover !important;
                background-position: center center !important;
                background-repeat: no-repeat !important;
                border-radius: 12px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def _inject_chat_layout_script(*, in_sidebar: bool = False) -> None:
    """Sidebar açıkken/kapalıyken composer'ın stMain ile hizalı kalması için
    iki CSS değişkenini günceller: --ha-main-left, --ha-main-width.

    Composer 'position: fixed' kullanır; CSS değişkenleri sayesinde stMain'in
    gerçek bounding rect'ine göre konumlanır.

    --ha-sidebar-left/width değişkenleri sidebar ölçümü için kullanılır."""
    script_html = (
        """
        <script>
          (function () {
            try {
              const doc = window.parent && window.parent.document
                ? window.parent.document
                : document;
              if (!doc) return;
              const root = doc.documentElement;
              function update() {
                try {
                  const main = doc.querySelector('[data-testid="stMain"]');
                  if (main) {
                    const rect = main.getBoundingClientRect();
                    root.style.setProperty('--ha-main-left', rect.left + 'px');
                    root.style.setProperty('--ha-main-width', rect.width + 'px');
                  }
                  const sidebar = doc.querySelector(
                    'section[data-testid="stSidebar"], [data-testid="stSidebar"]'
                  );
                  if (sidebar) {
                    const srect = sidebar.getBoundingClientRect();
                    root.style.setProperty('--ha-sidebar-left', srect.left + 'px');
                    root.style.setProperty('--ha-sidebar-width', srect.width + 'px');
                  }
                } catch (e) {}
              }

              update();
              const win = window.parent || window;
              win.addEventListener('resize', update, { passive: true });

              // Sidebar / main boyut değişimlerini izle
              try {
                const targets = [
                  doc.querySelector('[data-testid="stMain"]'),
                  doc.querySelector('section[data-testid="stSidebar"]'),
                  doc.querySelector('[data-testid="stSidebar"]'),
                  doc.querySelector('[data-testid="stAppViewContainer"]'),
                ].filter(Boolean);
                if (window.ResizeObserver && targets.length) {
                  const ro = new ResizeObserver(update);
                  targets.forEach((t) => ro.observe(t));
                }
              } catch (e) {}

              // Streamlit re-render sonrası DOM yenilenmesi için emniyet kemeri
              setInterval(update, 600);
            } catch (e) {}
          })();
        </script>
        """
    )
    if in_sidebar:
        with st.sidebar:
            components.html(script_html, height=0, width=0)
    else:
        components.html(script_html, height=0, width=0)


def _force_sidebar_collapsed_on_load() -> None:
    """
    Streamlit persists sidebar open/closed state in browser localStorage.
    This forces the sidebar to start collapsed again on refresh/re-run.
    """
    components.html(
        """
        <script>
          (function () {
            try {
              const flagKey = "ha_sidebar_collapsed_reset";
              if (window.sessionStorage && window.sessionStorage.getItem(flagKey) === "1") {
                return;
              }
              if (window.sessionStorage) {
                window.sessionStorage.setItem(flagKey, "1");
              }
              for (const k of Object.keys(window.localStorage || {})) {
                if (k.toLowerCase().includes("sidebar")) {
                  window.localStorage.removeItem(k);
                }
              }
              window.location.reload();
            } catch (e) {
              // If anything fails, fall back to normal Streamlit behavior.
            }
          })();
        </script>
        """,
        height=0,
    )
