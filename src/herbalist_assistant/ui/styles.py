"""CSS injection and layout scripts for the Streamlit UI.

Contains all ``st.markdown`` style blocks and ``components.html`` scripts
that control the visual presentation of the application.
"""

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Login hero: bundled logo (replaces "Welcome" heading when present).
_AUTH_LOGO_PATH = Path(__file__).resolve().parent / "static" / "herbalist_logo.png"
# Login background: soft botanical scene shown only on guest Login page.
_AUTH_BG_PATH = Path(__file__).resolve().parent / "static" / "login_background.png"

@st.cache_data(show_spinner=False)
def _auth_hero_logo_data_uri() -> str:
    if not _AUTH_LOGO_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_LOGO_PATH.read_bytes()
    ).decode("ascii")


@st.cache_data(show_spinner=False)
def _auth_background_data_uri() -> str:
    if not _AUTH_BG_PATH.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(
        _AUTH_BG_PATH.read_bytes()
    ).decode("ascii")

def _inject_guest_login_fullbleed_styles() -> None:
    """Guest Login only: remove sidebar column and expand main to full width (Streamlit 1.38+)."""
    bg_uri = _auth_background_data_uri()
    if bg_uri:
        st.markdown(
            f"""
            <style>
            /* Login kutusunun arkaplanı: botanik desen + üzerinde hafif beyaz overlay
               (form okunabilirliği için) — sayfa arkaplanı sade gri/beyaz kalır. */
            .st-key-ha_auth_card {{
                background-color: #ffffff !important;
                background-image:
                    linear-gradient(rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.82)),
                    url("{bg_uri}") !important;
                background-size: cover, cover !important;
                background-position: center, center !important;
                background-repeat: no-repeat, no-repeat !important;
                border: 1px solid rgba(0, 0, 0, 0.06) !important;
                border-radius: 18px !important;
                box-shadow: 0 12px 40px rgba(40, 55, 45, 0.10) !important;
            }}
            /* Kart içindeki iç bloklar şeffaf ki desen görünsün */
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"],
            .st-key-ha_auth_card [data-testid="column"] > [data-testid="stVerticalBlock"],
            .st-key-ha_auth_card .ha-lux-welcome,
            .st-key-ha_auth_card .ha-lux-welcome__inner,
            .st-key-ha_auth_form_card,
            .st-key-ha_auth_form_card [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {{
                background: transparent !important;
            }}
            /* Karta bindirilen ince halka deseni iptal — sade sadece botanik görünsün. */
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before,
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::after {{
                content: none !important;
                display: none !important;
                background: none !important;
                background-image: none !important;
            }}
            /* Sol sütundaki dekoratif sarmaşık/yaprak süslemeleri de kapat */
            .st-key-ha_auth_card .ha-lux-welcome__vine,
            .st-key-ha_auth_card .ha-lux-welcome__ghost-leaf,
            .st-key-ha_auth_card .ha-lux-welcome__rule,
            .st-key-ha_auth_card .ha-lux-botanical,
            .st-key-ha_auth_card .ha-lux-botanical__accent,
            .st-key-ha_auth_card .ha-lux-botanical__photo,
            .st-key-ha_auth_card .ha-lux-pot {{
                display: none !important;
                background: none !important;
                background-image: none !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <style>
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
        /* Guest login: normal document scroll (tall form / small viewports) */
        html, body {
            overflow-y: auto !important;
            overflow-x: hidden !important;
            min-height: 100dvh !important;
        }
        [data-testid="stApp"] {
            min-height: 100dvh !important;
            overflow: visible !important;
        }
        [data-testid="stAppViewContainer"] {
            min-height: calc(100dvh - 3.5rem) !important;
            overflow: visible !important;
            align-items: stretch !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stMain"],
        [data-testid="stAppViewContainer"] section.main {
            overflow: visible !important;
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            display: flex !important;
            flex-direction: column !important;
        }
        [data-testid="stMainBlockContainer"] {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: center !important;
            padding: 0.5rem 0 1.5rem 0 !important;
        }
        section.main .block-container {
            flex: 0 0 auto !important;
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            box-sizing: border-box !important;
            width: 100% !important;
        }
        .main .block-container:has(.st-key-ha_auth_shell) {
            min-height: 0 !important;
            height: auto !important;
            max-height: none !important;
            padding-top: 0 !important;
            padding-bottom: 0.5rem !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: center !important;
        }
        .st-key-ha_auth_shell {
            width: 100% !important;
            max-width: 1020px !important;
            flex: 0 0 auto !important;
            margin: 0 auto !important;
            overflow: visible !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            justify-content: flex-start !important;
            gap: 0.2rem !important;
        }
        .st-key-ha_auth_shell [data-testid="element-container"]:has(button[data-testid="baseButton-tertiary"]) {
            margin-bottom: 0 !important;
        }
        /* "Misafir olarak devam et" butonu: login kartının üstünde sağa
           hizalı, belirgin pill stilinde — kullanıcı login olmadan misafir
           Chat'e dönebilsin diye her zaman görünür kalır. SADECE bu key
           hedefleniyor; Forgot Password tertiary butonu etkilenmiyor. */
        .st-key-ha_auth_back_chat {
            display: flex !important;
            justify-content: flex-end !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.1rem !important;
        }
        .st-key-ha_auth_back_chat button {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(0, 0, 0, 0.10) !important;
            border-radius: 999px !important;
            padding: 0.4rem 0.95rem !important;
            min-height: 2.1rem !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: var(--ha-text, #1f2937) !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06) !important;
            text-decoration: none !important;
        }
        .st-key-ha_auth_back_chat button:hover {
            background: #ffffff !important;
            border-color: rgba(0, 0, 0, 0.22) !important;
            color: var(--ha-text, #1f2937) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_card,
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            flex: 0 1 auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome {
            min-height: 0 !important;
            max-height: none !important;
            padding: 1.2rem 1.25rem 1rem 1.25rem !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome h1 {
            font-size: clamp(2.05rem, 3.7vw, 2.75rem) !important;
            margin: 0 0 0.5rem 0 !important;
            text-align: center !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo {
            margin: 2.75rem auto !important;
            max-width: min(100%, 560px) !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__logo img {
            display: block !important;
            width: 100% !important;
            max-width: 560px !important;
            height: auto !important;
            margin: 0 auto !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__rule {
            margin: 0 auto 0.6rem auto !important;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__lead {
            font-size: 1.12rem !important;
            line-height: 1.58 !important;
            text-align: center !important;
            max-width: 40ch !important;
            margin-left: auto !important;
            margin-right: auto !important;
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
        .st-key-ha_auth_shell .ha-lux-welcome h1 {
            text-align: center !important;
        }
        [data-testid="stMarkdown"] .ha-lux-welcome h1,
        [data-testid="stMarkdown"] .ha-lux-welcome p.ha-lux-welcome__lead,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1,
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            text-align: center !important;
        }
        .st-key-ha_auth_shell .ha-lux-botanical {
            display: none !important;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
            padding: 2.5rem 2.5rem 3rem 2.5rem !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
            justify-content: flex-start !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-title {
            font-size: 1.35rem !important;
            margin-top: 0 !important;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub {
            font-size: 0.82rem !important;
            margin-bottom: 0.35rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-testid="element-container"] {
            margin-bottom: 0.4rem !important;
        }
        @media (min-width: 901px) {
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(1) > div[data-testid="stVerticalBlock"] {
                height: 100% !important;
                min-height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                flex: 0 1 auto !important;
                min-height: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__inner {
                align-items: center !important;
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome:has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
                min-height: min(42vh, 24rem) !important;
                justify-content: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome h1 {
                text-align: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__rule {
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome__lead {
                text-align: center !important;
            }
        }
        @media (max-width: 900px) {
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1)
                > div[data-testid="stVerticalBlock"] {
                justify-content: center !important;
                align-items: center !important;
            }
            .st-key-ha_auth_shell .ha-lux-welcome {
                min-height: 0 !important;
                max-height: none !important;
                justify-content: center !important;
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
            .st-key-ha_auth_shell .ha-lux-welcome__rule {
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                min-height: 0 !important;
                max-height: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Inter:wght@400;500;600&display=swap");
        :root {
            /* Soft, minimalist (ChatGPT-vari) palette: nötr beyaz/gri tonlar */
            --ha-bg: #ffffff;
            --ha-bg-2: #f7f7f8;
            --ha-sidebar-bg: #f9f9f9;
            --ha-text: #1f1f1f;
            --ha-text-soft: #6b7280;
            --ha-border: rgba(0, 0, 0, 0.08);
            --ha-surface: #ffffff;
            --ha-chat-ink: #1a1a1a;
            /* Çok hafif sıcak vurgu (herbal kimliğe minik bir gönderme) */
            --ha-accent-soft: #5b6e63;
            --ha-shell-1: #ffffff;
            --ha-shell-2: #ffffff;
            --ha-shell-3: #ffffff;
            --ha-shell-4: #ffffff;
            --ha-card-edge: rgba(0, 0, 0, 0.06);
            --ha-card-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
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
            padding-top: 0.35rem !important;
            padding-bottom: 1.35rem !important;
            position: relative;
            z-index: 1;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--ha-bg) !important;
        }
        html,
        body,
        #root {
            background: var(--ha-bg) !important;
        }
        [data-testid="stApp"] {
            background: var(--ha-bg) !important;
        }
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: transparent !important;
        }
        footer,
        [data-testid="stFooter"] {
            background: var(--ha-bg) !important;
        }
        section.main {
            position: relative;
            z-index: 0;
            background: transparent !important;
        }
        /* Sade görünüm: arka plan dokusu (film noise) tamamen kapatıldı */
        [data-testid="stAppViewContainer"] section.main::before {
            content: none !important;
            display: none !important;
        }
        .main .block-container:has(.st-key-ha_auth_shell) {
            max-width: 1120px;
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            min-height: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .st-key-ha_auth_shell {
            --ha-lux-ink: #4B5940;
            --ha-lux-moss: #708260;
            --ha-lux-cream: #F7F8F3;
            /* soft minimum height only; form can grow and the page scrolls */
            --ha-lux-card-h: min(32rem, 60vh);
            width: 100%;
            max-width: 1020px;
            margin: 0 auto;
            font-family: "Inter", system-ui, -apple-system, sans-serif;
            color: var(--ha-lux-ink);
        }
        .st-key-ha_auth_shell .st-key-ha_auth_card {
            width: 100%;
        }
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
            position: relative;
            isolation: isolate;
            align-items: stretch !important;
            gap: 0 !important;
            background: #fbfbf9 !important;
            border-radius: 22px !important;
            overflow: hidden;
            box-shadow:
                0 24px 48px rgba(75, 89, 64, 0.1),
                0 4px 14px rgba(75, 89, 64, 0.06) !important;
            border: 1px solid rgba(112, 130, 96, 0.2) !important;
        }
        /* İnce halka deseni: tüm kart (sol + sağ) üstünde, sütun renkleri üzerine rgba ile bırakılır */
        .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before {
            content: "";
            position: absolute;
            inset: 0;
            z-index: 0;
            border-radius: 22px;
            pointer-events: none;
            background-image:
                radial-gradient(
                    circle 200px at 6% 4%,
                    transparent 0%,
                    transparent 86%,
                    rgba(100, 112, 98, 0.1) 86%,
                    rgba(100, 112, 98, 0.1) 90.5%,
                    transparent 90.5%
                ),
                radial-gradient(
                    circle 240px at 96% 2%,
                    transparent 0%,
                    transparent 87%,
                    rgba(100, 112, 98, 0.08) 87%,
                    rgba(100, 112, 98, 0.08) 91%,
                    transparent 91%
                ),
                radial-gradient(
                    circle 300px at 50% 36%,
                    transparent 0%,
                    transparent 89%,
                    rgba(100, 112, 98, 0.065) 89%,
                    rgba(100, 112, 98, 0.065) 92.5%,
                    transparent 92.5%
                ),
                radial-gradient(
                    circle 180px at 2% 48%,
                    transparent 0%,
                    transparent 84%,
                    rgba(100, 112, 98, 0.07) 84%,
                    rgba(100, 112, 98, 0.07) 88%,
                    transparent 88%
                ),
                radial-gradient(
                    circle 220px at 100% 42%,
                    transparent 0%,
                    transparent 86%,
                    rgba(100, 112, 98, 0.075) 86%,
                    rgba(100, 112, 98, 0.075) 90%,
                    transparent 90%
                ),
                radial-gradient(
                    circle 160px at 20% 72%,
                    transparent 0%,
                    transparent 83%,
                    rgba(100, 112, 98, 0.06) 83%,
                    rgba(100, 112, 98, 0.06) 87%,
                    transparent 87%
                ),
                radial-gradient(
                    circle 280px at 80% 78%,
                    transparent 0%,
                    transparent 90%,
                    rgba(100, 112, 98, 0.055) 90%,
                    rgba(100, 112, 98, 0.055) 93.5%,
                    transparent 93.5%
                ),
                radial-gradient(
                    circle 130px at 12% 92%,
                    transparent 0%,
                    transparent 80%,
                    rgba(100, 112, 98, 0.065) 80%,
                    rgba(100, 112, 98, 0.065) 85%,
                    transparent 85%
                ),
                radial-gradient(
                    circle 200px at 55% 100%,
                    transparent 0%,
                    transparent 86%,
                    rgba(100, 112, 98, 0.05) 86%,
                    rgba(100, 112, 98, 0.05) 90%,
                    transparent 90%
                ) !important;
            background-repeat: no-repeat;
            background-size: 100% 100%;
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
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2) {
            flex: 1.1 1 0% !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(1)
            > div[data-testid="stVerticalBlock"] {
            position: relative;
            padding: 0 !important;
            margin: 0 !important;
            /* Yarı saydam: kart genelindeki halka deseni sol panelde de görünsün */
            background-color: rgba(240, 242, 237, 0.78) !important;
            background-image:
                radial-gradient(ellipse 125% 90% at 88% 8%, rgba(180, 195, 172, 0.34) 0%, transparent 58%),
                radial-gradient(ellipse 100% 80% at -5% 78%, rgba(200, 208, 194, 0.3) 0%, transparent 55%),
                radial-gradient(ellipse 70% 55% at 40% 95%, rgba(186, 198, 178, 0.18) 0%, transparent 50%),
                linear-gradient(
                    168deg,
                    rgba(243, 244, 241, 0.92) 0%,
                    rgba(240, 242, 237, 0.88) 40%,
                    rgba(230, 232, 226, 0.9) 100%
                ) !important;
            background-repeat: no-repeat !important;
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
            justify-content: center !important;
            align-items: center !important;
        }
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            > div[data-testid="stVerticalBlock"] {
            position: relative;
            /* Açık zemin: ince halkalar üst seviyede (stHorizontalBlock::before) */
            background: rgba(251, 251, 249, 0.82) !important;
            border: none !important;
            padding: 2.5rem 2.5rem 3rem 2.5rem !important;
            min-height: var(--ha-lux-card-h);
            box-sizing: border-box;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            align-items: stretch !important;
        }
        /* Form / üst bar iç sarmalayıcıları şeffaf: yuvarlak desen tüm kutu alanında görünsün */
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            [data-testid="stVerticalBlock"]
            [data-testid="stVerticalBlock"] {
            background: transparent !important;
        }
        .st-key-ha_auth_card
            [data-testid="stHorizontalBlock"]
            > [data-testid="column"]:nth-of-type(2)
            [data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
        }
        /* —— Welcome column (markdown); gradient is on the column stVerticalBlock —— */
        .st-key-ha_auth_shell .ha-lux-welcome {
            position: relative;
            min-height: var(--ha-lux-card-h);
            box-sizing: border-box;
            padding: 2.2rem 2.05rem 2rem 2.05rem;
            display: flex;
            flex-direction: column;
            background: transparent;
            flex: 1 1 auto;
            width: 100%;
            overflow: hidden;
            justify-content: center;
            align-items: center;
        }
        @media (min-width: 901px) {
        .st-key-ha_auth_shell .ha-lux-welcome {
            min-height: 0;
            flex: 0 1 auto;
            align-self: center;
        }
        .st-key-ha_auth_shell .ha-lux-welcome:has(.ha-lux-welcome__logo) .ha-lux-welcome__inner {
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
        .st-key-ha_auth_shell .ha-lux-welcome__logo {
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
        .st-key-ha_auth_shell .ha-lux-welcome h1 {
            font-family: "Lora", Georgia, "Times New Roman", serif;
            font-size: clamp(2.1rem, 3.5vw, 2.8rem);
            font-weight: 700;
            color: var(--ha-lux-ink);
            margin: 0 0 0.9rem 0;
            line-height: 1.18;
            letter-spacing: -0.025em;
            text-align: center;
            width: 100%;
        }
        .st-key-ha_auth_shell .ha-lux-welcome__rule {
            width: 52px;
            height: 3px;
            border-radius: 2px;
            margin: 0 auto 1.05rem auto;
            background: linear-gradient(90deg, var(--ha-lux-moss), rgba(112, 130, 96, 0.35));
        }
        .st-key-ha_auth_shell .ha-lux-welcome__lead {
            margin: 0;
            max-width: 40ch;
            font-size: 1.12rem;
            line-height: 1.65;
            color: #5a6652;
            font-weight: 400;
            text-align: center;
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
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome h1 {
            font-size: clamp(2.1rem, 3.5vw, 2.8rem) !important;
        }
        .st-key-ha_auth_shell [data-testid="stMarkdownContainer"] .ha-lux-welcome p.ha-lux-welcome__lead {
            font-size: 1.12rem !important;
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
        /* —— Auth language switcher (compact, top-right) —— */
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header {
            width: 100%;
            margin: 0 0 0.1rem 0;
            padding: 0;
            display: flex !important;
            justify-content: flex-end !important;
            align-items: flex-start !important;
            opacity: 0.92;
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
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stElementContainer"] {
            flex: 0 0 auto !important;
            width: auto !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] {
            width: auto !important;
            max-width: 100% !important;
            margin: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] [data-baseweb="button-group"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] {
            gap: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: rgba(247, 248, 243, 0.65) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(75, 89, 64, 0.07) !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"] {
            border-radius: 999px !important;
            min-height: 20px !important;
            min-width: 0 !important;
            padding: 0.1rem 0.4rem !important;
            font-size: 0.62rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.01em !important;
            line-height: 1.1 !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
            background-image: none !important;
            color: #4b5940 !important;
            transition:
                background-color 0.16s ease,
                color 0.16s ease,
                box-shadow 0.16s ease !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"]:hover,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"]:hover {
            background: rgba(75, 89, 64, 0.06) !important;
            color: #4a5640 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"],
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] {
            background: rgba(75, 89, 64, 0.2) !important;
            background-image: none !important;
            color: #3d4a35 !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"]:hover,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"]:hover {
            background: rgba(75, 89, 64, 0.26) !important;
            color: #323c2c !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="false"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="false"] span,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="false"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="false"] span {
            color: #6a7562 !important;
            -webkit-text-fill-color: #6a7562 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] span,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] p,
        .st-key-ha_auth_shell .st-key-ha_auth_lang_header [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] span {
            color: #3d4a35 !important;
            -webkit-text-fill-color: #3d4a35 !important;
            font-weight: 600 !important;
        }
        /* Top: EN/TR sağda, hemen altında Login | Create Account (mockup ile aynı sıra) */
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar {
            width: 100%;
            margin: 0 0 0.15rem 0;
            display: flex !important;
            flex-direction: column !important;
            align-items: stretch !important;
            gap: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_lang_header {
            align-self: flex-end !important;
            width: auto !important;
            max-width: 100% !important;
            margin: 0 0 0.4rem 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row {
            width: 100%;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_top_bar .st-key-ha_auth_tab_row div[role="radiogroup"] {
            justify-content: center !important;
            margin: 0 0 1.05rem 0 !important;
        }
        /* Login / Create Account tabs — scoped so language control is not styled as a pill track */
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] {
            justify-content: center !important;
            gap: 0.5rem !important;
            margin: 0.55rem 0 1.05rem 0 !important;
            width: 100%;
            padding: 0.2rem !important;
            background: rgba(247, 248, 243, 0.85) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(112, 130, 96, 0.14) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label {
            border-radius: 999px !important;
            padding: 0.34rem 1.2rem !important;
            font-size: 0.84rem !important;
            font-weight: 500 !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            margin: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label p {
            font-size: 0.84rem !important;
            text-align: center !important;
            margin: 0 !important;
            color: var(--ha-lux-ink) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) {
            background: var(--ha-lux-ink) !important;
            border-color: var(--ha-lux-ink) !important;
            box-shadow: 0 2px 8px rgba(75, 89, 64, 0.18) !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_tab_row div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        /* —— Form stack —— */
        .st-key-ha_auth_shell .ha-lux-form-title {
            font-family: "Lora", Georgia, serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--ha-lux-ink);
            text-align: center;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .st-key-ha_auth_shell .ha-lux-form-sub {
            text-align: center;
            font-size: 0.92rem;
            line-height: 1.5;
            color: #5f6b56;
            margin: 0 auto 1.35rem auto;
            max-width: 38ch;
        }
        .st-key-ha_auth_form_card [data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"],
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="select"] {
            border-radius: 14px !important;
            min-height: 56px !important;
            background: rgba(247, 248, 243, 0.95) !important;
            border: 1px solid rgba(112, 130, 96, 0.2) !important;
            box-shadow: none !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] [data-baseweb="input"] input {
            padding: 0.65rem 1.1rem !important;
            line-height: 1.45 !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] input::placeholder,
        .st-key-ha_auth_shell [data-testid="stForm"] input::-webkit-input-placeholder {
            color: #7a8578 !important;
            opacity: 1 !important;
        }
        .st-key-ha_auth_shell [data-testid="stForm"] div[data-baseweb="input"]:focus-within {
            border-color: var(--ha-lux-moss) !important;
            box-shadow: 0 0 0 3px rgba(112, 130, 96, 0.2) !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-ha_lux_remember_row [data-testid="stVerticalBlock"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            margin-top: 0.1rem !important;
            margin-bottom: 0.45rem !important;
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
        .st-key-ha_lux_remember_row [data-testid="column"]:last-child {
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stFormSubmitButton"] {
            flex: 0 0 auto !important;
            margin-bottom: 0 !important;
        }
        .st-key-ha_lux_remember_row [data-testid="stFormSubmitButton"] [data-testid="stMarkdownContainer"] p {
            margin: 0 !important;
            line-height: 1.3 !important;
        }
        /* Forgot password butonu: form'un DIŞINDA, Login butonunun hemen
           altında ortalı tertiary link olarak. Form içinde tek submit
           (Login) bulunduğundan Enter doğrudan Login'i tetikler. */
        .st-key-ha_lux_forgot_row {
            margin-top: 0.35rem !important;
            display: flex !important;
            justify-content: center !important;
        }
        .st-key-ha_lux_forgot_row [data-testid="stElementContainer"] {
            width: auto !important;
            margin: 0 !important;
        }
        .st-key-ha_lux_forgot_row [data-testid="stButton"] {
            display: inline-flex !important;
            justify-content: center !important;
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
        }
        .st-key-ha_lux_forgot_row button:hover {
            background: transparent !important;
            color: var(--ha-text, #1f1f1f) !important;
            text-decoration-color: currentColor;
        }
        /* Primary submit — smaller pill, centered; does not touch card edges (column padding) */
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit {
            display: flex !important;
            justify-content: center !important;
            width: 100% !important;
            margin-top: 0.25rem !important;
            margin-bottom: 0.75rem !important;
            padding: 0 0.75rem 0.5rem 0.75rem !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="element-container"] {
            max-width: 180px !important;
            width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        .st-key-ha_auth_shell .st-key-ha_auth_primary_submit [data-testid="stFormSubmitButton"] {
            margin-bottom: 0 !important;
            width: 100% !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"],
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] {
            border-radius: 12px !important;
            min-height: 32px !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.02em !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            background-color: #6d7f62 !important;
            background-image: linear-gradient(180deg, #95a88a 0%, #6d7f62 100%) !important;
            border: 1px solid #5d6e54 !important;
            box-shadow:
                0 3px 10px rgba(75, 89, 64, 0.14),
                inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
            transition:
                box-shadow 0.2s ease,
                transform 0.2s ease,
                filter 0.2s ease !important;
        }
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] *,
        .st-key-ha_auth_shell [data-testid="stFormSubmitButton"] button[kind="primary"] * {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
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
            margin-bottom: 0.65rem;
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
        [data-testid="stSidebar"] {
            position: relative;
            z-index: 0;
            border-right: 1px solid var(--ha-border) !important;
            background: var(--ha-sidebar-bg) !important;
        }
        /* Sidebar üzerindeki noise dokusu sade görünüm için kapatıldı */
        [data-testid="stAppViewContainer"] [data-testid="stSidebar"]::before {
            content: none !important;
            display: none !important;
        }
        [data-testid="stSidebar"] > * {
            position: relative;
            z-index: 1;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.65rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        /* Sidebar: position: relative + tam viewport yüksekliği. Bu sayede
           içindeki absolute konumlu login bloğu (bottom: 0) gerçek viewport
           dibine yapışır, içerik kısa olsa bile yukarıya kaymaz.
           stSidebarUserContent ve block-container ek positioning context
           oluşturmaması için sıfırlanır. Block-container'a ise login
           bloğunun altta kalacağı kadar bottom padding veriyoruz. */
        [data-testid="stSidebar"] {
            position: relative !important;
            min-height: 100vh !important;
            height: 100vh !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            position: static !important;
            min-height: 100% !important;
            height: auto !important;
        }
        [data-testid="stSidebar"] .block-container {
            min-height: 100vh !important;
        }
        [data-testid="stSidebar"] .block-container:has(.st-key-ha_sidebar_login_row) {
            padding-bottom: 12rem !important;
        }
        /* Login satırı (başlık + açıklama + buton) sidebar'ın TAM dipine
           absolute olarak yapıştırılır. left/right 0 + bottom 0 → kenarlarla
           ve alt kenarla flush. İç padding ile içeriği konumluyoruz. */
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> .st-key-ha_sidebar_login_row),
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.st-key-ha_sidebar_login_row) {
            position: absolute !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0.85rem 0.85rem 0.85rem 0.85rem !important;
            border-top: 1px solid var(--ha-border) !important;
            background: var(--ha-sidebar-bg) !important;
            z-index: 5 !important;
        }
        .st-key-ha_sidebar_login_row {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
        }
        /* Login kartı: başlık + açıklama (sade, kutusuz) */
        .ha-sidebar-login-card {
            margin: 0 0 0.55rem 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        .ha-sidebar-login-title {
            margin: 0 0 0.22rem 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            line-height: 1.3 !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
        }
        /* Açıklama metni: küçük, açık gri, sade (kutusuz) */
        .ha-sidebar-login-hint {
            margin: 0 !important;
            padding: 0 !important;
            font-family: "Inter", system-ui, -apple-system, sans-serif !important;
            font-size: 0.72rem !important;
            font-weight: 400 !important;
            line-height: 1.4 !important;
            color: var(--ha-text-soft) !important;
            -webkit-text-fill-color: var(--ha-text-soft) !important;
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
        }
        /* Premium header: nav label + user card */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) {
            margin-bottom: 0.15rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.ha-sidebar-header) p {
            margin: 0 !important;
            color: inherit !important;
        }
        .ha-sidebar-header {
            margin: 0 0 0.75rem 0;
        }
        .ha-sidebar-header__eyebrow {
            display: flex !important;
            align-items: center;
            gap: 0.45rem;
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
            padding: 0.5rem 0.65rem;
            background: #ffffff;
            border: 1px solid var(--ha-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ha-sidebar-header__avatar {
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 10px;
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
            background: linear-gradient(145deg, #4a4a4a 0%, #1f1f1f 92%);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
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
            border-radius: 12px !important;
            overflow: hidden;
            background: var(--ha-surface) !important;
            margin-bottom: 0.65rem !important;
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
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="secondary"],
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            background: transparent !important;
            background-image: none !important;
            border: 1px solid transparent !important;
            color: var(--ha-text) !important;
            -webkit-text-fill-color: var(--ha-text) !important;
            border-radius: 8px !important;
            min-height: 2.1rem !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            padding: 0.4rem 0.6rem !important;
            box-shadow: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
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
        [data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[kind="secondary"]:hover,
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
            background: rgba(0, 0, 0, 0.05) !important;
            filter: none !important;
            border-color: transparent !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover [data-testid="stIconMaterial"] {
            color: var(--ha-text) !important;
        }

        /* Sidebar dipindeki Login (CTA) butonu prominent kalsın — minimal
           kuralı override ederek koyu pill stilini geri veriyoruz. */
        .st-key-ha_sidebar_login_btn button {
            background: #2c2c2c !important;
            background-image: none !important;
            border: 1px solid #2c2c2c !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-radius: 10px !important;
            min-height: 2.4rem !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            padding: 0.55rem 0.85rem !important;
            text-align: center !important;
            justify-content: center !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        }
        .st-key-ha_sidebar_login_btn button p,
        .st-key-ha_sidebar_login_btn button span {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-weight: 600 !important;
        }
        .st-key-ha_sidebar_login_btn button:hover {
            background: #1f1f1f !important;
            border-color: #1f1f1f !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
        }
        [data-testid="stSidebar"] .ha-sidebar-title {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--ha-text-soft);
            margin: 0.65rem 0 0.28rem 0;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.85;
        }
        [data-testid="stSidebar"] .ha-sidebar-subtitle {
            font-size: 0.74rem;
            color: var(--ha-text-soft);
            opacity: 0.85;
            margin-bottom: 0.32rem;
            font-weight: 500;
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
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 9px;
            padding: 0.34rem 0.55rem;
            margin-bottom: 0.08rem;
            transition: background-color 0.15s ease;
            font-size: 0.9rem !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(0, 0, 0, 0.04);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(0, 0, 0, 0.06);
            border: 1px solid transparent !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
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
        }
        /* stMainBlockContainer: TEK scroll alanı (mesajlar). Composer'ın altta
           kapladığı alan kadar bottom padding ekleriz, son mesaj gizlenmesin. */
        [data-testid="stAppViewContainer"]:has(.st-key-ha_chat_composer_row) [data-testid="stMainBlockContainer"] {
            height: 100% !important;
            max-height: 100% !important;
            min-height: 0 !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-bottom: calc(7.5rem + env(safe-area-inset-bottom, 0px)) !important;
            scroll-behavior: smooth !important;
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
            bottom: 0 !important;
            top: auto !important;
            left: var(--ha-main-left, 0) !important;
            right: auto !important;
            width: var(--ha-main-width, 100vw) !important;
            max-width: none !important;
            z-index: 999 !important;
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
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
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
            pointer-events: auto;
            max-width: 48rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: max(0.6rem, env(safe-area-inset-bottom, 0px)) !important;
            padding: 0.5rem 0.65rem !important;
            background: #ffffff !important;
            border: 1px solid var(--ha-card-edge) !important;
            border-radius: 14px !important;
            box-shadow:
                0 -4px 14px rgba(40, 55, 45, 0.06),
                var(--ha-card-shadow) !important;
        }
        @media (max-width: 720px) {
            .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
                margin-left: 0.5rem !important;
                margin-right: 0.5rem !important;
                margin-bottom: max(0.5rem, env(safe-area-inset-bottom, 0px)) !important;
            }
        }
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] {
            align-items: flex-end !important;
            gap: 0.35rem !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            flex: 0 0 2.5rem !important;
            width: 2.5rem !important;
            min-width: 2.5rem !important;
            max-width: 2.5rem !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button {
            min-height: 2.35rem !important;
            width: 100% !important;
            max-width: none !important;
            padding: 0.15rem !important;
            border-radius: 11px !important;
            border: 1px solid rgba(72, 92, 78, 0.2) !important;
            background: #ffffff !important;
            color: #4d564e !important;
            -webkit-text-fill-color: #4d564e !important;
            box-shadow: 0 1px 2px rgba(40, 55, 45, 0.05) !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button:hover {
            background: #f9fbf9 !important;
            border-color: rgba(72, 92, 78, 0.28) !important;
            color: #3d433d !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stPopover"] button [data-testid="stMarkdownContainer"] {
            display: none !important;
        }
        .st-key-ha_chat_composer_row [data-testid="stChatInput"] > div {
            border-radius: 14px !important;
            border-color: var(--ha-border) !important;
            background: var(--ha-surface) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
        }
        /* Chat bubbles: ChatGPT tarzı sade — asistan baloncuğu çerçevesiz,
           kullanıcı baloncuğu hafif gri pill. */
        section.main [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.45rem 0.65rem !important;
            margin-bottom: 0.45rem !important;
            box-shadow: none !important;
        }
        /* Kullanıcı mesajı (sağ avatarlı): yumuşak gri pill */
        section.main [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ),
        section.main [data-testid="stChatMessage"].st-emotion-cache-user,
        section.main [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) {
            background: #f4f4f5 !important;
            border-radius: 14px !important;
            padding: 0.6rem 0.85rem !important;
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
        /* Assistant bubbles: compact copy + sources + merged 👍/👎 */
        [class*="st-key-ha_assistant_actions_row"] [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            flex-wrap: nowrap !important;
        }
        [class*="st-key-ha_assistant_copy_cell"] {
            overflow: visible !important;
        }
        [class*="st-key-ha_assistant_copy_cell"] iframe {
            height: 32px !important;
            min-height: 32px !important;
            max-height: 32px !important;
        }
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stPopover"] button {
            min-height: 1.72rem !important;
            padding: 0.14rem 0.55rem !important;
            font-size: 0.78rem !important;
            line-height: 1.25 !important;
            white-space: nowrap !important;
        }
        [class*="st-key-ha_assistant_sources_cell"] [data-testid="stCaptionContainer"] p {
            font-size: 0.72rem !important;
            margin: 0 !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            align-items: stretch !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 0 !important;
            min-width: 2.55rem !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"] {
            min-height: 1.65rem !important;
            padding: 0.08rem 0.32rem !important;
            font-size: 1.05rem !important;
            line-height: 1.15 !important;
            border-radius: 0 !important;
            border: 1px solid rgba(120, 120, 120, 0.32) !important;
            background: #ffffff !important;
            width: 100% !important;
            max-width: none !important;
        }
        [class*="st-key-ha_assistant_feedback_group"]
            [data-testid="stHorizontalBlock"]
            > div[data-testid="column"]:first-child
            button[data-testid^="baseButton"] {
            border-radius: 7px 0 0 7px !important;
        }
        [class*="st-key-ha_assistant_feedback_group"]
            [data-testid="stHorizontalBlock"]
            > div[data-testid="column"]:last-child
            button[data-testid^="baseButton"] {
            border-radius: 0 7px 7px 0 !important;
            border-left: none !important;
        }
        [class*="st-key-ha_assistant_feedback_group"] button[data-testid^="baseButton"]:hover {
            background: #f9fbf9 !important;
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
                justify-content: flex-start;
            }
            .st-key-ha_auth_shell {
                --ha-lux-card-h: auto;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                border-radius: 18px !important;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"]::before {
                border-radius: 18px;
            }
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(1),
            .st-key-ha_auth_card
                [data-testid="stHorizontalBlock"]
                > [data-testid="column"]:nth-of-type(2) {
                flex: 1 1 auto !important;
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
            .st-key-ha_auth_shell .ha-lux-welcome__rule {
                margin-left: auto;
                margin-right: auto;
            }
            .ha-lux-botanical {
                padding-top: 1.25rem;
            }
            .st-key-ha_auth_card [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) > div[data-testid="stVerticalBlock"] {
                min-height: auto;
                padding: 1.75rem 1.5rem 2.5rem 1.5rem !important;
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
            margin: -0.2rem 0 0.28rem 0;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

def _inject_chat_layout_script() -> None:
    """Sidebar açıkken/kapalıyken composer'ın stMain ile hizalı kalması için
    iki CSS değişkenini günceller: --ha-main-left, --ha-main-width.

    Composer 'position: fixed' kullanır; CSS değişkenleri sayesinde stMain'in
    gerçek bounding rect'ine göre konumlanır."""
    components.html(
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
                  if (!main) return;
                  const rect = main.getBoundingClientRect();
                  root.style.setProperty('--ha-main-left', rect.left + 'px');
                  root.style.setProperty('--ha-main-width', rect.width + 'px');
                } catch (e) {}
              }

              update();
              const win = window.parent || window;
              win.addEventListener('resize', update, { passive: true });

              // Sidebar / main boyut değişimlerini izle
              try {
                const targets = [
                  doc.querySelector('[data-testid="stMain"]'),
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
        """,
        height=0,
        width=0,
    )

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
