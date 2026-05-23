"""Admin panel: PDF management, reindexing, and feedback log."""

import html as _html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from herbalist_assistant import config

from ..components import (
    _normalize_sources,
    _render_advanced_settings_widgets,
    _source_entry_label,
)
from ..i18n import get_string
from ..resources import delete_pdfs_from_index, index_new_pdfs, reindex_pdfs
from ..state import iter_all_feedback
from .chat import _truncate_chat_title


def _list_pdf_files() -> List[Path]:
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(data_dir.glob("*.pdf"))


def _get_last_index_time() -> str:
    if st.session_state.last_index_time:
        return st.session_state.last_index_time

    chroma_db = Path(config.CHROMA_DIR) / "chroma.sqlite3"
    if chroma_db.exists():
        return datetime.fromtimestamp(chroma_db.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return "Not indexed yet"


def _require_admin() -> None:
    if st.session_state.role != "admin":
        st.error("Unauthorized access")
        st.stop()


def _html_admin_top_metrics(
    lang: str,
    *,
    total_pdfs: int,
    last_index: str,
    model: str,
) -> str:
    """Top admin metric row (HTML cards; values escaped)."""
    l1 = _html.escape(get_string(lang, "total_pdfs"))
    l2 = _html.escape(get_string(lang, "last_index_time"))
    l3 = _html.escape(get_string(lang, "active_model"))
    v2 = _html.escape(str(last_index))
    v3 = _html.escape(str(model))
    return (
        f'<div class="ha-admin-metric-row" role="group">'
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l1}</div>'
        f'<div class="ha-admin-metric-card__value">{int(total_pdfs)}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l2}</div>'
        f'<div class="ha-admin-metric-card__value ha-admin-metric-card__value--sm">{v2}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l3}</div>'
        f'<div class="ha-admin-metric-card__value ha-admin-metric-card__value--sm">{v3}</div>'
        f"</div>"
        f"</div>"
    )


def _html_admin_feedback_metrics(lang: str, *, total: int, ups: int, downs: int) -> str:
    """Feedback summary metric row."""
    l1 = _html.escape(get_string(lang, "feedback_log_total"))
    l2 = _html.escape(get_string(lang, "feedback_metric_helpful"))
    l3 = _html.escape(get_string(lang, "feedback_metric_unhelpful"))
    return (
        f'<div class="ha-admin-metric-row ha-admin-metric-row--tight" role="group">'
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l1}</div>'
        f'<div class="ha-admin-metric-card__value">{int(total)}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l2}</div>'
        f'<div class="ha-admin-metric-card__value">{int(ups)}</div>'
        f"</div>"
        f'<div class="ha-admin-metric-card">'
        f'<div class="ha-admin-metric-card__label">{l3}</div>'
        f'<div class="ha-admin-metric-card__value">{int(downs)}</div>'
        f"</div>"
        f"</div>"
    )


def _render_admin_feedback_log(*, lang: str) -> None:
    """Flat, newest-first list of feedback signals across every user and chat."""
    with st.container(key="ha_admin_feedback"):
        st.markdown(
            f'<header class="ha-admin-feedback-head" style="margin-top:1.1rem;">'
            f'<p class="ha-admin-feedback-head__title">{_html.escape(get_string(lang, "feedback_log_title"))}</p>'
            f'<p class="ha-admin-feedback-head__sub">{_html.escape(get_string(lang, "feedback_log_desc"))}</p>'
            f"</header>",
            unsafe_allow_html=True,
        )

        entries = iter_all_feedback()
        if not entries:
            st.info(get_string(lang, "feedback_log_empty"))
            return

        ups = sum(1 for e in entries if e.get("feedback") == "up")
        downs = sum(1 for e in entries if e.get("feedback") == "down")

        st.markdown(
            f'<p class="ha-admin-glance-label">{_html.escape(get_string(lang, "feedback_at_a_glance"))}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _html_admin_feedback_metrics(lang, total=len(entries), ups=ups, downs=downs),
            unsafe_allow_html=True,
        )

        filter_options = {
            "all": get_string(lang, "feedback_log_filter_all"),
            "up": get_string(lang, "feedback_log_filter_up"),
            "down": get_string(lang, "feedback_log_filter_down"),
        }
        with st.container(border=True):
            selected_filter = st.radio(
                get_string(lang, "feedback_log_filter"),
                options=list(filter_options.keys()),
                format_func=lambda k: filter_options[k],
                horizontal=True,
                key="admin_feedback_filter",
            )

        filtered = (
            entries
            if selected_filter == "all"
            else [e for e in entries if e.get("feedback") == selected_filter]
        )

        if not filtered:
            st.info(get_string(lang, "feedback_log_empty"))
            return

        for entry in filtered:
            icon = (
                get_string(lang, "feedback_expander_helpful")
                if entry.get("feedback") == "up"
                else get_string(lang, "feedback_expander_unhelpful")
            )
            user = str(entry.get("username", "?"))
            chat_part = _truncate_chat_title(entry.get("chat_title", ""), 48)
            ts = str(entry.get("feedback_at", "") or entry.get("timestamp", ""))
            title = f"{icon}  {user}  ·  {chat_part}"
            with st.expander(title, expanded=False):
                st.markdown(
                    f'<p class="ha-admin-entry-meta">{_html.escape(ts)}</p>',
                    unsafe_allow_html=True,
                )
                question = entry.get("question") or ""
                if question:
                    st.markdown(f"**{get_string(lang, 'feedback_log_question')}**")
                    st.markdown(question)
                st.markdown(f"**{get_string(lang, 'feedback_log_answer')}**")
                st.markdown(entry.get("answer", ""))
                sources = entry.get("sources", []) or []
                normalized_sources = _normalize_sources(sources)
                if normalized_sources:
                    st.caption(
                        get_string(lang, "feedback_log_sources")
                        + ": "
                        + ", ".join(_source_entry_label(s) for s in normalized_sources)
                    )


def _render_admin_panel() -> None:
    lang = st.session_state.get("language", "en")
    _require_admin()
    pdf_files = _list_pdf_files()
    with st.container(key="ha_admin_panel"):
        st.markdown(
            f'<header class="ha-admin-hero">'
            f'<h1 class="ha-admin-hero__title">{_html.escape(get_string(lang, "admin_dashboard"))}</h1>'
            f'<p class="ha-admin-hero__sub">{_html.escape(get_string(lang, "admin_subtitle"))}</p>'
            f"</header>",
            unsafe_allow_html=True,
        )

        with st.expander(get_string(lang, "advanced_settings"), expanded=False):
            _render_advanced_settings_widgets(
                model_key="admin_adv_model", web_key="admin_adv_web"
            )

        st.markdown(
            _html_admin_top_metrics(
                lang,
                total_pdfs=len(pdf_files),
                last_index=_get_last_index_time(),
                model=str(st.session_state.selected_model),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<h2 class="ha-admin-section-h">{_html.escape(get_string(lang, "admin_op_section"))}</h2>',
            unsafe_allow_html=True,
        )

        col_u, col_r, col_d = st.columns(3)
        with col_u:
            with st.container(border=True):
                st.markdown(
                    f'<p class="ha-admin-op-title">{_html.escape(get_string(lang, "admin_card_upload"))}</p>',
                    unsafe_allow_html=True,
                )
                uploaded = st.file_uploader(
                    get_string(lang, "upload_btn"),
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="admin_pdf_uploader",
                )
                if uploaded:
                    data_dir = Path(config.DATA_DIR)
                    for file in uploaded:
                        target = data_dir / file.name
                        with target.open("wb") as f:
                            f.write(file.getbuffer())
                    st.success(f"{len(uploaded)} PDF file(s) uploaded.")

        with col_r:
            with st.container(border=True):
                st.markdown(
                    f'<p class="ha-admin-op-title">{_html.escape(get_string(lang, "admin_card_reindex"))}</p>',
                    unsafe_allow_html=True,
                )
                st.caption(get_string(lang, "reindex_desc"))
                if st.button(
                    get_string(lang, "index_new_btn"),
                    use_container_width=True,
                ):
                    with st.spinner(get_string(lang, "index_new_spinner")):
                        stats = index_new_pdfs()
                    st.session_state.last_index_time = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    indexed = stats.get("indexed") or []
                    skipped = stats.get("skipped") or []
                    if indexed:
                        st.success(
                            get_string(lang, "index_new_done").format(
                                count=len(indexed),
                                files=", ".join(indexed),
                            )
                        )
                    else:
                        st.info(get_string(lang, "index_new_none").format(count=len(skipped)))
                if st.button(
                    get_string(lang, "reindex_btn"),
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(get_string(lang, "reindex_spinner")):
                        reindex_pdfs()
                    st.session_state.last_index_time = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.success(get_string(lang, "db_rebuilt"))

        with col_d:
            with st.container(border=True):
                st.markdown(
                    f'<p class="ha-admin-op-title">{_html.escape(get_string(lang, "admin_card_delete"))}</p>',
                    unsafe_allow_html=True,
                )
                if pdf_files:
                    selected_delete = st.multiselect(
                        get_string(lang, "delete_select"),
                        options=[p.name for p in pdf_files],
                        key="admin_pdf_delete_select",
                    )
                    if st.button(get_string(lang, "delete_btn"), use_container_width=True):
                        if not selected_delete:
                            st.warning(get_string(lang, "select_pdf_warn"))
                        else:
                            data_dir = Path(config.DATA_DIR)
                            removed_from_disk: list[str] = []
                            for filename in selected_delete:
                                target = data_dir / filename
                                if target.exists():
                                    target.unlink()
                                    removed_from_disk.append(filename)
                            if removed_from_disk:
                                with st.spinner(get_string(lang, "delete_index_spinner")):
                                    delete_pdfs_from_index(removed_from_disk)
                            st.success(
                                get_string(lang, "pdf_deleted").format(
                                    count=len(removed_from_disk)
                                )
                            )
                            st.rerun()
                else:
                    st.info(get_string(lang, "no_pdf"))

        _render_admin_feedback_log(lang=lang)
