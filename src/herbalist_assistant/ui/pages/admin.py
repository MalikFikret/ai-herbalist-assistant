"""Admin panel: PDF management, reindexing, and feedback log."""

import html as _html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from herbalist_assistant import config
from herbalist_assistant.settings_manager import get_setting, save_settings

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
    data_dir = Path(get_setting("DATA_DIR"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(data_dir.glob("*.pdf"))

from herbalist_assistant.db import repository as db_repository
from herbalist_assistant.ui.auth import _load_runtime_admin_credentials


def _get_last_index_time() -> str:
    if st.session_state.last_index_time:
        return st.session_state.last_index_time

    chroma_db = Path(get_setting("CHROMA_DIR")) / "chroma.sqlite3"
    if chroma_db.exists():
        return datetime.fromtimestamp(chroma_db.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return "Not indexed yet"


def _require_admin() -> None:
    if st.session_state.get("role") != "admin" and not st.session_state.get("admin_is_logged_in"):
        st.error("Unauthorized access")
        st.stop()


def _html_admin_top_metrics(
    lang: str,
    *,
    total_users: int,
    total_pdfs: int,
    total_feedback: int,
    last_index: str,
    model: str,
) -> str:
    """Top admin metric row (HTML cards; values escaped)."""
    l0 = _html.escape(get_string(lang, "total_users") or "Total Users")
    l1 = _html.escape(get_string(lang, "total_pdfs"))
    l_fb = _html.escape(get_string(lang, "total_feedback") or "Total Feedback")
    l2 = _html.escape(get_string(lang, "last_index_time"))
    l3 = _html.escape(get_string(lang, "active_model"))
    v2 = _html.escape(str(last_index))
    v3 = _html.escape(str(model))
    
    card_style = (
        "background-color: #FFFFFF; "
        "border-radius: 16px; "
        "padding: 24px; "
        "box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1); "
        "border-left: 6px solid #5c6f5e; "
        "display: flex; "
        "flex-direction: column; "
        "justify-content: center;"
    )
    
    label_style = "font-size: 14px; font-weight: 600; color: #475569; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;"
    value_style = "font-size: 28px; font-weight: 800; color: #0f172a; margin: 0; line-height: 1.2;"
    value_sm_style = "font-size: 20px; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.2;"
    
    return (
        f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; margin-top: 16px; margin-bottom: 32px;">'
        
        f'<div style="{card_style}">'
        f'<div style="{label_style}">{l0}</div>'
        f'<div style="{value_style}">{int(total_users)}</div>'
        f"</div>"
        
        f'<div style="{card_style}">'
        f'<div style="{label_style}">{l1}</div>'
        f'<div style="{value_style}">{int(total_pdfs)}</div>'
        f"</div>"
        
        f'<div style="{card_style}">'
        f'<div style="{label_style}">{l_fb}</div>'
        f'<div style="{value_style}">{int(total_feedback)}</div>'
        f"</div>"
        
        f'<div style="{card_style}">'
        f'<div style="{label_style}">{l3}</div>'
        f'<div style="{value_sm_style}">{v3}</div>'
        f"</div>"
        
        f'<div style="{card_style}">'
        f'<div style="{label_style}">{l2}</div>'
        f'<div style="{value_sm_style}">{v2}</div>'
        f"</div>"
        
        f"</div>"
    )


def _render_admin_feedback_log(*, lang: str) -> None:
    """Flat, newest-first list of feedback signals across every user and chat."""
    entries = iter_all_feedback()
    
    total = len(entries)
    ups = sum(1 for e in entries if e.get("feedback") == "up")
    downs = sum(1 for e in entries if e.get("feedback") == "down")
    rate = f"{(ups / total * 100):.1f}%" if total > 0 else "0%"

    # 1. Stats
    st.markdown("<h4 style='color:#3e4e35; margin-bottom: 12px; margin-top:0;'>📊 Feedback Overview</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Feedback", total)
    c2.metric("👍 Helpful", ups)
    c3.metric("👎 Unhelpful", downs)
    c4.metric("📈 Helpful Rate", rate)
        
    st.markdown("<hr style='margin: 16px 0;'/>", unsafe_allow_html=True)
    
    # 2 & 3. Search and Filter
    col_search, col_filter = st.columns([2, 1])
    search_q = col_search.text_input(
        "Search feedback...", 
        label_visibility="collapsed", 
        placeholder="Search feedback...", 
        key="admin_feedback_search"
    )
    
    selected_filter = col_filter.radio(
        "Filter", 
        options=["All", "Helpful", "Unhelpful"], 
        horizontal=True, 
        label_visibility="collapsed", 
        key="admin_feedback_filter"
    )

    filtered = entries
    if selected_filter == "Helpful":
        filtered = [e for e in filtered if e.get("feedback") == "up"]
    elif selected_filter == "Unhelpful":
        filtered = [e for e in filtered if e.get("feedback") == "down"]
    
    if search_q:
        q = search_q.lower()
        filtered = [e for e in filtered if q in (e.get("question") or "").lower() or q in (e.get("username") or "").lower()]

    if not filtered:
        st.info("No feedback found.")
        return

    # 4. Table
    with st.container(border=True):
        st.markdown("""
        <style>
        .ha-fb-row-marker {
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 14px;
            color: #1e293b;
        }
        div[data-testid="stHorizontalBlock"]:has(.ha-fb-row-marker) {
            align-items: center;
            padding: 4px 8px;
            border-bottom: 1px solid #f1f5f9;
        }
        div[data-testid="stHorizontalBlock"]:has(.ha-fb-row-marker):hover {
            background-color: #f8fafc;
        }
        div[data-testid="stHorizontalBlock"]:has(.ha-fb-row-marker) button {
            padding: 0 !important;
            min-height: 28px !important;
            height: 28px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        h1, h2, h3, h4, h5 = st.columns([1.5, 1, 4, 1.5, 1])
        h1.markdown("<div style='background-color:#e2e8f0; padding:6px; border-radius:4px; font-weight:700; font-size:13px;'>User</div>", unsafe_allow_html=True)
        h2.markdown("<div style='background-color:#e2e8f0; padding:6px; border-radius:4px; font-weight:700; font-size:13px;'>Rating</div>", unsafe_allow_html=True)
        h3.markdown("<div style='background-color:#e2e8f0; padding:6px; border-radius:4px; font-weight:700; font-size:13px;'>Question</div>", unsafe_allow_html=True)
        h4.markdown("<div style='background-color:#e2e8f0; padding:6px; border-radius:4px; font-weight:700; font-size:13px;'>Date</div>", unsafe_allow_html=True)
        h5.markdown("<div style='background-color:#e2e8f0; padding:6px; border-radius:4px; font-weight:700; font-size:13px; text-align:center;'>Action</div>", unsafe_allow_html=True)

        for i, entry in enumerate(filtered):
            c1, c2, c3, c4, c5 = st.columns([1.5, 1, 4, 1.5, 1])
            user = _html.escape(str(entry.get("username", "•")))
            is_up = entry.get("feedback") == "up"
            rating = "👍 Helpful" if is_up else "👎 Unhelpful"
            rating_color = "#166534" if is_up else "#991b1b"
            
            question = entry.get("question") or ""
            q_trunc = _html.escape(question[:50] + ("..." if len(question) > 50 else question))
            ts_raw = str(entry.get("feedback_at", "") or entry.get("timestamp", ""))
            ts = ts_raw.split(".")[0] if "." in ts_raw else ts_raw
            
            c1.markdown(f"<span class='ha-fb-row-marker'>👤 {user}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span class='ha-fb-row-marker' style='color:{rating_color}; font-weight:600;'>{rating}</span>", unsafe_allow_html=True)
            c3.markdown(f"<span class='ha-fb-row-marker' title='{_html.escape(question)}'>{q_trunc}</span>", unsafe_allow_html=True)
            c4.markdown(f"<span class='ha-fb-row-marker' style='color:#64748b; font-size:12px;'>{_html.escape(ts)}</span>", unsafe_allow_html=True)
            
            with c5:
                if hasattr(st, "popover"):
                    with st.popover("👁️ View", use_container_width=True):
                        st.markdown(f"**User:** {user}  \n**Rating:** {rating}  \n**Date:** {ts}")
                        if question:
                            st.markdown("**Question:**")
                            st.info(question)
                        st.markdown("**Answer:**")
                        st.success(entry.get("answer", ""))
                        sources = entry.get("sources", [])
                        if sources:
                            st.caption("Sources: " + ", ".join(_source_entry_label(s) for s in _normalize_sources(sources)))
                else:
                    if st.button("👁️", key=f"fb_view_{i}", use_container_width=True, help="View Details"):
                        pass # Fallback for old streamlit versions


def _render_admin_panel() -> None:
    lang = st.session_state.get("language", "en")
    _require_admin()
    pdf_files = _list_pdf_files()
    
    with st.container(key="ha_admin_panel"):
        st.markdown("""
        <style>
        /* Dashboard Background */
        [data-testid="stAppViewContainer"] {
            background-color: #F2F6F0;
        }
        
        /* White Cards with Light Green Borders */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border-color: #DCE5DA !important;
            border-radius: 8px !important;
        }
        
        /* Dark Green Accents for Headers inside cards */
        div[data-testid="stVerticalBlockBorderWrapper"] h1,
        div[data-testid="stVerticalBlockBorderWrapper"] h2,
        div[data-testid="stVerticalBlockBorderWrapper"] h3,
        div[data-testid="stVerticalBlockBorderWrapper"] h4,
        div[data-testid="stVerticalBlockBorderWrapper"] h5 {
            color: #4F6F52 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(
            f'<header class="ha-admin-hero" style="padding-top: 1rem;">'
            f'<h1 class="ha-admin-hero__title" style="color: #4F6F52;">{_html.escape(get_string(lang, "admin_dashboard"))}</h1>'
            f'<p class="ha-admin-hero__sub" style="color: #5c6f5e;">{_html.escape(get_string(lang, "admin_subtitle"))}</p>'
            f"</header>",
            unsafe_allow_html=True,
        )

        tab_overview, tab_users, tab_docs, tab_feedback, tab_settings = st.tabs([
            "Overview", 
            "Users",
            "Documents", 
            "Feedback", 
            "Settings"
        ])

        with tab_overview:
            import os
            # 1. Keep Top Metrics
            all_users_list = db_repository.get_all_users()
            all_feedback_list = list(iter_all_feedback())
            last_idx = _get_last_index_time()
            st.markdown(
                _html_admin_top_metrics(
                    lang,
                    total_users=len(all_users_list),
                    total_pdfs=len(pdf_files),
                    total_feedback=len(all_feedback_list),
                    last_index=last_idx,
                    model=str(st.session_state.selected_model),
                ),
                unsafe_allow_html=True,
            )
            
            # 2. Dashboard Grid layout
            st.markdown("<hr style='margin: 0 0 24px 0;'/>", unsafe_allow_html=True)
            col_left, col_right = st.columns(2, gap="large")
            
            with col_left:
                # 1. Quick Actions
                with st.container(border=True):
                    st.markdown("##### ⚡ Quick Actions")
                    c1, c2 = st.columns(2)
                    c1.button("📤 Upload PDF", use_container_width=True, help="Navigate to Documents tab")
                    if c2.button("🔄 Full Re-index", use_container_width=True, type="primary"):
                        with st.spinner("Re-indexing..."):
                            reindex_pdfs()
                        st.session_state.last_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("Re-indexed successfully!")
                        st.rerun()
                    
                    c3, c4 = st.columns(2)
                    c3.button("👥 Manage Users", use_container_width=True, help="Navigate to Users tab")
                    c4.button("⚙️ Open Settings", use_container_width=True, help="Navigate to Settings tab")

                # 3. Recent Activity
                with st.container(border=True):
                    st.markdown("##### 🕒 Recent Activity")
                    st.markdown("<p style='font-size:14px; margin-bottom:8px;'>• <strong>Last Index:</strong> " + last_idx + "</p>", unsafe_allow_html=True)
                    if pdf_files:
                        st.markdown("<p style='font-size:14px; margin-bottom:8px;'>• <strong>Latest PDF:</strong> " + pdf_files[-1].name + "</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='font-size:14px; margin-bottom:8px;'>• <strong>Latest PDF:</strong> None</p>", unsafe_allow_html=True)
                    
                    if all_users_list:
                        st.markdown("<p style='font-size:14px; margin-bottom:8px;'>• <strong>Latest User:</strong> " + all_users_list[-1]['username'] + "</p>", unsafe_allow_html=True)
                    
                    if all_feedback_list:
                        st.markdown("<p style='font-size:14px; margin-bottom:0;'>• <strong>Latest Feedback:</strong> " + ("👍" if all_feedback_list[0].get("feedback") == "up" else "👎") + " by " + str(all_feedback_list[0].get("username", "•")) + "</p>", unsafe_allow_html=True)
                
                # 5. PDF Statistics
                with st.container(border=True):
                    st.markdown("##### 📚 PDF Statistics")
                    st.markdown(f"**Total PDFs:** {len(pdf_files)}")
                    if last_idx == "Not indexed yet":
                        st.markdown("**Indexed PDFs:** 0")
                        st.markdown(f"**Pending PDFs:** {len(pdf_files)}")
                        st.progress(0.0)
                    else:
                        st.markdown(f"**Indexed PDFs:** {len(pdf_files)}")
                        st.markdown("**Pending PDFs:** 0")
                        st.progress(1.0)

            with col_right:
                # 2. System Health
                with st.container(border=True):
                    st.markdown("##### 🩺 System Health")
                    api_is_ok = os.environ.get("GROQ_API_KEY")
                    api_color = "#22c55e" if api_is_ok else "#ef4444"
                    api_status = "Online" if api_is_ok else "Error"
                    search_provider = st.session_state.get("web_search_provider", "Tavily")
                    
                    st.markdown(f"""
                    <div style='display:flex; flex-direction:column; gap:8px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; background:#f8fafc; padding:8px 12px; border-radius:6px;'>
                            <span style='font-size:14px; font-weight:600;'>SQLite DB</span>
                            <span style='font-size:13px;'><span style='color:#22c55e;'>🟢</span> Online</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; background:#f8fafc; padding:8px 12px; border-radius:6px;'>
                            <span style='font-size:14px; font-weight:600;'>Vector DB (Chroma)</span>
                            <span style='font-size:13px;'><span style='color:#22c55e;'>🟢</span> Online</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; background:#f8fafc; padding:8px 12px; border-radius:6px;'>
                            <span style='font-size:14px; font-weight:600;'>LLM API</span>
                            <span style='font-size:13px;'><span style='color:{api_color};'>🟢</span> {api_status}</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; background:#f8fafc; padding:8px 12px; border-radius:6px;'>
                            <span style='font-size:14px; font-weight:600;'>Embedding Model</span>
                            <span style='font-size:13px;'><span style='color:#22c55e;'>🟢</span> Ready</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; background:#f8fafc; padding:8px 12px; border-radius:6px;'>
                            <span style='font-size:14px; font-weight:600;'>Search ({search_provider})</span>
                            <span style='font-size:13px;'><span style='color:#22c55e;'>🟢</span> Ready</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 4. Feedback Summary
                with st.container(border=True):
                    st.markdown("##### 💬 Feedback Summary")
                    total_fb = len(all_feedback_list)
                    ups = sum(1 for e in all_feedback_list if e.get("feedback") == "up")
                    downs = sum(1 for e in all_feedback_list if e.get("feedback") == "down")
                    rate = f"{(ups / total_fb * 100):.1f}%" if total_fb > 0 else "0%"
                    
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("👍 Helpful", ups)
                    sc2.metric("👎 Unhelpful", downs)
                    sc3.metric("📈 Satisfaction", rate)
                    
                    if total_fb > 0:
                        st.progress(ups / total_fb)
                    else:
                        st.progress(0.0)

                # 6. RAG Summary
                with st.container(border=True):
                    st.markdown("##### ⚙️ RAG Summary")
                    st.markdown(f"""
                    <div style='display:flex; gap:8px; flex-wrap:wrap;'>
                        <span style='background:#dbeafe; color:#1e3a8a; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:600;'>Chunk Size: {get_setting('CHUNK_SIZE')}</span>
                        <span style='background:#dcfce7; color:#166534; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:600;'>Overlap: {get_setting('CHUNK_OVERLAP')}</span>
                        <span style='background:#fef08a; color:#854d0e; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:600;'>Retriever K: {get_setting('RETRIEVER_K')}</span>
                        <span style='background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:12px; font-size:13px; font-weight:600;'>Temp: {get_setting('LLM_TEMPERATURE')}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_users:
            all_users = db_repository.get_all_users()
            
            with st.container(border=True):
                st.markdown("<div class='ha-users-table-wrapper'></div>", unsafe_allow_html=True)
                st.markdown("""
                <style>
                /* Remove all vertical gap between Streamlit rows inside this specific container */
                div[data-testid="stVerticalBlockBorderWrapper"]:has(.ha-users-table-wrapper) > div[data-testid="stVerticalBlock"] {
                    gap: 0 !important;
                }
                
                /* Search box spacing adjustment */
                .ha-users-search-container {
                    margin-bottom: 8px;
                }
                
                /* Header row styling */
                div[data-testid="stHorizontalBlock"]:has(.ha-user-header) {
                    background-color: #eef2eb;
                    padding: 6px 12px;
                    border-bottom: 2px solid #5c6f5e;
                    border-radius: 4px 4px 0 0;
                    align-items: center;
                }
                
                /* Data row styling */
                div[data-testid="stHorizontalBlock"]:has(.ha-user-row) {
                    padding: 2px 12px;
                    border-bottom: 1px solid #e2e8f0;
                    height: 44px;
                    max-height: 44px;
                    align-items: center;
                    transition: background-color 0.2s;
                }
                div[data-testid="stHorizontalBlock"]:has(.ha-user-row):hover {
                    background-color: #f8fafc;
                }
                
                /* Compact action buttons */
                div[data-testid="stHorizontalBlock"]:has(.ha-user-row) button {
                    min-height: 28px !important;
                    height: 28px !important;
                    width: 28px !important;
                    padding: 0 !important;
                    font-size: 14px !important;
                    line-height: 1 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="ha-users-search-container">', unsafe_allow_html=True)
                search_query = st.text_input("Search users", key="admin_users_search", placeholder="Search by username...", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if search_query:
                    all_users = [u for u in all_users if search_query.lower() in u["username"].lower()]
                
                if not all_users:
                    st.info("No users found.")
                else:
                    # Table Header
                    h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
                    h1.markdown("<span class='ha-user-header' style='font-size: 13px; font-weight: 800; color: #3e4e35; text-transform: uppercase;'>Username</span>", unsafe_allow_html=True)
                    h2.markdown("<span style='font-size: 13px; font-weight: 800; color: #3e4e35; text-transform: uppercase;'>Role</span>", unsafe_allow_html=True)
                    h3.markdown("<span style='font-size: 13px; font-weight: 800; color: #3e4e35; text-transform: uppercase;'>Registered</span>", unsafe_allow_html=True)
                    h4.markdown("<span style='font-size: 13px; font-weight: 800; color: #3e4e35; text-transform: uppercase;'>Actions</span>", unsafe_allow_html=True)
                    
                    # Table Body
                    for u in all_users:
                        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                        
                        avatar = f"<div style='width: 24px; height: 24px; border-radius: 50%; background: #e2e8f0; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; color: #475569; margin-right: 8px; vertical-align: middle;'>{u['username'][0].upper()}</div>"
                        c1.markdown(f"<span class='ha-user-row'>{avatar}<span style='font-size: 14px; color: #1e293b; font-weight: 600; vertical-align: middle;'>{_html.escape(u['username'])}</span></span>", unsafe_allow_html=True)
                        
                        role_html = f"<span style='background:#dcfce7; color:#166534; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:700;'>Admin</span>" if u['role'] == 'admin' else f"<span style='font-size: 14px; color: #475569;'>{_html.escape(u['role'])}</span>"
                        c2.markdown(role_html, unsafe_allow_html=True)
                        
                        created_dt = str(u.get('created_at', '')).split(' ')[0]
                        c3.markdown(f"<span style='font-size: 14px; color: #475569;'>{_html.escape(created_dt)}</span>", unsafe_allow_html=True)
                        
                        with c4:
                            ac1, ac2 = st.columns(2)
                            with ac1:
                                with st.popover("👁️", help="View Profile"):
                                    profile = db_repository.get_user_profile(u['username']) or {}
                                    
                                    def _pv(val):
                                        if not val or (isinstance(val, str) and not val.strip()):
                                            return "<span style='color:#94a3b8; font-style:italic;'>Not provided</span>"
                                        return _html.escape(str(val))
                                        
                                    p_name = _pv(profile.get("name"))
                                    p_age = _pv(profile.get("age"))
                                    p_gender = _pv(profile.get("gender"))
                                    p_allergies = _pv(profile.get("allergies"))
                                    p_medical = _pv(profile.get("medical_conditions"))
                                    
                                    st.markdown(f"""
                                    <div style='background:#ffffff; border:1px solid #DCE5DA; border-radius:8px; padding:16px; min-width:240px;'>
                                        <h4 style='color:#4F6F52; margin-top:0; margin-bottom:12px; font-size:16px; border-bottom:1px solid #f1f5f9; padding-bottom:8px;'>User Profile</h4>
                                        <div style='margin-bottom:8px;'><span style='font-size:16px; margin-right:8px;'>👤</span><strong style='color:#475569; font-size:14px;'>Name:</strong> <div style='margin-left:28px; font-size:14px; color:#1e293b;'>{p_name}</div></div>
                                        <div style='margin-bottom:8px;'><span style='font-size:16px; margin-right:8px;'>🎂</span><strong style='color:#475569; font-size:14px;'>Age:</strong> <div style='margin-left:28px; font-size:14px; color:#1e293b;'>{p_age}</div></div>
                                        <div style='margin-bottom:8px;'><span style='font-size:16px; margin-right:8px;'>🚻</span><strong style='color:#475569; font-size:14px;'>Gender:</strong> <div style='margin-left:28px; font-size:14px; color:#1e293b;'>{p_gender}</div></div>
                                        <div style='margin-bottom:8px;'><span style='font-size:16px; margin-right:8px;'>🤧</span><strong style='color:#475569; font-size:14px;'>Allergies:</strong> <div style='margin-left:28px; font-size:14px; color:#1e293b;'>{p_allergies}</div></div>
                                        <div style='margin-bottom:4px;'><span style='font-size:16px; margin-right:8px;'>🏥</span><strong style='color:#475569; font-size:14px;'>Medical Conditions:</strong> <div style='margin-left:28px; font-size:14px; color:#1e293b;'>{p_medical}</div></div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            with ac2:
                                active_admin = _load_runtime_admin_credentials().get("username", "")
                                if u['username'] == active_admin:
                                    st.button("🗑️", disabled=True, key=f"del_d_{u['id']}", help="Cannot delete current admin")
                                else:
                                    with st.popover("🗑️", help="Delete User"):
                                        st.warning(f"Delete '{u['username']}'•")
                                        if st.button("Confirm", key=f"del_c_{u['id']}", type="primary"):
                                            db_repository.delete_user(u['username'])
                                            st.rerun()

        with tab_docs:
            st.markdown(
                '<h2 class="ha-admin-section-h" style="margin-top:0; margin-bottom: 1rem;">Document Management</h2>',
                unsafe_allow_html=True,
            )
            
            col_left, col_right = st.columns(2, gap="large")
            
            with col_left:
                # 1. PDF Library
                with st.container(border=True):
                    st.markdown("<h4 style='color:#3e4e35; margin-bottom: 12px;'>📚 PDF Library</h4>", unsafe_allow_html=True)
                    if pdf_files:
                        st.markdown("""
                        <style>
                        /* Remove border from inner scroll container */
                        div[data-testid="stVerticalBlockBorderWrapper"]:has(.ha-pdf-scroll-marker) {
                            border: none !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        with st.container(height=350):
                            st.markdown("<span class='ha-pdf-scroll-marker'></span>", unsafe_allow_html=True)
                            for p in pdf_files:
                                st.markdown(f"📄 **{_html.escape(p.name)}**")
                        st.markdown("<hr style='margin: 12px 0;'/>", unsafe_allow_html=True)
                        
                        selected_delete = st.multiselect(
                            get_string(lang, "delete_select"),
                            options=[p.name for p in pdf_files],
                            key="admin_pdf_delete_select",
                            label_visibility="collapsed",
                            placeholder="Select PDFs to delete..."
                        )
                        if st.button("🗑️ Delete Selected", use_container_width=True):
                            if selected_delete:
                                data_dir = Path(get_setting("DATA_DIR"))
                                removed = []
                                for filename in selected_delete:
                                    target = data_dir / filename
                                    if target.exists():
                                        target.unlink()
                                        removed.append(filename)
                                if removed:
                                    with st.spinner(get_string(lang, "delete_index_spinner")):
                                        delete_pdfs_from_index(removed)
                                    st.success(get_string(lang, "pdf_deleted").format(count=len(removed)))
                                    st.rerun()
                            else:
                                st.warning("Please select at least one PDF.")
                    else:
                        st.info(get_string(lang, "no_pdf"))

            with col_right:
                # 3. Vector Database
                with st.container(border=True):
                    st.markdown("<h4 style='color:#3e4e35; margin-bottom: 12px;'>🧠 Vector Database</h4>", unsafe_allow_html=True)
                    last_idx = _get_last_index_time()
                    
                    st.markdown(f"""
                    <div style='background-color:#f8fafc; padding:12px; border-radius:6px; margin-bottom:12px;'>
                        <div style='margin-bottom:4px;'><strong>Status:</strong> {'🟢 Active' if last_idx != 'Not indexed yet' else '🔴 Needs Indexing'}</div>
                        <div style='margin-bottom:4px;'><strong>Last Indexed:</strong> {last_idx}</div>
                        <div><strong>Total Documents Indexed:</strong> {len(pdf_files)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    if c1.button(get_string(lang, "index_new_btn"), use_container_width=True):
                        with st.spinner(get_string(lang, "index_new_spinner")):
                            stats = index_new_pdfs()
                        st.session_state.last_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        indexed = stats.get("indexed") or []
                        skipped = stats.get("skipped") or []
                        if indexed:
                            st.success(get_string(lang, "index_new_done").format(count=len(indexed), files=", ".join(indexed)))
                        else:
                            st.info(get_string(lang, "index_new_none").format(count=len(skipped)))
                    
                    if c2.button(get_string(lang, "reindex_btn"), type="primary", use_container_width=True):
                        with st.spinner(get_string(lang, "reindex_spinner")):
                            reindex_pdfs()
                        st.session_state.last_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success(get_string(lang, "db_rebuilt"))

                # 2. Upload PDF
                with st.container(border=True):
                    st.markdown("<h4 style='color:#3e4e35; margin-bottom: 12px;'>☁️ Upload PDF</h4>", unsafe_allow_html=True)
                    uploaded = st.file_uploader(
                        get_string(lang, "upload_btn"),
                        type=["pdf"],
                        accept_multiple_files=True,
                        key="admin_pdf_uploader",
                        label_visibility="collapsed"
                    )
                    if uploaded:
                        data_dir = Path(get_setting("DATA_DIR"))
                        for file in uploaded:
                            target = data_dir / file.name
                            with target.open("wb") as f:
                                f.write(file.getbuffer())
                        st.success(f"{len(uploaded)} PDF file(s) uploaded.")
                        st.rerun()

        with tab_feedback:
            _render_admin_feedback_log(lang=lang)

        with tab_settings:
            import os
            
            api_is_ok = os.environ.get("GROQ_API_KEY")
            api_status_text = "Connected" if api_is_ok else "Missing Key"
            api_color = "#22c55e" if api_is_ok else "#ef4444"

            st.markdown(f"""
            <h4 style='color:#3e4e35; margin-bottom: 10px; margin-top:0;'>🩺 System Health</h4>
            <div style="display: flex; gap: 12px; margin-bottom: 4px;">
                <div style="flex: 1; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                    <span style="font-size: 13px; font-weight: 600; color: #475569;">Chroma DB</span>
                    <span style="font-size: 12px; color: #0f172a; font-weight: 600; display:flex; align-items:center; gap:5px;">
                        <span style="height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span> Online
                    </span>
                </div>
                <div style="flex: 1; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                    <span style="font-size: 13px; font-weight: 600; color: #475569;">SQLite DB</span>
                    <span style="font-size: 12px; color: #0f172a; font-weight: 600; display:flex; align-items:center; gap:5px;">
                        <span style="height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span> Online
                    </span>
                </div>
                <div style="flex: 1; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                    <span style="font-size: 13px; font-weight: 600; color: #475569;">LLM API</span>
                    <span style="font-size: 12px; color: #0f172a; font-weight: 600; display:flex; align-items:center; gap:5px;">
                        <span style="height: 8px; width: 8px; background-color: {api_color}; border-radius: 50%; display: inline-block;"></span> {api_status_text}
                    </span>
                </div>
                <div style="flex: 1; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                    <span style="font-size: 13px; font-weight: 600; color: #475569;">Embedding API</span>
                    <span style="font-size: 12px; color: #0f172a; font-weight: 600; display:flex; align-items:center; gap:5px;">
                        <span style="height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span> Ready
                    </span>
                </div>
            </div>
            <hr style='margin: 16px 0;'/>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:#3e4e35; margin-bottom: 12px;'>⚙️ System Configuration</h4>", unsafe_allow_html=True)
            
            # 2-column layout for config items
            c1, c2 = st.columns(2, gap="large")
            
            with c1:
                with st.container(border=True):
                    st.markdown("##### 🤖 LLM Model")
                    st.code(get_setting("GROQ_MODEL"), language="text")
                
                with st.container(border=True):
                    st.markdown("##### 📁 Data Directories")
                    st.markdown(f"**PDF Directory:** `{get_setting('DATA_DIR')}`")
                    st.markdown(f"**Vector DB Path:** `{get_setting('CHROMA_DIR')}`")
                    st.markdown(f"**SQLite Path:** `{get_setting('DB_PATH')}`")
                
                with st.container(border=True):
                    st.markdown("##### 🔍 Search Configuration")
                    search_provider = st.session_state.get("web_search_provider", "Tavily")
                    st.markdown(f"**Provider:** {search_provider}")
                    st.markdown(f"**Trusted Domain Min Results:** `{get_setting('TRUSTED_DOMAIN_MIN_RESULTS')}`")
                    st.markdown("**Trusted Domains:**")
                    domains_html = "".join([f"<span style='background:#f1f5f9; border: 1px solid #e2e8f0; color:#475569; padding:2px 8px; border-radius:12px; font-size:12px; font-family:monospace;'>{domain}</span>" for domain in get_setting("TRUSTED_HERB_DOMAINS")])
                    st.markdown(f"<div style='display:flex; gap:6px; flex-wrap:wrap; margin-top:4px;'>{domains_html}</div>", unsafe_allow_html=True)

            with c2:
                with st.container(border=True):
                    st.markdown("##### 🧠 Embedding Model")
                    st.code(get_setting("EMBEDDING_MODEL"), language="text")
                
                with st.container(border=True):
                    st.markdown("##### ⚙️ Retrieval & Generation (RAG)")
                    
                    import json
                    _src = "config.py"
                    _sp = Path("settings.json")
                    if _sp.exists():
                        try:
                            with open(_sp, "r", encoding="utf-8") as _f:
                                if "RETRIEVER_K" in json.load(_f):
                                    _src = "settings.json"
                        except:
                            pass
                            
                    st.info(f"**Current settings source:** `{_src}`  \n"
                            f"• **CHUNK_SIZE:** `{get_setting('CHUNK_SIZE')}`  \n"
                            f"• **CHUNK_OVERLAP:** `{get_setting('CHUNK_OVERLAP')}`  \n"
                            f"• **RETRIEVER_K:** `{get_setting('RETRIEVER_K')}`  \n"
                            f"• **LLM_TEMPERATURE:** `{get_setting('LLM_TEMPERATURE')}`")
                            
                    with st.form("admin_rag_settings_form"):
                        st.markdown("<p style='font-size:13px; color:#475569; margin-bottom:12px; line-height:1.4;'>Configure document chunking and retrieval parameters.<br/><span style='color:#b45309; font-weight:600;'>Warning:</span> Changing Chunk Size or Overlap requires a <b>Full Re-index</b> on the Documents page to apply to existing files.</p>", unsafe_allow_html=True)
                        
                        col_rag1, col_rag2 = st.columns(2)
                        new_chunk_size = col_rag1.slider("Chunk Size", min_value=200, max_value=2000, value=get_setting("CHUNK_SIZE"), step=50)
                        new_overlap = col_rag2.slider("Chunk Overlap", min_value=50, max_value=500, value=get_setting("CHUNK_OVERLAP"), step=10)
                        
                        col_rag3, col_rag4 = st.columns(2)
                        new_k = col_rag3.slider("Retriever K", min_value=1, max_value=10, value=get_setting("RETRIEVER_K"), step=1)
                        new_temp = col_rag4.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=get_setting("LLM_TEMPERATURE"), step=0.1)
                        
                        submitted = st.form_submit_button("Save Settings", use_container_width=True)
                        if submitted:
                            save_settings({
                                "CHUNK_SIZE": new_chunk_size,
                                "CHUNK_OVERLAP": new_overlap,
                                "RETRIEVER_K": new_k,
                                "LLM_TEMPERATURE": new_temp
                            })
                            st.success("Settings saved successfully! They will take effect immediately. Remember to Full Re-index if Chunk parameters were modified.")
                            st.rerun()
                

