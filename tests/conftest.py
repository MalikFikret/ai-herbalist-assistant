"""Test configuration.

We add ``src/`` to ``sys.path`` so tests can import the package without an
editable install, and install minimal stubs for Streamlit + the heavy
third-party libraries so the UI modules can be imported offline. Tests
exercise only pure-Python helpers; no UI rendering or real LLM / vector
store call is triggered.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep the test suite offline: any code path that reads these env vars at
# import time should still see SOMETHING rather than raise.
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("LANGSMITH_TRACING", "false")


def _make_module(name: str, attrs: dict | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _install_streamlit_stub() -> None:
    """Install a minimal Streamlit stub so UI modules can import offline."""
    if "streamlit" in sys.modules:
        return

    fake = types.ModuleType("streamlit")

    class _Session(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    fake.session_state = _Session()

    def _noop(*_a, **_kw):
        return None

    def _identity_decorator(*args, **_kw):
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap

    fake.markdown = _noop
    fake.toast = _noop
    fake.rerun = _noop
    fake.cache_resource = _identity_decorator
    fake.cache_data = _identity_decorator

    components_pkg = types.ModuleType("streamlit.components")
    components_v1 = types.ModuleType("streamlit.components.v1")
    components_v1.html = _noop
    components_pkg.v1 = components_v1
    fake.components = components_pkg

    sys.modules["streamlit"] = fake
    sys.modules["streamlit.components"] = components_pkg
    sys.modules["streamlit.components.v1"] = components_v1


def _install_heavy_dependency_stubs() -> None:
    """Stub heavy third-party modules imported at package load time.

    ``streamlit_app`` indirectly imports ``chromadb`` / ``langchain_chroma`` /
    ``sentence_transformers`` through the resources layer. The unit tests
    never exercise those paths, so we replace them with lightweight
    placeholders that are import-compatible.
    """
    if "langchain_chroma" not in sys.modules:
        class _Chroma:
            def __init__(self, *_a, **_kw):
                pass

            @classmethod
            def from_documents(cls, *_a, **_kw):
                return cls()

            def as_retriever(self, *_a, **_kw):
                return self

        _make_module("langchain_chroma", {"Chroma": _Chroma})

    if "langchain_groq" not in sys.modules:
        class _ChatGroq:
            def __init__(self, *_a, **_kw):
                pass

            def invoke(self, *_a, **_kw):
                raise RuntimeError("ChatGroq stub is not callable in unit tests")

            def with_structured_output(self, *_a, **_kw):
                return self

        _make_module("langchain_groq", {"ChatGroq": _ChatGroq})

    if "langchain_huggingface" not in sys.modules:
        class _HFEmbeddings:
            def __init__(self, *_a, **_kw):
                pass

        _make_module("langchain_huggingface", {"HuggingFaceEmbeddings": _HFEmbeddings})

    if "langchain_community" not in sys.modules:
        _make_module("langchain_community")
        doc_loaders = _make_module("langchain_community.document_loaders")

        class _PyPDFLoader:
            def __init__(self, *_a, **_kw):
                pass

            def load(self):
                return []

        doc_loaders.PyPDFLoader = _PyPDFLoader

    if "langchain_text_splitters" not in sys.modules:
        class _RecursiveCharacterTextSplitter:
            def __init__(self, *_a, **_kw):
                pass

            def split_documents(self, docs):
                return docs

        _make_module(
            "langchain_text_splitters",
            {"RecursiveCharacterTextSplitter": _RecursiveCharacterTextSplitter},
        )


_install_streamlit_stub()
_install_heavy_dependency_stubs()
