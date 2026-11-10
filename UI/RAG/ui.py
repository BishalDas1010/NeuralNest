"""ui.py — Streamlit UI. Talks to FastAPI over HTTP only."""

import os
import uuid
import requests
import streamlit as st

API = os.getenv("API_BASE", "http://localhost:8000/api")
TIMEOUT = 180  # generous for LLM + embeddings

# ---------------------------------------------------------------------------
# SESSION
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
SID = st.session_state.session_id

for _k, _v in {
    "messages": [],
    "test_questions": None,
    "test_results": None,
    "active_page": "Home",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def fetch_status() -> dict:
    try:
        r = requests.get(f"{API}/session/{SID}/status", timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"has_kb": False, "documents": []}


# ---------------------------------------------------------------------------
# PAGE CONFIG + STYLES
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyRAG AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; }
    .feature-card { border-radius: 12px; padding: 20px; height: 100%; transition: transform 0.2s; }
    .feature-card:hover { transform: translateY(-2px); }
    .card-purple { background-color: #F4EFFF; border: 1px solid #E5D5FF; }
    .card-red { background-color: #FFF0F0; border: 1px solid #FFD5D5; }
    .card-blue { background-color: #EFF6FF; border: 1px solid #D5E5FF; }
    .card-green { background-color: #EFFFF4; border: 1px solid #D5FFE5; }
    .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; color: #1a1a1a; }
    .result-card { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .result-pass { background-color: #EFFFF4; border: 2px solid #34D399; }
    .result-fail { background-color: #FFF0F0; border: 2px solid #F87171; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------
def render_home():
    st.markdown("## ✨ AI Study Assistant")
    st.markdown(
        "<p style='color: gray; margin-bottom: 2rem;'>Upload your notes, learn with AI, and take mock tests.</p>",
        unsafe_allow_html=True,
    )

    status = fetch_status()
    # FIX: Safe dictionary access
    if status.get("has_kb", False):
        st.success(f"✅ **Active Knowledge Base:** {len(status.get('documents', []))} document(s) loaded.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="feature-card card-purple"><div style="font-size: 2rem;">📝</div><div class="card-title">Text Upload</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card card-red"><div style="font-size: 2rem;">📄</div><div class="card-title">PDF Upload</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card card-blue"><div style="font-size: 2rem;">💬</div><div class="card-title">AI Chat</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="feature-card card-green"><div style="font-size: 2rem;">📋</div><div class="card-title">Mock Test</div></div>',
            unsafe_allow_html=True,
        )


def render_upload():
    st.markdown("## 📥 Upload & Learn")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

    if st.button("Process & Build Vector DB", type="primary") and uploaded:
        with st.status("Uploading & indexing...", expanded=True) as w:
            try:
                files = {"file": (uploaded.name, uploaded.getvalue())}
                r = requests.post(
                    f"{API}/upload",
                    params={"session_id": SID},
                    files=files,
                    timeout=TIMEOUT,
                )
                if not r.ok:
                    raise RuntimeError(f"{r.status_code}: {r.text}")
                data = r.json()
                w.update(label="Complete!", state="complete", expanded=False)
                # FIX: Safe dictionary access for upload response
                st.success(
                    f"Indexed **{data.get('filename', 'file')}** — {data.get('pages', 1)} pages, "
                    f"{data.get('chunks', 0)} chunks. Go to **AI Chat** or **Mock Test**."
                )
            except Exception as e:
                w.update(label="Failed", state="error")
                st.error(f"Upload failed: {e}")


def render_chat():
    st.markdown("## 💬 AI Chat")
    # FIX: Safe dictionary access
    if not fetch_status().get("has_kb", False):
        st.warning("⚠️ Please upload a document first.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    r = requests.post(
                        f"{API}/chat",
                        json={"session_id": SID, "question": prompt},
                        timeout=TIMEOUT,
                    )
                    if not r.ok:
                        raise RuntimeError(f"{r.status_code}: {r.text}")
                    answer = r.json()["answer"]
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


def render_mock_test():
    st.markdown("## 📋 AI Mock Test")
    # FIX: Safe dictionary access
    if not fetch_status().get("has_kb", False):
        st.warning("⚠️ Please upload a document in 'Upload & Learn' first.")
        return

    # STATE 1 — Setup
    if st.session_state.test_questions is None and st.session_state.test_results is None:
        st.info("The AI will analyze your document and generate a test.")
        num_q = st.radio("How many questions?", [5, 10, 20], horizontal=True)

        if st.button("🚀 Generate Test", type="primary"):
            with st.spinner(f"AI is generating {num_q} questions..."):
                try:
                    r = requests.post(
                        f"{API}/mock-test/generate",
                        json={"session_id": SID, "num_questions": num_q},
                        timeout=TIMEOUT,
                    )
                    if not r.ok:
                        raise RuntimeError(f"{r.status_code}: {r.text}")
                    st.session_state.test_questions = r.json()["questions"]
                    st.session_state.test_results = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    # STATE 2 — Taking the test
    elif st.session_state.test_questions and st.session_state.test_results is None:
        st.write(f"### Answer the following {len(st.session_state.test_questions)} questions:")

        with st.form("mock_test_form"):
            answers = {}
            for i, q in enumerate(st.session_state.test_questions):
                st.markdown(f"**Q{i + 1}: {q['question']}**")
                answers[i] = st.radio(
                    "Select an answer:",
                    options=q["options"],
                    index=None,
                    key=f"q_{i}",
                    label_visibility="collapsed",
                )
                st.write("---")

            submitted = st.form_submit_button("✅ Submit Test", type="primary")

            if submitted:
                if any(a is None for a in answers.values()):
                    st.warning("⚠️ Please answer all questions before submitting.")
                else:
                    try:
                        r = requests.post(
                            f"{API}/mock-test/submit",
                            json={
                                "session_id": SID,
                                "answers": [answers[i] for i in range(len(answers))],
                            },
                            timeout=TIMEOUT,
                        )
                        if not r.ok:
                            raise RuntimeError(f"{r.status_code}: {r.text}")
                        st.session_state.test_results = r.json()
                        st.session_state.test_questions = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Submit failed: {e}")

    # STATE 3 — Results
    else:
        res = st.session_state.test_results
        css_class = "result-pass" if res["passed"] else "result-fail"
        label = "Pass" if res["passed"] else "Fail"
        st.markdown(
            f"""
        <div class="result-card {css_class}">
            <h2>Your Score: {res['score']} / {res['total']} ({res['percentage']:.1f}%)</h2>
            <h3>{label}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔍 Review Your Answers")
        for i, d in enumerate(res["results"]):
            emoji = "✅" if d["is_correct"] else "❌"
            with st.expander(f"Q{i + 1}: {d['question']} {emoji}"):
                user_text = (
                    d["options"][d["user_index"]]
                    if d["user_index"] is not None
                    else "Not Answered"
                )
                correct_text = d["options"][d["correct_index"]]
                st.write(f"**Your Answer:** {user_text}")
                st.write(f"**Correct Answer:** {correct_text}")
                if d["is_correct"]:
                    st.success("Correct!")
                else:
                    st.error("Incorrect.")

        if st.button("🔄 Take Another Test", type="primary"):
            st.session_state.test_questions = None
            st.session_state.test_results = None
            st.rerun()


# ---------------------------------------------------------------------------
# SIDEBAR & ROUTER
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 StudyRAG AI")
    st.divider()

    pages = {
        "Home": "🏠 Home",
        "Upload & Learn": "📤 Upload & Learn",
        "AI Chat": "💬 AI Chat",
        "Mock Test": "📋 Mock Test",
    }
    for key, label in pages.items():
        if st.button(
            label,
            use_container_width=True,
            type="primary" if st.session_state.active_page == key else "secondary",
        ):
            st.session_state.active_page = key
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        try:
            requests.delete(f"{API}/session/{SID}", timeout=15)
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.test_questions = None
        st.session_state.test_results = None
        st.rerun()


if st.session_state.active_page == "Home":
    render_home()
elif st.session_state.active_page == "Upload & Learn":
    render_upload()
elif st.session_state.active_page == "AI Chat":
    render_chat()
elif st.session_state.active_page == "Mock Test":
    render_mock_test()