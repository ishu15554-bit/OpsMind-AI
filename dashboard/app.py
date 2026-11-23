import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="OpsMind AI | AIOps Platform",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)


# ==========================================
# 2. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []


# ==========================================
# 2A. DATABASE-BACKED PREDICTION HISTORY
# ==========================================

def fetch_prediction_history(limit=1000):
    """Fetch saved predictions from persistent SQLite database."""

    try:
        response = requests.get(
            API_URL + "/predictions/history",
            params={"limit": limit},
            timeout=10
        )

        response.raise_for_status()
        payload = response.json()

        records = payload.get("history", [])
        history_df = pd.DataFrame(records)

        if history_df.empty:
            return history_df, None

        history_df = history_df.rename(columns={
            "created_at": "Time",
            "model_name": "Model",
            "input_text": "Input",
            "prediction": "Prediction",
            "id": "ID"
        })

        if "Time" in history_df.columns:
            history_df["Time"] = pd.to_datetime(
                history_df["Time"],
                errors="coerce"
            )

        return history_df, None

    except requests.RequestException as error:
        return pd.DataFrame(), str(error)

    except (ValueError, TypeError) as error:
        return pd.DataFrame(), f"Invalid history response: {error}"


history_df, history_error = fetch_prediction_history()

total_predictions = len(history_df)


# ==========================================
# 3. PROFESSIONAL DARK THEME
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{--bg:#070a16;--panel:#10162b;--panel2:#171d38;--border:#30345d;--purple:#a855f7;--cyan:#22c7ff;--green:#16d6a0;--orange:#f5a623;--text:#eef0ff;--muted:#a4aecb;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;color:var(--text)!important;}
.stApp{background:radial-gradient(ellipse at 78% 8%,rgba(107,52,220,.20),transparent 32%),radial-gradient(ellipse at 15% 85%,rgba(49,80,190,.13),transparent 35%),linear-gradient(135deg,#070a16 0%,#0b1022 55%,#080b19 100%)!important;animation:appFadeIn .7s ease both;}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important}[data-testid="stToolbar"]{display:none;}
.block-container{max-width:1600px;padding:1.05rem 2rem 2.3rem;}
h1,h2,h3,h4{color:#f3f1ff!important;font-weight:800!important;letter-spacing:-.55px!important;}h1{text-shadow:0 0 22px rgba(168,85,247,.22);}h2{font-size:1.65rem!important;}h3{font-size:1.2rem!important;}
p,label,.stCaption,[data-testid="stCaption"]{color:var(--muted)!important;}hr{border-color:rgba(143,105,255,.25)!important;}
/* SIDEBAR / BRAND */
section[data-testid="stSidebar"]{background:radial-gradient(ellipse at 50% 100%,rgba(118,64,255,.25),transparent 38%),linear-gradient(180deg,#090d1b,#10152a 75%,#0b1020)!important;border-right:1px solid #29284c!important;}
section[data-testid="stSidebar"]>div{padding:1.05rem .85rem;}
section[data-testid="stSidebar"] h2{font-size:1.18rem!important;text-align:center;text-shadow:0 0 18px rgba(168,85,247,.55);animation:softGlow 4s ease-in-out infinite;}
section[data-testid="stSidebar"] h2:before{content:'✦';display:block;color:#b879ff;font-size:1.65rem;line-height:1.2;text-shadow:0 0 15px #a855f7;}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{font-size:.78rem;}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:7px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label{background:rgba(19,25,48,.82);border:1px solid rgba(77,75,125,.35);border-radius:9px;padding:8px 10px;transition:all .22s ease;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(106,61,194,.24);border-color:rgba(168,85,247,.8);transform:translateX(3px);box-shadow:0 0 16px rgba(137,75,255,.14);}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:linear-gradient(100deg,rgba(126,53,220,.55),rgba(72,45,153,.42));border-color:#9b5cff;box-shadow:0 0 18px rgba(137,75,255,.22);}
/* KPI CARDS */
div[data-testid="stMetric"]{position:relative;overflow:hidden;background:radial-gradient(ellipse at 90% 100%,rgba(168,85,247,.18),transparent 45%),linear-gradient(135deg,#19203c,#0c1226)!important;border:1px solid #34365f!important;border-radius:14px!important;padding:17px 18px!important;box-shadow:0 8px 25px rgba(0,0,0,.25);transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;animation:cardRise .65s ease both;}
div[data-testid="stMetric"]:after{content:'';position:absolute;width:48%;height:70%;right:-8%;bottom:-42%;border-radius:50%;background:rgba(139,92,246,.23);filter:blur(18px);pointer-events:none;}
div[data-testid="stMetric"]:hover{transform:translateY(-4px);border-color:var(--purple)!important;box-shadow:0 9px 28px rgba(124,58,237,.22);}
div[data-testid="stMetricLabel"]{color:#aeb8d7!important;}div[data-testid="stMetricValue"]{color:#fff!important;font-weight:800!important;}
/* INPUTS */
.stTextArea textarea,.stTextInput input,div[data-baseweb="select"]>div{background:linear-gradient(135deg,#151c36,#10172d)!important;color:#f5f3ff!important;border:1px solid #4a4777!important;border-radius:10px!important;transition:border-color .2s ease,box-shadow .2s ease;}
.stTextArea textarea:focus,.stTextInput input:focus,div[data-baseweb="select"]>div:focus-within{border-color:var(--purple)!important;box-shadow:0 0 0 2px rgba(168,85,247,.16),0 0 18px rgba(124,58,237,.12)!important;}
.stTextArea textarea::placeholder,.stTextInput input::placeholder{color:#8792b1!important;}
/* BUTTONS */
.stButton>button{background:linear-gradient(110deg,#a855f7,#7138e8 55%,#5426c8)!important;color:#fff!important;border:1px solid rgba(208,174,255,.42)!important;border-radius:10px!important;padding:.62rem 1.25rem!important;font-weight:700!important;box-shadow:0 5px 18px rgba(123,64,225,.28);transition:all .22s ease;}
.stButton>button:hover{transform:translateY(-2px);filter:brightness(1.12);box-shadow:0 8px 26px rgba(137,75,255,.42);}.stButton>button:active{transform:translateY(0) scale(.98);}
/* CUSTOM / RESULT PANELS */
.custom-card,.result-card{background:radial-gradient(circle at 90% 0%,rgba(123,58,237,.15),transparent 38%),linear-gradient(135deg,rgba(25,30,62,.97),rgba(13,19,40,.98));border:1px solid #454071;border-radius:13px;padding:19px;margin:12px 0;box-shadow:0 8px 25px rgba(0,0,0,.22);transition:all .25s ease;animation:cardRise .6s ease both;}
.custom-card:hover,.result-card:hover{border-color:rgba(168,85,247,.85);box-shadow:0 0 24px rgba(139,92,246,.17);transform:translateY(-2px);}.result-card{border-color:#8b5cf6;box-shadow:0 0 22px rgba(155,92,255,.16);}
.result-value{font-size:27px;font-weight:800;color:#c49aff;margin-top:8px;text-shadow:0 0 18px rgba(168,85,247,.35);}.muted{color:var(--muted);font-size:13px;}
/* ANALYTICS + CHARTS */
[data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"],[data-testid="stPlotlyChart"]{background:linear-gradient(135deg,#11182e,#0d1327)!important;border:1px solid #34365f!important;border-radius:14px!important;padding:10px!important;box-shadow:0 8px 24px rgba(0,0,0,.2);transition:all .25s ease;}
[data-testid="stVegaLiteChart"]:hover,[data-testid="stPlotlyChart"]:hover{border-color:rgba(168,85,247,.7)!important;box-shadow:0 0 20px rgba(168,85,247,.14);}
/* HISTORY TABLES */
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #454071!important;border-radius:12px!important;overflow:hidden!important;box-shadow:0 8px 24px rgba(0,0,0,.22);}
[data-testid="stDataFrame"]{background:#10162b!important;}
[data-testid="stDataFrame"] *{color:#e8eaff!important;}
/* CHAT */
[data-testid="stChatMessage"]{background:rgba(24,29,54,.76);border:1px solid #303455;border-radius:12px;}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li{color:#edf0ff!important;}
[data-testid="stChatInput"]{background:#111831!important;border:1px solid #45416d!important;border-radius:14px!important;box-shadow:0 8px 28px rgba(0,0,0,.25)!important;transition:all .2s ease;}
[data-testid="stChatInput"]:focus-within{border-color:var(--purple)!important;box-shadow:0 0 0 2px rgba(168,85,247,.16)!important;}
[data-testid="stChatInput"] textarea{color:#fff!important;-webkit-text-fill-color:#fff!important;caret-color:#c49aff!important;}
[data-testid="stChatInput"] textarea::placeholder{color:#8e99b7!important;-webkit-text-fill-color:#8e99b7!important;}
[data-testid="stChatInput"] button{background:linear-gradient(110deg,#a855f7,#6636dc)!important;color:white!important;border-radius:9px!important;}
[data-testid="stAlert"]{border-radius:10px!important;}
.brand-logo{filter:drop-shadow(0 0 9px rgba(168,85,247,.65));transition:transform .3s ease,filter .3s ease;}.brand-logo:hover{transform:scale(1.06);filter:drop-shadow(0 0 16px rgba(168,85,247,.9));}
@keyframes appFadeIn{from{opacity:0}to{opacity:1}}
@keyframes cardRise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
@keyframes softGlow{0%,100%{box-shadow:0 0 8px rgba(168,85,247,.10)}50%{box-shadow:0 0 22px rgba(168,85,247,.24)}}
@media(max-width:900px){.block-container{padding:1rem .9rem 2rem;}}
@media(max-width:600px){h1{font-size:1.7rem!important}h2{font-size:1.35rem!important}div[data-testid="stMetric"]{padding:13px!important;}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important;}}
</style>
""", unsafe_allow_html=True)



# ==========================================
# 4. BACKEND CONNECTION
# ==========================================

def check_backend():
    try:
        response = requests.get(API_URL, timeout=3)
        return response.status_code == 200

    except requests.RequestException:
        return False


backend_online = check_backend()


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def run_prediction(model, input_text):

    endpoints = {
        "ITSM Ticket Classification": (
            "/predict/ticket", "ticket_text"
        ),
        "HDFS Anomaly Detection": (
            "/predict/hdfs", "log_text"
        ),
        "OpenStack Anomaly Detection": (
            "/predict/openstack", "log_text"
        )
    }

    endpoint, field_name = endpoints[model]

    payload = {field_name: input_text}

    response = requests.post(
        API_URL + endpoint,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()["prediction"]


# ==========================================
# 5A. RAG API FUNCTION
# ==========================================

def ask_rag(question):

    response = requests.post(
        API_URL + "/rag/ask",
        json={"log_text": question},
        timeout=120
    )

    response.raise_for_status()

    return response.json()


# ==========================================
# 6. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## OpsMind AI")
    st.caption("ENTERPRISE AIOPS PLATFORM")

    st.divider()

    st.markdown("##### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Predict Incident",
            "RAG Assistant",
            "Analytics",
            "Logs & History",
            "Model Status",
            "System Health",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("##### BACKEND STATUS")

    if backend_online:
        st.success("API Connected")
    else:
        st.error("API Disconnected")

    st.divider()

    st.caption("OpsMind AI | Version 1.0")
    st.caption("Intelligent Incident Platform")


# ==========================================
# 7. HEADER AND KPI CARDS
# ==========================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.title("OpsMind AI")
    st.caption(
        "Enterprise AIOps Incident Intelligence Platform"
    )

with header_right:
    if backend_online:
        st.success("SYSTEM ONLINE")
    else:
        st.error("SYSTEM OFFLINE")

    st.caption(datetime.now().strftime("%d %B %Y"))

st.divider()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("AI Models", "03")

with k2:
    st.metric("Predictions Run", total_predictions)

with k3:
    st.metric(
        "API Status",
        "Online" if backend_online else "Offline"
    )

with k4:
    st.metric("Prediction History", total_predictions)

if history_error:
    st.warning(
        "Prediction history service is currently unavailable. "
        "KPI history values may be incomplete."
    )

st.divider()


# ==========================================
# 8. PREDICTION WORKSPACE
# ==========================================

if page == "Predict Incident":

    st.header("Prediction Workspace")

    st.caption(
        "Select an AI model and submit incident or log data."
    )

    selected_model = st.selectbox(
        "Select AI Model",
        [
            "ITSM Ticket Classification",
            "HDFS Anomaly Detection",
            "OpenStack Anomaly Detection"
        ]
    )

    descriptions = {
        "ITSM Ticket Classification":
            "Classify incoming IT service desk tickets.",

        "HDFS Anomaly Detection":
            "Detect abnormal HDFS event sequences.",

        "OpenStack Anomaly Detection":
            "Identify abnormal OpenStack system log messages."
    }

    st.markdown(
        f"""
        <div class="custom-card">
            <h4>{selected_model}</h4>
            <p class="muted">{descriptions[selected_model]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    input_text = st.text_area(
        "Incident / Log Details",
        height=150,
        placeholder="Enter incident description or log details..."
    )

    if st.button("Run Prediction", use_container_width=False):

        if not backend_online:
            st.error(
                "Backend is offline. Start the FastAPI server first."
            )

        elif not input_text.strip():
            st.warning("Please enter incident or log details.")

        else:
            try:
                with st.spinner("Analyzing input with AI model..."):

                    result = run_prediction(
                        selected_model,
                        input_text.strip()
                    )

                st.session_state.history.append({
                    "Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),
                    "Model": selected_model,
                    "Input": input_text.strip(),
                    "Prediction": str(result)
                })

                st.success("Prediction completed successfully.")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <p class="muted">PREDICTION RESULT</p>
                        <h3>{selected_model}</h3>
                        <div class="result-value">
                            {result}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Refresh database-backed history after prediction.
                history_df, history_error = fetch_prediction_history()
                total_predictions = len(history_df)

            except requests.RequestException as error:
                st.error(f"Prediction request failed: {error}")

            except (KeyError, ValueError):
                st.error("Unexpected response received from backend.")


# ==========================================
# 8A. RAG ASSISTANT
# ==========================================

elif page == "RAG Assistant":

    st.header("OpsMind AI - RAG Assistant")

    st.caption(
        "AI-powered IT troubleshooting using "
        "your knowledge base and Groq LLM."
    )

    st.markdown(
        """
        <div class="custom-card">
            <h4>Intelligent Troubleshooting Assistant</h4>
            <p class="muted">
                Ask questions about password reset, network connectivity,
                email service and IT troubleshooting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    for message in st.session_state.rag_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):
                st.caption(
                    "Sources: " + ", ".join(message["sources"])
                )

    question = st.chat_input(
        "Ask an IT troubleshooting question..."
    )

    if question:

        st.session_state.rag_messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            if not backend_online:

                st.error(
                    "Backend is offline. Start FastAPI first."
                )

            else:

                try:

                    with st.spinner(
                        "OpsMind AI is analyzing your question..."
                    ):

                        result = ask_rag(question)

                    answer = result.get(
                        "answer",
                        "No answer received from AI."
                    )

                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:

                        st.markdown("**Source Documents**")

                        for source in sources:
                            st.write(f"- {source}")

                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except requests.RequestException as error:

                    st.error(
                        f"RAG API request failed: {error}"
                    )

                except Exception as error:

                    st.error(
                        f"Unexpected RAG error: {error}"
                    )

    if st.button("Clear Chat"):

        st.session_state.rag_messages = []

        st.rerun()


# ==========================================
# 9. ANALYTICS
# ==========================================

elif page == "Analytics":

    st.header("Prediction Analytics")

    if history_error:

        st.error(
            f"Could not load prediction analytics: {history_error}"
        )

    elif not history_df.empty:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Predictions",
                len(history_df)
            )

        with col2:
            st.metric(
                "Models Used",
                history_df["Model"].nunique()
                if "Model" in history_df.columns else 0
            )

        st.subheader("Predictions by Model")

        if "Model" in history_df.columns:
            model_counts = history_df["Model"].value_counts()
            st.bar_chart(model_counts)

        st.subheader("Recent Predictions")

        display_columns = [
            column for column in ["Time", "Model", "Prediction"]
            if column in history_df.columns
        ]

        st.dataframe(
            history_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No prediction history available yet. "
            "Run a prediction to populate analytics."
        )


# ==========================================
# 10. LOGS AND HISTORY
# ==========================================

elif page == "Logs & History":

    st.header("Prediction History")

    st.caption(
        "Persistent prediction records loaded from the "
        "OpsMind AI database."
    )

    if history_error:

        st.error(
            f"Could not load prediction history: {history_error}"
        )

    elif not history_df.empty:

        st.metric(
            "Total Saved Records",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "History is stored in the backend database. "
            "Records are preserved across dashboard and API restarts."
        )

    else:

        st.info(
            "No prediction history available in the database."
        )


# ==========================================
# 11. MODEL STATUS
# ==========================================

elif page == "Model Status":

    st.header("AI Model Status")

    models = [
        ("ITSM Ticket Classifier", "Ticket classification"),
        ("HDFS Anomaly Detector", "HDFS event sequence analysis"),
        ("OpenStack Anomaly Detector", "OpenStack log analysis")
    ]

    for name, description in models:

        st.markdown(
            f"""
            <div class="custom-card">
                <h4>{name}</h4>
                <p class="muted">{description}</p>
                <p style="color:#34d399;">Configured</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# 12. SYSTEM HEALTH
# ==========================================

elif page == "System Health":

    st.header("System Health")

    if backend_online:
        st.success("FastAPI backend is responding.")
    else:
        st.error("FastAPI backend is not responding.")

    st.markdown(
        f"""
        <div class="custom-card">
            <p class="muted">BACKEND URL</p>
            <h4>{API_URL}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Refresh Status"):
        st.rerun()


# ==========================================
# 13. ABOUT
# ==========================================

elif page == "About":

    st.header("About OpsMind AI")

    st.write(
        "OpsMind AI is an AI-powered enterprise "
        "IT operations intelligence platform."
    )

    st.markdown("""
    **Core capabilities**

    - ITSM ticket classification
    - HDFS log anomaly detection
    - OpenStack log anomaly detection
    - Machine learning prediction services
    - FastAPI backend integration
    - RAG-based IT troubleshooting assistant

    **Technology Stack**

    - Python and Scikit-learn
    - FastAPI
    - Streamlit
    - Pandas
    - FAISS Vector Search
    - Sentence Transformers
    - Groq LLM
    """)


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "OPSMIND AI  |  ENTERPRISE AIOPS INTELLIGENCE PLATFORM"
)