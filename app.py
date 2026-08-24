"""
SpamGuard AI - Email Spam Classifier Web App
"""

import re
import os
import json
import string
import time
import datetime
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SpamGuard AI | Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Always resolve relative to this script's own folder, regardless of the
# directory the app was launched from, so saved data is always found reliably.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data.json")

# =========================================================
# PERSISTENT STORAGE HELPERS
# =========================================================
def load_all_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def save_all_data(data):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

def persist_user_data():
    all_data = load_all_data()
    all_data.setdefault("accounts", {})
    all_data["accounts"] = st.session_state.accounts
    all_data.setdefault("users", {})
    all_data["users"][st.session_state.username] = {
        "history": st.session_state.history,
        "notifications": st.session_state.notifications,
        "theme": st.session_state.theme,
        "language": st.session_state.language,
    }
    save_all_data(all_data)


# =========================================================
# SESSION STATE INIT
# =========================================================
_all_data = load_all_data()

defaults = {
    "logged_in": False,
    "username": "",
    "auth_page": "login",
    "accounts": _all_data.get("accounts", {"demo": "demo123"}),
    "history": [],
    "notifications": [],
    "theme": "light",
    "language": "en",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# TRANSLATIONS (English / Urdu script)
# =========================================================
T = {
    "en": {
        "app_name": "SpamGuard AI",
        "tagline": "Smart Email & SMS Spam Detection",
        "nav_home": "Home",
        "nav_classify": "Classify Message",
        "nav_performance": "Model Performance",
        "nav_history": "History",
        "nav_about": "How It Works",
        "nav_notifications": "Notifications",
        "nav_settings": "Settings",
        "logout": "Logout",
        "welcome": "Welcome back",
        "sign_in": "Sign In",
        "create_account": "Create Account",
        "username": "Username",
        "password": "Password",
        "confirm_password": "Confirm Password",
        "email": "Email",
        "no_account": "Don't have an account?",
        "have_account": "Already have an account?",
        "go_signup": "Create one",
        "go_login": "Sign in",
        "signup_success": "Account created! Please sign in.",
        "login_fail": "Invalid username or password.",
        "fill_fields": "Please fill in all fields.",
        "password_mismatch": "Passwords do not match.",
        "user_exists": "That username is already taken.",
        "classify_title": "Classify a Message",
        "classify_sub": "Paste an email or SMS below to check whether it's spam",
        "quick_examples": "Quick examples",
        "custom_option": "-- Type your own --",
        "message_label": "Message text",
        "message_placeholder": "Type or paste the email/SMS content here...",
        "classify_btn": "Classify Message",
        "warning_empty": "Please enter a message first.",
        "analyzing": "Analyzing message...",
        "spam_detected": "SPAM DETECTED",
        "not_spam": "NOT SPAM",
        "confidence": "Confidence score",
        "confidence_note_spam": "(higher = more confident)",
        "confidence_note_ham": "(more negative = more confident it's not spam)",
        "see_cleaned": "See preprocessed text (what the model actually saw)",
        "performance_title": "Model Performance",
        "performance_sub": "Comparison of all trained models on the test set",
        "chart_title": "Model Comparison Chart",
        "cm_title": "Confusion Matrix (Best Model)",
        "best_model_note": "was selected as the best model based on F1-score, balancing precision (avoiding false spam flags) and recall (catching actual spam).",
        "history_title": "History",
        "history_sub": "All messages you've classified, saved permanently to your account",
        "no_history": "No messages classified yet. Go to Classify Message to try it out.",
        "clear_history": "Clear History",
        "about_title": "How It Works",
        "about_sub": "Understanding the spam detection pipeline",
        "notif_title": "Notifications",
        "notif_sub": "Recent alerts from your classification activity",
        "no_notif": "No notifications yet.",
        "clear_notif": "Clear Notifications",
        "settings_title": "Settings",
        "settings_sub": "Customize your app experience",
        "appearance": "Appearance",
        "theme_label": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language_label": "Language",
        "account": "Account",
        "logged_in_as": "Logged in as",
        "metric_accuracy": "Model Accuracy",
        "metric_f1": "F1-Score",
        "metric_best": "Best Model",
        "metric_checked": "Total Messages Checked",
        "metric_spam": "Spam Caught",
        "metric_ham": "Safe Messages",
        "nav_hint": "Use the sidebar to check a new message, review your history, or explore how the model performs.",
        "recent_activity": "Recent Activity",
        "no_activity": "Nothing checked yet — try the Classify Message page.",
        "quick_tips": "Tips for Spotting Spam",
        "tip1": "Be cautious of urgent language demanding immediate action.",
        "tip2": "Unexpected prize or lottery notifications are almost always spam.",
        "tip3": "Check sender links carefully before clicking hover to preview the URL.",
        "tip4": "Legitimate companies rarely ask for passwords or card details by message.",
        "breakdown": "Spam vs Safe Breakdown",
    },
    "ur": {
        "app_name": "سپیم گارڈ اے آئی",
        "tagline": "ای میل اور ایس ایم ایس کے لیے ذہین سپیم شناخت",
        "nav_home": "ہوم",
        "nav_classify": "پیغام کی جانچ کریں",
        "nav_performance": "ماڈل کی کارکردگی",
        "nav_history": "سابقہ ریکارڈ",
        "nav_about": "یہ کیسے کام کرتا ہے",
        "nav_notifications": "اطلاعات",
        "nav_settings": "ترتیبات",
        "logout": "لاگ آؤٹ",
        "welcome": "خوش آمدید",
        "sign_in": "سائن ان",
        "create_account": "اکاؤنٹ بنائیں",
        "username": "صارف کا نام",
        "password": "پاس ورڈ",
        "confirm_password": "پاس ورڈ دوبارہ درج کریں",
        "email": "ای میل",
        "no_account": "اکاؤنٹ نہیں ہے؟",
        "have_account": "پہلے سے اکاؤنٹ موجود ہے؟",
        "go_signup": "نیا اکاؤنٹ بنائیں",
        "go_login": "سائن ان کریں",
        "signup_success": "اکاؤنٹ بن گیا! اب سائن ان کریں۔",
        "login_fail": "صارف کا نام یا پاس ورڈ درست نہیں۔",
        "fill_fields": "براہ کرم تمام خانے پُر کریں۔",
        "password_mismatch": "پاس ورڈ مماثل نہیں ہیں۔",
        "user_exists": "یہ صارف نام پہلے سے موجود ہے۔",
        "classify_title": "پیغام کی جانچ کریں",
        "classify_sub": "نیچے ای میل یا ایس ایم ایس درج کریں اور معلوم کریں کہ یہ سپیم ہے یا نہیں",
        "quick_examples": "فوری مثالیں",
        "custom_option": "-- خود لکھیں --",
        "message_label": "پیغام کا متن",
        "message_placeholder": "یہاں ای میل یا ایس ایم ایس کا متن لکھیں یا چسپاں کریں...",
        "classify_btn": "پیغام کی جانچ کریں",
        "warning_empty": "براہ کرم پہلے کوئی پیغام درج کریں۔",
        "analyzing": "پیغام کا تجزیہ ہو رہا ہے...",
        "spam_detected": "سپیم پایا گیا",
        "not_spam": "سپیم نہیں ہے",
        "confidence": "اعتماد کا سکور",
        "confidence_note_spam": "(زیادہ عدد = زیادہ یقین)",
        "confidence_note_ham": "(زیادہ منفی عدد = زیادہ یقین کہ یہ سپیم نہیں)",
        "see_cleaned": "پروسیس شدہ متن دیکھیں (ماڈل نے اصل میں کیا دیکھا)",
        "performance_title": "ماڈل کی کارکردگی",
        "performance_sub": "ٹیسٹ ڈیٹا پر تمام تربیت یافتہ ماڈلز کا موازنہ",
        "chart_title": "ماڈل موازنہ چارٹ",
        "cm_title": "کنفیوژن میٹرکس (بہترین ماڈل)",
        "best_model_note": "کو F1-سکور کی بنیاد پر بہترین ماڈل منتخب کیا گیا، جو درستگی (غلط سپیم شناخت سے بچاؤ) اور بازیافت (اصل سپیم کی شناخت) دونوں میں توازن رکھتا ہے۔",
        "history_title": "سابقہ ریکارڈ",
        "history_sub": "آپ کے جانچے گئے تمام پیغامات، جو آپ کے اکاؤنٹ میں مستقل محفوظ ہیں",
        "no_history": "ابھی تک کوئی پیغام نہیں جانچا گیا۔ 'پیغام کی جانچ کریں' صفحے پر جائیں۔",
        "clear_history": "ریکارڈ صاف کریں",
        "about_title": "یہ کیسے کام کرتا ہے",
        "about_sub": "سپیم کی شناخت کا طریقہ کار سمجھیں",
        "notif_title": "اطلاعات",
        "notif_sub": "آپ کی جانچ کی سرگرمی کی حالیہ اطلاعات",
        "no_notif": "ابھی تک کوئی اطلاع موجود نہیں۔",
        "clear_notif": "اطلاعات صاف کریں",
        "settings_title": "ترتیبات",
        "settings_sub": "اپنی ایپ کا تجربہ اپنی مرضی کے مطابق بنائیں",
        "appearance": "ظاہری شکل",
        "theme_label": "تھیم",
        "light": "روشن",
        "dark": "تاریک",
        "language_label": "زبان",
        "account": "اکاؤنٹ",
        "logged_in_as": "اس نام سے لاگ ان ہیں",
        "metric_accuracy": "ماڈل کی درستگی",
        "metric_f1": "ایف ون سکور",
        "metric_best": "بہترین ماڈل",
        "metric_checked": "کل جانچے گئے پیغامات",
        "metric_spam": "پکڑا گیا سپیم",
        "metric_ham": "محفوظ پیغامات",
        "nav_hint": "سائیڈ بار سے نیا پیغام جانچیں، اپنا ریکارڈ دیکھیں، یا ماڈل کی کارکردگی معلوم کریں۔",
        "recent_activity": "حالیہ سرگرمی",
        "no_activity": "ابھی تک کچھ نہیں جانچا گیا . 'پیغام کی جانچ کریں' صفحہ آزمائیں۔",
        "quick_tips": "سپیم پہچاننے کے مفید نکات",
        "tip1": "فوری کارروائی کا مطالبہ کرنے والی زبان سے ہوشیار رہیں۔",
        "tip2": "غیر متوقع انعام یا لاٹری کی اطلاعات تقریباً ہمیشہ سپیم ہوتی ہیں۔",
        "tip3": "کلک کرنے سے پہلے لنکس کو غور سے جانچیں۔",
        "tip4": "معتبر کمپنیاں پیغام کے ذریعے پاس ورڈ یا کارڈ کی تفصیلات نہیں مانگتیں۔",
        "breakdown": "سپیم بمقابلہ محفوظ پیغامات کا خلاصہ",
    }
}

def tr(key):
    return T[st.session_state.language].get(key, key)

# =========================================================
# THEME CSS
# =========================================================
def inject_css():
    dark = st.session_state.theme == "dark"

    if dark:
        bg = "linear-gradient(160deg, #131629 0%, #1e1b3f 55%, #241a3d 100%)"
        card_bg = "#1c1f3a"
        text_color = "#f1f3ff"
        subtext = "#b8bddd"
        header_grad = "linear-gradient(120deg, #0369a1 0%, #7c3aed 100%)"
        border = "#3a3764"
        input_bg = "#242850"
        accent = "#38bdf8"
        accent2 = "#a78bfa"
        sidebar_bg = "#171936"
        sidebar_text = "#f1f3ff"
        sidebar_muted = "#b8bddd"
    else:
        bg = "linear-gradient(160deg, #eaf5ff 0%, #f1eeff 55%, #f6eeff 100%)"
        card_bg = "#ffffff"
        text_color = "#241f3d"
        subtext = "#6b6690"
        header_grad = "linear-gradient(120deg, #0ea5e9 0%, #8b5cf6 100%)"
        border = "#e3ddfb"
        input_bg = "#e7e9f2"
        accent = "#0ea5e9"
        accent2 = "#8b5cf6"
        sidebar_bg = "#f8f9fc"
        sidebar_text = "#241f3d"
        sidebar_muted = "#6b6690"

    st.markdown(f"""
    <style>
        /* =====================================================
           GLOBAL THEME
           ===================================================== */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{ background: transparent; }}

        .stApp {{
            background: {bg};
        }}

        .stApp, .stMarkdown, p, span, label, div {{
            color: {text_color};
        }}

        /* Streamlit text / headings */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {text_color};
        }}

        /* =====================================================
           SIDEBAR - FIX USERNAME + LOGOUT + NAV TEXT
           ===================================================== */
        section[data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {border};
        }}

        section[data-testid="stSidebar"] > div {{
            background: {sidebar_bg} !important;
        }}

        section[data-testid="stSidebar"] * {{
            color: {sidebar_text} !important;
        }}

        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {{
            color: {sidebar_muted} !important;
        }}

        /* Sidebar username */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {{
            color: {sidebar_text} !important;
            font-weight: 700;
        }}

        /* Sidebar navigation */
        section[data-testid="stSidebar"] .nav-link {{
            color: {sidebar_text} !important;
        }}

        section[data-testid="stSidebar"] .nav-link:hover {{
            color: #ffffff !important;
        }}

        section[data-testid="stSidebar"] .nav-link-selected {{
            color: #ffffff !important;
        }}

        section[data-testid="stSidebar"] .nav-link-selected span,
        section[data-testid="stSidebar"] .nav-link-selected i {{
            color: #ffffff !important;
        }}

        /* Sidebar Logout button */
        section[data-testid="stSidebar"] .stButton > button {{
            background: {"#242850" if dark else "#ffffff"} !important;
            color: {sidebar_text} !important;
            border: 1px solid {border} !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }}

        section[data-testid="stSidebar"] .stButton > button p,
        section[data-testid="stSidebar"] .stButton > button span {{
            color: {sidebar_text} !important;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover {{
            border-color: {accent} !important;
            color: {sidebar_text} !important;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: {border} !important;
        }}

        /* =====================================================
           MAIN HEADER
           ===================================================== */
        .main-header {{
            background: {header_grad};
            padding: 2rem 2.5rem;
            border-radius: 14px;
            color: white !important;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(15, 61, 62, 0.25);
        }}

        .main-header h1,
        .main-header p {{
            color: white !important;
        }}

        .main-header h1 {{
            margin: 0;
            font-size: 2rem;
        }}

        .main-header p {{
            margin: 0.3rem 0 0 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }}

        /* =====================================================
           CARDS
           ===================================================== */
        .metric-card {{
            background: {card_bg};
            padding: 1.3rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid {border};
            border-left: 4px solid {accent};
        }}

        .metric-card h3 {{
            margin: 0;
            font-size: 1.7rem;
            color: {text_color} !important;
        }}

        .metric-card p {{
            margin: 0.2rem 0 0 0;
            color: {subtext} !important;
            font-size: 0.85rem;
        }}

        .section-card {{
            background: {card_bg};
            padding: 1.5rem 1.7rem;
            border-radius: 12px;
            border: 1px solid {border};
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }}

        .section-card h4 {{
            margin-top: 0;
            color: {text_color} !important;
        }}

        .activity-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.6rem 0;
            border-bottom: 1px solid {border};
            font-size: 0.88rem;
            color: {text_color} !important;
        }}

        .activity-row:last-child {{
            border-bottom: none;
        }}

        .tag-spam {{
            color: #ff6b6b !important;
            font-weight: 600;
        }}

        .tag-ham {{
            color: {accent} !important;
            font-weight: 600;
        }}

        /* =====================================================
           RESULTS / NOTIFICATIONS / TIPS
           ===================================================== */
        .result-spam {{
            background: {"linear-gradient(120deg, #3a1a1a, #4a1f1f)" if dark else "linear-gradient(120deg, #ffe3e3, #ffd0d0)"};
            border-left: 5px solid #e03131;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-top: 1rem;
        }}

        .result-spam h3, .result-spam p {{
            color: {text_color} !important;
        }}

        .result-ham {{
            background: {"linear-gradient(120deg, #0d3330, #114440)" if dark else "linear-gradient(120deg, #dcf6f0, #c8f0e6)"};
            border-left: 5px solid {accent};
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-top: 1rem;
        }}

        .result-ham h3, .result-ham p {{
            color: {text_color} !important;
        }}

        .notif-item {{
            background: {card_bg};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
            color: {text_color} !important;
        }}

        .tip-item {{
            padding: 0.5rem 0;
            font-size: 0.9rem;
            color: {text_color} !important;
        }}

        /* =====================================================
           AUTH
           ===================================================== */
        .auth-box {{
            max-width: 440px;
            margin: 3rem auto;
            background: {card_bg};
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(15, 61, 62, 0.15);
            text-align: center;
            border: 1px solid {border};
        }}

        .auth-box h2 {{
            color: {text_color} !important;
            margin-bottom: 0.3rem;
        }}

        .auth-box p {{
            color: {subtext} !important;
            margin-bottom: 1.5rem;
            font-size: 0.9rem;
        }}

        /* =====================================================
           STREAMLIT WIDGETS - FIX INVISIBLE TEXT
           ===================================================== */
        /* =====================================================
           BUTTONS - FIX CLEAR HISTORY / CLEAR NOTIFICATIONS
           ===================================================== */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 600;
            color: {text_color} !important;
            background-color: {card_bg} !important;
            border: 1px solid {border} !important;
        }}

        .stButton > button p,
        .stButton > button span,
        .stButton > button div {{
            color: {text_color} !important;
        }}

        .stButton > button:hover {{
            color: {text_color} !important;
            border-color: {accent} !important;
            background-color: {input_bg} !important;
        }}

        .stButton > button:hover p,
        .stButton > button:hover span,
        .stButton > button:hover div {{
            color: {text_color} !important;
        }}

        /* Primary buttons keep white text */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(120deg, #0ea5e9, #7c3aed) !important;
            color: #ffffff !important;
            border: none !important;
        }}

        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div {{
            color: #ffffff !important;
        }}

        /* Input fields - slightly darker in light mode for visibility */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {{
            border-radius: 10px;
            background-color: {input_bg} !important;
            color: {text_color} !important;
            -webkit-text-fill-color: {text_color} !important;
            caret-color: {accent} !important;
            border: 1px solid {border} !important;
        }}

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            -webkit-text-fill-color: {text_color} !important;
            border-color: {accent} !important;
        }}

        /* Selectbox */
        div[data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            border-color: {border} !important;
        }}

        div[data-baseweb="select"] span {{
            color: {text_color} !important;
        }}

        /* Dropdown menu */
        div[role="listbox"],
        div[role="option"] {{
            background-color: {card_bg} !important;
            color: {text_color} !important;
        }}

        div[role="option"]:hover {{
            background-color: {input_bg} !important;
        }}

        /* Radio buttons */
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] label span {{
            color: {text_color} !important;
        }}

        /* Expander */
        div[data-testid="stExpander"] {{
            background-color: {card_bg} !important;
            border: 1px solid {border} !important;
        }}

        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span {{
            color: {text_color} !important;
        }}

        /* Alerts */
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {{
            color: {text_color} !important;
        }}

        /* Captions */
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {subtext} !important;
        }}

        /* Dataframes / tables */
        div[data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
        }}

        /* Markdown links */
        a {{
            color: {accent} !important;
        }}
    </style>
    """, unsafe_allow_html=True)


inject_css()

# =========================================================
# MODEL LOADING + PREPROCESSING
# =========================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# =========================================================
# AUTH PAGES
# =========================================================
def show_login():
    st.markdown(f"""
        <div class="auth-box">
            <h2>📧 {tr('app_name')}</h2>
            <p>{tr('tagline')}</p>
        </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        with st.form("login_form"):
            st.markdown(f"#### {tr('sign_in')}")
            username = st.text_input(tr("username"), key="login_user")
            password = st.text_input(tr("password"), type="password", key="login_pass")
            submitted = st.form_submit_button(tr("sign_in"), use_container_width=True, type="primary")

            if submitted:
                if not username.strip() or not password.strip():
                    st.warning(tr("fill_fields"))
                elif st.session_state.accounts.get(username.strip()) == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()

                    all_data = load_all_data()
                    user_record = all_data.get("users", {}).get(st.session_state.username, {})
                    st.session_state.history = user_record.get("history", [])
                    st.session_state.notifications = user_record.get("notifications", [])
                    st.session_state.theme = user_record.get("theme", "light")
                    st.session_state.language = user_record.get("language", "en")
                    st.rerun()
                else:
                    st.error(tr("login_fail"))

        st.write("")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption(tr("no_account"))
        with c2:
            if st.button(tr("go_signup"), use_container_width=True):
                st.session_state.auth_page = "signup"
                st.rerun()


def show_signup():
    st.markdown(f"""
        <div class="auth-box">
            <h2>📧 {tr('app_name')}</h2>
            <p>{tr('tagline')}</p>
        </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        with st.form("signup_form"):
            st.markdown(f"#### {tr('create_account')}")
            new_user = st.text_input(tr("username"), key="signup_user")
            new_email = st.text_input(tr("email"), key="signup_email")
            new_pass = st.text_input(tr("password"), type="password", key="signup_pass")
            confirm_pass = st.text_input(tr("confirm_password"), type="password", key="signup_confirm")
            submitted = st.form_submit_button(tr("create_account"), use_container_width=True, type="primary")

            if submitted:
                if not new_user.strip() or not new_email.strip() or not new_pass or not confirm_pass:
                    st.warning(tr("fill_fields"))
                elif new_pass != confirm_pass:
                    st.error(tr("password_mismatch"))
                elif new_user.strip() in st.session_state.accounts:
                    st.error(tr("user_exists"))
                else:
                    st.session_state.accounts[new_user.strip()] = new_pass
                    all_data = load_all_data()
                    all_data.setdefault("accounts", {})
                    all_data["accounts"] = st.session_state.accounts
                    save_all_data(all_data)

                    st.success(tr("signup_success"))
                    time.sleep(1.2)
                    st.session_state.auth_page = "login"
                    st.rerun()

        st.write("")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption(tr("have_account"))
        with c2:
            if st.button(tr("go_login"), use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()


# =========================================================
# HOME PAGE
# =========================================================
def show_home():
    st.markdown(f"""
        <div class="main-header">
            <h1>📧 {tr('app_name')}</h1>
            <p>{tr('tagline')}</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        results_df = pd.read_csv("model_results.csv")
        best = results_df.sort_values("F1-Score", ascending=False).iloc[0]
    except FileNotFoundError:
        best = None

    hist = st.session_state.history
    spam_count = sum(1 for h in hist if h["result"] == "SPAM")
    ham_count = sum(1 for h in hist if h["result"] == "HAM")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if best is not None:
            st.markdown(f"""<div class="metric-card"><h3>{best['Accuracy']*100:.1f}%</h3>
                         <p>{tr('metric_accuracy')}</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><h3>{len(hist)}</h3>
                     <p>{tr('metric_checked')}</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><h3>{spam_count}</h3>
                     <p>{tr('metric_spam')}</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><h3>{ham_count}</h3>
                     <p>{tr('metric_ham')}</p></div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown(f"### {tr('welcome')}, {st.session_state.username} 👋")
    st.caption(tr("nav_hint"))
    st.write("")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown(f'<div class="section-card"><h4>📋 {tr("recent_activity")}</h4>', unsafe_allow_html=True)
        if not hist:
            st.markdown(f'<p style="opacity:0.7">{tr("no_activity")}</p></div>', unsafe_allow_html=True)
        else:
            rows_html = ""
            for h in hist[-6:][::-1]:
                tag_class = "tag-spam" if h["result"] == "SPAM" else "tag-ham"
                tag_label = tr("spam_detected") if h["result"] == "SPAM" else tr("not_spam")
                rows_html += f"""<div class="activity-row">
                    <span>{h['time']} — {h['message']}</span>
                    <span class="{tag_class}">{tag_label}</span>
                </div>"""
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="section-card"><h4>💡 {tr("quick_tips")}</h4>', unsafe_allow_html=True)
        tips_html = "".join(f'<div class="tip-item">• {tr(f"tip{i}")}</div>' for i in range(1, 5))
        st.markdown(tips_html + "</div>", unsafe_allow_html=True)

        if best is not None:
            st.markdown(f'<div class="section-card"><h4>🏆 {tr("metric_best")}</h4>'
                        f'<p style="font-size:1.1rem; font-weight:600; margin:0;">{best["Model"]}</p>'
                        f'<p style="opacity:0.7; margin:0.2rem 0 0 0;">{tr("metric_f1")}: {best["F1-Score"]*100:.1f}%</p></div>',
                        unsafe_allow_html=True)


# =========================================================
# CLASSIFY PAGE
# =========================================================
def show_classify(model, vectorizer):
    st.markdown(f"""
        <div class="main-header">
            <h1>🔍 {tr('classify_title')}</h1>
            <p>{tr('classify_sub')}</p>
        </div>
    """, unsafe_allow_html=True)

    example_messages = [
        tr("custom_option"),
        "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!!!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: Your account has been suspended. Verify your details immediately at this link.",
        "Can you send me the report before end of day? Thanks."
    ]
    example_choice = st.selectbox(tr("quick_examples"), example_messages)
    default_text = "" if example_choice == tr("custom_option") else example_choice

    user_input = st.text_area(tr("message_label"), value=default_text, height=150,
                               placeholder=tr("message_placeholder"))

    if st.button(f"🚀 {tr('classify_btn')}", type="primary"):
        if not user_input.strip():
            st.warning(tr("warning_empty"))
        else:
            with st.spinner(tr("analyzing")):
                time.sleep(0.4)
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                prediction = model.predict(vec)[0]
                decision_score = model.decision_function(vec)[0]

            label = "SPAM" if prediction == 1 else "HAM"
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.session_state.history.append({
                "time": now_str,
                "message": user_input[:80] + ("..." if len(user_input) > 80 else ""),
                "result": label,
                "score": round(float(decision_score), 2)
            })

            notif_text = (
                f"🚫 Spam message detected at {now_str}" if prediction == 1
                else f"✅ Message checked at {now_str} — looks safe"
            )
            st.session_state.notifications.insert(0, notif_text)
            persist_user_data()

            if prediction == 1:
                st.toast("Spam detected!", icon="🚫")
                st.markdown(f"""
                    <div class="result-spam">
                        <h3>🚫 {tr('spam_detected')}</h3>
                        <p>{tr('confidence')}: <b>{decision_score:.2f}</b> {tr('confidence_note_spam')}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.toast("Message looks safe", icon="✅")
                st.markdown(f"""
                    <div class="result-ham">
                        <h3>✅ {tr('not_spam')}</h3>
                        <p>{tr('confidence')}: <b>{decision_score:.2f}</b> {tr('confidence_note_ham')}</p>
                    </div>
                """, unsafe_allow_html=True)

            with st.expander(tr("see_cleaned")):
                st.code(cleaned if cleaned else "(empty after cleaning)")


# =========================================================
# PERFORMANCE PAGE
# =========================================================
def show_performance():
    st.markdown(f"""
        <div class="main-header">
            <h1>📊 {tr('performance_title')}</h1>
            <p>{tr('performance_sub')}</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        results_df = pd.read_csv("model_results.csv").sort_values("F1-Score", ascending=False).reset_index(drop=True)
        results_df.index = results_df.index + 1
        results_df.index.name = "Rank"
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                "Recall": "{:.2%}", "F1-Score": "{:.2%}"
            }).background_gradient(cmap="PuBu", subset=["F1-Score"]),
            use_container_width=True
        )
        best_model_name = results_df.iloc[0]["Model"]
    except FileNotFoundError:
        best_model_name = None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{tr('chart_title')}**")
        st.image("model_comparison.png", use_container_width=True)
    with col2:
        st.markdown(f"**{tr('cm_title')}**")
        st.image("confusion_matrix.png", use_container_width=True)

    if best_model_name:
        st.info(f"**{best_model_name}** {tr('best_model_note')}")


# =========================================================
# HISTORY PAGE
# =========================================================
def show_history():
    st.markdown(f"""
        <div class="main-header">
            <h1>📜 {tr('history_title')}</h1>
            <p>{tr('history_sub')}</p>
        </div>
    """, unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.info(tr("no_history"))
    else:
        spam_count = sum(1 for h in hist if h["result"] == "SPAM")
        ham_count = len(hist) - spam_count

        c1, c2, c3 = st.columns(3)
        c1.metric(tr("metric_checked"), len(hist))
        c2.metric(tr("metric_spam"), spam_count)
        c3.metric(tr("metric_ham"), ham_count)

        st.write("")
        hist_df = pd.DataFrame(hist[::-1]).reset_index(drop=True)
        hist_df.index = hist_df.index + 1
        st.dataframe(hist_df, use_container_width=True)

        if st.button(tr("clear_history")):
            st.session_state.history = []
            persist_user_data()
            st.rerun()


# =========================================================
# NOTIFICATIONS PAGE
# =========================================================
def show_notifications():
    st.markdown(f"""
        <div class="main-header">
            <h1>🔔 {tr('notif_title')}</h1>
            <p>{tr('notif_sub')}</p>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.notifications:
        st.info(tr("no_notif"))
    else:
        for note in st.session_state.notifications:
            st.markdown(f'<div class="notif-item">{note}</div>', unsafe_allow_html=True)
        if st.button(tr("clear_notif")):
            st.session_state.notifications = []
            persist_user_data()
            st.rerun()


# =========================================================
# ABOUT / HOW IT WORKS PAGE
# =========================================================
def show_about():
    st.markdown(f"""
        <div class="main-header">
            <h1>ℹ️ {tr('about_title')}</h1>
            <p>{tr('about_sub')}</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.language == "ur":
        st.markdown("""
        **یہ ایپ کیا کرتی ہے:**
        یہ ایپ کسی بھی ای میل یا ایس ایم ایس پیغام کو پڑھ کر بتاتی ہے کہ وہ **سپیم** ہے یا **عام (محفوظ)** پیغام۔

        **یہ کیسے کام کرتی ہے:**
        1. **متن کی صفائی** پیغام کو چھوٹے حروف میں بدلا جاتا ہے، لنکس، اعداد اور رموزِ اوقاف ہٹا دیے جاتے ہیں، اور عام الفاظ (جیسے "the"، "is"، "and") نکال دیے جاتے ہیں۔
        2. **خصوصیات کا استخراج (TF-IDF)** صاف شدہ متن کو اعداد میں تبدیل کیا جاتا ہے جو بتاتے ہیں کہ اس پیغام میں کون سے الفاظ زیادہ اہم ہیں۔
        3. **مشین لرننگ ماڈل** ایک تربیت یافتہ درجہ بندی کنندہ (سپورٹ ویکٹر مشین) ان اعداد کو دیکھ کر فیصلہ کرتا ہے کہ پیغام سپیم ہے یا نہیں۔
        4. **اعتماد کا سکور** ایپ یہ بھی دکھاتی ہے کہ ماڈل اپنے فیصلے پر کتنا پُراعتماد ہے۔

        یہ ماڈل ہزاروں لیبل شدہ پیغامات پر تربیت یافتہ ہے اور اُن نمونوں کو پہچانتا ہے جو عام طور پر سپیم پیغامات میں پائے جاتے ہیں — جیسے فوری پیشکشیں، انعامی دعوے، یا مشکوک لنکس۔
        """)
    else:
        st.markdown("""
        **What this app does:**
        This app reads any email or SMS message and predicts whether it is **spam** or a **normal (safe)** message.

        **How it works:**
        1. **Text Cleaning**: The message is lowercased, links/numbers/punctuation are stripped, and common filler words (like "the", "is", "and") are removed.
        2. **Feature Extraction (TF-IDF)**: The cleaned text is converted into numerical features that capture which words matter most in the message.
        3. **Machine Learning Model**: A trained classifier (Support Vector Machine) looks at these features and decides whether the message is spam.
        4. **Confidence Score**: The app also shows how confident the model is in its decision.

        The model was trained on thousands of labeled messages, learning the patterns commonly found in spam like urgent offers, prize claims, or suspicious links.
        """)


# =========================================================
# SETTINGS PAGE
# =========================================================
def show_settings():
    st.markdown(f"""
        <div class="main-header">
            <h1>⚙️ {tr('settings_title')}</h1>
            <p>{tr('settings_sub')}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"#### {tr('appearance')}")
    theme_choice = st.radio(
        tr("theme_label"),
        options=["light", "dark"],
        format_func=lambda x: tr("light") if x == "light" else tr("dark"),
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        persist_user_data()
        st.rerun()

    lang_choice = st.radio(
        tr("language_label"),
        options=["en", "ur"],
        format_func=lambda x: "English" if x == "en" else "اردو",
        index=0 if st.session_state.language == "en" else 1,
        horizontal=True
    )
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice
        persist_user_data()
        st.rerun()

    st.divider()
    st.markdown(f"#### {tr('account')}")
    st.write(f"{tr('logged_in_as')}: **{st.session_state.username}**")


# =========================================================
# MAIN APP FLOW
# =========================================================
if not st.session_state.logged_in:
    if st.session_state.auth_page == "signup":
        show_signup()
    else:
        show_login()
else:
    model, vectorizer = load_artifacts()

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        unread = len(st.session_state.notifications)
        bell = f"🔔 ({unread})" if unread else "🔔"
        st.caption(bell)
        st.divider()

        nav_options = [tr("nav_home"), tr("nav_classify"), tr("nav_performance"),
                       tr("nav_history"), tr("nav_notifications"), tr("nav_about"), tr("nav_settings")]
        nav_icons = ["house", "search", "bar-chart", "clock-history", "bell", "info-circle", "gear"]

        selected = option_menu(
            menu_title=None,
            options=nav_options,
            icons=nav_icons,
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#0ea5e9" if st.session_state.theme == "light" else "#38bdf8", "font-size": "16px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "3px 0", "border-radius": "8px"},
                "nav-link-selected": {"background-color": "#8b5cf6" if st.session_state.theme == "light" else "#7c3aed"},
            }
        )

        st.divider()
        if st.button(f"🚪 {tr('logout')}", use_container_width=True):
            persist_user_data()
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    if selected == tr("nav_home"):
        show_home()
    elif selected == tr("nav_classify"):
        show_classify(model, vectorizer)
    elif selected == tr("nav_performance"):
        show_performance()
    elif selected == tr("nav_history"):
        show_history()
    elif selected == tr("nav_notifications"):
        show_notifications()
    elif selected == tr("nav_about"):
        show_about()
    elif selected == tr("nav_settings"):
        show_settings()