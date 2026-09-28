import streamlit as st
import pandas as pd
import socket
import requests

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .stApp {
        background: #f5f7fb;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #172554, #1e3a8a);
        padding: 40px 45px;
        border-radius: 18px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 10px;
        font-weight: 700;
    }

    .hero p {
        font-size: 18px;
        margin: 0;
        color: #dbeafe;
    }

    /* Cards */
    .card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 22px;
        font-weight: 700;
        color: #1d4ed8;
    }

    /* Result cards */
    .safe-result {
        background: #ecfdf5;
        border-left: 6px solid #10b981;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
    }

    .safe-result h2 {
        color: #047857;
        margin-bottom: 5px;
    }

    .warning-result {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
    }

    .warning-result h2 {
        color: #c2410c;
        margin-bottom: 5px;
    }

    /* Section titles */
    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #172554;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px 0 10px 0;
        font-size: 14px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🛡️ Phishing Website Detector</h1>

<p>
Machine Learning powered website security analysis
using URL and static webpage features.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv("data/dataset.csv")

    return df


df = load_dataset()


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report"
]


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURE_COLUMNS]
    y = data["Result"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model(df)


# ============================================================
# CALCULATE MODEL ACCURACY
# ============================================================

@st.cache_data
def calculate_accuracy(data):

    X = data[FEATURE_COLUMNS]
    y = data["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    evaluation_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


accuracy = calculate_accuracy(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

SHORTENING_SERVICES = [
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at"
]


def is_ip_address(hostname):

    try:
        socket.inet_aton(hostname)
        return True

    except:

        return False


def count_subdomains(hostname):

    if not hostname:
        return 0

    parts = hostname.split(".")

    return max(len(parts) - 2, 0)


def analyze_webpage(url):

    features = {
        "Request_URL": 0,
        "URL_of_Anchor": 0,
        "Links_in_tags": 0,
        "SFH": 0,
        "Submitting_to_email": 0,
        "on_mouseover": 0,
        "RightClick": 0,
        "popUpWidnow": 0,
        "Iframe": 0
    }

    try:

        response = requests.get(
            url,
            timeout=5,
            allow_redirects=False,
            headers={
                "User-Agent": "PhishingWebsiteDetector/1.0"
            }
        )

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if "text/html" not in content_type:

            return features

        html = response.text[:100000]

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # ----------------------------------------------------
        # Forms
        # ----------------------------------------------------

        forms = soup.find_all("form")

        if forms:

            features["SFH"] = 1

            for form in forms:

                action = form.get("action", "")

                if action.startswith("mailto:"):

                    features["Submitting_to_email"] = 1

                if action in ["", "#"]:

                    features["SFH"] = -1

        # ----------------------------------------------------
        # Anchors
        # ----------------------------------------------------

        anchors = soup.find_all("a")

        if anchors:

            total = len(anchors)

            suspicious = 0

            for anchor in anchors:

                href = anchor.get("href", "")

                if (
                    not href
                    or href.startswith("#")
                    or href.startswith("javascript:")
                ):

                    suspicious += 1

            ratio = suspicious / total

            if ratio > 0.5:

                features["URL_of_Anchor"] = -1

            elif ratio > 0.2:

                features["URL_of_Anchor"] = 0

            else:

                features["URL_of_Anchor"] = 1

        # ----------------------------------------------------
        # Scripts / Links
        # ----------------------------------------------------

        scripts = soup.find_all("script")
        links = soup.find_all("link")

        if scripts or links:

            total_tags = len(scripts) + len(links)

            if total_tags > 20:

                features["Links_in_tags"] = -1

            elif total_tags > 10:

                features["Links_in_tags"] = 0

            else:

                features["Links_in_tags"] = 1

        # ----------------------------------------------------
        # Iframe
        # ----------------------------------------------------

        if soup.find("iframe"):

            features["Iframe"] = -1

        else:

            features["Iframe"] = 1

        # ----------------------------------------------------
        # Mouseover
        # ----------------------------------------------------

        html_lower = html.lower()

        if "onmouseover" in html_lower:

            features["on_mouseover"] = -1

        else:

            features["on_mouseover"] = 1

        # ----------------------------------------------------
        # Right click
        # ----------------------------------------------------

        if (
            "contextmenu" in html_lower
            or "event.button==2" in html_lower
            or "event.button == 2" in html_lower
        ):

            features["RightClick"] = -1

        else:

            features["RightClick"] = 1

        # ----------------------------------------------------
        # Popups
        # ----------------------------------------------------

        if (
            "window.open" in html_lower
            or "popup" in html_lower
        ):

            features["popUpWidnow"] = -1

        else:

            features["popUpWidnow"] = 1

        # ----------------------------------------------------
        # External resources
        # ----------------------------------------------------

        resources = (
            soup.find_all("img")
            + soup.find_all("script")
            + soup.find_all("link")
        )

        if resources:

            external_count = 0

            page_domain = urlparse(url).netloc

            for resource in resources:

                src = (
                    resource.get("src")
                    or resource.get("href")
                    or ""
                )

                if src.startswith("http"):

                    resource_domain = urlparse(src).netloc

                    if (
                        resource_domain
                        and resource_domain != page_domain
                    ):

                        external_count += 1

            ratio = external_count / len(resources)

            if ratio > 0.5:

                features["Request_URL"] = -1

            elif ratio > 0.2:

                features["Request_URL"] = 0

            else:

                features["Request_URL"] = 1

    except Exception:

        pass

    return features


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

def extract_features(url):

    parsed = urlparse(url)

    hostname = parsed.netloc.lower()

    if ":" in hostname:

        hostname = hostname.split(":")[0]

    path = parsed.path

    query = parsed.query

    features = {}

    # --------------------------------------------------------
    # URL FEATURES
    # --------------------------------------------------------

    features["having_IP_Address"] = (
        -1 if is_ip_address(hostname) else 1
    )

    if len(url) < 54:

        features["URL_Length"] = 1

    elif len(url) <= 75:

        features["URL_Length"] = 0

    else:

        features["URL_Length"] = -1

    features["Shortining_Service"] = (
        -1
        if any(service in hostname for service in SHORTENING_SERVICES)
        else 1
    )

    features["having_At_Symbol"] = (
        -1 if "@" in url else 1
    )

    features["double_slash_redirecting"] = (
        -1
        if "//" in url[7:]
        else 1
    )

    features["Prefix_Suffix"] = (
        -1
        if "-" in hostname
        else 1
    )

    subdomains = count_subdomains(hostname)

    if subdomains == 0:

        features["having_Sub_Domain"] = 1

    elif subdomains == 1:

        features["having_Sub_Domain"] = 0

    else:

        features["having_Sub_Domain"] = -1

    features["SSLfinal_State"] = (
        1
        if parsed.scheme == "https"
        else -1
    )

    # --------------------------------------------------------
    # FEATURES REQUIRING EXTERNAL DOMAIN DATA
    # --------------------------------------------------------

    features["Domain_registeration_length"] = 0

    features["Favicon"] = 0

    # --------------------------------------------------------
    # PORT
    # --------------------------------------------------------

    port = parsed.port

    if port is None:

        features["port"] = 1

    elif port in [80, 443]:

        features["port"] = 1

    else:

        features["port"] = -1

    # --------------------------------------------------------
    # HTTPS TOKEN
    # --------------------------------------------------------

    features["HTTPS_token"] = (
        -1
        if "https" in hostname
        else 1
    )

    # --------------------------------------------------------
    # WEBPAGE ANALYSIS
    # --------------------------------------------------------

    webpage_features = analyze_webpage(url)

    features.update(webpage_features)

    # --------------------------------------------------------
    # ABNORMAL URL
    # --------------------------------------------------------

    if hostname:

        features["Abnormal_URL"] = 1

    else:

        features["Abnormal_URL"] = -1

    # --------------------------------------------------------
    # REDIRECT
    # --------------------------------------------------------

    if "//" in path or "//" in query:

        features["Redirect"] = -1

    else:

        features["Redirect"] = 1

    # --------------------------------------------------------
    # DNS RECORD
    # --------------------------------------------------------

    try:

        socket.gethostbyname(hostname)

        features["DNSRecord"] = 1

    except:

        features["DNSRecord"] = -1

    # --------------------------------------------------------
    # FEATURES REQUIRING EXTERNAL SERVICES
    # --------------------------------------------------------

    features["age_of_domain"] = 0
    features["web_traffic"] = 0
    features["Page_Rank"] = 0
    features["Google_Index"] = 0
    features["Links_pointing_to_page"] = 0
    features["Statistical_report"] = 0

    return features


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Analyze a Website</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a website URL below to analyze its URL structure "
    "and available static webpage features."
)

url = st.text_input(
    "Website URL",
    placeholder="https://example.com",
    label_visibility="collapsed"
)

analyze_button = st.button(
    "🔍 Analyze Website"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not url:

        st.warning("⚠️ Please enter a website URL.")

    else:

        # ----------------------------------------------------
        # Add protocol if missing
        # ----------------------------------------------------

        if not url.startswith(("http://", "https://")):

            url = "https://" + url

        parsed_url = urlparse(url)

        if not parsed_url.netloc:

            st.error(
                "❌ Invalid URL. Please enter a valid website address."
            )

        else:

            with st.spinner(
                "Analyzing website and extracting features..."
            ):

                try:

                    extracted_features = extract_features(url)

                    feature_df = pd.DataFrame(
                        [extracted_features]
                    )

                    # Make sure the columns are exactly
                    # the same as the trained model

                    feature_df = feature_df[
                        FEATURE_COLUMNS
                    ]

                    prediction = model.predict(
                        feature_df
                    )[0]

                    probabilities = model.predict_proba(
                        feature_df
                    )[0]

                    classes = list(
                        model.classes_
                    )

                    prediction_index = classes.index(
                        prediction
                    )

                    confidence = (
                        probabilities[prediction_index]
                        * 100
                    )

                except Exception as e:

                    st.error(
                        f"❌ Unable to analyze this URL.\n\n{e}"
                    )

                    st.stop()

            # ====================================================
            # WEBSITE INFORMATION
            # ====================================================

            st.markdown(
                '<div class="section-title">🌐 Website Information</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="card-title">
                    🌍 Domain
                    </div>

                    <div class="card-value">
                    {parsed_url.netloc}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="card-title">
                    🔐 Protocol
                    </div>

                    <div class="card-value">
                    {parsed_url.scheme.upper()}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="card-title">
                    📏 URL Length
                    </div>

                    <div class="card-value">
                    {len(url)} characters
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # RESULT
            # ====================================================

            if prediction == -1:

                st.markdown(
                    """
                    <div class="warning-result">

                    <h2>⚠️ Potentially Phishing</h2>

                    <p>
                    The machine learning model detected patterns
                    that are commonly associated with phishing websites.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="safe-result">

                    <h2>✅ Likely Legitimate</h2>

                    <p>
                    The machine learning model did not detect
                    strong phishing patterns in the analyzed features.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ====================================================
            # CONFIDENCE
            # ====================================================

            st.markdown(
                '<div class="section-title">📊 Prediction Confidence</div>',
                unsafe_allow_html=True
            )

            confidence_col1, confidence_col2 = st.columns(
                [1, 3]
            )

            with confidence_col1:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with confidence_col2:

                st.progress(
                    int(min(confidence, 100))
                )


            # ====================================================
            # MODEL INFORMATION
            # ====================================================

            st.markdown(
                '<div class="section-title">🤖 Model Information</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Model",
                    "Random Forest"
                )

            with col2:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Features",
                    "30"
                )

            with col4:

                st.metric(
                    "Dataset",
                    "11,055 URLs"
                )


            # ====================================================
            # FEATURES
            # ====================================================

            st.markdown(
                '<div class="section-title">🧩 Extracted Features</div>',
                unsafe_allow_html=True
            )

            display_features = feature_df.T.reset_index()

            display_features.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                display_features,
                use_container_width=True,
                hide_index=True
            )


            # ====================================================
            # DISCLAIMER
            # ====================================================

            st.warning(
                "⚠️ This project is an educational machine learning "
                "prototype. The prediction should not be treated as "
                "a guaranteed security verdict."
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown(
    '<div class="section-title">📘 About the Project</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

<p>
<strong>Phishing Website Detector</strong> is a machine learning
project that analyzes website URLs and static webpage characteristics
to identify patterns associated with phishing websites.
</p>

<p>
The project uses the <strong>UCI Phishing Websites Dataset</strong>
containing 11,055 website records and 30 input features.
A <strong>Random Forest Classifier</strong> is used for classification.
</p>

<p>
The application combines machine learning with basic URL and webpage
analysis to provide an interactive demonstration of phishing detection.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🛡️ Phishing Website Detector<br>

Built using Python • Streamlit • Scikit-learn • Pandas • BeautifulSoup

</div>
""", unsafe_allow_html=True)