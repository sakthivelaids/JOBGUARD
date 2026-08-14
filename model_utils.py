import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler


def _clean_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_demo_dataset() -> pd.DataFrame:
    rows = [
        {
            "title": "Junior Data Analyst",
            "description": "We are hiring a data analyst to analyze sales data and create dashboards using Python and Excel.",
            "requirements": "Bachelor degree, Excel, SQL, Python, communication skills",
            "benefits": "Health insurance, paid time off",
            "location": "New York, NY",
            "company_profile": "A growing technology company focused on logistics analytics.",
            "industry": "Technology",
            "fraudulent": 0,
        },
        {
            "title": "Remote Customer Support",
            "description": "Work from home and receive payment through a quick upfront deposit. Must send personal details and pay a registration fee.",
            "requirements": "Need to pay upfront fee, provide bank account, no experience needed",
            "benefits": "Fast cash, instant earnings",
            "location": "Remote",
            "company_profile": "Anonymous online job offer with limited verified company information.",
            "industry": "Services",
            "fraudulent": 1,
        },
        {
            "title": "Business Development Executive",
            "description": "Responsible for client outreach and sales coordination with growth opportunities in a reputable firm.",
            "requirements": "Strong communication skills, business development, CRM, presentation skills",
            "benefits": "Performance bonus, travel allowance",
            "location": "Chicago, IL",
            "company_profile": "Established B2B software company with a proven track record.",
            "industry": "Software",
            "fraudulent": 0,
        },
        {
            "title": "Urgent Payment Processor",
            "description": "Need a person to process payments and transfer funds immediately. You will be asked to pay a processing fee before starting.",
            "requirements": "Must pay processing fee, have bank account, no prior experience",
            "benefits": "High commission, weekly payout",
            "location": "Remote",
            "company_profile": "No credible company details available.",
            "industry": "Finance",
            "fraudulent": 1,
        },
        {
            "title": "Machine Learning Engineer",
            "description": "Build and deploy models for recommendation systems and predictive analytics in a collaborative environment.",
            "requirements": "Python, scikit-learn, deep learning, MLOps, Docker",
            "benefits": "Remote flexibility, annual bonus",
            "location": "Austin, TX",
            "company_profile": "AI startup building enterprise-scale products.",
            "industry": "Artificial Intelligence",
            "fraudulent": 0,
        },
        {
            "title": "Recruitment Agent",
            "description": "Before we start, you need to pay a small registration fee and send your personal identity details to confirm eligibility.",
            "requirements": "Pay fee, submit ID, no real interview process",
            "benefits": "Guaranteed earnings",
            "location": "Remote",
            "company_profile": "No company credentials or public listing found.",
            "industry": "Recruitment",
            "fraudulent": 1,
        },
    ]
    return pd.DataFrame(rows)


def preprocess_df(df: pd.DataFrame, text_column: str = None, target: str = "fraudulent"):
    if target not in df.columns:
        raise ValueError(f"Dataset must include a '{target}' column.")

    df = df.copy()
    if text_column is None:
        text_columns = [
            "title",
            "location",
            "company_profile",
            "description",
            "requirements",
            "benefits",
            "industry",
        ]
        available = [col for col in text_columns if col in df.columns]
        if available:
            df["text"] = df[available].fillna(" ").astype(str).agg(" ".join, axis=1)
        elif "text" in df.columns:
            df["text"] = df["text"].fillna(" ").astype(str)
        else:
            object_cols = df.select_dtypes(include=["object"]).columns.tolist()
            df["text"] = df[object_cols].fillna(" ").astype(str).agg(" ".join, axis=1)
    else:
        df["text"] = df[text_column].fillna(" ").astype(str)

    df["text"] = df["text"].apply(_clean_text)

    X = df["text"]
    y = df[target].astype(int)

    if y.nunique() < 2:
        raise ValueError("Target column must contain both fake and real labels for training.")

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X.to_frame(), y)
    X_res = X_res["text"].values

    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_vect = vect.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    return X_train, X_test, y_train, y_test, vect


def train_model(X_train, y_train, algorithm: str = "MultinomialNB"):
    if algorithm == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def predict_text(model, vectorizer, text: str):
    t = _clean_text(text)
    X = vectorizer.transform([t])
    return int(model.predict(X)[0])
