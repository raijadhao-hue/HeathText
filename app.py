import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from supabase import create_client

st.set_page_config(page_title="HealthText", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f8f7; color: #17352f; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#123f37 0%,#0b2925 100%); }
[data-testid="stSidebar"] * { color: #f1faf6 !important; }
.hero { background: linear-gradient(120deg,#103e35,#1d6a58); padding: 28px 30px; border-radius: 20px; color: white; margin-bottom: 22px; }
.hero h1 { color: white; font-family: Manrope,sans-serif; font-size: 2rem; margin: 0 0 6px 0; }
.hero p { color: #d7eee5; margin: 0; font-size: 1rem; }
.section-title {font-family: Manrope,sans-serif;font-size:1.35rem;font-weight:800;color:#16483d;margin:10px 0 12px 0;}
.metric-card { background:white;border:1px solid #e0ebe6;border-radius:16px;padding:18px 20px;box-shadow:0 4px 16px rgba(18,63,53,.04); }
.metric-label { color:#607a72;font-size:.86rem; }
.metric-value { color:#174a3e;font-family:Manrope,sans-serif;font-size:1.75rem;font-weight:800; }
.small-muted {color:#647d75;font-size:.88rem;}
div.stButton > button { border-radius:10px; font-weight:700; }
.stButton > button[kind="primary"] { background:#16745d; border-color:#16745d; }
div[data-testid="stForm"] { background:#fff;border:1px solid #e1ebe6;padding:18px;border-radius:16px; }
.auth-wrap {max-width:760px;margin:auto;}
</style>
""", unsafe_allow_html=True)

# ---------- Demo text classification dataset ----------
DATA = [
    ("I have a dry cough and difficulty breathing", "Respiratory"),
    ("My throat hurts and I have a runny nose", "Respiratory"),
    ("I am experiencing wheezing and chest tightness", "Respiratory"),
    ("I have a fever with a persistent cough", "Respiratory"),
    ("My nose is blocked and I keep sneezing", "Respiratory"),
    ("I feel short of breath while walking", "Respiratory"),
    ("There is mucus when I cough in the morning", "Respiratory"),
    ("My breathing sounds noisy at night", "Respiratory"),
    ("I have a sore throat and congestion", "Respiratory"),
    ("I feel pressure in my chest when breathing", "Respiratory"),
    ("I have stomach pain after eating", "Digestive"),
    ("I feel nauseated and have vomited twice", "Digestive"),
    ("I have diarrhea and abdominal cramps", "Digestive"),
    ("I am constipated and feel bloated", "Digestive"),
    ("I get acid reflux after spicy food", "Digestive"),
    ("My stomach feels upset and gassy", "Digestive"),
    ("I have heartburn after meals", "Digestive"),
    ("I feel pain in my upper abdomen", "Digestive"),
    ("I have indigestion and loss of appetite", "Digestive"),
    ("I feel bloated throughout the day", "Digestive"),
    ("I have an itchy red rash on my arm", "Dermatology"),
    ("My skin is dry and peeling", "Dermatology"),
    ("I noticed red patches on my face", "Dermatology"),
    ("My scalp is itchy and flaky", "Dermatology"),
    ("I have swelling and itching after a bite", "Dermatology"),
    ("There are small bumps on my skin", "Dermatology"),
    ("My skin becomes irritated after using soap", "Dermatology"),
    ("I have a painful boil on my skin", "Dermatology"),
    ("My hands are dry and cracked", "Dermatology"),
    ("I have an itchy rash that gets worse at night", "Dermatology"),
    ("I have a headache and feel dizzy", "Neurology"),
    ("I am experiencing a migraine with light sensitivity", "Neurology"),
    ("My hands feel numb and tingly", "Neurology"),
    ("I have recurring headaches in the morning", "Neurology"),
    ("I feel dizzy when I stand up", "Neurology"),
    ("I have a sharp headache on one side", "Neurology"),
    ("I am having trouble concentrating and feel confused", "Neurology"),
    ("There is tingling in my feet", "Neurology"),
    ("I feel unsteady and have a spinning sensation", "Neurology"),
    ("My head hurts and I feel sensitive to sound", "Neurology"),
]
df = pd.DataFrame(DATA, columns=["text", "category"])
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.25, random_state=42, stratify=df["category"])
model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# ---------- Supabase / session ----------
def get_supabase():
    if "supabase_client" not in st.session_state:
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
            st.session_state.supabase_client = create_client(url, key)
        except Exception as exc:
            st.error("Supabase connection is not configured. Check .streamlit/secrets.toml.")
            st.code('SUPABASE_URL = "https://your-project.supabase.co"\nSUPABASE_KEY = "your-publishable-key"')
            st.stop()
    return st.session_state.supabase_client


def current_user():
    return st.session_state.get("user")


def set_user_from_auth(auth_response):
    user = getattr(auth_response, "user", None)
    if user:
        st.session_state.user = {"id": user.id, "email": user.email or ""}
        return True
    return False


def sign_out():
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("user", None)
    st.session_state.pop("auth_message", None)
    st.rerun()


def safe_rows(table, user_id):
    try:
        result = get_supabase().table(table).select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return result.data or []
    except Exception as exc:
        st.error(f"Could not load {table}. Check that the SQL tables and RLS policies were created. Details: {exc}")
        return []


def save_profile(user_id, values):
    payload = {"user_id": user_id, **values, "updated_at": datetime.now().astimezone().isoformat()}
    get_supabase().table("health_profiles").upsert(payload, on_conflict="user_id").execute()


def add_record(table, user_id, name_field, name_value, notes=""):
    payload = {"user_id": user_id, name_field: name_value.strip(), "notes": notes.strip()}
    get_supabase().table(table).insert(payload).execute()


def delete_record(table, record_id, user_id):
    get_supabase().table(table).delete().eq("id", record_id).eq("user_id", user_id).execute()

# ---------- Header ----------
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def auth_screen():
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    hero("HealthText", "A personal health space with healthcare text classification")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Your health, organised")
        st.write("Keep your health profile, conditions, allergies, medicines and personal notes together in one private dashboard.")
        st.info("The text classifier is an educational demo. It does not diagnose illnesses or replace a medical professional.")
    with right:
        login_tab, register_tab = st.tabs(["Login", "Create account"])
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email address", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
            if submitted:
                if not email.strip() or not password:
                    st.warning("Enter both email and password.")
                else:
                    try:
                        response = get_supabase().auth.sign_in_with_password({"email": email.strip(), "password": password})
                        if set_user_from_auth(response):
                            st.success("Login successful!")
                            st.rerun()
                        st.error("Login did not return a user. Check your credentials and email confirmation settings.")
                    except Exception as exc:
                        st.error(f"Login failed: {exc}")
        with register_tab:
            with st.form("register_form"):
                name = st.text_input("Full name")
                email_new = st.text_input("Email address", key="register_email")
                password_new = st.text_input("Create password (at least 6 characters)", type="password", key="register_password")
                password_again = st.text_input("Confirm password", type="password")
                submitted_new = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if submitted_new:
                if not name.strip() or not email_new.strip() or not password_new:
                    st.warning("Please fill in all fields.")
                elif len(password_new) < 6:
                    st.warning("Use a password with at least 6 characters.")
                elif password_new != password_again:
                    st.warning("Passwords do not match.")
                else:
                    try:
                        response = get_supabase().auth.sign_up({"email": email_new.strip(), "password": password_new, "options": {"data": {"full_name": name.strip()}}})
                        if set_user_from_auth(response):
                            try:
                                save_profile(st.session_state.user["id"], {"full_name": name.strip(), "age": None, "gender": "", "blood_group": ""})
                            except Exception:
                                pass
                            st.success("Account created!")
                            st.rerun()
                        else:
                            st.success("Account created. Check your email to confirm it, then log in here.")
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Authentication gate ----------
if not current_user():
    auth_screen()
    st.stop()

user = current_user()
supabase = get_supabase()

with st.sidebar:
    st.markdown("## 🩺 HealthText")
    st.caption(f"Signed in as {user.get('email', '')}")
    page = st.radio("NAVIGATION", ["Dashboard", "My Health Profile", "Health Records", "Health Journal", "Text Classifier", "Dataset Explorer", "Model Evaluation"], label_visibility="collapsed")
    st.divider()
    st.caption("Your records are protected by Supabase Row Level Security.")
    if st.button("Log out", use_container_width=True):
        sign_out()

# ---------- Pages ----------
if page == "Dashboard":
    profile_rows = []
    try:
        profile_rows = supabase.table("health_profiles").select("*").eq("user_id", user["id"]).limit(1).execute().data or []
    except Exception as exc:
        st.error(f"Could not load your profile: {exc}")
    profile = profile_rows[0] if profile_rows else {}
    conditions = safe_rows("health_conditions", user["id"])
    medications = safe_rows("health_medications", user["id"])
    allergies = safe_rows("health_allergies", user["id"])
    journal = safe_rows("health_journal", user["id"])
    hero(f"Welcome{', ' + profile.get('full_name', '').strip() if profile.get('full_name') else ''} 👋", "Your personal health overview, in one place.")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [(c1, "Conditions", len(conditions)), (c2, "Medications", len(medications)), (c3, "Allergies", len(allergies)), (c4, "Journal entries", len(journal))]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Your health snapshot</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Medical conditions")
        if conditions:
            for item in conditions[:5]: st.write("•", item.get("condition_name", "Condition"), (f"— {item.get('notes')}" if item.get("notes") else ""))
        else: st.caption("No conditions added yet. Add one under Health Records if you want to keep a record.")
        st.markdown("#### Allergies")
        if allergies:
            for item in allergies[:5]: st.write("•", item.get("allergy_name", "Allergy"))
        else: st.caption("No allergies saved yet.")
    with b:
        st.markdown("#### Current medications")
        if medications:
            for item in medications[:5]: st.write("•", item.get("medication_name", "Medication"), (f"— {item.get('notes')}" if item.get("notes") else ""))
        else: st.caption("No medicines added yet.")
        st.markdown("#### Quick note")
        with st.form("quick_journal"):
            note = st.text_area("Write a health note or symptom observation", placeholder="Example: headache after lunch…", height=90)
            save_note = st.form_submit_button("Save note", type="primary")
        if save_note:
            if note.strip():
                try:
                    supabase.table("health_journal").insert({"user_id": user["id"], "entry_text": note.strip()}).execute()
                    st.success("Note saved.")
                    st.rerun()
                except Exception as exc: st.error(f"Could not save note: {exc}")
            else: st.warning("Write a note first.")
    st.warning("HealthText is an educational project. Classifications are predictions from a small sample dataset, not medical diagnoses. Seek urgent medical care for severe or sudden symptoms.")

elif page == "My Health Profile":
    hero("My Health Profile", "Keep the personal health details you choose to record.")
    try:
        rows = supabase.table("health_profiles").select("*").eq("user_id", user["id"]).limit(1).execute().data or []
        profile = rows[0] if rows else {}
    except Exception as exc:
        st.error(f"Could not load profile: {exc}"); profile = {}
    with st.form("profile_form"):
        full_name = st.text_input("Full name", value=profile.get("full_name") or "")
        col1, col2, col3 = st.columns(3)
        with col1:
            age_value = profile.get("age")
            age = st.number_input("Age", min_value=0, max_value=120, value=int(age_value) if age_value else 0, help="Leave as 0 if you prefer not to provide it.")
        with col2:
            gender_options = ["Prefer not to say", "Female", "Male", "Non-binary", "Other"]
            saved_gender = profile.get("gender") or "Prefer not to say"
            gender = st.selectbox("Gender", gender_options, index=gender_options.index(saved_gender) if saved_gender in gender_options else 0)
        with col3:
            blood_options = ["Prefer not to say", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
            saved_blood = profile.get("blood_group") or "Prefer not to say"
            blood = st.selectbox("Blood group", blood_options, index=blood_options.index(saved_blood) if saved_blood in blood_options else 0)
        save = st.form_submit_button("Save profile", type="primary")
    if save:
        try:
            save_profile(user["id"], {"full_name": full_name.strip(), "age": int(age) if age else None, "gender": "" if gender == "Prefer not to say" else gender, "blood_group": "" if blood == "Prefer not to say" else blood})
            st.success("Your profile has been saved.")
        except Exception as exc: st.error(f"Could not save profile: {exc}")
    st.caption("Only add details you are comfortable storing. This is not an emergency service or a medical record system managed by a healthcare provider.")

elif page == "Health Records":
    hero("Health Records", "Add and manage conditions, medicines and allergies.")
    tabs = st.tabs(["Conditions", "Medications", "Allergies"])
    configs = [
        (tabs[0], "health_conditions", "condition_name", "Condition name", "Example: asthma", "Add condition"),
        (tabs[1], "health_medications", "medication_name", "Medication name", "Example: medicine name and dosage", "Add medication"),
        (tabs[2], "health_allergies", "allergy_name", "Allergy", "Example: peanuts", "Add allergy"),
    ]
    for tab, table, field, label, placeholder, button in configs:
        with tab:
            with st.form(f"add_{table}"):
                name = st.text_input(label, placeholder=placeholder)
                notes = st.text_area("Notes (optional)", placeholder="Details, date started, or reminders…", height=75)
                add = st.form_submit_button(button, type="primary")
            if add:
                if not name.strip(): st.warning(f"Enter a {label.lower()} first.")
                else:
                    try:
                        add_record(table, user["id"], field, name, notes)
                        st.success("Saved successfully."); st.rerun()
                    except Exception as exc: st.error(f"Could not save record: {exc}")
            st.markdown("#### Saved records")
            records = safe_rows(table, user["id"])
            if not records: st.caption("No records yet.")
            for item in records:
                with st.container(border=True):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{item.get(field, '')}**")
                        if item.get("notes"): st.write(item["notes"])
                        if item.get("created_at"): st.caption(str(item["created_at"])[:10])
                    with col2:
                        if st.button("Delete", key=f"delete_{table}_{item['id']}"):
                            try: delete_record(table, item["id"], user["id"]); st.success("Deleted."); st.rerun()
                            except Exception as exc: st.error(f"Could not delete record: {exc}")

elif page == "Health Journal":
    hero("Health Journal", "Record symptoms, appointments, questions or observations over time.")
    with st.form("journal_form"):
        entry = st.text_area("New journal entry", placeholder="What did you notice? When did it start? Any context you want to remember?", height=140)
        save_entry = st.form_submit_button("Save entry", type="primary")
    if save_entry:
        if not entry.strip(): st.warning("Write something before saving.")
        else:
            try: supabase.table("health_journal").insert({"user_id": user["id"], "entry_text": entry.strip()}).execute(); st.success("Journal entry saved."); st.rerun()
            except Exception as exc: st.error(f"Could not save journal entry: {exc}")
    st.markdown("#### Previous entries")
    entries = safe_rows("health_journal", user["id"])
    if not entries: st.info("Your journal is empty. New entries will appear here.")
    for item in entries:
        with st.container(border=True):
            st.caption(str(item.get("created_at", ""))[:19].replace("T", " "))
            st.write(item.get("entry_text", ""))
            if st.button("Delete entry", key=f"journal_delete_{item['id']}"):
                try: supabase.table("health_journal").delete().eq("id", item["id"]).eq("user_id", user["id"]).execute(); st.rerun()
                except Exception as exc: st.error(f"Could not delete entry: {exc}")

elif page == "Text Classifier":
    hero("Healthcare Text Classifier", "Enter a symptom-related sentence to see the model's predicted text category.")
    st.info("This machine-learning demo classifies text into broad categories: Respiratory, Digestive, Dermatology or Neurology. It does not diagnose a condition.")
    with st.form("classifier_form"):
        text_input = st.text_area("Describe the text", placeholder="Example: I have a dry cough and difficulty breathing", height=130)
        classify = st.form_submit_button("Classify text", type="primary")
    if classify:
        if len(text_input.strip()) < 4: st.warning("Enter a little more text to classify.")
        else:
            prediction = model.predict([text_input.strip()])[0]
            probabilities = model.predict_proba([text_input.strip()])[0]
            class_probs = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
            st.markdown("### Prediction")
            st.success(f"Predicted category: **{prediction}**")
            st.caption("Model confidence is not a measure of medical certainty.")
            prob_df = pd.DataFrame(class_probs, columns=["Category", "Model score"])
            prob_df["Model score"] = prob_df["Model score"] * 100
            fig = px.bar(prob_df, x="Model score", y="Category", orientation="h", text=prob_df["Model score"].map(lambda x: f"{x:.1f}%"), range_x=[0,100], title="Relative model scores")
            fig.update_layout(height=300, yaxis={"categoryorder":"total ascending"}, margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("What does this result mean?"):
                st.write("The model compares words and phrases with a small labelled sample dataset. Similar wording can change the prediction. Use this output only as a classroom demonstration, not to decide treatment or delay care.")

elif page == "Dataset Explorer":
    hero("Dataset Explorer", "Explore the small labelled dataset used for the text classifier.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total examples", len(df)); c2.metric("Categories", df["category"].nunique()); c3.metric("Examples per category", int(df["category"].value_counts().min()))
    category_filter = st.multiselect("Filter categories", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
    filtered = df[df["category"].isin(category_filter)]
    st.dataframe(filtered.rename(columns={"text":"Text example", "category":"Category"}), use_container_width=True, hide_index=True)
    counts = df["category"].value_counts().rename_axis("Category").reset_index(name="Examples")
    fig = px.pie(counts, names="Category", values="Examples", hole=0.55, title="Dataset category distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download filtered dataset (CSV)", filtered.to_csv(index=False).encode("utf-8"), file_name="healthtext_dataset.csv", mime="text/csv")

elif page == "Model Evaluation":
    hero("Model Evaluation", "Review the baseline TF-IDF + Logistic Regression model's test-set results.")
    st.caption("Metrics below use a held-out split of this small demo dataset. They should not be interpreted as clinical performance or validated medical accuracy.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{model_accuracy*100:.1f}%")
    c2.metric("Training examples", len(X_train))
    c3.metric("Test examples", len(X_test))
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(cm_df, text_auto=True, aspect="auto", labels=dict(x="Predicted category", y="Actual category", color="Count"), title="Confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index":"Category"})
    st.markdown("#### Classification report")
    st.dataframe(report_df.round(3), use_container_width=True, hide_index=True)
    st.markdown("#### Model configuration")
    st.write("**Text features:** TF-IDF word n-grams (1–2), English stop-word removal")
    st.write("**Classifier:** Logistic Regression")
    st.write("**Split:** 75% training / 25% testing, stratified by category")

st.markdown("---")
st.caption("HealthText • Student project • For educational demonstration only; not a diagnostic or emergency-care tool.")
