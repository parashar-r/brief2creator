import streamlit as st
import pandas as pd
import torch
import hashlib
from sentence_transformers import SentenceTransformer, util

# ------------------
# Helper Functions
# ------------------
def load_dataset(uploaded_file):
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

def validate_dataset(df, required_cols):
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must include the following columns: {required_cols}")

def display_creator_card(row):
    card_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
        <h3 style="margin-bottom: 5px;">{row['name']}</h3>
        <p>
            <strong>Niche:</strong> {row['niche']} |
            <strong>Location:</strong> {row['location']} |
            <strong>Audience:</strong> {row['audience_size']}
        </p>
        <p style="color: #006600;"><strong>Similarity Score:</strong> {row['similarity_score']:.4f}</p>
        <p><em>{row['bio']}</em></p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# -------------------------------
# Streamlit App – Creator Matching
# -------------------------------
st.set_page_config(page_title="Creator Matching App", layout="wide")
st.title("🔍 Creator Matching App")
st.markdown("### Step 1: Upload your creator dataset (CSV or Excel format)")

uploaded_file = st.file_uploader("📁 Upload your creator dataset", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Please upload your dataset to get started.")
    st.stop()

# Load dataset and validate structure
try:
    creators_df = load_dataset(uploaded_file)
    required_cols = {"name", "bio", "niche", "location", "audience_size"}
    validate_dataset(creators_df, required_cols)
    st.success(f"✅ Dataset loaded successfully with {len(creators_df)} creators.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Campaign brief input
st.markdown("### Step 2: Enter your campaign brief")
campaign_brief = st.text_area("📣 Campaign Brief", placeholder="e.g. Looking for Indian influencers who focus on eco-friendly fashion and modern lifestyle.")

if campaign_brief.strip() == "":
    st.info("Please enter a campaign brief to proceed.")
    st.stop()

# Recompute button
st.markdown("### Step 3: Generate or Refresh Results")
recompute = st.button("🔁 Recompute Embeddings")

# Create a unique hash key from file + brief
file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
brief = campaign_brief.strip()

# Check if we can reuse the cached result
cached = (
    "scored_df" in st.session_state and
    st.session_state.get("file_hash") == file_hash and
    st.session_state.get("brief") == brief and
    not recompute
)

if not cached:
    with st.spinner("Embedding and scoring creators..."):
        try:
            model = SentenceTransformer("BAAI/bge-m3")
            bio_embeddings = model.encode(creators_df["bio"].tolist(), convert_to_tensor=False, normalize_embeddings=True)
            creators_df["bio_embedding"] = bio_embeddings.tolist()

            query = "Represent this sentence for searching relevant passages: " + brief
            query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

            device = query_embedding.device
            creator_embeddings_tensor = torch.tensor(bio_embeddings, device=device)
            similarities = util.cos_sim(query_embedding, creator_embeddings_tensor)[0].cpu().numpy()
            creators_df["similarity_score"] = similarities

            st.session_state.scored_df = creators_df.drop(columns=["bio_embedding"])
            st.session_state.file_hash = file_hash
            st.session_state.brief = brief

        except Exception as e:
            st.error(f"Error embedding or scoring: {e}")
            st.stop()
else:
    creators_df = st.session_state.scored_df.copy()
    st.info("✅ Using cached results. Click 'Recompute Embeddings' to refresh.")

# -------------------------------
# Filter and Display Results
# -------------------------------
st.sidebar.header("🔧 Filter Results")
unique_niches = sorted(creators_df["niche"].dropna().unique())
unique_locations = sorted(creators_df["location"].dropna().unique())

selected_niche = st.sidebar.selectbox("Filter by Niche", ["All"] + unique_niches)
selected_location = st.sidebar.selectbox("Filter by Location", ["All"] + unique_locations)

filtered_df = creators_df.copy()
if selected_niche != "All":
    filtered_df = filtered_df[filtered_df["niche"] == selected_niche]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_location]

top_creators = filtered_df.sort_values(by="similarity_score", ascending=False).head(10)

st.markdown("### 🎯 Top Matching Creators")
if top_creators.empty:
    st.warning("No matching creators found based on the filters. Try modifying your filters or campaign brief.")
else:
    for _, row in top_creators.iterrows():
        display_creator_card(row)

# -------------------------------
# Download Button
# -------------------------------
st.markdown("### 📥 Download Results")
csv_data = top_creators.to_csv(index=False)
st.download_button(label="Download CSV", data=csv_data, file_name="top_creators.csv", mime="text/csv")