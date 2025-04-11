# Required libraries
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# CreatorMatcher class definition
# -------------------------------

class CreatorMatcher:
    """
    Handling matching a campaign brief to a set of creators
    based on semantic similarity between the brief and creator bios.
    """
    def __init__(self, creators_df, embedding_model_name="all-MiniLM-L6-v2"):
        self.creators_df = creators_df.copy()
        # Load pre-trained model from Sentence Transformers
        self.model = SentenceTransformer(embedding_model_name)
        self._embed_creator_bios()

    def _embed_creator_bios(self):
        # Convert all bios into sentence embeddings and store in a new column
        self.creators_df["bio_embedding"] = self.model.encode(
            self.creators_df["bio"].tolist(), convert_to_tensor=True
        )

    def match(self, campaign_brief, top_n=10, filters=None):
        # Convert the campaign brief to an embedding
        campaign_embedding = self.model.encode(campaign_brief, convert_to_tensor=True)

        # Compute similarity between the brief and each bio
        similarities = util.cos_sim(campaign_embedding, list(self.creators_df["bio_embedding"]))[0].cpu().numpy()

        # Store similarity scores in a new column
        self.creators_df["similarity_score"] = similarities

        # Apply optional filters (niche/location)
        df_filtered = self.creators_df
        if filters:
            if filters.get("niche"):
                df_filtered = df_filtered[df_filtered["niche"] == filters["niche"]]
            if filters.get("location"):
                df_filtered = df_filtered[df_filtered["location"] == filters["location"]]

        # Sort by similarity and return top results
        return df_filtered.sort_values(by="similarity_score", ascending=False).head(top_n).drop(columns=["bio_embedding"])


### Streamlit App UI Logic


# Configure page
st.set_page_config(page_title="Creator Matching App", layout="wide")
st.title("🔍 Creator Matching App")
st.markdown("Match your campaign brief with the most relevant creators based on their bios using semantic similarity.")

# Upload file (CSV or Excel)
uploaded_file = st.sidebar.file_uploader("📁 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

# Proceed only if file is uploaded
if uploaded_file:
    try:
        # Read file dynamically based on extension
        if uploaded_file.name.endswith(".csv"):
            creators_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            creators_df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop()

        # Ensure required columns exist
        required_cols = {"name", "bio", "niche", "location", "audience_size"}
        if not required_cols.issubset(set(creators_df.columns)):
            st.error("Uploaded file must include: name, bio, niche, location, audience_size")
            st.stop()

        # Create matcher object
        matcher = CreatorMatcher(creators_df)

        # Sidebar filters for niche and location
        st.sidebar.header("🔧 Filters")
        selected_niche = st.sidebar.selectbox("Filter by Niche", ["All"] + sorted(creators_df["niche"].unique().tolist()))
        selected_location = st.sidebar.selectbox("Filter by Location", ["All"] + sorted(creators_df["location"].unique().tolist()))

        # Campaign brief input area
        campaign_brief = st.text_area("📢 Enter your campaign brief", height=150, placeholder="e.g. Looking for creators who talk about sustainable fashion trends...")

        # Button to trigger matching logic
        if st.button("Find Matching Creators 🚀") and campaign_brief.strip():
            with st.spinner("Finding best matches..."):
                # Apply selected filters (if not 'All')
                filters = {
                    "niche": None if selected_niche == "All" else selected_niche,
                    "location": None if selected_location == "All" else selected_location
                }

                # Get top 10 matching creators
                top_creators = matcher.match(campaign_brief, top_n=10, filters=filters)

                st.subheader("🎯 Top Matching Creators")

                # Display results in styled HTML blocks
                for idx, row in top_creators.iterrows():
                    st.markdown(
                        f"""
                        <div style="border:1px solid #eee; padding:15px; border-radius:10px; margin-bottom:15px; box-shadow:0 2px 5px rgba(0,0,0,0.05)">
                            <h4 style="margin-bottom:5px;">{row['name']}</h4>
                            <strong>Niche:</strong> {row['niche']} |
                            <strong>Location:</strong> {row['location']} |
                            <strong>Audience:</strong> {row['audience_size']}<br>
                            <strong>Similarity Score:</strong> {row['similarity_score']:.4f}<br>
                            <em>{row['bio']}</em>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Option to download results as CSV
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=top_creators.to_csv(index=False),
                    file_name="top_creators.csv",
                    mime="text/csv"
                )

        else:
            st.info("Enter a campaign brief and hit the button to find matches.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a creator dataset file to get started.")
