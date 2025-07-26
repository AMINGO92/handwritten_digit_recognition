import streamlit as st
import pandas as pd
import joblib

# Streamlit page configuration
st.set_page_config(page_title="Digit Recognition App", layout="centered")
st.title("✍️ Handwritten Digit Recognition App")
st.markdown("Upload a CSV file containing handwritten digit features to predict the digits.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("digit_classifier_model.pkl")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file here", type=["csv"])

if model and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Drop unwanted columns
        drop_cols = ['target', 'label', 'Unnamed: 0']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # Validation: ensure at least 200 columns
        if df.shape[1] < 200:
            st.error("⚠️ Input file must have at least 200 feature columns.")
        else:
            X = df.iloc[:, :200]

            st.subheader("📄 Preview of Input Data")
            st.dataframe(df.head())

            # Perform prediction
            predictions = model.predict(X)
            df["Prediction"] = predictions

            st.success("✅ Digit prediction completed successfully!")
            st.subheader("🔢 Prediction Results")
            st.dataframe(df[["Prediction"]])

            # Prediction distribution
            st.subheader("📊 Prediction Distribution")
            st.bar_chart(df["Prediction"].value_counts().sort_index())

            # Download predictions
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="digit_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"⚠️ Error occurred during processing: {e}")
elif model is None:
    st.warning("⚠️ Model is not loaded properly. Please check your model file.")
else:
    st.info("📌 Please upload a CSV file with 200 features to get started.")