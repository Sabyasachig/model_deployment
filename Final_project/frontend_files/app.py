
import streamlit as st
import requests
import pandas as pd

# =========================
# CONFIGURATION
# =========================
API_BASE_URL = "https://sabyasachighosh-supercart-be.hf.space"

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    layout="wide"
)

st.title("🛒 SuperKart Sales Prediction System")
st.markdown(
    "Predict product store sales using a trained XGBoost model. "
    "Supports both **single predictions** and **batch predictions via CSV upload**."
)

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🔹 Single Prediction", "📁 Batch Prediction (CSV)"])

# =========================
# SINGLE PREDICTION TAB
# =========================
with tab1:
    st.subheader("Single Record Prediction")

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            Product_Weight = st.number_input("Product Weight", value=10.0)
            Product_Sugar_Content = st.selectbox(
                "Product Sugar Content",
                ["Low Sugar", "Regular", "No Sugar"]
            )
            Product_Allocated_Area = st.number_input(
                "Product Allocated Area", value=100
            )
            Product_Type = st.selectbox(
                "Product Type",
                [
                    "Dairy",
                    "Soft Drinks",
                    "Snack Foods",
                    "Fruits and Vegetables",
                    "Frozen Foods",
                    "Household",
                    "Meat",
                    "Canned",
                    "Baking Goods",
                    "Health and Hygiene",
                    "Breads",
                    "Breakfast",
                    "Seafood",
                    "Others"
                ]
            )
            Product_MRP = st.number_input("Product MRP", value=200)

        with col2:
            Store_Id = st.text_input("Store ID", "OUT049")
            Store_Establishment_Year = st.number_input(
                "Store Establishment Year", value=2000
            )
            Store_Size = st.selectbox(
                "Store Size", ["Small", "Medium", "High"]
            )
            Store_Location_City_Type = st.selectbox(
                "Store City Type", ["Tier 1", "Tier 2", "Tier 3"]
            )
            Store_Type = st.selectbox(
                "Store Type",
                [
                    "Supermarket Type1",
                    "Supermarket Type2",
                    "Supermarket Type3",
                    "Grocery Store"
                ]
            )

        submit_single = st.form_submit_button("Predict Sales")

    if submit_single:
        payload = {
            "Product_Weight": Product_Weight,
            "Product_Sugar_Content": Product_Sugar_Content,
            "Product_Allocated_Area": Product_Allocated_Area,
            "Product_Type": Product_Type,
            "Product_MRP": Product_MRP,
            "Store_Id": Store_Id,
            "Store_Establishment_Year": Store_Establishment_Year,
            "Store_Size": Store_Size,
            "Store_Location_City_Type": Store_Location_City_Type,
            "Store_Type": Store_Type
        }

        with st.spinner("Predicting sales..."):
            response = requests.post(
                f"{API_BASE_URL}/v1/predict",  # ✅ FIXED ENDPOINT
                json=payload,
                timeout=30
            )

        if response.status_code == 200:
            st.success(
                f"💰 Predicted Store Sales: ₹{response.json()['predicted_product_store_sales']}"
            )
        else:
            st.error(
                f"❌ Prediction failed: {response.json().get('error', 'Unknown error')}"
            )

# =========================
# BATCH PREDICTION TAB
# =========================
with tab2:
    st.subheader("Batch Prediction via CSV Upload")

    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=["csv"]
    )

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        st.markdown("### 📄 Uploaded Data Preview")
        st.dataframe(input_df.head())

        if st.button("Run Batch Prediction"):
            with st.spinner("Running batch prediction..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv"
                    )
                }

                response = requests.post(
                    f"{API_BASE_URL}/v1/predict/batch",
                    files=files,
                    timeout=60
                )

            if response.status_code == 200:
                preds = response.json()["predictions"]

                pred_df = pd.DataFrame(preds)
                result_df = pd.concat(
                    [input_df.reset_index(drop=True), pred_df],
                    axis=1
                )

                st.success("✅ Batch prediction completed successfully!")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="superkart_batch_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(
                    f"❌ Batch prediction failed: {response.json().get('error', 'Unknown error')}"
                )
