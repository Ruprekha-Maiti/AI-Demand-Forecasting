
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("📊 AI-Powered Demand Forecasting System")
st.subheader("Retail Business Analytics Dashboard")

st.divider()

# =====================================
# CSV UPLOAD
# =====================================

st.header("📁 Upload Sales Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="Upload a sales dataset in CSV format."
)

# NO DEMO DATA
if uploaded_file is None:

    st.info(
        "👆 Please upload a CSV sales dataset to get started."
    )

    st.markdown("""
    ### Welcome to Retail Business Analytics!

    Upload your own sales dataset to explore:

    - 📊 Business Overview
    - 📈 Sales Analytics
    - 🏆 Product Performance
    - 📦 Inventory Insights
    - 💡 Business Recommendations
    - 🤖 AI Demand Forecasting
    - ⚠️ Smart Inventory Alerts
    """)

    st.stop()

# =====================================
# LOAD UPLOADED DATASET ONLY
# =====================================

try:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("The uploaded CSV file is empty!")
        st.stop()

    st.success("Dataset loaded successfully!")

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# =====================================
# COLUMN NAME COMPATIBILITY
# =====================================

# Support the newly created CSV format too.

column_mapping = {
    "Units Sold": "Sales",
    "Unit Price": "Price"
}

df = df.rename(columns=column_mapping)

required_columns = [
    "Date", "Product", "Category",
    "Sales", "Stock", "Price"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "CSV is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Required columns: Date, Product, Category, "
        "Sales (or Units Sold), Stock, "
        "Price (or Unit Price)"
    )

    st.stop()

# =====================================
# DATA CLEANING
# =====================================

df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

for col in ["Sales", "Stock", "Price"]:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df["Product"] = df["Product"].astype("string").str.strip()
df["Category"] = df["Category"].astype("string").str.strip()

df = df.dropna(subset=required_columns)

df = df[
    (df["Product"] != "") &
    (df["Category"] != "")
]

if df.empty:
    st.error("No valid sales records found!")
    st.stop()

if (df["Sales"] < 0).any() or (df["Stock"] < 0).any():
    st.error("Sales and Stock cannot contain negative values.")
    st.stop()

if (df["Price"] < 0).any():
    st.error("Price cannot contain negative values.")
    st.stop()

df = df.sort_values("Date").reset_index(drop=True)

# =====================================
# DATASET PREVIEW
# =====================================

st.divider()

with st.expander("📋 Preview Sales Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

st.caption(
    f"Total valid records: {len(df)}"
)

# =====================================
# BUSINESS OVERVIEW
# =====================================

st.divider()
st.header("📊 Business Overview")

col1, col2, col3, col4 = st.columns(4)

total_sales = df["Sales"].sum()
total_products = df["Product"].nunique()
total_revenue = (df["Sales"] * df["Price"]).sum()

total_stock = (
    df.sort_values("Date")
    .groupby("Product")
    .tail(1)["Stock"]
    .sum()
)

col1.metric("Total Units Sold", int(total_sales))

col2.metric(
    "Total Products",
    total_products
)

col3.metric(
    "Total Revenue",
    f"₹{total_revenue:,.0f}"
)

col4.metric(
    "Total Stock",
    int(total_stock)
)

# =====================================
# SALES ANALYTICS
# =====================================

st.divider()
st.header("📈 Sales Analytics")

fig = px.line(
    df,
    x="Date",
    y="Sales",
    color="Product",
    markers=True,
    title="Product-wise Sales Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# PRODUCT PERFORMANCE
# =====================================

st.header("🏆 Product Performance")

product_sales = (
    df.groupby("Product")["Sales"]
    .sum()
    .reset_index()
    .sort_values("Sales", ascending=False)
)

fig2 = px.bar(
    product_sales,
    x="Product",
    y="Sales",
    color="Product",
    title="Total Sales by Product"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# INVENTORY INSIGHTS
# =====================================

st.header("📦 Inventory Insights")

latest_stock = (
    df.sort_values("Date")
    .groupby("Product")
    .tail(1)[["Product", "Stock"]]
)

st.dataframe(
    latest_stock,
    use_container_width=True
)

# =====================================
# BUSINESS RECOMMENDATIONS
# =====================================

st.header("💡 Business Recommendations")

best_product = product_sales.iloc[0]["Product"]

st.success(
    f"🏆 {best_product} is the best-selling product in this dataset."
)

st.info(
    "Monitor inventory regularly to avoid stock shortages."
)

# =====================================
# AI DEMAND FORECASTING
# =====================================

st.divider()
st.header("🤖 AI Demand Forecasting")

ml_data = df.copy()

if len(ml_data) < 4:

    st.warning(
        "Please upload at least 4 valid sales records "
        "for AI demand forecasting."
    )

    st.stop()

ml_data["Product_ID"] = pd.factorize(
    ml_data["Product"]
)[0]

ml_data["Day"] = (
    ml_data["Date"] - ml_data["Date"].min()
).dt.days

ml_data["Weekday"] = ml_data["Date"].dt.weekday

X = ml_data[
    ["Product_ID", "Day", "Weekday"]
]

y = ml_data["Sales"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.success(
    "Machine Learning model trained successfully!"
)

# =====================================
# FUTURE DEMAND PREDICTION
# =====================================

st.subheader("📈 Next 7 Days Demand Prediction")

future_dates = pd.date_range(
    start=ml_data["Date"].max() + pd.Timedelta(days=1),
    periods=7
)

forecast_rows = []

for product in ml_data["Product"].unique():

    product_id = ml_data.loc[
        ml_data["Product"] == product,
        "Product_ID"
    ].iloc[0]

    for date in future_dates:

        day = (date - ml_data["Date"].min()).days
        weekday = date.weekday()

        prediction = model.predict(
            pd.DataFrame([{
                "Product_ID": product_id,
                "Day": day,
                "Weekday": weekday
            }])
        )[0]

        forecast_rows.append({
            "Date": date,
            "Product": product,
            "Predicted Demand": max(
                0, int(round(prediction))
            )
        })

forecast_df = pd.DataFrame(forecast_rows)

fig3 = px.line(
    forecast_df,
    x="Date",
    y="Predicted Demand",
    color="Product",
    markers=True,
    title="AI Predicted Product Demand"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.dataframe(
    forecast_df,
    use_container_width=True
)

# =====================================
# DOWNLOAD FORECAST REPORT
# =====================================

st.subheader("📥 Download Forecast Report")

csv_report = forecast_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download AI Forecast Report",
    data=csv_report,
    file_name="AI_Demand_Forecast_Report.csv",
    mime="text/csv"
)

# =====================================
# SMART INVENTORY ALERTS
# =====================================

st.divider()
st.subheader("⚠️ Smart Inventory Alerts")

expected_demand = (
    forecast_df.groupby("Product")["Predicted Demand"]
    .sum()
    .reset_index()
)

inventory_check = latest_stock.merge(
    expected_demand,
    on="Product"
)

inventory_check["Status"] = inventory_check.apply(
    lambda row: "⚠️ Low Stock"
    if row["Stock"] < row["Predicted Demand"]
    else "✅ Stock Available",
    axis=1
)

st.dataframe(
    inventory_check,
    use_container_width=True
)

low_stock_count = (
    inventory_check["Status"] == "⚠️ Low Stock"
).sum()

if low_stock_count > 0:

    st.warning(
        f"{low_stock_count} products may need restocking."
    )

else:

    st.success(
        "All products have sufficient stock "
        "for predicted demand."
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.success(
    "AI Demand Forecasting and Business Analysis Completed!"
)

st.caption(
    "Academic Prototype | User Uploaded Dataset | "
    "Random Forest Regression"
)

st.caption(
    "Predictions are experimental and depend "
    "on the uploaded dataset."
)
