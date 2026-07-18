"""
app.py
تطبيق Streamlit للتنبؤ باحتمالية إدمان اللعب (addiction_binary)
باستخدام موديل AdaBoost المدرب في train_model.py

طريقة التشغيل:
    streamlit run app.py
(لازم تشغل train_model.py قبل كده عشان الملفات .pkl تتولد)
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Gaming Addiction Predictor", page_icon="🎮", layout="centered")

MODEL_FILES = ["ada_model.pkl", "scaler.pkl", "encoders.pkl", "meta.pkl"]

if not all(os.path.exists(f) for f in MODEL_FILES):
    st.error(
        "❌ ملفات الموديل مش موجودة.\n\n"
        "شغّل الأمر ده الأول في نفس الفولدر (لازم يكون فيه gaming_addiction.csv):\n\n"
        "`python train_model.py`"
    )
    st.stop()

model = joblib.load("ada_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
meta = joblib.load("meta.pkl")

cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]
feature_order = meta["feature_order"]

st.title("🎮 Gaming Addiction Predictor")
st.write("املأ بيانات اللاعب عشان نتوقع احتمالية الإدمان (Addiction Binary).")

with st.form("prediction_form"):
    st.subheader("البيانات الأساسية والفئوية")
    cat_inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(cat_cols):
        options = list(encoders[col].classes_)
        with cols[i % 2]:
            cat_inputs[col] = st.selectbox(col, options)

    st.subheader("البيانات الرقمية")
    num_inputs = {}
    cols2 = st.columns(3)
    for i, col in enumerate(num_cols):
        with cols2[i % 3]:
            num_inputs[col] = st.number_input(col, value=0.0, format="%.2f")

    submitted = st.form_submit_button("توقع الآن 🔮")

if submitted:
    row = {**cat_inputs, **num_inputs}
    input_df = pd.DataFrame([row])[feature_order]

    # تطبيق نفس الـ LabelEncoder اللي اتدرب في train_model.py
    for col in cat_cols:
        le = encoders[col]
        input_df[col] = input_df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        input_df[col] = le.transform(input_df[col])

    # تطبيق نفس الـ Scaler
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ النتيجة: اللاعب مصنّف كـ **مدمن** (Addiction = 1)")
    else:
        st.success(f"✅ النتيجة: اللاعب مصنّف كـ **غير مدمن** (Addiction = 0)")

    st.metric("احتمالية الإدمان", f"{proba * 100:.1f}%")
