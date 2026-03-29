import os
# Warning কমানোর জন্য (train.py এর মতো)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Smart Garbage Classifier", 
    page_icon="🗑️", 
    layout="centered"
)

# Load model with cache
@st.cache_resource
def load_garbage_model():
    model = load_model('models/model.keras', compile=False)   # compile=False দিয়ে লোড করি
    # আবার compile করি (warning দূর করার জন্য)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = load_garbage_model()

# Class names (train এ যেভাবে ছিল ঠিক সেভাবে রাখো)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# বাংলায় সুন্দর ব্যাখ্যা
explanations = {
    'cardboard': '🟫 এটা **কার্ডবোর্ড**। খুব সহজে রিসাইকেল হয়।',
    'glass': '🟢 এটা **কাঁচ**। ১০০% রিসাইকেলযোগ্য।',
    'metal': '⚙️ এটা **ধাতু** (টিন/ক্যান)। মূল্যবান আবর্জনা।',
    'paper': '📄 এটা **কাগজ**। গাছ বাঁচাতে সাহায্য করে।',
    'plastic': '🟦 এটা **প্লাস্টিক**। আলাদা করে ফেললে পরিবেশ ভালো থাকে।',
    'trash': '🗑️ এটা **মিক্সড/সাধারণ আবর্জনা**।'
}

# ====================== UI ======================
st.title("🗑️ স্মার্ট আবর্জনা শনাক্তকারী")
st.subheader("Tech Fest প্রজেক্ট — ছবি দাও, AI বলে দেবে কোন ধরনের আবর্জনা!")

st.write("**📸 ক্যামেরা দিয়ে তুলুন অথবা ছবি আপলোড করুন**")

col1, col2 = st.columns(2)

with col1:
    camera_img = st.camera_input("লাইভ ক্যামেরা")

with col2:
    uploaded_file = st.file_uploader("ছবি আপলোড করুন", type=["jpg", "jpeg", "png"])

# যেকোনো একটা ছবি নেবে
img_file = camera_img if camera_img is not None else uploaded_file

if img_file is not None:
    try:
        # Image preprocessing
        img = Image.open(img_file).convert('RGB')
        st.image(img, caption="📷 আপনার দেওয়া ছবি", use_column_width=True)
        
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_idx]
        confidence = float(np.max(prediction)) * 100

        # Result Display
        st.success(f"✅ **এটা {predicted_class.upper()} ধরনের আবর্জনা!**")
        st.info(f"🔢 Confidence: **{confidence:.2f}%**")

        st.subheader("📝 বিস্তারিত ব্যাখ্যা:")
        st.write(explanations.get(predicted_class, "ব্যাখ্যা পাওয়া যায়নি।"))

        st.caption("💡 টিপস: সবসময় আবর্জনা আলাদা করে ফেলুন — পরিবেশ বাঁচুক!")

    except Exception as e:
        st.error(f"❌ কিছু সমস্যা হয়েছে: {str(e)}")

st.caption("Made with ❤️ for Tech Fest | TensorFlow + Streamlit")