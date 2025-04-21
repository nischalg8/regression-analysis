import streamlit as st
import numpy as np
import pickle


model = pickle.load(open('best_random_forest.pkl', 'rb'))
scaler1 = pickle.load(open('scaler1.pkl', 'rb')) 

st.title("📱 Mobile Price Predictor")

st.markdown("Fill in the mobile phone details below to predict its estimated market price.")


st.header("📐 Physical Specifications")

# RAM 
ram = st.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16, 24])

# Internal Storage options
inbuilt_memory = st.selectbox("Inbuilt Memory (GB)", [8, 16, 32, 64, 128, 256, 512])

# Battery
battery = st.number_input("Battery Capacity (mAh)", min_value=1000, max_value=7000, step=100)
# Display size
screen_size = st.number_input("Display Size (inches)", min_value=4.0, max_value=7.5, step=0.1)

st.header("📏 Screen Resolution")
width = st.number_input("Screen Width (pixels)", min_value=1, step=1)
height = st.number_input("Screen Height (pixels)", min_value=1, step=1)

#PPI
if width > 0 and height > 0 and screen_size > 0:
    ppi = ((width**2 + height**2) ** 0.5) / screen_size
else:
    ppi = 0 

st.write(f"PPI (Pixels Per Inch): {ppi:.2f}")


st.header("📸 Camera")

back_cam = st.slider("Back Camera (MP)", 2, 200, step=1)
front_cam = st.slider("Front Camera (MP)", 2, 60, step=1)
rear_cam_count = st.selectbox("Number of Rear Cameras", [1, 2, 3, 4])

st.header("⚙️ Brand & Processor")

brands = [
    "ASUS", "COOLPAD", "GIONEE", "GOOGLE", "HONOR", "HUAWEI", "IQOO", "ITEL",
    "LAVA", "LENOVO", "LG", "MOTOROLA", "NOTHING", "ONEPLUS", "OPPO", "POCO",
    "REALME", "SAMSUNG", "TCL", "TECNO", "VIVO", "XIAOMI"
]
brand = st.selectbox("Select Brand", brands)
brand_encoding = [1 if b == brand else 0 for b in brands]

processors = ["dimensity", "exynos", "helio", "other", "snapdragon", "tensor", "tiger", "unisoc"]
processor = st.selectbox("Select Processor", processors)
processor_encoding = [1 if p == processor else 0 for p in processors]


input_features = np.array([
    ram, battery, screen_size, inbuilt_memory, back_cam, front_cam, rear_cam_count, ppi
] + brand_encoding + processor_encoding).reshape(1, -1)


scaled_numeric = scaler1.transform(input_features[:, :8])

input_scaled = np.concatenate([scaled_numeric, input_features[:, 8:]], axis=1)


if st.button("Predict Price 💰"):
    predicted_price = model.predict(input_scaled)[0]
    st.success(f"📱 Estimated Price: ₹{predicted_price:,.2f}")
