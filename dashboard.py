import streamlit as st
import sys
import os
from PIL import Image

# -------------------------------
# FIX PATH
# -------------------------------
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.main import AdversarialSystem

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# LOAD SYSTEM
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_system():
    return AdversarialSystem("models/saved/cifar_model.pth")

system = load_system()

# -------------------------------
# CIFAR-10 CLASS NAMES
# -------------------------------
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -------------------------------
# HEADER
# -------------------------------
st.title("🛡️ AI Adversarial Detection System")
st.markdown("### Secure AI + IPFS + Blockchain Audit Trail")
st.markdown("---")

# -------------------------------
# UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    # Save temp file
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 1])

    # -------------------------------
    # LEFT → IMAGE
    # -------------------------------
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    # -------------------------------
    # BUTTON
    # -------------------------------
    if st.button("🚀 Run Detection"):

        with st.spinner("🔍 Analyzing Image..."):
            result = system.process_image("temp.png")

        # -------------------------------
        # RIGHT → RESULT
        # -------------------------------
        with col2:

            st.subheader("📊 Detection Result")

            # -------------------------------
            # STATUS
            # -------------------------------
            if result["status"] == "SAFE":
                st.success("✅ SAFE IMAGE")
            else:
                st.error("🚨 ADVERSARIAL ATTACK DETECTED")

            # -------------------------------
            # RISK LEVEL
            # -------------------------------
            if result["risk_level"] == "LOW":
                st.info("🟢 Risk: LOW")
            elif result["risk_level"] == "MEDIUM":
                st.warning("🟡 Risk: MEDIUM")
            elif result["risk_level"] == "HIGH":
                st.error("🔴 Risk: HIGH")

            st.markdown("---")

            # -------------------------------
            # CONFIDENCE METRICS
            # -------------------------------
            st.markdown("### 📈 Confidence Metrics")

            m1, m2, m3 = st.columns(3)

            m1.metric("Confidence Before", f"{result['conf_before']:.3f}")
            m2.metric("Confidence After", f"{result['conf_after']:.3f}")
            m3.metric("Drop", f"{result['confidence_drop']:.4f}")

            st.markdown("---")

            # -------------------------------
            # MODEL BEHAVIOR
            # -------------------------------
            st.markdown("### 🤖 Model Behavior")

            st.write(f"Prediction Before: **{classes[result['pred_before']]}**")
            st.write(f"Prediction After: **{classes[result['pred_after']]}**")

            st.markdown("---")

            # -------------------------------
            # AI INSIGHT 🔥
            # -------------------------------
            st.markdown("### 🧠 AI Insight")

            if result["status"] == "SAFE":
                st.success("Model prediction is stable. No adversarial behavior detected.")
            else:
                st.warning("Prediction changed under perturbation → possible adversarial attack.")

            st.markdown("---")

            # -------------------------------
            # ACTION
            # -------------------------------
            st.markdown("### ⚙️ System Action")
            st.write(f"👉 {result['action']}")

            st.markdown("---")

            # -------------------------------
            # SECURITY STORAGE
            # -------------------------------
            st.markdown("### 🔐 Security & Storage")

            if result["status"] == "SAFE":
                st.info("No storage needed (image is safe)")
            else:
                if result["file_hash"]:
                    st.write("🔑 File Hash")
                    st.code(result["file_hash"])

                if result["ipfs_url"]:
                    st.success("🌐 Stored on IPFS")
                    st.markdown(f"[🔗 View File]({result['ipfs_url']})")

                if result["tx_hash"]:
                    st.success("⛓ Stored on Blockchain")
                    st.code(result["tx_hash"])

            st.markdown("---")

            # -------------------------------
            # TIMESTAMP
            # -------------------------------
            if "timestamp" in result:
                st.caption(f"🕒 {result['timestamp']}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built with ❤️ for Secure AI Systems")