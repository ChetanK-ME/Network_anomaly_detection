import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Section
st.title("🔒 Network Anomaly Detection System")
st.write("This application uses machine learning to detect potential cybersecurity threats in network traffic. Enter the network traffic attributes below to analyze for anomalies.")

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

if model is None or scaler is None:
    st.error("Please train the model first using train_model.py")
    st.stop()

# Create input fields in three columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Network Connection Details")
    src_port = st.number_input("Source Port", min_value=0, max_value=65535, value=80, help="Source port number (0-65535)")
    dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=443, help="Destination port number (0-65535)")
    protocol = st.number_input("Protocol", min_value=0, value=6, help="Protocol number (e.g., 6 for TCP, 17 for UDP)")
    flow_duration = st.number_input("Flow Duration (ms)", min_value=0, value=1000, help="Duration of the network flow in milliseconds")

with col2:
    st.subheader("Packet Statistics")
    pkt_count = st.number_input("Packet Count", min_value=0, value=10, help="Total number of packets in the flow")
    pkt_size_avg = st.number_input("Average Packet Size", min_value=0, value=1000, help="Average size of packets in bytes")
    bytes_sent = st.number_input("Bytes Sent", min_value=0, value=1000, help="Total bytes sent in the flow")
    bytes_received = st.number_input("Bytes Received", min_value=0, value=1000, help="Total bytes received in the flow")

with col3:
    st.subheader("Connection Patterns")
    conn_count_last_10s = st.number_input("Connection Count (Last 10s)", min_value=0, value=5, help="Number of connections in the last 10 seconds")
    same_dst_count = st.number_input("Same Destination Count", min_value=0, value=3, help="Number of connections to the same destination")
    srv_serror_rate = st.number_input("Service Error Rate", min_value=0.0, max_value=1.0, value=0.0, help="Rate of service errors (0-1)")
    dst_host_srv_count = st.number_input("Destination Host Service Count", min_value=0, value=5, help="Number of services on destination host")
    dst_host_same_srv_rate = st.number_input("Destination Host Same Service Rate", min_value=0.0, max_value=1.0, value=0.5, help="Rate of same service connections (0-1)")
    entropy = st.number_input("Entropy", min_value=0.0, value=1.0, help="Entropy value of the flow (0-1)")
    honeypot_flag = st.number_input("Honeypot Flag", min_value=0, max_value=1, value=0, help="Indicates if the connection is from a honeypot (0 or 1)")

# Create a button for prediction
if st.button("🔍 Detect Threat"):
    try:
        # Prepare input data
        input_data = np.array([[
            src_port, dst_port, protocol, flow_duration, pkt_count,
            pkt_size_avg, bytes_sent, bytes_received, conn_count_last_10s,
            same_dst_count, srv_serror_rate, dst_host_srv_count,
            dst_host_same_srv_rate, entropy, honeypot_flag
        ]])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Display results
        if prediction[0] == -1:
            st.error("🚨 Anomaly Detected! Potential security threat identified.")
            st.write("### Recommended Actions:")
            st.write("- Investigate the network traffic")
            st.write("- Check firewall logs")
            st.write("- Review security policies")
            st.write("- Consider blocking the source IP")
        else:
            st.success("✅ No anomaly detected. Network traffic appears normal.")
            
        # Show anomaly score with a progress bar
        anomaly_score = model.score_samples(scaled_data)
        score = anomaly_score[0]
        
        # Normalize score to be between 0 and 1
        min_score = -0.5
        max_score = 0.5
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        st.write("### Anomaly Score")
        st.progress(normalized_score)
        st.write(f"Score: {score:.4f}")
        
        # Interpretation of the score
        st.write("### Score Interpretation:")
        st.write("- Lower scores indicate higher likelihood of anomaly")
        st.write("- Scores closer to 0 suggest normal behavior")
        st.write("- Negative scores typically indicate anomalies")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")