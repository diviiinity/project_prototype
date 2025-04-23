import streamlit as st
import time
import os
import pandas as pd
import piexif
import base64
from openai import OpenAI
from PIL import Image
from opencage.geocoder import OpenCageGeocode

# Load datasets
dumping_data = pd.read_csv("data/dumping_types.csv")
tips_data = pd.read_csv("data/reporting_tips.csv")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Function to analyze pollution
def get_ai_analysis(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI expert in illegal dumping analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Feature1: Extract location from image EXIF metadata
def extract_location(image_file):
    try:
        img = Image.open(image_file)
        exif_dict = piexif.load(img.info['exif'])

        gps = exif_dict.get("GPS")
        if not gps:
            return "Unknown Location"

        def convert_to_degrees(value):
            d = value[0][0] / value[0][1]
            m = value[1][0] / value[1][1]
            s = value[2][0] / value[2][1]
            return d + (m / 60.0) + (s / 3600.0)

        lat = convert_to_degrees(gps[piexif.GPSIFD.GPSLatitude])
        if gps[piexif.GPSIFD.GPSLatitudeRef] != b'N':
            lat = -lat

        lon = convert_to_degrees(gps[piexif.GPSIFD.GPSLongitude])
        if gps[piexif.GPSIFD.GPSLongitudeRef] != b'E':
            lon = -lon

        return reverse_geocode(lat, lon)
    except Exception as e:
        return f"Unknown Location (Error: {e})"

# Reverse geocode
OPENCAGE_API_KEY = "d023d87ba6be4e3785be92480a92d27c"
geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
def reverse_geocode(lat, lon):
    try:
        results = geocoder.reverse_geocode(lat, lon)
        if results and len(results):
            return results[0]['formatted']
    except Exception as e:
        print("Reverse geocoding failed:", e)
    return f"{lat:.6f}, {lon:.6f}"

# Feature2: Detect waste type using OpenAI Vision
def detect_waste_type(image_file):
    image_bytes = image_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_file.seek(0)

    vision_prompt = [
        {
            "role": "system",
            "content": "You are a waste classification expert. You analyze images and return the type of waste present (e.g., plastic, oil, metal, glass, organic, tire, etc)."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Identify the types of waste in this image (e.g., plastic, oil, glass, food, e-waste, etc)."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=vision_prompt,
        max_tokens=200
    )

    result_text = response.choices[0].message.content.strip()
    detected_items = [item.strip().lower() for item in result_text.split(",") if item]
    return detected_items or ["unknown"]

# Feature3: Contact local authorities based on detected waste types
def contact_authorities(location, detected_types):
    matched_agencies = set()

    for dtype in detected_types:
        for _, row in dumping_data.iterrows():
            if dtype.lower() in row['type'].lower():
                matched_agencies.add(row['disposal_agency'])

    if matched_agencies:
        agencies_str = ", ".join(matched_agencies)
    else:
        agencies_str = "No specific agencies found."

    return f"Local authorities and cleanup agencies contacted at **{location}**: {agencies_str}."

# Get tips based on dump categories
def get_reporting_tips(detected_types):
    categories = set()
    for dtype in detected_types:
        matches = dumping_data[dumping_data['type'].str.lower().str.contains(dtype.lower())]
        categories.update(matches['category'].tolist())
    matched_tips = tips_data[tips_data['category'].isin(categories)]
    return matched_tips['tips'].tolist()


# Streamlit CSS Background & Heading
page_element="""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/15265064/pexels-photo-15265064/free-photo-of-seagulls-on-the-shore.jpeg");
    background-size: cover;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: white;
    text-align: center;
}

[data-testid="stSidebar"] p {
    line-height: 1.75;
}

[data-testid="stMain"] p {
    font-size: 1.2em;
}
[data-testid="stExpander"]{
    background-color: transparent;
}

{
    background-color: lightsteelblue;
}

[data-testid="stExpander"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stNumberInputContainer"] div, 
[data-testid="stNumberInputContainer"] div button, 
.st-ay, .st-aw  {
    background-color: lightsteelblue;
    color: black;
}

p {
    font-weight: bold;
}

summary:hover {
    color: steelblue !important;
}

[data-testid="stMarkdown"] {
    background-color: rgba(255, 255, 255, .80);
    padding: 10px;
    border-radius: 10px;
}

button:hover, button:focus {
    background-color: steelblue !important;
    border-color: white !important;
    color: white !important;
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns([1, 1, 1])
with col2:
    st.sidebar.title("WELCOME!")
    st.sidebar.image("images/aqua.PNG")
    st.sidebar.divider()
    st.sidebar.markdown("Aqua Alert was made to help local communities care for their waterways using an accessible and intuitive app!")
    st.sidebar.markdown("For a demo of our app, please visit this link here: -insert link in the future-.")


# --- Streamlit UI ---
st.title("AquaAlert: Illegal Dumping Reporting Tool")
st.subheader("📸 Just upload a photo — we’ll take care of the rest!")
st.markdown("AI-powered tool to report illegal dumping, contact agencies, and learn how to reduce pollution.")

# Upload section
uploaded_file = st.file_uploader("Upload a photo of the polluted site", type=["jpg", "jpeg", "png"])
description = st.text_area("Optional: Add a brief description (if you'd like)", placeholder="e.g., Looks like a spill near the river...")

if st.button("Report"):
    if uploaded_file:
        with st.spinner("Analyzing the image and contacting local agencies..."):
            time.sleep(2)

            # Step 1: Detect waste and location
            detected_types = detect_waste_type(uploaded_file)
            uploaded_file.seek(0)  # reset to read again
            location = extract_location(uploaded_file)

            # Step 2: Display basic info
            st.subheader("📍 Report Summary")
            st.markdown(f"**Location Detected:** {location}")
            st.markdown(f"**Waste Type Detected:** {', '.join([w.title() for w in detected_types])}")

            # Step 3: AI Summary
            ai_prompt = f"""
            Analyze this illegal dumping report. Location: {location}.
            Detected waste types: {', '.join(detected_types)}.
            Additional description: {description}.
            What kind of pollution is this and what actions can help?
            """
            analysis_result = get_ai_analysis(ai_prompt)

            st.subheader("🌍 Pollution Incident Summary:")
            st.write(analysis_result)

            # Step 4: Notify authorities
            st.subheader("📡 Authorities Notified:")
            st.success(contact_authorities(location, detected_types))

            # Step 5: Show helpful tips
            st.subheader("💡 Helpful Tips:")
            for tip in get_reporting_tips(detected_types):
                st.markdown(f"- {tip}")

            st.success("✅ Thanks for reporting and making a difference!")
    else:
        st.warning("Please upload an image to begin.")
