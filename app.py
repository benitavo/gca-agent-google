
import streamlit as st
import google.generativeai as genai
import fitz
import json

st.set_page_config(page_title="GCA Extraction Agent", page_icon="⚡")

st.title("⚡ GCA Extraction Agent")

FIELDS = [
    "project",
    "grid_operator",
    "company",
    "type",
    "reference",
    "location",
    "date_of_signature",
    "date_initial_gco_request",
    "injection_capacity",
    "consumption_capacity",
    "grid_voltage",
    "inverters",
    "reactive_energy_requirements",
    "plant_substation",
    "grid_substation",
    "connection_works",
    "equipment_plant_substation",
    "hv_protection_category",
    "hz_filter",
    "downtime",
    "other",
    "total_costs_excl_vat",
    "quote_part_excl_vat",
    "timing"
]

PROMPT = """
You extract structured information from French grid connection agreements (CRAC / Enedis).

Return ONLY valid JSON.

Fields:
project
grid_operator
company
type
reference
location
date_of_signature
date_initial_gco_request
injection_capacity
consumption_capacity
grid_voltage
inverters
reactive_energy_requirements
plant_substation
grid_substation
connection_works
equipment_plant_substation
hv_protection_category
hz_filter
downtime
other
total_costs_excl_vat
quote_part_excl_vat
timing

All answers must be in English.

If information is missing return "Info not found".
"""

def extract_pdf_text(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

uploaded = st.file_uploader("Upload CRAC / GCA PDF", type="pdf")

if uploaded:

    st.info(uploaded.name)

    text = extract_pdf_text(uploaded)

    if st.button("Extract data"):

        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        response = model.generate_content(
            PROMPT + "\n\nDOCUMENT:\n" + text[:20000]
        )

        try:

            data = json.loads(response.text)

            st.success("Extraction complete")

            st.json(data)

            csv = "\n".join(
                f"{k},{data.get(k,'')}" for k in FIELDS
            )

            st.download_button(
                "Download CSV",
                csv,
                file_name="gca_extraction.csv"
            )

        except:

            st.write(response.text)
