import streamlit as st
import fitz  # PyMuPDF
import requests
import json

st.set_page_config(page_title="GCA Extraction Agent", page_icon="⚡", layout="centered")
st.markdown("""
<div style="background:#1c2b3a; color:#e8dfc8; padding:20px; border-radius:8px;">
<h2>⚡ GCA Extraction Agent</h2>
<p>Upload CRAC / GCA PDF for automated data extraction</p>
</div>
""", unsafe_allow_html=True)

FIELDS = [
    "project", "grid_operator", "company", "type", "reference", "location",
    "date_of_signature", "date_initial_gco_request", "injection_capacity",
    "consumption_capacity", "grid_voltage", "inverters",
    "reactive_energy_requirements", "plant_substation", "grid_substation",
    "connection_works", "equipment_plant_substation", "hv_protection_category",
    "hz_filter", "downtime", "other", "total_costs_excl_vat",
    "quote_part_excl_vat", "timing"
]

PROMPT = """
You extract structured information from French grid connection agreements (CRAC / Enedis).
Return ONLY valid JSON.
All answers must be in English.
If a field is missing return "Info not found".
Fields: """ + ", ".join(FIELDS)

def extract_pdf_text(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

uploaded = st.file_uploader("Upload CRAC / GCA PDF", type="pdf")

if uploaded:
    st.info(f"📄 {uploaded.name}")
    text = extract_pdf_text(uploaded)

    if st.button("⚡ Extract Data"):
        with st.spinner("Extracting data using Hugging Face API…"):
            try:
                headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}
                payload = {
                    "inputs": PROMPT + "\n\nDOCUMENT:\n" + text[:15000],
                    "parameters": {"max_new_tokens": 1500}
                }

                response = requests.post(
                    "https://router.huggingface.co/models/NousResearch/Llama-2-7b-hf",
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                # Robust parsing of HF router response
                resp_json = response.json()
                st.write(resp_json)  # for debugging if needed

                if "generated_text" in resp_json:
                    output_text = resp_json["generated_text"]
                elif "data" in resp_json and len(resp_json["data"]) > 0:
                    output_text = resp_json["data"][0]["generated_text"]
                else:
                    output_text = ""

                if not output_text.strip():
                    st.error("❌ Hugging Face model returned empty output")
                else:
                    data = json.loads(output_text)
                    st.success("✅ Extraction complete")
                    st.json(data)

                    # Export CSV
                    csv_content = "Field,Value\n" + "\n".join(
                        f"{field},{data.get(field,'')}" for field in FIELDS
                    )

                    st.download_button(
                        "⬇ Download CSV",
                        csv_content,
                        file_name=f"GCA_{data.get('project','output')}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")
                if 'response' in locals():
                    st.text(response.text)
