import streamlit as st
import fitz
import requests
import json
import ast

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
You are an expert at reading French grid connection agreements (CRAC / Enedis).
Respond ONLY with a valid JSON object — do NOT write explanations, examples, or markdown.
Use standard ASCII quotes (") only.
All values must be in English. If a field is missing, return "Info not found".
Fields: """ + ", ".join(FIELDS)

def extract_pdf_text(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def parse_json_safe(output_text):
    start = output_text.find("{")
    end = output_text.rfind("}") + 1
    if start == -1 or end == -1:
        return None
    json_str = output_text[start:end]
    json_str = json_str.replace("“", "\"").replace("”", "\"").replace("’", "'")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_str)
        except:
            return None

def call_model(text_chunk):
    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "messages": [{"role": "user", "content": PROMPT + "\n\nDOCUMENT:\n" + text_chunk}],
        "max_tokens": 2000
    }
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=300
    )
    if response.status_code != 200:
        st.error(f"❌ Hugging Face API returned {response.status_code}")
        st.text(response.text)
        return None
    result = response.json()
    output_text = result["choices"][0]["message"]["content"] if "choices" in result and len(result["choices"])>0 else ""
    return parse_json_safe(output_text)

uploaded = st.file_uploader("Upload CRAC / GCA PDF", type="pdf")

if uploaded:
    st.info(f"📄 {uploaded.name}")
    text = extract_pdf_text(uploaded)

    if st.button("⚡ Extract Data"):
        with st.spinner("Extracting data in chunks…"):
            chunk_size = 3000
            data = {}
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                partial_data = call_model(chunk)
                if partial_data:
                    # Fusionner les champs partiels
                    data.update({k: v for k, v in partial_data.items() if v != "Info not found"})
            if data:
                # remplir les champs manquants avec "Info not found"
                for f in FIELDS:
                    if f not in data:
                        data[f] = "Info not found"

                st.success("✅ Extraction complete")
                st.json(data)

                csv_content = "Field,Value\n" + "\n".join(f"{field},{data.get(field,'')}" for field in FIELDS)
                st.download_button(
                    "⬇ Download CSV",
                    csv_content,
                    file_name=f"GCA_{data.get('project','output')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ JSON could not be decoded")
