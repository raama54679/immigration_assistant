# app.py
import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

API_URL = os.environ.get("IMMIGRATION_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Immigration Assistant", page_icon="🧭", layout="wide")
st.title("🧭 Immigration Assistant (Info Search)")
st.caption("This tool helps you find nearby informational resources. Not legal advice.")

with st.form("intake"):
    col1, col2 = st.columns(2)

    with col1:
        location = st.text_input("Your City/Region", placeholder="e.g., Ann Arbor, MI")
        desired_choice = st.selectbox(
            "What are you trying to do?",
            [
                "Work Authorization",
                "Student Visa",
                "Lawful Permanent Residence",
                "Citizenship",
                "Other",
            ],
        )
        if desired_choice == "Other":
            other_goal = st.text_input(
                "Describe your goal",
                placeholder="e.g., Asylum referral, TPS, DACA, etc."
            )
            desired = other_goal.strip()
        else:
            desired = desired_choice

    with col2:
        creds = st.multiselect(
            "What documents/credentials do you currently have?",
            [
                "Passport", "I-94", "Birth Certificate", "SEVIS I-20",
                "Green Card", "Job Offer", "Sponsor", "Tax Returns"
            ],
        )
        radius = st.slider("Search radius (km)", 10, 200, 50, 10)
        max_iter = st.slider("Max search iterations", 1, 8, 5)

    submitted = st.form_submit_button("Find Nearby Help")

if submitted:
    if not location or not desired:
        st.error("Please fill in your location and goal.")
    else:
        payload = {
            "user_location": location,
            "desired_outcome": desired,
            "credentials": creds or [],
            "radius_km": radius,
            "max_iterations": max_iter,
        }

        with st.spinner("Searching…"):
            try:
                r = requests.post(f"{API_URL}/infer", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        left, right = st.columns([2, 1])

        # --- Left column: candidates ---
        with left:
            st.subheader("Top candidates")
            candidates = data.get("candidates", [])
            if not candidates:
                st.write("No candidates found.")
            else:
                for c in candidates:
                    title = c.get("title", "(no title)")
                    url = c.get("url", "")
                    st.markdown(f"### [{title}]({url})" if url else f"### {title}")

                    # Confidence (0–1) -> percentage
                    score = c.get("score")
                    if isinstance(score, (int, float)):
                        st.caption(f"Confidence: {score*100:.0f}%")

                    st.write(c.get("content", ""))

                    # Keep other metadata if you want it
                    meta = {k: v for k, v in c.items() if k in ("query",)}
                    if meta:
                        st.caption(json.dumps(meta, indent=2))

                    st.divider()

        # --- Right column: result + trace ---
        with right:
            st.subheader("Result")
            if data.get("success"):
                best = data.get("nearest_option") or {}
                best_title = best.get("title", "(no title)")
                best_url = best.get("url", "")
                st.success("A nearby option was found based on your inputs.")
                if best_url:
                    st.markdown(f"**Recommendation:** [{best_title}]({best_url})")
                else:
                    st.markdown(f"**Recommendation:** {best_title}")

                # Confidence for the recommended item if present
                best_score = best.get("score")
                if isinstance(best_score, (int, float)):
                    st.caption(f"Confidence: {best_score*100:.0f}%")

            else:
                st.warning(
                    "No confident match within the iteration limit. "
                    "Consider broadening your search terms or radius."
                )

            missing = data.get("missing_requirements", [])
            if missing:
                st.info("You may want to gather the following before contacting providers:")
                for m in missing:
                    st.write(f"• {m}")

            st.subheader("Trace")
            for step in data.get("trace", []):
                st.code(step)
