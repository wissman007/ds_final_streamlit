# Entrypoint for the project

import streamlit as st

intro_page = st.Page("overview.py", title="Overview")
past_page = st.Page("past_disasters.py", title="Past disaster analysis")
model_page = st.Page("model_page.py", title="Model development & training")
future_page = st.Page("damage_estimator.py", title="Damage estimator")

pg = st.navigation([intro_page, past_page, model_page, future_page])
st.set_page_config(page_title="Disaster damage estimator")
pg.run()