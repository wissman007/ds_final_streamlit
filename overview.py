import streamlit as st

st.title("Overview")

st.header("Starting point - the xView2 challenge")

st.markdown("""
* **GOAL**:
    * Automatically assess building damage.
    * Localize Buildings and Score Damage.
* **AIM**: Help disaster response teams to assess damage rapidly and allocate resources effectively when a disaster occur
""")

st.header("The dataset")

col1, col2, col3 = st.columns(3)

col1.success("18 336 images")
col2.success("850 000 buildings")
col3.success("15 countries")

col1.image("data/pre-intro.jpg")
col2.image("data/post-intro.jpg")
col3.image("data/mask-intro.jpg")

st.header("Approach")

st.write("Image segmentation")
