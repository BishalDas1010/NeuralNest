import streamlit as st

st.title("NurelNest")

st.subheader("Upload the pdf,docoments and other notes")

# this section used to upload the pdf doc images
uploaded_files = st.file_uploader(label="this only accept",type = ["pdf","docx"],accept_multiple_files=True)

if st.button("Upload PDF", type="primary"):
    if uploaded_files:
        st.success("PDF uploaded successfully!")

        for file in uploaded_files:
            st.write("📄", file.name)
    else:
        st.warning("Please select a PDF first.")