import streamlit as st

st.title("💰 회계내역 오류 검사기")

st.write("PDF나 이미지를 업로드하면 금액 오류를 검사하는 앱입니다.")

uploaded_file = st.file_uploader(
    "회계내역 PDF 또는 이미지 업로드",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.success("파일 업로드 성공!")
    st.write("파일 이름:", uploaded_file.name)
