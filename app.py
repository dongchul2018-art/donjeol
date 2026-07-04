import re
from io import BytesIO

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image


st.set_page_config(page_title="회계내역 오류 검사기", page_icon="💰")

st.title("💰 회계내역 오류 검사기")
st.write("PDF나 이미지를 업로드하면 금액을 읽고 합계 오류를 검사합니다.")


MONEY_PATTERN = re.compile(
    r"[-+]?\s*(?:₩\s*)?\d{1,3}(?:,\d{3})+(?:\s*원)?|[-+]?\s*(?:₩\s*)?\d+(?:\s*원)"
)

INCOME_WORDS = ["수입", "입금", "매출", "받음"]
EXPENSE_WORDS = ["지출", "출금", "비용", "결제", "사용", "지급"]
TOTAL_WORDS = ["합계", "총계", "계"]
BALANCE_WORDS = ["잔액", "남은돈", "남은 금액", "차액"]


def clean_money(text):
    text = text.replace("₩", "")
    text = text.replace("원", "")
    text = text.replace(",", "")
    text = text.replace(" ", "")
    return int(text)


def extract_money_from_line(line):
    found = MONEY_PATTERN.findall(line)
    amounts = []

    for item in found:
        try:
            amount = clean_money(item)
            if abs(amount) >= 100:
                amounts.append(amount)
        except:
            pass

    return amounts


def classify_line(line):
    if any(word in line for word in BALANCE_WORDS):
        return "잔액"

    has_income = any(word in line for word in INCOME_WORDS)
    has_expense = any(word in line for word in EXPENSE_WORDS)
    has_total = any(word in line for word in TOTAL_WORDS)

    if has_income and has_total:
        return "수입합계"

    if has_expense and has_total:
        return "지출합계"

    if has_income:
        return "수입"

    if has_expense:
        return "지출"

    if has_total:
        return "합계"

    return "기타"


def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    # PDF 안에 글자가 있는 경우
    for page in doc:
        text += page.get_text("text", sort=True) + "\n"

    # 스캔 PDF처럼 글자가 거의 안 읽히면 OCR 시도
    if len(text.strip()) < 30:
        text = ""
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = Image.open(BytesIO(pix.tobytes("png")))

            try:
                page_text = pytesseract.image_to_string(image, lang="kor+eng")
            except:
                page_text = pytesseract.image_to_string(image)

            text += page_text + "\n"

    return text


def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)

    try:
        text = pytesseract.image_to_string(image, lang="kor+eng")
    except:
        text = pytesseract.image_to_string(image)

    return text


def analyze_text(text):
    rows = []

    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        amounts = extract_money_from_line(line)

        if not amounts:
            continue

        category = classify_line(line)

        for amount in amounts:
            rows.append(
                {
                    "줄 번호": i,
                    "분류": category,
                    "금액": amount,
                    "원문": line,
                }
            )

    return pd.DataFrame(rows)


def check_result(df):
    messages = []

    if df.empty:
        return ["금액을 찾지 못했습니다. 이미지가 흐리거나 글자가 인식되지 않았을 수 있습니다."]

    income_sum = df[df["분류"] == "수입"]["금액"].sum()
    expense_sum = df[df["분류"] == "지출"]["금액"].sum()

    income_total = df[df["분류"] == "수입합계"]
    expense_total = df[df["분류"] == "지출합계"]
    balance = df[df["분류"] == "잔액"]

    if not income_total.empty:
        written_income = income_total.iloc[-1]["금액"]
        diff = income_sum - written_income

        if diff == 0:
            messages.append(f"✅ 수입 합계 정상: {income_sum:,}원")
        else:
            messages.append(
                f"❌ 수입 합계 오류 의심: 실제 계산 {income_sum:,}원 / 문서 표기 {written_income:,}원 / 차이 {diff:,}원"
            )
    else:
        messages.append(f"ℹ️ 문서에서 수입 합계를 찾지 못했습니다. 계산된 수입 합계는 {income_sum:,}원입니다.")

    if not expense_total.empty:
        written_expense = expense_total.iloc[-1]["금액"]
        diff = expense_sum - written_expense

        if diff == 0:
            messages.append(f"✅ 지출 합계 정상: {expense_sum:,}원")
        else:
            messages.append(
                f"❌ 지출 합계 오류 의심: 실제 계산 {expense_sum:,}원 / 문서 표기 {written_expense:,}원 / 차이 {diff:,}원"
            )
    else:
        messages.append(f"ℹ️ 문서에서 지출 합계를 찾지 못했습니다. 계산된 지출 합계는 {expense_sum:,}원입니다.")

    calculated_balance = income_sum - expense_sum

    if not balance.empty:
        written_balance = balance.iloc[-1]["금액"]
        diff = calculated_balance - written_balance

        if diff == 0:
            messages.append(f"✅ 잔액 정상: {calculated_balance:,}원")
        else:
            messages.append(
                f"❌ 잔액 오류 의심: 계산상 잔액 {calculated_balance:,}원 / 문서 표기 잔액 {written_balance:,}원 / 차이 {diff:,}원"
            )
    else:
        messages.append(f"ℹ️ 문서에서 잔액을 찾지 못했습니다. 계산상 잔액은 {calculated_balance:,}원입니다.")

    return messages


uploaded_file = st.file_uploader(
    "회계내역 PDF 또는 이미지 업로드",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.success("파일 업로드 성공!")

    if st.button("검사 시작"):
        file_type = uploaded_file.name.split(".")[-1].lower()

        with st.spinner("문서를 읽고 금액을 분석하는 중입니다..."):
            if file_type == "pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_image(uploaded_file)

            df = analyze_text(text)
            results = check_result(df)

        st.subheader("1. 읽은 텍스트")
        st.text_area("PDF/이미지에서 추출된 내용", text, height=250)

        st.subheader("2. 찾은 금액 목록")
        if df.empty:
            st.warning("금액을 찾지 못했습니다.")
        else:
            st.dataframe(df)

        st.subheader("3. 검사 결과")
        for result in results:
            if "❌" in result:
                st.error(result)
            elif "✅" in result:
                st.success(result)
            else:
                st.info(result)
