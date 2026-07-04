import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="회계내역 오류 검사기",
    page_icon="💰",
    layout="wide",
)


# -------------------------------------------------
# 기본 설정
# -------------------------------------------------

MONEY_RE = re.compile(
    r"""
    [-+]?
    \s*
    (?:₩\s*)?
    (?:
        \d{1,3}(?:[,\.\s]\d{3})+
        |
        \d+
    )
    \s*
    원?
    """,
    re.VERBOSE,
)

HEADER_WORDS = ["순번", "내용", "날짜", "비고", "입금", "출금", "잔액"]


# -------------------------------------------------
# 금액 처리 함수
# -------------------------------------------------

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("₩", "")
    text = text.replace("，", ",")
    text = text.replace("ㆍ", ".")
    text = text.replace("·", ".")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    return text


def money_to_int(raw) -> Optional[int]:
    if raw is None:
        return None

    if isinstance(raw, (int, np.integer)):
        return int(raw)

    if isinstance(raw, float):
        if np.isnan(raw):
            return None
        return int(raw)

    s = normalize_text(str(raw))

    # OCR에서 자주 틀리는 문자 보정
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("I", "1").replace("l", "1").replace("|", "1")
    s = s.replace("원", "")
    s = s.replace(" ", "")
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.replace("₩", "")
    s = s.strip()

    if s in ["", "-", "+"]:
        return None

    sign = -1 if s.startswith("-") else 1
    s = s.replace("-", "").replace("+", "")

    if not s.isdigit():
        return None

    return sign * int(s)


def extract_money_values(text: str) -> List[int]:
    if not text:
        return []

    text = normalize_text(text)
    matches = MONEY_RE.findall(text)

    values = []
    for m in matches:
        value = money_to_int(m)
        if value is not None:
            values.append(value)

    return values


def fmt_won(value) -> str:
    value = money_to_int(value)
    if value is None:
        return ""
    return f"{value:,}원"


# -------------------------------------------------
# 파일 처리
# -------------------------------------------------

def pdf_to_images(file_bytes: bytes, zoom: float = 2.5) -> List[Image.Image]:
    images = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page in doc:
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)

    return images


def uploaded_file_to_images(uploaded_file) -> List[Image.Image]:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return pdf_to_images(file_bytes)

    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return [img]


# -------------------------------------------------
# 이미지 전처리
# -------------------------------------------------

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]

    # 글자가 작으면 확대
    if max(h, w) < 2000:
        gray = cv2.resize(
            gray,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return Image.fromarray(binary)


# -------------------------------------------------
# 표 세로선 감지
# -------------------------------------------------

def merge_close_positions(values: List[int], gap: int = 12) -> List[int]:
    if not values:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [int(np.mean(g)) for g in groups]


def detect_vertical_lines(img: Image.Image) -> List[int]:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape[:2]
    kernel_height = max(25, h // 25)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, kernel_height),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        vertical,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    xs = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        if ch > h * 0.12:
            xs.append(x)
            xs.append(x + cw)

    xs = merge_close_positions(xs, gap=max(8, w // 250))

    # 너무 가까운 선 제거
    filtered = []
    for x in xs:
        if not filtered or abs(x - filtered[-1]) > w * 0.015:
            filtered.append(x)

    return filtered


def build_money_column_zones(img: Image.Image) -> Dict[str, Tuple[int, int]]:
    """
    입금 / 출금 / 잔액 열 위치 추정.
    표 세로선을 찾으면 오른쪽 3개 열을 사용하고,
    실패하면 이미지 오른쪽 비율로 대략 잡음.
    """
    w, h = img.size
    xs = detect_vertical_lines(img)

    # 표가 잘 감지된 경우: 마지막 3칸 = 입금, 출금, 잔액
    if len(xs) >= 8:
        return {
            "입금": (xs[-4], xs[-3]),
            "출금": (xs[-3], xs[-2]),
            "잔액": (xs[-2], xs[-1]),
        }

    # 감지 실패 시 기본값
    return {
        "입금": (int(w * 0.64), int(w * 0.77)),
        "출금": (int(w * 0.77), int(w * 0.89)),
        "잔액": (int(w * 0.89), int(w * 0.995)),
    }


# -------------------------------------------------
# OCR
# -------------------------------------------------

def run_ocr_words(img: Image.Image, page_num: int) -> pd.DataFrame:
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

    data = pytesseract.image_to_data(
        img,
        lang="kor+eng",
        config=config,
        output_type=pytesseract.Output.DATAFRAME,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]

    data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(-1)

    # 신뢰도 낮아도 숫자는 살림
    def keep_row(row):
        text = str(row["text"])
        conf = float(row["conf"])

        if conf >= 20:
            return True

        if re.search(r"\d", text):
            return True

        return False

    data = data[data.apply(keep_row, axis=1)].copy()

    if data.empty:
        return pd.DataFrame()

    for col in ["left", "top", "width", "height"]:
        data[col] = data[col].astype(int)

    data["right"] = data["left"] + data["width"]
    data["bottom"] = data["top"] + data["height"]
    data["cx"] = data["left"] + data["width"] / 2
    data["cy"] = data["top"] + data["height"] / 2
    data["page"] = page_num

    return data[
        [
            "page",
            "text",
            "conf",
            "left",
            "top",
            "width",
            "height",
            "right",
            "bottom",
            "cx",
            "cy",
        ]
    ].reset_index(drop=True)


def group_words_into_rows(words: pd.DataFrame) -> List[pd.DataFrame]:
    if words.empty:
        return []

    words = words.sort_values(["cy", "left"]).copy()

    median_h = max(10, int(words["height"].median()))
    row_gap = max(14, int(median_h * 0.9))

    rows = []
    current = []

    current_y = None

    for _, word in words.iterrows():
        cy = float(word["cy"])

        if current_y is None:
            current = [word]
            current_y = cy
            continue

        if abs(cy - current_y) <= row_gap:
            current.append(word)
            current_y = np.mean([float(w["cy"]) for w in current])
        else:
            rows.append(pd.DataFrame(current))
            current = [word]
            current_y = cy

    if current:
        rows.append(pd.DataFrame(current))

    return rows


def text_from_words(row_words: pd.DataFrame) -> str:
    if row_words.empty:
        return ""

    row_words = row_words.sort_values("left")
    return " ".join(row_words["text"].astype(str).tolist())


def words_in_zone(row_words: pd.DataFrame, x1: int, x2: int) -> pd.DataFrame:
    if row_words.empty:
        return row_words

    # 중앙점이 영역 안에 있는 단어
    return row_words[
        (row_words["cx"] >= x1) &
        (row_words["cx"] <= x2)
    ].copy()


def extract_amount_from_zone(row_words: pd.DataFrame, zone: Tuple[int, int], mode: str) -> Optional[int]:
    x1, x2 = zone
    zone_words = words_in_zone(row_words, x1, x2)

    if zone_words.empty:
        return None

    zone_text = text_from_words(zone_words)
    values = extract_money_values(zone_text)

    if not values:
        # OCR이 쉼표나 원 표시를 놓친 경우를 대비
        raw_digits = []
        for t in zone_words["text"].astype(str).tolist():
            if re.search(r"\d", t):
                raw_digits.append(t)

        values = []
        for item in raw_digits:
            value = money_to_int(item)
            if value is not None:
                values.append(value)

    if not values:
        return None

    # 입금/출금 칸에 금액이 여러 개 잡히면 합산
    if mode in ["입금", "출금"]:
        return int(sum(values))

    # 잔액은 마지막 값을 사용
    return int(values[-1])


def is_header_row(text: str) -> bool:
    hit = 0
    for word in HEADER_WORDS:
        if word in text:
            hit += 1
    return hit >= 2


def extract_rows_from_page(img: Image.Image, page_num: int) -> Tuple[pd.DataFrame, Image.Image, Dict[str, Tuple[int, int]]]:
    processed_img = preprocess_for_ocr(img)

    zones = build_money_column_zones(processed_img)
    words = run_ocr_words(processed_img, page_num=page_num)

    if words.empty:
        return pd.DataFrame(), processed_img, zones

    grouped_rows = group_words_into_rows(words)

    extracted = []

    income_x1 = zones["입금"][0]

    for row_words in grouped_rows:
        full_text = text_from_words(row_words)

        if not full_text:
            continue

        if is_header_row(full_text):
            continue

        입금 = extract_amount_from_zone(row_words, zones["입금"], "입금")
        출금 = extract_amount_from_zone(row_words, zones["출금"], "출금")
        잔액 = extract_amount_from_zone(row_words, zones["잔액"], "잔액")

        if 입금 is None and 출금 is None and 잔액 is None:
            continue

        # 내용은 입금 열보다 왼쪽에 있는 글자를 대략 추출
        content_words = row_words[row_words["cx"] < income_x1].copy()
        내용 = text_from_words(content_words)

        top = int(row_words["top"].min())

        extracted.append(
            {
                "사용": True,
                "페이지": page_num,
                "내용": 내용,
                "입금": 입금,
                "출금": 출금,
                "문서잔액": 잔액,
                "원문": full_text,
                "y좌표": top,
            }
        )

    df = pd.DataFrame(extracted)

    if not df.empty:
        df = df.sort_values(["페이지", "y좌표"]).reset_index(drop=True)

    return df, processed_img, zones


def extract_rows_from_images(images: List[Image.Image]):
    all_rows = []
    processed_images = []
    all_zones = []

    for i, img in enumerate(images, start=1):
        df, processed_img, zones = extract_rows_from_page(img, page_num=i)
        processed_images.append(processed_img)
        all_zones.append(zones)

        if not df.empty:
            all_rows.append(df)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True), processed_images, all_zones

    return pd.DataFrame(), processed_images, all_zones


# -------------------------------------------------
# 검산
# -------------------------------------------------

def clean_table_for_check(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = ["사용", "페이지", "내용", "입금", "출금", "문서잔액", "원문"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["사용"] = df["사용"].fillna(True).astype(bool)

    for col in ["입금", "출금", "문서잔액"]:
        df[col] = df[col].apply(money_to_int)

    df["페이지"] = pd.to_numeric(df["페이지"], errors="coerce").fillna(1).astype(int)
    df["내용"] = df["내용"].fillna("").astype(str)
    df["원문"] = df["원문"].fillna("").astype(str)

    return df


def check_ledger(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = clean_table_for_check(df)

    df = df[df["사용"] == True].copy().reset_index(drop=True)

    if df.empty:
        return df, {
            "total": 0,
            "normal": 0,
            "error": 0,
            "warning": 0,
        }

    result_rows = []

    previous_balance = None
    pending_income = 0
    pending_expense = 0

    total_check_points = 0
    normal_count = 0
    error_count = 0
    warning_count = 0

    for idx, row in df.iterrows():
        income = row["입금"] if pd.notna(row["입금"]) else None
        expense = row["출금"] if pd.notna(row["출금"]) else None
        balance = row["문서잔액"] if pd.notna(row["문서잔액"]) else None

        income_value = income if income is not None else 0
        expense_value = expense if expense is not None else 0

        pending_income += income_value
        pending_expense += expense_value

        expected_balance = None
        diff = None
        status = ""
        detail = ""

        if balance is None:
            status = "대기"
            detail = "잔액이 없는 행입니다. 다음 잔액 행까지 입금/출금을 합산합니다."
            warning_count += 1

        else:
            total_check_points += 1

            if previous_balance is None:
                # 첫 잔액 행
                if pending_income != 0 or pending_expense != 0:
                    expected_balance = pending_income - pending_expense
                    diff = expected_balance - balance

                    if diff == 0:
                        status = "정상"
                        detail = "시작 잔액이 입금/출금과 일치합니다."
                        normal_count += 1
                    else:
                        status = "확인 필요"
                        detail = "첫 행 이전 잔액이 따로 있을 수 있습니다."
                        warning_count += 1
                else:
                    expected_balance = balance
                    diff = 0
                    status = "시작 잔액"
                    detail = "첫 번째 잔액으로 기준을 잡았습니다."
                    normal_count += 1

            else:
                expected_balance = previous_balance + pending_income - pending_expense
                diff = expected_balance - balance

                if diff == 0:
                    status = "정상"
                    detail = "이전 잔액 + 입금 - 출금 = 문서 잔액"
                    normal_count += 1
                else:
                    status = "오류 의심"
                    error_count += 1

                    if diff > 0:
                        detail = f"문서 잔액이 계산값보다 {abs(diff):,}원 작습니다."
                    else:
                        detail = f"문서 잔액이 계산값보다 {abs(diff):,}원 큽니다."

            previous_balance = balance
            pending_income = 0
            pending_expense = 0

        result_rows.append(
            {
                "행": idx + 1,
                "페이지": row["페이지"],
                "내용": row["내용"],
                "입금": income,
                "출금": expense,
                "문서잔액": balance,
                "계산잔액": expected_balance,
                "차이": diff,
                "상태": status,
                "설명": detail,
                "원문": row["원문"],
            }
        )

    # 마지막에 잔액 없는 거래가 남은 경우
    if pending_income != 0 or pending_expense != 0:
        warning_count += 1

    result_df = pd.DataFrame(result_rows)

    summary = {
        "total": int(total_check_points),
        "normal": int(normal_count),
        "error": int(error_count),
        "warning": int(warning_count),
    }

    return result_df, summary


def style_result_table(df: pd.DataFrame):
    def style_row(row):
        if row["상태"] == "오류 의심":
            return ["background-color: #ffdddd"] * len(row)
        if row["상태"] in ["확인 필요", "대기"]:
            return ["background-color: #fff3cd"] * len(row)
        if row["상태"] in ["정상", "시작 잔액"]:
            return ["background-color: #ddffdd"] * len(row)
        return [""] * len(row)

    display_df = df.copy()

    for col in ["입금", "출금", "문서잔액", "계산잔액", "차이"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_won)

    return display_df.style.apply(style_row, axis=1)


# -------------------------------------------------
# Streamlit 화면
# -------------------------------------------------

st.title("💰 회계내역 오류 검사기")
st.write(
    "PDF나 이미지를 업로드하면 입금, 출금, 잔액을 읽고 "
    "`이전 잔액 + 입금 - 출금 = 현재 잔액`이 맞는지 검사합니다."
)

with st.expander("사용 방법"):
    st.markdown(
        """
        1. 회계내역 PDF 또는 이미지를 업로드합니다.  
        2. 앱이 입금 / 출금 / 잔액을 자동으로 읽습니다.  
        3. OCR 결과가 틀리면 표에서 직접 수정합니다.  
        4. 아래 검사 결과에서 오류 의심 행을 확인합니다.  

        더 정확하게 하려면 표 부분만 잘라서 업로드하는 것이 좋습니다.
        """
    )

uploaded_file = st.file_uploader(
    "회계내역 PDF 또는 이미지 업로드",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("PDF 또는 이미지를 업로드해 주세요.")

else:
    with st.spinner("문서를 분석하는 중입니다..."):
        images = uploaded_file_to_images(uploaded_file)
        raw_df, processed_images, zones_list = extract_rows_from_images(images)

    st.subheader("1. 업로드한 문서 미리보기")

    preview_cols = st.columns(min(2, len(images)))

    for i, img in enumerate(images[:2]):
        with preview_cols[i % len(preview_cols)]:
            st.image(img, caption=f"원본 페이지 {i + 1}", use_container_width=True)

    with st.expander("OCR용 전처리 이미지 보기"):
        for i, img in enumerate(processed_images):
            st.image(img, caption=f"OCR 전처리 페이지 {i + 1}", use_container_width=True)

    if raw_df.empty:
        st.error(
            "금액 표를 자동으로 읽지 못했습니다. "
            "표 부분만 잘라서 다시 업로드하거나 이미지 화질을 높여 주세요."
        )

    else:
        st.subheader("2. OCR로 읽은 회계 표")
        st.write("잘못 읽힌 숫자는 여기서 직접 고칠 수 있습니다.")

        edit_df = raw_df[
            ["사용", "페이지", "내용", "입금", "출금", "문서잔액", "원문"]
        ].copy()

        edited_df = st.data_editor(
            edit_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "사용": st.column_config.CheckboxColumn(
                    "사용",
                    help="검산에 포함할 행만 체크하세요.",
                    default=True,
                ),
                "페이지": st.column_config.NumberColumn("페이지"),
                "내용": st.column_config.TextColumn("내용"),
                "입금": st.column_config.NumberColumn("입금", format="%d"),
                "출금": st.column_config.NumberColumn("출금", format="%d"),
                "문서잔액": st.column_config.NumberColumn("문서잔액", format="%d"),
                "원문": st.column_config.TextColumn("OCR 원문"),
            },
            hide_index=True,
        )

        st.subheader("3. 검산 결과")

        result_df, summary = check_ledger(edited_df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("검사 지점", f"{summary['total']}개")
        col2.metric("정상", f"{summary['normal']}개")
        col3.metric("오류 의심", f"{summary['error']}개")
        col4.metric("확인 필요", f"{summary['warning']}개")

        if summary["error"] == 0:
            st.success("잔액 흐름에서 뚜렷한 오류를 찾지 못했습니다.")
        else:
            st.error(f"오류 의심 행이 {summary['error']}개 있습니다.")

        st.dataframe(
            style_result_table(result_df),
            use_container_width=True,
            hide_index=True,
        )

        error_rows = result_df[result_df["상태"] == "오류 의심"]

        if not error_rows.empty:
            st.subheader("4. 오류 의심 상세 설명")

            for _, row in error_rows.iterrows():
                st.error(
                    f"""
                    행 {row['행']} / 페이지 {row['페이지']}

                    내용: {row['내용']}

                    입금: {fmt_won(row['입금'])}
                    출금: {fmt_won(row['출금'])}
                    문서상 잔액: {fmt_won(row['문서잔액'])}
                    계산상 잔액: {fmt_won(row['계산잔액'])}
                    차이: {fmt_won(row['차이'])}

                    {row['설명']}
                    """
                )

        st.subheader("5. 결과 다운로드")

        download_df = result_df.copy()

        csv = download_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="검사 결과 CSV 다운로드",
            data=csv,
            file_name="account_check_result.csv",
            mime="text/csv",
        )
