from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    file_bytes:
        Uploaded PDF as bytes.

    Returns
    -------
    str
        Extracted text.
    """

    reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page in reader.pages:

        try:
            text = page.extract_text()
        except Exception:
            text = ""

        if text:
            pages.append(text)

    return "\n".join(pages).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from a TXT file.
    """

    return file_bytes.decode(
        "utf-8",
        errors="replace"
    ).strip()


def extract_contract_text(
    file_bytes: bytes,
    filename: str
) -> str:
    """
    Automatically extract text based on file extension.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":

        return extract_text_from_pdf(
            file_bytes
        )

    if extension == ".txt":

        return extract_text_from_txt(
            file_bytes
        )

    raise ValueError(
        "Unsupported file type. "
        "Please upload a PDF or TXT file."
    )