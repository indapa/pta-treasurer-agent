#!/usr/bin/env python3
"""
PTA Cash Verification Agent (local Ollama vision model, or Gemini)
Extracts the counted totals from a PTA Cash Verification Form and the deposited
amount from the bank deposit slip/receipt, then verifies the two amounts agree.
"""

import argparse
import base64
import io
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

from PIL import Image
from fpdf import FPDF, FontFace, XPos, YPos
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# LLM_PROVIDER selects "gemini" (default, needs GOOGLE_API_KEY) or "ollama" (local).
# The --provider and --model command-line options override these.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:7b")
# Local vision models slow down sharply on large images; cap the longest side.
OLLAMA_MAX_IMAGE_SIDE = 1600
GEMINI_MODEL = "gemini-3.8-flash"

# Silence the google-genai AFC advisory triggered by langchain's internal calls.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


# ==========================================
# 1. Pydantic Schemas
# ==========================================

class DenominationLine(BaseModel):
    denomination: str = Field(description="Coin or bill denomination as printed, e.g. '25¢', '$1', '$20'")
    denomination_value: float = Field(description="Value of one coin or bill in dollars, e.g. 0.25, 1.0, 20.0")
    count: int = Field(description="Handwritten number of coins or bills")
    amount: float = Field(description="Handwritten dollar amount written for this line")


class CashVerificationForm(BaseModel):
    unit_name: Optional[str] = Field(default="Unknown Unit", description="Handwritten UNIT NAME")
    activity: Optional[str] = Field(default="Unknown Activity", description="Handwritten ACTIVITY")
    form_date: Optional[str] = Field(default="Unknown Date", description="Handwritten DATE")
    coin_lines: List[DenominationLine] = Field(default_factory=list, description="Filled-in coin lines in the counted section")
    currency_lines: List[DenominationLine] = Field(default_factory=list, description="Filled-in currency lines in the counted section")
    coins_total: float = Field(default=0.0, description="Handwritten coins TOTAL $")
    currency_total: float = Field(default=0.0, description="Handwritten currency TOTAL $")
    checks_total: float = Field(default=0.0, description="Handwritten checks TOTAL $")
    grand_total: float = Field(description="Handwritten GRAND TOTAL $")
    signature_count: int = Field(default=0, description="Number of signature lines that are signed")


class DepositReceipt(BaseModel):
    institution: Optional[str] = Field(default="Unknown Bank", description="Bank or credit union name")
    deposit_date: Optional[str] = Field(default="Unknown Date", description="Date of the deposit")
    account: Optional[str] = Field(default="", description="Masked account number, e.g. XXXXXX4031")
    deposit_amount: float = Field(description="Total amount deposited (e.g. 'Amount:' or 'Deposit Total')")
    cash_amount: Optional[float] = Field(default=None, description="Cash portion of the deposit, if listed separately")
    check_amount: Optional[float] = Field(default=None, description="Check portion of the deposit, if listed separately")


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    FORM_MATH_ERROR = "FORM_MATH_ERROR"


class VerificationReport(BaseModel):
    unit_name: str
    activity: str
    form_date: str
    deposit_date: str
    form_grand_total: float
    deposit_amount: float
    difference: float
    status: VerificationStatus
    form: CashVerificationForm
    deposit: DepositReceipt
    notes: List[str]


# ==========================================
# 2. File Conversion Utilities
# ==========================================

def _fit_for_model(img: Image.Image) -> Image.Image:
    """Downscales large pages for local Ollama models; Gemini gets the full resolution."""
    if LLM_PROVIDER == "ollama" and max(img.size) > OLLAMA_MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((OLLAMA_MAX_IMAGE_SIDE, OLLAMA_MAX_IMAGE_SIDE), Image.LANCZOS)
    return img


def load_file_as_base64_images(file_path: str) -> List[str]:
    """
    Accepts a path to a PDF or an Image file (.png, .jpeg, .jpg, .webp).
    Returns a list of base64-encoded PNG strings.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    base64_images = []

    if ext == ".pdf":
        images = convert_from_path(file_path)
        for img in images:
            img = _fit_for_model(img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with Image.open(file_path) as img:
            buffered = io.BytesIO()
            _fit_for_model(img.convert("RGB")).save(buffered, format="PNG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file {file_path}")

    return base64_images


def _image_payload(prompt: str, file_path: str) -> list:
    payload = [{"type": "text", "text": prompt}]
    for b64 in load_file_as_base64_images(file_path):
        payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    return payload


# ==========================================
# 3. Agent Graph State & Nodes
# ==========================================

def active_model() -> str:
    return GEMINI_MODEL if LLM_PROVIDER == "gemini" else OLLAMA_MODEL


def make_structured_llm(schema: type[BaseModel]):
    """Returns a vision LLM bound to a Pydantic output schema for the configured provider."""
    if LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0).with_structured_output(schema)
    # Ollama constrains decoding to the JSON schema; num_ctx leaves room for image tokens.
    # Small local models skip optional fields, so every field is marked required here.
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0, num_ctx=8192)
    json_schema = _require_all_fields(schema.model_json_schema())
    return llm.with_structured_output(json_schema, method="json_schema") | schema.model_validate


def _require_all_fields(node):
    """Marks every property of every object in a JSON schema as required."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
        for value in node.values():
            _require_all_fields(value)
    elif isinstance(node, list):
        for value in node:
            _require_all_fields(value)
    return node


class AgentState(BaseModel):
    form_path: str
    deposit_path: str
    form: Optional[CashVerificationForm] = None
    deposit: Optional[DepositReceipt] = None
    report: Optional[VerificationReport] = None


def process_cash_form_node(state: AgentState) -> dict:
    """Extracts the handwritten counts and totals from the Cash Verification Form."""
    structured_llm = make_structured_llm(CashVerificationForm)

    prompt = (
        "This is a PTA Cash Verification Form. Extract the HANDWRITTEN values only. "
        "The blank form has pre-printed '0.00' placeholders on many lines; ignore them and "
        "treat a line as empty unless a count or amount has been written in by hand. "
        "Ignore the 'START UP CASH' section at the top; extract only the counted section "
        "below UNIT NAME / ACTIVITY (coins, currency, checks, and GRAND TOTAL). "
        "Each coin/currency row has three columns: the PRINTED denomination, the handwritten "
        "NUMBER of coins or bills (count), and the handwritten dollar AMOUNT. For example, '11' "
        "written on the '$5' row with '55.00' gives denomination '$5', denomination_value 5.0, "
        "count 11, amount 55.0. Only include rows where something is handwritten."
    )
    form = structured_llm.invoke([{"role": "user", "content": _image_payload(prompt, state.form_path)}])
    return {"form": form}


def process_deposit_node(state: AgentState) -> dict:
    """Extracts the deposited amount from the bank deposit slip or receipt."""
    structured_llm = make_structured_llm(DepositReceipt)

    prompt = (
        "This is a bank deposit slip or teller deposit receipt. Extract the institution, "
        "deposit date, masked account number, and the total amount deposited. Do NOT use the "
        "account's new or available balance as the deposit amount. If the cash and check "
        "portions of the deposit are listed separately, extract those too."
    )
    deposit = structured_llm.invoke([{"role": "user", "content": _image_payload(prompt, state.deposit_path)}])
    return {"deposit": deposit}


def verify_amounts_node(state: AgentState) -> dict:
    """Performs deterministic math to verify the form totals and the deposited amount agree."""
    form = state.form
    deposit = state.deposit
    notes = []
    form_math_ok = True

    # Rule 1: Each line's count x denomination must equal the amount written.
    for line in form.coin_lines + form.currency_lines:
        expected = round(line.count * line.denomination_value, 2)
        if abs(expected - line.amount) > 0.01:
            form_math_ok = False
            notes.append(
                f"[Form] {line.count} x {line.denomination} should be ${expected:.2f}, "
                f"but ${line.amount:.2f} is written."
            )

    # Rule 2: Section totals must equal the sum of their lines.
    for section, lines, written in (
        ("Coins", form.coin_lines, form.coins_total),
        ("Currency", form.currency_lines, form.currency_total),
    ):
        line_sum = round(sum(line.amount for line in lines), 2)
        if abs(line_sum - written) > 0.01:
            form_math_ok = False
            notes.append(
                f"[Form] {section} lines add up to ${line_sum:.2f}, "
                f"but the {section.lower()} TOTAL is ${written:.2f}."
            )

    # Rule 3: Grand total must equal coins + currency + checks.
    subtotal_sum = round(form.coins_total + form.currency_total + form.checks_total, 2)
    if abs(subtotal_sum - form.grand_total) > 0.01:
        form_math_ok = False
        notes.append(
            f"[Form] Coins + currency + checks = ${subtotal_sum:.2f}, "
            f"but the GRAND TOTAL is ${form.grand_total:.2f}."
        )

    # Rule 4: Cash and check portions of the deposit, when listed, should match the form.
    form_cash = round(form.coins_total + form.currency_total, 2)
    if deposit.cash_amount is not None and abs(deposit.cash_amount - form_cash) > 0.01:
        notes.append(
            f"[Deposit] Cash deposited (${deposit.cash_amount:.2f}) differs from cash counted "
            f"on the form (${form_cash:.2f})."
        )
    if deposit.check_amount is not None and abs(deposit.check_amount - form.checks_total) > 0.01:
        notes.append(
            f"[Deposit] Checks deposited (${deposit.check_amount:.2f}) differ from checks listed "
            f"on the form (${form.checks_total:.2f})."
        )

    # California State PTA requires two people to count and sign.
    if form.signature_count < 2:
        notes.append(f"[Form] Only {form.signature_count} signature(s) found; two counters should sign the form.")

    form_total = round(form.grand_total, 2)
    deposited = round(deposit.deposit_amount, 2)
    diff = round(deposited - form_total, 2)

    # Determine final status
    if abs(diff) > 0.01:
        status = VerificationStatus.AMOUNT_MISMATCH
        if diff > 0:
            notes.append(f"DISCREPANCY: Deposit (${deposited:.2f}) exceeds the form grand total (${form_total:.2f}) by ${diff:.2f}.")
        else:
            notes.append(f"DISCREPANCY: Deposit (${deposited:.2f}) is ${abs(diff):.2f} short of the form grand total (${form_total:.2f}).")
    elif not form_math_ok:
        status = VerificationStatus.FORM_MATH_ERROR
        notes.append("Deposit matches the form grand total, but the form's own arithmetic has errors.")
    else:
        status = VerificationStatus.VERIFIED
        notes.append("DEPOSIT VERIFIED: Deposited amount matches the Cash Verification Form grand total.")

    report = VerificationReport(
        unit_name=form.unit_name or "Unknown Unit",
        activity=form.activity or "Unknown Activity",
        form_date=form.form_date or "Unknown Date",
        deposit_date=deposit.deposit_date or "Unknown Date",
        form_grand_total=form_total,
        deposit_amount=deposited,
        difference=diff,
        status=status,
        form=form,
        deposit=deposit,
        notes=notes,
    )

    return {"report": report}


# ==========================================
# 4. Build LangGraph Workflow
# ==========================================

def build_verification_graph():
    builder = StateGraph(AgentState)

    builder.add_node("process_form", process_cash_form_node)
    builder.add_node("process_deposit", process_deposit_node)
    builder.add_node("verify", verify_amounts_node)

    builder.set_entry_point("process_form")
    builder.add_edge("process_form", "process_deposit")
    builder.add_edge("process_deposit", "verify")
    builder.add_edge("verify", END)

    return builder.compile()


# ==========================================
# 5. PDF Report
# ==========================================

STATUS_COLORS = {
    VerificationStatus.VERIFIED: (22, 128, 60),
    VerificationStatus.AMOUNT_MISMATCH: (190, 30, 45),
    VerificationStatus.FORM_MATH_ERROR: (200, 110, 0),
}

# fpdf2's built-in fonts only cover Latin-1, so map common typographic characters.
_PDF_REPLACEMENTS = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "*", "…": "...", "™": "(TM)",
}


def _pdf_text(text) -> str:
    text = str(text if text is not None else "")
    for char, replacement in _PDF_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def write_pdf_report(report: VerificationReport, form_path: str, deposit_path: str, output_path: Path) -> None:
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text: str, size: int) -> None:
        pdf.set_font("Helvetica", "B", size)
        pdf.multi_cell(0, size * 0.5, _pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    heading("PTA Cash Deposit Verification Report", 18)

    # Summary
    pdf.set_font("Helvetica", size=11)
    summary = [
        ("Unit", report.unit_name),
        ("Activity", report.activity),
        ("Status", report.status.value),
        ("Form Grand Total", f"${report.form_grand_total:.2f}"),
        ("Deposit Amount", f"${report.deposit_amount:.2f}"),
        ("Variance", f"${report.difference:.2f}"),
        ("Form Date", report.form_date),
        ("Deposit Date", report.deposit_date),
        ("Cash Verification Form", Path(form_path).name),
        ("Deposit Slip", Path(deposit_path).name),
    ]
    with pdf.table(col_widths=(50, 130), first_row_as_headings=False, line_height=7) as table:
        for label, value in summary:
            row = table.row()
            row.cell(label, style=FontFace(emphasis="BOLD"))
            if label == "Status":
                row.cell(value, style=FontFace(emphasis="BOLD", color=STATUS_COLORS[report.status]))
            else:
                row.cell(_pdf_text(value))
    pdf.ln(6)

    # Form count
    heading("Cash Verification Form Count", 14)
    form = report.form
    rows = [(line.denomination, str(line.count), f"${line.amount:.2f}") for line in form.coin_lines + form.currency_lines]
    rows += [
        ("Coins Total", "", f"${form.coins_total:.2f}"),
        ("Currency Total", "", f"${form.currency_total:.2f}"),
        ("Checks Total", "", f"${form.checks_total:.2f}"),
        ("Grand Total", "", f"${form.grand_total:.2f}"),
    ]
    with pdf.table(col_widths=(90, 40, 50), text_align=("LEFT", "RIGHT", "RIGHT"), line_height=6,
                   headings_style=FontFace(emphasis="BOLD", fill_color=(230, 230, 230))) as table:
        table.row(["Denomination", "Count", "Amount"])
        for denomination, count, amount in rows:
            table.row([_pdf_text(denomination), count, amount])
    pdf.ln(5)

    # Deposit
    heading("Deposit Slip", 14)
    deposit = report.deposit
    deposit_rows = [
        ("Institution", deposit.institution or "Unknown"),
        ("Account", deposit.account or "Unknown"),
        ("Date", deposit.deposit_date or "Unknown"),
        ("Deposit Amount", f"${deposit.deposit_amount:.2f}"),
    ]
    if deposit.cash_amount is not None:
        deposit_rows.append(("Cash", f"${deposit.cash_amount:.2f}"))
    if deposit.check_amount is not None:
        deposit_rows.append(("Checks", f"${deposit.check_amount:.2f}"))
    for label, value in deposit_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.write(5, f"{label}: ")
        pdf.set_font("Helvetica", size=10)
        pdf.write(5, _pdf_text(value))
        pdf.ln(5)
    pdf.ln(5)

    # Notes
    heading("Verification Notes & Warnings", 14)
    pdf.set_font("Helvetica", size=10)
    for note in report.notes:
        pdf.multi_cell(0, 5, _pdf_text(f"- {note}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.output(str(output_path))


# ==========================================
# 6. CLI Entrypoint
# ==========================================

def main():
    global LLM_PROVIDER, OLLAMA_MODEL, GEMINI_MODEL
    parser = argparse.ArgumentParser(
        description="PTA Cash Verification Agent: Verify a bank deposit slip matches the Cash Verification Form."
    )

    parser.add_argument(
        "-f", "--cash_verification_form",
        required=True,
        type=str,
        help="Path to the PDF scan or image of the completed Cash Verification Form."
    )

    parser.add_argument(
        "-d", "--deposit_slip",
        required=True,
        type=str,
        help="Path to the PDF scan or image of the bank deposit slip or receipt."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output prefix; the report is also written to <prefix>_cash_deposit_slip_verification.pdf."
    )

    parser.add_argument(
        "-p", "--provider",
        choices=["gemini", "ollama"],
        default=LLM_PROVIDER if LLM_PROVIDER in ("ollama", "gemini") else None,
        help="LLM provider: 'gemini' (needs GOOGLE_API_KEY) or 'ollama' (local). "
             "Defaults to $LLM_PROVIDER, or 'gemini'."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help=f"Model name for the chosen provider (defaults: ollama={OLLAMA_MODEL}, gemini={GEMINI_MODEL})."
    )

    args = parser.parse_args()

    if args.provider is None:
        parser.error(f"LLM_PROVIDER must be 'ollama' or 'gemini', got '{LLM_PROVIDER}'.")

    LLM_PROVIDER = args.provider
    if args.model and LLM_PROVIDER == "gemini":
        GEMINI_MODEL = args.model
    elif args.model:
        OLLAMA_MODEL = args.model
    if LLM_PROVIDER == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[+] Initializing PTA Cash Verification Agent ({LLM_PROVIDER}: {active_model()})...")
    print(f"    Cash Verification Form: {args.cash_verification_form}")
    print(f"    Deposit Slip:           {args.deposit_slip}")

    graph = build_verification_graph()

    initial_state = AgentState(
        form_path=args.cash_verification_form,
        deposit_path=args.deposit_slip
    )

    try:
        final_state = graph.invoke(initial_state)
        report: VerificationReport = final_state["report"]

        print("\n" + "="*50)
        print("       PTA CASH DEPOSIT VERIFICATION REPORT       ")
        print("="*50)
        print(f"Unit:              {report.unit_name}")
        print(f"Activity:          {report.activity}")
        print(f"Status:            {report.status.value}")
        print(f"Form Grand Total:  ${report.form_grand_total:.2f}")
        print(f"Deposit Amount:    ${report.deposit_amount:.2f}")
        print(f"Variance:          ${report.difference:.2f}")
        print(f"Form Date:         {report.form_date}")
        print(f"Deposit Date:      {report.deposit_date}")
        print("-"*50)
        print("CASH VERIFICATION FORM COUNT:")
        for line in report.form.coin_lines + report.form.currency_lines:
            print(f"  {line.count:>4} x {line.denomination:<5} = ${line.amount:.2f}")
        print(f"  Coins: ${report.form.coins_total:.2f} | Currency: ${report.form.currency_total:.2f} "
              f"| Checks: ${report.form.checks_total:.2f}")
        print("-"*50)
        print("DEPOSIT SLIP:")
        dep = report.deposit
        print(f"  {dep.institution} | Acct: {dep.account} | Date: {dep.deposit_date} | Amount: ${dep.deposit_amount:.2f}")
        print("-"*50)
        print("VERIFICATION NOTES & WARNINGS:")
        for note in report.notes:
            print(f"  * {note}")
        print("="*50 + "\n")

        if args.output:
            # Accept "-o Name.pdf" as the prefix "Name".
            prefix = args.output.removesuffix(".pdf")
            output_path = Path(f"{prefix}_cash_deposit_slip_verification.pdf")
            write_pdf_report(report, args.cash_verification_form, args.deposit_slip, output_path)
            print(f"[+] PDF report written to {output_path}\n")

    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
