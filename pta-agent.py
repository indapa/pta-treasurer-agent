#!/usr/bin/env python3
"""
PTA Reimbursement Audit Agent (Powered by Gemini)
Extracts reimbursement form data and supporting receipt scans (PDFs or Images),
cross-validates claim math, and outputs an audit result.
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
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

GEMINI_MODEL = "gemini-3.8-flash"

# Silence the google-genai AFC advisory triggered by langchain's internal calls.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


# ==========================================
# 1. Pydantic Schemas
# ==========================================

class ReceiptItem(BaseModel):
    description: str = Field(description="Description of item purchased")
    amount: float = Field(description="Total price for this line item")


class ParsedReceipt(BaseModel):
    vendor_name: Optional[str] = Field(default="Unknown Vendor", description="Vendor/Store Name")
    receipt_date: Optional[str] = Field(default="Unknown Date", description="Date on receipt")
    items: List[ReceiptItem] = Field(default_factory=list, description="Itemized list of purchases")
    tax_amount: float = Field(default=0.0, description="Sales tax amount")
    tip_amount: float = Field(default=0.0, description="Tip or service fee")
    total_amount: float = Field(description="Final grand total listed on receipt")


class ReimbursementClaim(BaseModel):
    claimant_name: str = Field(description="Full name of person requesting reimbursement")
    category_or_event: str = Field(description="PTA event, committee, or budget category")
    claimed_amount: float = Field(description="Total amount claimed on the reimbursement form")
    form_notes: Optional[str] = Field(default="", description="Any additional comments on form")


class AuditStatus(str, Enum):
    APPROVED = "APPROVED"
    MATH_MISMATCH = "MATH_MISMATCH"
    UNITEMIZED = "UNITEMIZED_RECEIPT"
    FORM_EXTRACTION_ERROR = "FORM_EXTRACTION_ERROR"


class AuditReport(BaseModel):
    claimant_name: str
    category: str
    claimed_amount: float
    supported_amount: float
    difference: float
    status: AuditStatus
    receipt_breakdown: List[ParsedReceipt]
    discrepancy_notes: List[str]


# ==========================================
# 2. File Conversion Utilities
# ==========================================

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
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        with Image.open(file_path) as img:
            buffered = io.BytesIO()
            img.convert("RGB").save(buffered, format="PNG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file {file_path}")

    return base64_images


# ==========================================
# 3. Agent Graph State & Nodes (Gemini Powered)
# ==========================================

class AgentState(BaseModel):
    form_path: str
    receipt_paths: List[str]
    claim: Optional[ReimbursementClaim] = None
    parsed_receipts: List[ParsedReceipt] = Field(default_factory=list)
    audit_report: Optional[AuditReport] = None


def process_claim_form_node(state: AgentState) -> dict:
    """Extracts structured reimbursement claim information using Gemini."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
    structured_llm = llm.with_structured_output(ReimbursementClaim)

    base64_images = load_file_as_base64_images(state.form_path)
    
    content_payload = [{"type": "text", "text": "Extract reimbursement claim form details."}]
    for b64 in base64_images:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    claim_data = structured_llm.invoke([{"role": "user", "content": content_payload}])
    return {"claim": claim_data}


def process_receipts_node(state: AgentState) -> dict:
    """Extracts itemized details from every submitted receipt using Gemini."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.0)
    structured_llm = llm.with_structured_output(ParsedReceipt)
    
    parsed_list = []
    
    for receipt_path in state.receipt_paths:
        # All pages of one file form a single receipt, so a total repeated on a
        # later page (e.g. a payment confirmation) is not counted twice.
        base64_images = load_file_as_base64_images(receipt_path)
        payload = [
            {
                "type": "text",
                "text": (
                    "These images are all pages of ONE receipt. Extract vendor name, date, "
                    "itemized prices, tax, tip, and grand total for the receipt as a whole. "
                    "Do not duplicate items or totals that repeat across pages."
                )
            }
        ]
        for b64 in base64_images:
            payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        receipt_obj = structured_llm.invoke([{"role": "user", "content": payload}])
        parsed_list.append(receipt_obj)

    return {"parsed_receipts": parsed_list}


def audit_and_cross_check_node(state: AgentState) -> dict:
    """Performs deterministic math calculations to verify receipts match the form claim."""
    claim = state.claim
    receipts = state.parsed_receipts
    notes = []
    
    total_supported = 0.0
    has_unitemized = False

    for idx, receipt in enumerate(receipts, start=1):
        total_supported += receipt.total_amount
        vendor = receipt.vendor_name or f"Receipt #{idx}"

        # Rule 1: Check for unitemized submissions
        if not receipt.items:
            has_unitemized = True
            notes.append(f"[{vendor}] Receipt lacks itemized line items.")

        # Rule 2: Verify receipt internal arithmetic
        if receipt.items:
            calculated_sum = round(sum(i.amount for i in receipt.items) + receipt.tax_amount + receipt.tip_amount, 2)
            if abs(calculated_sum - receipt.total_amount) > 0.05:
                notes.append(
                    f"[{vendor}] Receipt internal math inconsistency: Sum of items (${calculated_sum:.2f}) "
                    f"does not match grand total on receipt (${receipt.total_amount:.2f})."
                )

    claimed = round(claim.claimed_amount, 2)
    supported = round(total_supported, 2)
    diff = round(claimed - supported, 2)

    # Determine final status
    if abs(diff) <= 0.01 and not has_unitemized:
        status = AuditStatus.APPROVED
        notes.append("CLAIM VERIFIED: Receipts fully support the claimed amount.")
    elif has_unitemized:
        status = AuditStatus.UNITEMIZED
    else:
        status = AuditStatus.MATH_MISMATCH
        if diff > 0:
            notes.append(f"DISCREPANCY: Claimed amount (${claimed:.2f}) exceeds supported receipts (${supported:.2f}) by ${diff:.2f}.")
        else:
            notes.append(f"OVER-DOCUMENTED: Total receipts (${supported:.2f}) exceed claimed amount (${claimed:.2f}) by ${abs(diff):.2f}.")

    report = AuditReport(
        claimant_name=claim.claimant_name,
        category=claim.category_or_event,
        claimed_amount=claimed,
        supported_amount=supported,
        difference=diff,
        status=status,
        receipt_breakdown=receipts,
        discrepancy_notes=notes
    )

    return {"audit_report": report}


# ==========================================
# 4. Build LangGraph Workflow
# ==========================================

def build_audit_graph():
    builder = StateGraph(AgentState)

    builder.add_node("process_form", process_claim_form_node)
    builder.add_node("process_receipts", process_receipts_node)
    builder.add_node("audit_check", audit_and_cross_check_node)

    builder.set_entry_point("process_form")
    builder.add_edge("process_form", "process_receipts")
    builder.add_edge("process_receipts", "audit_check")
    builder.add_edge("audit_check", END)

    return builder.compile()


# ==========================================
# 5. Markdown Report
# ==========================================

def _md_cell(text) -> str:
    """Escapes a value for use inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(report: AuditReport, form_path: str, receipt_paths: List[str]) -> str:
    lines = [
        "# PTA Reimbursement Verification Report",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Claimant | {_md_cell(report.claimant_name)} |",
        f"| Category/Event | {_md_cell(report.category)} |",
        f"| Status | **{report.status.value}** |",
        f"| Claimed Amount | ${report.claimed_amount:.2f} |",
        f"| Supported Amount | ${report.supported_amount:.2f} |",
        f"| Variance | ${report.difference:.2f} |",
        f"| Reimbursement Form | `{Path(form_path).name}` |",
        "",
        "## Extracted Receipt Breakdown",
        "",
    ]

    for idx, (rec, path) in enumerate(zip(report.receipt_breakdown, receipt_paths), start=1):
        lines += [
            f"### {idx}. {_md_cell(rec.vendor_name)}",
            "",
            f"- **File:** `{Path(path).name}`",
            f"- **Date:** {rec.receipt_date}",
            f"- **Total:** ${rec.total_amount:.2f}",
            "",
            "| Item | Amount |",
            "|---|---:|",
        ]
        for item in rec.items:
            lines.append(f"| {_md_cell(item.description)} | ${item.amount:.2f} |")
        if rec.tax_amount > 0:
            lines.append(f"| *Tax* | ${rec.tax_amount:.2f} |")
        if rec.tip_amount > 0:
            lines.append(f"| *Tip* | ${rec.tip_amount:.2f} |")
        lines.append("")

    lines += ["## Audit Notes & Warnings", ""]
    lines += [f"- {note}" for note in report.discrepancy_notes]
    lines.append("")

    return "\n".join(lines)


# ==========================================
# 6. CLI Entrypoint
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="PTA Reimbursement Audit Agent (Gemini): Cross-check reimbursement claim form against supporting receipts."
    )
    
    parser.add_argument(
        "-f", "--reimbursement_form",
        required=True,
        type=str,
        help="Path to the PDF scan or image of the completed reimbursement form."
    )
    
    parser.add_argument(
        "-r", "--supporting_receipts",
        required=True,
        nargs="+",
        help="Path(s) to supporting receipt PDF scans or image files (space-separated)."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output prefix; the report is also written to <prefix>_reimbursement_verification.md."
    )

    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("\n[+] Initializing PTA Reimbursement Agent (Powered by Gemini)...")
    print(f"    Reimbursement Form: {args.reimbursement_form}")
    print(f"    Supporting Receipts ({len(args.supporting_receipts)}): {', '.join(args.supporting_receipts)}")
    
    graph = build_audit_graph()
    
    initial_state = AgentState(
        form_path=args.reimbursement_form,
        receipt_paths=args.supporting_receipts
    )
    
    try:
        final_state = graph.invoke(initial_state)
        report: AuditReport = final_state["audit_report"]
        
        print("\n" + "="*50)
        print("         PTA REIMBURSEMENT VERIFICATION REPORT        ")
        print("="*50)
        print(f"Claimant:          {report.claimant_name}")
        print(f"Category/Event:    {report.category}")
        print(f"Status:            {report.status.value}")
        print(f"Claimed Amount:    ${report.claimed_amount:.2f}")
        print(f"Supported Amount:  ${report.supported_amount:.2f}")
        print(f"Variance:          ${report.difference:.2f}")
        print("-"*50)
        print("EXTRACTED RECEIPT BREAKDOWN:")
        for idx, rec in enumerate(report.receipt_breakdown, start=1):
            print(f"  {idx}. Vendor: {rec.vendor_name} | Date: {rec.receipt_date} | Total: ${rec.total_amount:.2f}")
            for item in rec.items:
                print(f"     - {item.description}: ${item.amount:.2f}")
            if rec.tax_amount > 0:
                print(f"     - [Tax]: ${rec.tax_amount:.2f}")
            if rec.tip_amount > 0:
                print(f"     - [Tip]: ${rec.tip_amount:.2f}")
        print("-"*50)
        print("AUDIT NOTES & WARNINGS:")
        for note in report.discrepancy_notes:
            print(f"  * {note}")
        print("="*50 + "\n")

        if args.output:
            output_path = Path(f"{args.output}_reimbursement_verification.md")
            output_path.write_text(
                render_markdown_report(report, args.reimbursement_form, args.supporting_receipts),
                encoding="utf-8"
            )
            print(f"[+] Markdown report written to {output_path}\n")

    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()