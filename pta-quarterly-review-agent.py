#!/usr/bin/env python3
"""
PTA Quarterly Financial Review Agent (Gemini, or a local Ollama vision model)
Extracts a quarter's monthly bank statements and QuickBooks reconciliation reports,
cross-checks them, and builds the treasurer's quarterly report: balance on hand,
receipts, disbursements, ending balance, deposits not yet credited, and uncleared checks.
"""

import argparse
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional

from fpdf import FontFace
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from pta_common import (
    GREEN, ORANGE, RED, TABLE_HEADING_FILL,
    add_llm_arguments, configure_llm_from_args, describe_llm, image_payload,
    make_structured_llm, new_report_pdf, pdf_heading, pdf_notes, pdf_text,
    require_api_key,
)

# Uncleared items older than this (relative to the quarter end) are flagged as stale.
STALE_DAYS = 90


# ==========================================
# 1. Pydantic Schemas
# ==========================================

class TransactionKind(str, Enum):
    DEPOSIT = "deposit"
    CHECK = "check"
    CASH_WITHDRAWAL = "cash_withdrawal"
    OTHER_WITHDRAWAL = "other_withdrawal"


class BankTransaction(BaseModel):
    date: str = Field(description="Posting date as MM/DD/YYYY; take the year from the statement period")
    amount: float = Field(description="Signed amount: positive for deposits, negative for checks and withdrawals (a trailing '-' means negative)")
    description: str = Field(description="Short description without the TYPE/ID/CO details, e.g. 'Deposit ACH Zeffy US', 'Deposit Check', 'Check 1089', 'Withdrawal Cash'")
    check_number: Optional[str] = Field(default=None, description="Check number for checks, e.g. '1089'")
    kind: TransactionKind = Field(description="'deposit', 'check', 'cash_withdrawal' (e.g. 'Withdrawal Cash'), or 'other_withdrawal' (ACH debits, fees, transfers out)")


class BankStatement(BaseModel):
    institution: Optional[str] = Field(default="Unknown Bank", description="Bank or credit union name")
    account: Optional[str] = Field(default="", description="Account name and number as printed, e.g. 'Bus Standard Checking 4279'")
    period_start: str = Field(description="First day of the statement period as MM/DD/YYYY")
    period_end: str = Field(description="Last day of the statement period as MM/DD/YYYY")
    beginning_balance: float = Field(description="Beginning Balance from the summary box")
    ending_balance: float = Field(description="Ending Balance from the summary box")
    deposits_count: int = Field(default=0, description="Number of deposits from the summary box")
    deposits_total: float = Field(description="Total of deposits from the summary box, as a positive number")
    withdrawals_count: int = Field(default=0, description="Number of withdrawals from the summary box")
    withdrawals_total: float = Field(description="Total of withdrawals from the summary box, as a positive number")
    transactions: List[BankTransaction] = Field(default_factory=list, description="Every row of the transaction table, in order")


class ReconLine(BaseModel):
    date: str = Field(description="DATE column as MM/DD/YYYY")
    type: str = Field(description="TYPE column, e.g. 'Check', 'Expense', 'Deposit'")
    ref_no: Optional[str] = Field(default=None, description="REF NO. column (check number), if any")
    payee: Optional[str] = Field(default=None, description="PAYEE column, if any")
    amount: float = Field(description="AMOUNT (USD) as printed, including the minus sign")


class ReconciliationReport(BaseModel):
    organization: Optional[str] = Field(default="", description="Organization name at the top of the report")
    account: Optional[str] = Field(default="", description="Account name at the top, e.g. 'Bus Standard Checking 4279 - 1'")
    period_end: str = Field(description="'Period Ending' date as MM/DD/YYYY")
    reconciled_on: Optional[str] = Field(default="", description="'Reconciled on' date")
    reconciled_by: Optional[str] = Field(default="", description="'Reconciled by' name")
    statement_beginning_balance: float = Field(description="Statement beginning balance")
    statement_ending_balance: float = Field(description="Statement ending balance")
    cleared_payments_total: float = Field(description="Checks and payments cleared total from the summary (negative)")
    cleared_deposits_total: float = Field(description="Deposits and other credits cleared total from the summary")
    uncleared_total: float = Field(description="'Uncleared transactions as of <period end>' from the summary")
    register_balance: float = Field(description="'Register balance as of <period end>' from the summary, NOT a later date")
    cleared_payments: List[ReconLine] = Field(default_factory=list, description="Rows of the 'Checks and payments cleared' table")
    cleared_deposits: List[ReconLine] = Field(default_factory=list, description="Rows of the 'Deposits and other credits cleared' table")
    uncleared_payments: List[ReconLine] = Field(default_factory=list, description="Rows of 'Uncleared checks and payments as of <period end>' only")
    uncleared_deposits: List[ReconLine] = Field(default_factory=list, description="Rows of 'Uncleared deposits and other credits as of <period end>' only")
    later_items: List[ReconLine] = Field(default_factory=list, description="Rows of any 'Uncleared ... after <period end>' sections, as printed")


class ReviewStatus(str, Enum):
    BALANCED = "BALANCED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DISCREPANCY = "DISCREPANCY"


class MonthSummary(BaseModel):
    period_end: str
    beginning_balance: float
    receipts: float
    disbursements: float
    ending_balance: float
    register_balance: Optional[float]
    reconciled_by: str


class QuarterlyReview(BaseModel):
    organization: str
    account: str
    period_start: str
    period_end: str
    beginning_balance: float
    receipts: float
    receipts_count: int
    disbursements: float
    disbursements_count: int
    ending_balance: float
    deposits_in_transit: List[ReconLine]
    deposits_in_transit_total: float
    uncleared_checks: List[ReconLine]
    uncleared_checks_total: float
    adjusted_balance: float
    register_balance: Optional[float]
    months: List[MonthSummary]
    receipt_lines: List[BankTransaction]
    disbursement_lines: List[BankTransaction]
    status: ReviewStatus
    notes: List[str]


# ==========================================
# 2. Agent Graph State & Nodes
# ==========================================

class AgentState(BaseModel):
    statement_paths: List[str]
    reconciliation_paths: List[str]
    statements: List[BankStatement] = Field(default_factory=list)
    reconciliations: List[ReconciliationReport] = Field(default_factory=list)
    review: Optional[QuarterlyReview] = None


def process_statements_node(state: AgentState) -> dict:
    """Extracts balances and transactions from each monthly bank statement."""
    structured_llm = make_structured_llm(BankStatement)

    prompt = (
        "This is a monthly bank account statement. Extract the statement period, the "
        "Beginning Balance, Ending Balance, deposit and withdrawal counts and totals from the "
        "account summary box, and every row of the transaction table. Amounts with a trailing "
        "'-' are withdrawals and must be negative. Dates in the table omit the year; use the "
        "year of the statement period. Ignore the check-number summary, fee table, disclosures, "
        "and privacy notice pages."
    )
    statements = []
    for path in state.statement_paths:
        statement = structured_llm.invoke([{"role": "user", "content": image_payload(prompt, path)}])
        statements.append(_normalize_statement(statement))
    return {"statements": statements}


def process_reconciliations_node(state: AgentState) -> dict:
    """Extracts the summary and line items from each QuickBooks reconciliation report."""
    structured_llm = make_structured_llm(ReconciliationReport)

    prompt = (
        "This is a QuickBooks bank RECONCILIATION REPORT. Extract the header (organization, "
        "account, period ending, reconciled on/by), the Summary amounts, and every row of the "
        "cleared and uncleared tables, which may continue across pages. For the uncleared "
        "tables, use ONLY the sections titled 'as of <period end>'; put the rows of any sections "
        "titled 'after <period end>' in later_items instead. Use the register balance as of the period end, "
        "not a later date. Include zero-amount rows (voided checks) as printed."
    )
    reconciliations = []
    for path in state.reconciliation_paths:
        recon = structured_llm.invoke([{"role": "user", "content": image_payload(prompt, path)}])
        reconciliations.append(_normalize_reconciliation(recon))
    return {"reconciliations": reconciliations}


def _normalize_statement(statement: BankStatement) -> BankStatement:
    """Forces sign conventions, since models sometimes drop a trailing '-'."""
    statement.deposits_total = abs(statement.deposits_total)
    statement.withdrawals_total = abs(statement.withdrawals_total)
    for txn in statement.transactions:
        txn.amount = abs(txn.amount) if txn.kind == TransactionKind.DEPOSIT else -abs(txn.amount)
    return statement


def _normalize_reconciliation(recon: ReconciliationReport) -> ReconciliationReport:
    """Forces payments negative and deposits positive."""
    recon.cleared_payments_total = -abs(recon.cleared_payments_total)
    recon.cleared_deposits_total = abs(recon.cleared_deposits_total)
    for line in recon.cleared_payments + recon.uncleared_payments:
        line.amount = -abs(line.amount)
    for line in recon.cleared_deposits + recon.uncleared_deposits:
        line.amount = abs(line.amount)
    return recon


# ---------- review helpers ----------

class Item(NamedTuple):
    cents: int
    ref: Optional[str]
    label: str


def parse_date(text: Optional[str]) -> Optional[date]:
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime((text or "").strip(), fmt).date()
        except ValueError:
            continue
    return None


def cents(amount: float) -> int:
    return round(amount * 100)


def money(amount: float) -> str:
    return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"


def check_ref(value: Optional[str]) -> Optional[str]:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits or None


def describe_line(line: ReconLine) -> str:
    parts = [line.date, line.type, f"#{line.ref_no}" if line.ref_no else None, line.payee]
    return " ".join(p for p in parts if p) + f" {money(line.amount)}"


def recon_item(line: ReconLine) -> Item:
    return Item(cents(line.amount), check_ref(line.ref_no), describe_line(line))


def bank_item(txn: BankTransaction) -> Item:
    return Item(cents(txn.amount), check_ref(txn.check_number), f"{txn.date} {txn.description} {money(txn.amount)}")


def match_items(left: List[Item], right: List[Item]) -> tuple[List[Item], List[Item]]:
    """
    Pairs items with the same check number and amount first, then the same amount alone
    (QuickBooks may record a cleared check as an Expense without its number).
    Returns the unmatched items from each side.
    """
    remaining = list(right)

    def take(predicate) -> bool:
        for i, item in enumerate(remaining):
            if predicate(item):
                remaining.pop(i)
                return True
        return False

    deferred = [item for item in left
                if not (item.ref and take(lambda r: r.cents == item.cents and r.ref == item.ref))]
    unmatched = [item for item in deferred if not take(lambda r: r.cents == item.cents)]
    return unmatched, remaining


def month_label(text: str) -> str:
    parsed = parse_date(text)
    return parsed.strftime("%m/%Y") if parsed else text


def compress_ranges(numbers: List[int]) -> str:
    """[1090, 1091, 1092, 1102] -> '1090-1092, 1102'"""
    ranges, start = [], None
    for i, n in enumerate(numbers):
        if start is None:
            start = n
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            ranges.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ", ".join(ranges)


def review_node(state: AgentState) -> dict:
    """Deterministic cross-checks and quarterly totals; no LLM."""
    errors: List[str] = []
    flags: List[str] = []

    statements = sorted(state.statements, key=lambda s: parse_date(s.period_end) or date.max)
    recons_by_month = {}
    for recon in state.reconciliations:
        key = month_label(recon.period_end)
        if key in recons_by_month:
            errors.append(f"[QuickBooks] Two reconciliation reports are for {key}.")
        recons_by_month[key] = recon

    months: List[MonthSummary] = []
    paired: List[ReconciliationReport] = []
    prev_statement: Optional[BankStatement] = None
    prev_recon: Optional[ReconciliationReport] = None
    voided_refs: set[str] = set()

    for statement in statements:
        month = month_label(statement.period_end)
        tag = f"[Bank {month}]"

        # Rule 1: Statement math, from the summary box and from the transaction rows.
        expected_end = statement.beginning_balance + statement.deposits_total - statement.withdrawals_total
        if cents(expected_end) != cents(statement.ending_balance):
            errors.append(f"{tag} Beginning {money(statement.beginning_balance)} + deposits "
                          f"{money(statement.deposits_total)} - withdrawals {money(statement.withdrawals_total)} = "
                          f"{money(expected_end)}, but the ending balance is {money(statement.ending_balance)}.")
        txn_deposits = sum(t.amount for t in statement.transactions if t.amount > 0)
        txn_withdrawals = -sum(t.amount for t in statement.transactions if t.amount < 0)
        if cents(txn_deposits) != cents(statement.deposits_total):
            errors.append(f"{tag} Deposit rows add up to {money(txn_deposits)}, "
                          f"but the summary shows {money(statement.deposits_total)}.")
        if cents(txn_withdrawals) != cents(statement.withdrawals_total):
            errors.append(f"{tag} Withdrawal rows add up to {money(txn_withdrawals)}, "
                          f"but the summary shows {money(statement.withdrawals_total)}.")

        # Rule 2: Continuity with the prior month.
        if prev_statement is not None:
            if cents(statement.beginning_balance) != cents(prev_statement.ending_balance):
                errors.append(f"{tag} Beginning balance {money(statement.beginning_balance)} doesn't match the "
                              f"prior statement's ending balance {money(prev_statement.ending_balance)}.")
            prev_end, start = parse_date(prev_statement.period_end), parse_date(statement.period_start)
            if prev_end and start and (start - prev_end).days != 1:
                errors.append(f"{tag} Statement period starts {statement.period_start}, but the prior one "
                              f"ended {prev_statement.period_end}; a month may be missing.")

        recon = recons_by_month.pop(month, None)
        if recon is None:
            errors.append(f"[QuickBooks {month}] No reconciliation report was provided for this month.")
        else:
            paired.append(recon)
            qtag = f"[QuickBooks {month}]"
            xtag = f"[Bank vs QuickBooks {month}]"

            # Rule 3: QuickBooks summary agrees with the bank statement.
            for label, ours, theirs in (
                ("beginning balance", recon.statement_beginning_balance, statement.beginning_balance),
                ("ending balance", recon.statement_ending_balance, statement.ending_balance),
                ("cleared payments", -recon.cleared_payments_total, statement.withdrawals_total),
                ("cleared deposits", recon.cleared_deposits_total, statement.deposits_total),
            ):
                if cents(ours) != cents(theirs):
                    errors.append(f"{xtag} QuickBooks {label} {money(ours)} differs from the bank's {money(theirs)}.")

            # Rule 4: Every bank transaction has a cleared QuickBooks line, and vice versa.
            cleared = [recon_item(line) for line in recon.cleared_payments + recon.cleared_deposits if line.amount]
            bank_only, books_only = match_items([bank_item(t) for t in statement.transactions], cleared)
            for item in bank_only:
                errors.append(f"{xtag} On the bank statement but not cleared in QuickBooks: {item.label}.")
            for item in books_only:
                errors.append(f"{xtag} Cleared in QuickBooks but not on the bank statement: {item.label}.")

            # Rule 5: Reconciliation report math.
            cleared_end = recon.statement_beginning_balance + recon.cleared_payments_total + recon.cleared_deposits_total
            if cents(cleared_end) != cents(recon.statement_ending_balance):
                errors.append(f"{qtag} Beginning + cleared items = {money(cleared_end)}, but the statement "
                              f"ending balance is {money(recon.statement_ending_balance)}.")
            uncleared_sum = sum(line.amount for line in recon.uncleared_payments + recon.uncleared_deposits)
            if cents(uncleared_sum) != cents(recon.uncleared_total):
                errors.append(f"{qtag} Uncleared rows add up to {money(uncleared_sum)}, but the summary "
                              f"shows {money(recon.uncleared_total)}.")
            if cents(recon.statement_ending_balance + recon.uncleared_total) != cents(recon.register_balance):
                errors.append(f"{qtag} Statement ending {money(recon.statement_ending_balance)} + uncleared "
                              f"{money(recon.uncleared_total)} doesn't equal the register balance "
                              f"{money(recon.register_balance)}.")

            all_lines = recon.cleared_payments + recon.cleared_deposits + recon.uncleared_payments + recon.uncleared_deposits
            voided_refs |= {check_ref(line.ref_no) for line in all_lines + recon.later_items
                            if not line.amount and check_ref(line.ref_no)}

            # Rule 6: Items uncleared last month must clear or stay uncleared, not disappear.
            if prev_recon is not None:
                previously_uncleared = [recon_item(line) for line in prev_recon.uncleared_payments + prev_recon.uncleared_deposits if line.amount]
                now_listed = [recon_item(line) for line in all_lines if line.amount]
                removed, _ = match_items(previously_uncleared, now_listed)
                if removed:
                    total = sum(item.cents for item in removed) / 100
                    flags.append(f"{qtag} {len(removed)} item(s) uncleared at {prev_recon.period_end} neither cleared "
                                 f"nor remain uncleared (net {money(total)}); they were deleted, voided or edited in "
                                 f"QuickBooks. Confirm each change was intended:")
                    flags.extend(f"    removed: {item.label}" for item in removed)
            prev_recon = recon

        months.append(MonthSummary(
            period_end=statement.period_end,
            beginning_balance=statement.beginning_balance,
            receipts=statement.deposits_total,
            disbursements=statement.withdrawals_total,
            ending_balance=statement.ending_balance,
            register_balance=recon.register_balance if recon else None,
            reconciled_by=(recon.reconciled_by or "") if recon else "Not reconciled",
        ))
        prev_statement = statement

    for month in recons_by_month:
        errors.append(f"[QuickBooks {month}] Reconciliation report doesn't match any bank statement provided.")

    if not statements:
        raise ValueError("No bank statements could be read.")

    first, last = statements[0], statements[-1]
    last_recon = paired[-1] if paired and month_label(paired[-1].period_end) == month_label(last.period_end) else None
    quarter_end = parse_date(last.period_end)
    quarter_start = parse_date(first.period_start)

    receipt_lines = [t for s in statements for t in s.transactions if t.amount > 0]
    disbursement_lines = [t for s in statements for t in s.transactions if t.amount < 0]
    receipts = round(sum(s.deposits_total for s in statements), 2)
    disbursements = round(sum(s.withdrawals_total for s in statements), 2)

    in_transit = [line for line in last_recon.uncleared_deposits if line.amount] if last_recon else []
    uncleared = [line for line in last_recon.uncleared_payments if line.amount] if last_recon else []
    in_transit_total = round(sum(line.amount for line in in_transit), 2)
    uncleared_total = round(-sum(line.amount for line in uncleared), 2)
    adjusted = round(last.ending_balance + in_transit_total - uncleared_total, 2)
    register = last_recon.register_balance if last_recon else None

    # Rule 7: Adjusted bank balance agrees with the QuickBooks register.
    if register is not None and cents(adjusted) != cents(register):
        errors.append(f"[Review] Adjusted bank balance {money(adjusted)} doesn't equal the QuickBooks register "
                      f"balance {money(register)} at {last.period_end}.")

    # Treasurer flags: stale uncleared items, cash withdrawals, unaccounted check numbers.
    for line in in_transit + uncleared:
        issued = parse_date(line.date)
        if issued and quarter_end and (quarter_end - issued).days > STALE_DAYS:
            action = ("confirm it reached the bank or correct the QuickBooks entry" if line.amount > 0
                      else "follow up with the payee, or void and reissue")
            flags.append(f"[Stale] {describe_line(line)} has been uncleared for {(quarter_end - issued).days} days; {action}.")

    for txn in disbursement_lines:
        if txn.kind == TransactionKind.CASH_WITHDRAWAL:
            flags.append(f"[Cash] Cash withdrawal on {txn.date} for {money(txn.amount)}; confirm it was approved "
                         f"and is supported by receipts or a cash advance form.")

    recon_lines = [line for r in state.reconciliations
                   for line in r.cleared_payments + r.cleared_deposits + r.uncleared_payments + r.uncleared_deposits]
    issued_this_quarter = {int(ref) for line in recon_lines
                           if (ref := check_ref(line.ref_no)) and (d := parse_date(line.date))
                           and quarter_start and quarter_end and quarter_start <= d <= quarter_end}
    known = {int(ref) for ref in voided_refs}
    known |= {int(ref) for r in state.reconciliations for line in recon_lines + r.later_items
              if (ref := check_ref(line.ref_no))}
    known |= {int(ref) for t in disbursement_lines if (ref := check_ref(t.check_number))}
    if issued_this_quarter:
        missing = [n for n in range(min(issued_this_quarter), max(issued_this_quarter) + 1) if n not in known]
        if missing:
            flags.append(f"[Checks] Check number(s) {compress_ranges(missing)} fall within the range used this "
                         f"quarter but don't appear on any statement or report; confirm they are unused or voided "
                         f"(keep voided checks on file).")
    if voided_refs:
        flags.append(f"[Checks] Voided check(s): {', '.join(sorted(voided_refs))}. Keep the voided checks on file.")

    if errors:
        status = ReviewStatus.DISCREPANCY
        summary = "DISCREPANCY: The bank statements and QuickBooks reports don't agree; see the notes above."
    elif flags:
        status = ReviewStatus.NEEDS_REVIEW
        summary = "The quarter balances, but the items above need the treasurer's attention."
    else:
        status = ReviewStatus.BALANCED
        summary = "BALANCED: The bank statements and QuickBooks reconciliations agree for the quarter."

    review = QuarterlyReview(
        organization=(last_recon.organization if last_recon and last_recon.organization else "PTA"),
        account=" ".join(p for p in (last.institution, last.account) if p),
        period_start=first.period_start,
        period_end=last.period_end,
        beginning_balance=first.beginning_balance,
        receipts=receipts,
        receipts_count=sum(s.deposits_count for s in statements),
        disbursements=disbursements,
        disbursements_count=sum(s.withdrawals_count for s in statements),
        ending_balance=last.ending_balance,
        deposits_in_transit=in_transit,
        deposits_in_transit_total=in_transit_total,
        uncleared_checks=uncleared,
        uncleared_checks_total=uncleared_total,
        adjusted_balance=adjusted,
        register_balance=register,
        months=months,
        receipt_lines=receipt_lines,
        disbursement_lines=disbursement_lines,
        status=status,
        notes=errors + flags + [summary],
    )
    return {"review": review}


# ==========================================
# 3. Build LangGraph Workflow
# ==========================================

def build_review_graph():
    builder = StateGraph(AgentState)

    builder.add_node("process_statements", process_statements_node)
    builder.add_node("process_reconciliations", process_reconciliations_node)
    builder.add_node("review", review_node)

    builder.set_entry_point("process_statements")
    builder.add_edge("process_statements", "process_reconciliations")
    builder.add_edge("process_reconciliations", "review")
    builder.add_edge("review", END)

    return builder.compile()


# ==========================================
# 4. Report Output
# ==========================================

STATUS_COLORS = {
    ReviewStatus.BALANCED: GREEN,
    ReviewStatus.NEEDS_REVIEW: ORANGE,
    ReviewStatus.DISCREPANCY: RED,
}


def treasurer_report_rows(review: QuarterlyReview) -> List[tuple[str, str]]:
    """The six-line treasurer's report plus the adjusted balance check."""
    rows = [
        (f"1. Balance on hand, {review.period_start}", money(review.beginning_balance)),
        (f"2. Receipts ({review.receipts_count} deposits)", f"+ {money(review.receipts)}"),
        (f"3. Disbursements ({review.disbursements_count} withdrawals)", f"- {money(review.disbursements)}"),
        (f"4. Balance on hand, {review.period_end}", money(review.ending_balance)),
        (f"5. Deposits not yet credited ({len(review.deposits_in_transit)})", f"+ {money(review.deposits_in_transit_total)}"),
        (f"6. Uncleared checks ({len(review.uncleared_checks)})", f"- {money(review.uncleared_checks_total)}"),
        ("   Adjusted bank balance", money(review.adjusted_balance)),
    ]
    if review.register_balance is not None:
        rows.append(("   QuickBooks register balance", money(review.register_balance)))
    return rows


def print_review(review: QuarterlyReview) -> None:
    width = 64
    print("\n" + "=" * width)
    print("PTA QUARTERLY FINANCIAL REVIEW".center(width))
    print("=" * width)
    print(f"Organization:  {review.organization}")
    print(f"Account:       {review.account}")
    print(f"Quarter:       {review.period_start} - {review.period_end}")
    print(f"Status:        {review.status.value}")
    print("-" * width)
    for label, amount in treasurer_report_rows(review):
        print(f"{label:<46}{amount:>18}")
    print("-" * width)
    print("MONTHLY BREAKDOWN:")
    for m in review.months:
        print(f"  {m.period_end}: begin {money(m.beginning_balance)} | +{money(m.receipts)} | "
              f"-{money(m.disbursements)} | end {money(m.ending_balance)} | reconciled by {m.reconciled_by}")
    print("-" * width)
    print("DEPOSITS NOT YET CREDITED:")
    for line in review.deposits_in_transit:
        print(f"  {describe_line(line)}")
    if not review.deposits_in_transit:
        print("  (none)")
    print("UNCLEARED CHECKS:")
    for line in review.uncleared_checks:
        print(f"  {describe_line(line)}")
    if not review.uncleared_checks:
        print("  (none)")
    print("-" * width)
    print("REVIEW NOTES & WARNINGS:")
    for note in review.notes:
        print(f"  * {note}")
    print("=" * width + "\n")


def _pdf_table(pdf, headings, widths, aligns, rows) -> None:
    with pdf.table(col_widths=widths, text_align=aligns, line_height=6,
                   headings_style=FontFace(emphasis="BOLD", fill_color=TABLE_HEADING_FILL)) as table:
        table.row(headings)
        for row in rows:
            table.row([pdf_text(cell) for cell in row])
    pdf.ln(5)


def write_pdf_report(review: QuarterlyReview, statement_paths: List[str], reconciliation_paths: List[str],
                     output_path: Path) -> None:
    pdf = new_report_pdf()
    pdf_heading(pdf, "PTA Quarterly Financial Review", 18)

    # Summary
    pdf.set_font("Helvetica", size=10)
    summary = [
        ("Organization", review.organization),
        ("Account", review.account),
        ("Quarter", f"{review.period_start} - {review.period_end}"),
        ("Status", review.status.value),
        ("Bank Statements", ", ".join(Path(p).name for p in statement_paths)),
        ("Reconciliations", ", ".join(Path(p).name for p in reconciliation_paths)),
    ]
    with pdf.table(col_widths=(40, 140), first_row_as_headings=False, line_height=6) as table:
        for label, value in summary:
            row = table.row()
            row.cell(label, style=FontFace(emphasis="BOLD"))
            if label == "Status":
                row.cell(value, style=FontFace(emphasis="BOLD", color=STATUS_COLORS[review.status]))
            else:
                row.cell(pdf_text(value))
    pdf.ln(6)

    # Treasurer's report
    pdf_heading(pdf, "Treasurer's Report", 14)
    pdf.set_font("Helvetica", size=10)
    with pdf.table(col_widths=(130, 50), text_align=("LEFT", "RIGHT"), first_row_as_headings=False,
                   line_height=7) as table:
        for label, amount in treasurer_report_rows(review):
            bold = label.startswith(("1.", "4.", "   Adjusted"))
            style = FontFace(emphasis="BOLD") if bold else None
            row = table.row()
            row.cell(pdf_text(label.strip()), style=style)
            row.cell(amount, style=style)
    pdf.ln(6)

    pdf_heading(pdf, "Monthly Breakdown", 14)
    pdf.set_font("Helvetica", size=9)
    _pdf_table(
        pdf, ["Month End", "Beginning", "Receipts", "Disbursements", "Ending", "Register", "Reconciled By"],
        (22, 26, 22, 26, 26, 26, 32), ("LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "LEFT"),
        [(m.period_end, money(m.beginning_balance), money(m.receipts), money(m.disbursements),
          money(m.ending_balance), money(m.register_balance) if m.register_balance is not None else "-",
          m.reconciled_by) for m in review.months],
    )

    for title, lines in (("Deposits Not Yet Credited", review.deposits_in_transit),
                         ("Uncleared Checks", review.uncleared_checks)):
        pdf_heading(pdf, title, 14)
        pdf.set_font("Helvetica", size=9)
        _pdf_table(
            pdf, ["Date", "Type", "Ref No.", "Payee", "Amount"],
            (25, 25, 20, 80, 30), ("LEFT", "LEFT", "LEFT", "LEFT", "RIGHT"),
            [(line.date, line.type, line.ref_no or "", line.payee or "", money(line.amount)) for line in lines]
            or [("(none)", "", "", "", "")],
        )

    for title, txns in (("Receipts Detail", review.receipt_lines),
                        ("Disbursements Detail", review.disbursement_lines)):
        pdf_heading(pdf, title, 14)
        pdf.set_font("Helvetica", size=9)
        _pdf_table(
            pdf, ["Date", "Description", "Check No.", "Amount"],
            (25, 105, 20, 30), ("LEFT", "LEFT", "LEFT", "RIGHT"),
            [(t.date, t.description, t.check_number or "", money(t.amount)) for t in txns]
            or [("(none)", "", "", "")],
        )

    pdf_notes(pdf, "Review Notes & Warnings", review.notes)

    pdf.output(str(output_path))


# ==========================================
# 5. CLI Entrypoint
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="PTA Quarterly Financial Review Agent: Build the treasurer's quarterly report from bank "
                    "statements and QuickBooks reconciliation reports."
    )

    parser.add_argument(
        "-s", "--statements",
        required=True,
        nargs="+",
        type=str,
        help="Paths to the quarter's monthly bank statements (PDF or image), in any order."
    )

    parser.add_argument(
        "-r", "--reconciliations",
        required=True,
        nargs="+",
        type=str,
        help="Paths to the matching QuickBooks reconciliation reports (PDF or image), in any order."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output prefix; the report is also written to <prefix>_quarterly_financial_review.pdf."
    )

    add_llm_arguments(parser)

    args = parser.parse_args()
    configure_llm_from_args(parser, args)
    require_api_key()

    print(f"\n[+] Initializing PTA Quarterly Financial Review Agent ({describe_llm()})...")
    print(f"    Bank Statements ({len(args.statements)}): {', '.join(args.statements)}")
    print(f"    Reconciliations ({len(args.reconciliations)}): {', '.join(args.reconciliations)}")

    graph = build_review_graph()

    initial_state = AgentState(
        statement_paths=args.statements,
        reconciliation_paths=args.reconciliations,
    )

    try:
        final_state = graph.invoke(initial_state)
        review: QuarterlyReview = final_state["review"]
        print_review(review)

        if args.output:
            # Accept "-o Name.pdf" as the prefix "Name".
            prefix = args.output.removesuffix(".pdf")
            output_path = Path(f"{prefix}_quarterly_financial_review.pdf")
            write_pdf_report(review, args.statements, args.reconciliations, output_path)
            print(f"[+] PDF report written to {output_path}\n")

    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
