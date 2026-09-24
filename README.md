# PTA Reimbursement Agent

A command-line tool that audits PTA reimbursement requests. It reads a completed
reimbursement form and the supporting receipts (PDF scans or images), extracts
the key details with Google Gemini, and checks that the receipts actually support
the amount being claimed.

## What it does

The agent is a small [LangGraph](https://github.com/langchain-ai/langgraph) workflow with three steps:

1. **Read the claim form** – extracts the claimant name, PTA event or budget
   category, claimed amount, and any notes.
2. **Read each receipt** – extracts the vendor, date, line items, tax, tip, and
   grand total. All pages of a receipt file are treated as one receipt, so a total
   repeated on a second page (for example, a payment confirmation) is not counted twice.
3. **Audit the claim** – plain arithmetic, no AI:
   - sums the receipt totals and compares them to the claimed amount
   - flags receipts with no itemized line items
   - flags receipts whose items + tax + tip don't add up to the receipt's total

The result is one of these statuses:

| Status | Meaning |
|---|---|
| `APPROVED` | Receipts are itemized and match the claimed amount (within $0.01). |
| `MATH_MISMATCH` | The claimed amount is higher or lower than the receipts support. |
| `UNITEMIZED_RECEIPT` | At least one receipt has no line items. |

The report is printed to the terminal and can also be saved as a PDF file.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Poppler](https://poppler.freedesktop.org/), used to convert PDFs to images
  (macOS: `brew install poppler`, Debian/Ubuntu: `apt install poppler-utils`)
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

```bash
git clone https://github.com/indapa/pta-reimbursement-agent.git
cd pta-reimbursement-agent
uv sync
```

Store your API key in a `.env` file in the project folder (it is already in
`.gitignore`, so it won't be committed):

```bash
echo 'GOOGLE_API_KEY=your-key-here' > .env
chmod 600 .env
```

## Usage

```bash
uv run --env-file .env pta-agent.py \
  -f reimbursement_form.pdf \
  -r receipt1.pdf receipt2.jpg receipt3.pdf \
  -o Jane_Doe
```

| Option | Description |
|---|---|
| `-f`, `--reimbursement_form` | The completed reimbursement form (PDF or image). Required. |
| `-r`, `--supporting_receipts` | One or more receipt files (PDF, PNG, JPG, JPEG, or WEBP), space-separated. Required. |
| `-o`, `--output` | Optional output prefix. The report is also written to `<prefix>_reimbursement_verification.pdf`, e.g. `Jane_Doe_reimbursement_verification.pdf`. |

Example terminal output:

```
==================================================
         PTA REIMBURSEMENT VERIFICATION REPORT
==================================================
Claimant:          Jane Doe
Category/Event:    Classroom Supplies
Status:            APPROVED
Claimed Amount:    $84.21
Supported Amount:  $84.21
Variance:          $0.00
--------------------------------------------------
EXTRACTED RECEIPT BREAKDOWN:
  1. Vendor: Office Depot | Date: 09/08/2026 | Total: $69.46
     - Folders (30): $50.40
     - Shipping: $15.16
     - [Tax]: $3.90
  2. Vendor: FedEx | Date: 09/10/2026 | Total: $14.75
     - Printing: $14.75
--------------------------------------------------
AUDIT NOTES & WARNINGS:
  * CLAIM VERIFIED: Receipts fully support the claimed amount.
==================================================
```

## Notes

- **Model:** the Gemini model is set by `GEMINI_MODEL` near the top of `pta-agent.py`.
- **API limits:** each run makes one request for the form and one per receipt file.
  Gemini's free tier allows only a few requests per minute per model, so a claim
  with many receipts may fail with `429 RESOURCE_EXHAUSTED`. Wait a minute and
  retry, or enable billing on your Google AI project.
- **Review the results:** the AI can misread a scan. Check the extracted amounts
  against the receipts before approving a reimbursement.
