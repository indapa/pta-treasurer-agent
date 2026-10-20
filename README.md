# PTA Treasurer Agent

Command-line AI agents that help a PTA treasurer check paperwork. Each tool reads
PDF scans or photos, extracts the key details with a vision LLM, and then checks
the numbers with plain arithmetic, no AI.

| Tool | Script | What it checks |
|---|---|---|
| [Reimbursement Agent](#reimbursement-agent) | `pta-reimbursement-agent.py` | The receipts support the amount claimed on a reimbursement form. |
| [Cash Verification Agent](#cash-verification-agent) | `pta-cash-agent.py` | The bank deposit matches the amount counted on the Cash Verification Form. |
| [Quarterly Financial Review Agent](#quarterly-financial-review-agent) | `pta-quarterly-review-agent.py` | The quarter's bank statements and QuickBooks reconciliations agree, and builds the treasurer's quarterly report. |

## Project layout

| File | Contents |
|---|---|
| `pta-reimbursement-agent.py` | Reimbursement Agent: schemas, prompts, audit rules, report. |
| `pta-cash-agent.py` | Cash Verification Agent: schemas, prompts, verification rules, report. |
| `pta-quarterly-review-agent.py` | Quarterly Financial Review Agent: schemas, prompts, cross-checks, report. |
| `pta_common.py` | Shared by all agents: LLM provider setup, loading PDFs and images for the model, PDF report helpers. |

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Poppler](https://poppler.freedesktop.org/), used to convert PDFs to images
  (macOS: `brew install poppler`, Debian/Ubuntu: `apt install poppler-utils`)
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).
  The Cash Verification Agent can instead run locally with [Ollama](https://ollama.com/)
  (see [Running locally with Ollama](#running-locally-with-ollama)).

## Setup

```bash
git clone https://github.com/indapa/pta-treasurer-agent.git
cd pta-treasurer-agent
uv sync
```

Store your API key in a `.env` file in the project folder (it is already in
`.gitignore`, so it won't be committed):

```bash
echo 'GOOGLE_API_KEY=your-key-here' > .env
chmod 600 .env
```

Both tools share this environment and API key.

---

## Reimbursement Agent

Audits PTA reimbursement requests. It reads a completed reimbursement form and
the supporting receipts, and checks that the receipts actually support the
amount being claimed.

### What it does

A small [LangGraph](https://github.com/langchain-ai/langgraph) workflow with three steps:

1. **Read the claim form** – extracts the claimant name, PTA event or budget
   category, claimed amount, and any notes.
2. **Read each receipt** – extracts the vendor, date, line items, tax, tip, and
   grand total. All pages of a receipt file are treated as one receipt, so a total
   repeated on a second page (for example, a payment confirmation) is not counted twice.
3. **Audit the claim** – plain arithmetic, no AI:
   - sums the receipt totals and compares them to the claimed amount
   - flags receipts with no itemized line items
   - flags receipts whose items + tax + tip don't add up to the receipt's total

```mermaid
flowchart TD
    form[/"Reimbursement form<br/>-f (PDF or image)"/]
    receipts[/"Receipt files<br/>-r (PDF or image)"/]
    gemini{{"Google Gemini<br/>structured output"}}

    start([START]) --> process_form
    form --> process_form
    process_form["<b>process_form</b><br/>Extract claimant, category,<br/>claimed amount, notes"]
    process_form --> process_receipts
    receipts --> process_receipts
    process_receipts["<b>process_receipts</b><br/>One call per receipt file:<br/>vendor, date, items, tax, tip, total"]
    process_receipts --> audit_check
    audit_check["<b>audit_check</b><br/>Plain arithmetic, no AI:<br/>sum totals, check itemization<br/>and each receipt's math"]

    process_form -.-> gemini
    process_receipts -.-> gemini

    audit_check --> decision{"Receipts itemized<br/>and total = claim?"}
    decision -->|yes| approved["APPROVED"]
    decision -->|"a receipt has no items"| unitemized["UNITEMIZED_RECEIPT"]
    decision -->|"totals differ"| mismatch["MATH_MISMATCH"]

    approved --> finish([END])
    unitemized --> finish
    mismatch --> finish
    finish --> report[/"Terminal report<br/>+ optional PDF (-o)"/]
```

The result is one of these statuses:

| Status | Meaning |
|---|---|
| `APPROVED` | Receipts are itemized and match the claimed amount (within $0.01). |
| `MATH_MISMATCH` | The claimed amount is higher or lower than the receipts support. |
| `UNITEMIZED_RECEIPT` | At least one receipt has no line items. |

The report is printed to the terminal and can also be saved as a PDF file.

### Usage

```bash
uv run --env-file .env pta-reimbursement-agent.py \
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

---

## Cash Verification Agent

Checks PTA cash deposits. It reads a completed California State PTA **Cash
Verification Form** and the bank **deposit slip or receipt**, and checks that
the amount deposited matches the amount counted on the form.

### What it does

A small [LangGraph](https://github.com/langchain-ai/langgraph) workflow with three steps:

```mermaid
flowchart TD
    form[/"Cash Verification Form<br/>(PDF or image)"/] --> process_form
    deposit[/"Deposit slip<br/>(PDF or image)"/] --> process_deposit

    START([START]) --> process_form["<b>process_form</b><br/>LLM extracts handwritten<br/>counts and totals"]
    process_form --> process_deposit["<b>process_deposit</b><br/>LLM extracts the<br/>deposit amount"]
    process_deposit --> verify{"<b>verify</b><br/>Deterministic math checks<br/>(no AI)"}
    verify --> END([END])

    verify -. deposit ≠ grand total .-> mismatch["AMOUNT_MISMATCH"]
    verify -. amounts match,<br/>form math wrong .-> matherr["FORM_MATH_ERROR"]
    verify -. amounts match,<br/>form math correct .-> verified["VERIFIED"]

    classDef ai fill:#e8f0fe,stroke:#4285f4
    classDef ok fill:#e6f4ea,stroke:#16803c
    classDef warn fill:#fef3e0,stroke:#c86e00
    classDef bad fill:#fce8e6,stroke:#be1e2d
    class process_form,process_deposit ai
    class verified ok
    class matherr warn
    class mismatch bad
```

1. **Read the Cash Verification Form**: extracts the unit name, activity, and date,
   then the handwritten count for each coin and bill, the coins, currency and checks
   totals, the grand total, and how many signature lines are signed. The form's
   pre-printed `0.00` placeholders and the "Start Up Cash" section are ignored.
2. **Read the deposit slip**: extracts the bank, date, masked account number, and
   the deposit amount (not the account's new or available balance), plus the cash
   and check portions if the slip lists them separately.
3. **Verify the deposit** using plain arithmetic, with no AI:
   - compares the deposit amount to the form's grand total
   - checks each line on the form (count × denomination = amount)
   - checks the coins and currency totals against their lines, and that
     coins + currency + checks = grand total
   - compares the cash and check portions of the deposit to the form, when the slip lists them
   - warns if fewer than two people signed the form

The result is one of these statuses:

| Status | Meaning |
|---|---|
| `VERIFIED` | The deposit matches the form's grand total (within $0.01) and the form's math is correct. |
| `AMOUNT_MISMATCH` | The deposit is more or less than the form's grand total. |
| `FORM_MATH_ERROR` | The deposit matches the grand total, but a line or subtotal on the form doesn't add up. |

The report is printed to the terminal and can also be saved as a PDF file.

### Usage

```bash
uv run --env-file .env pta-cash-agent.py \
  -f 5th_grade_popsicle_sales_20261002.pdf \
  -d 5th_grade_popsicle_sales_20261002.png \
  -o 5th_grade_popsicle_sales_20261002
```

To run locally instead, add `--provider ollama` (the `--env-file` is then optional).

| Option | Description |
|---|---|
| `-f`, `--cash_verification_form` | The completed Cash Verification Form (PDF or image). Required. |
| `-d`, `--deposit_slip` | The bank deposit slip or teller receipt (PDF, PNG, JPG, JPEG, or WEBP). Required. |
| `-o`, `--output` | Optional output prefix. The report is also written to `<prefix>_cash_deposit_slip_verification.pdf`. |
| `-p`, `--provider` | `gemini` (default) or `ollama` (local). Overrides `LLM_PROVIDER`. |
| `-m`, `--model` | Model for the chosen provider, e.g. `gemma3:12b` or `gemini-3.8-flash`. Overrides `OLLAMA_MODEL` / `GEMINI_MODEL`. |

Example terminal output:

```
==================================================
       PTA CASH DEPOSIT VERIFICATION REPORT
==================================================
Unit:              5TH GRADE PROMOTION COMMITTEE
Activity:          ICE CREAM SALES
Status:            VERIFIED
Form Grand Total:  $184.00
Deposit Amount:    $184.00
Variance:          $0.00
Form Date:         10/2/26
Deposit Date:      10/09/26
--------------------------------------------------
CASH VERIFICATION FORM COUNT:
    16 x 25¢   = $4.00
    75 x $1    = $75.00
    11 x $5    = $55.00
     1 x $10   = $10.00
     2 x $20   = $40.00
  Coins: $4.00 | Currency: $180.00 | Checks: $0.00
--------------------------------------------------
DEPOSIT SLIP:
  MISSION FED CREDIT UNION | Acct: XXXXXX4031 | Date: 10/09/26 | Amount: $184.00
--------------------------------------------------
VERIFICATION NOTES & WARNINGS:
  * DEPOSIT VERIFIED: Deposited amount matches the Cash Verification Form grand total.
==================================================
```

### Running locally with Ollama

Pass `--provider ollama` to the Cash Verification Agent. No API key is needed,
and your scans never leave your machine, but small local models misread
handwritten tables more often than Gemini.

```bash
brew install ollama          # or download the app from https://ollama.com/download
ollama serve                 # not needed if the Ollama menu-bar app is running
ollama pull qwen2.5vl:7b     # vision model, ~6 GB
```

Use a different model with `--model`, for example
`uv run pta-cash-agent.py --provider ollama --model gemma3:12b ...`. The model
must support images.

## Quarterly Financial Review Agent

Builds the treasurer's quarterly financial review. It reads the quarter's monthly
**bank statements** and the matching **QuickBooks reconciliation reports**, checks
that they agree, and produces the report:

1. Balance on hand at the start of the quarter
2. Receipts (deposits)
3. Disbursements
4. Balance on hand at the end of the quarter
5. Deposits not yet credited (added to the balance)
6. Uncleared checks (subtracted from the balance)

The adjusted balance (4 + 5 − 6) should equal the QuickBooks register balance at the end of the quarter.

### What it does

```mermaid
flowchart TD
    statements[/"Monthly bank statements<br/>(PDF or image)"/] --> process_statements
    recons[/"QuickBooks reconciliation<br/>reports (PDF or image)"/] --> process_reconciliations

    START([START]) --> process_statements["<b>process_statements</b><br/>LLM extracts balances<br/>and transactions"]
    process_statements --> process_reconciliations["<b>process_reconciliations</b><br/>LLM extracts summary,<br/>cleared and uncleared items"]
    process_reconciliations --> review{"<b>review</b><br/>Deterministic cross-checks<br/>(no AI)"}
    review --> END([END])

    review -. math or matching fails .-> discrepancy["DISCREPANCY"]
    review -. balances, items to follow up .-> needsreview["NEEDS_REVIEW"]
    review -. balances, nothing flagged .-> balanced["BALANCED"]

    classDef ai fill:#e8f0fe,stroke:#4285f4
    classDef ok fill:#e6f4ea,stroke:#16803c
    classDef warn fill:#fef3e0,stroke:#c86e00
    classDef bad fill:#fce8e6,stroke:#be1e2d
    class process_statements,process_reconciliations ai
    class balanced ok
    class needsreview warn
    class discrepancy bad
```

1. **Read each bank statement**: the period, beginning and ending balances,
   deposit and withdrawal totals, and every transaction.
2. **Read each reconciliation report**: the summary, the cleared items, and the
   uncleared items *as of* the period end. Sections for transactions after the period
   end are ignored.
3. **Review** the quarter using plain arithmetic, with no AI. Files are paired by
   month, so they can be given in any order. For each month it checks:
   - the statement's math, and that its beginning balance equals the prior month's ending balance
   - that QuickBooks shows the same beginning and ending balances and cleared totals as the bank
   - that every bank transaction has a matching cleared item in QuickBooks, and vice versa
     (matched by check number and amount, then amount alone)
   - the reconciliation math: statement ending balance + uncleared items = register balance
   - that the adjusted bank balance at quarter end equals the QuickBooks register balance

   It also flags items for the treasurer to follow up on:
   - uncleared checks or deposits older than 90 days
   - cash withdrawals
   - items uncleared last month that neither cleared nor are still listed this month
     (deleted, voided or edited in QuickBooks)
   - check numbers within the quarter's range that don't appear anywhere, and voided checks

| Status | Meaning |
|---|---|
| `BALANCED` | Everything agrees and nothing was flagged. |
| `NEEDS_REVIEW` | Everything agrees, but some items need follow-up (stale, cash, removed or missing checks). |
| `DISCREPANCY` | A balance, total, or transaction doesn't match between the bank and QuickBooks. |

### Usage

```bash
uv run --env-file .env pta-quarterly-review-agent.py \
  -s "Account Statements_2026-07-31.pdf" "Account Statements_2026-08-31.pdf" "Account Statements_2026-09-30.pdf" \
  -r reconciliation_report_july.pdf reconciliation_report_aug.pdf reconciliation_report_sept.pdf \
  -o Q1_2026
```

| Option | Description |
|---|---|
| `-s`, `--statements` | The monthly bank statements for the quarter (PDF or image), in any order. Required. |
| `-r`, `--reconciliations` | The matching QuickBooks reconciliation reports, in any order. Required. |
| `-o`, `--output` | Optional output prefix. The report is also written to `<prefix>_quarterly_financial_review.pdf`. |
| `-p`, `--provider` | `gemini` (default) or `ollama` (local). Overrides `LLM_PROVIDER`. |
| `-m`, `--model` | Model for the chosen provider. |

Example terminal output (abridged):

```
================================================================
                 PTA QUARTERLY FINANCIAL REVIEW
================================================================
Organization:  Willard B. Hage Elementary PTA
Account:       Mission Fed Bus Standard Checking 4279
Quarter:       07/01/2026 - 09/30/2026
Status:        NEEDS_REVIEW
----------------------------------------------------------------
1. Balance on hand, 07/01/2026                        $51,879.58
2. Receipts (17 deposits)                           + $10,904.18
3. Disbursements (15 withdrawals)                    - $9,712.67
4. Balance on hand, 09/30/2026                        $53,071.09
5. Deposits not yet credited (1)                         + $8.25
6. Uncleared checks (3)                                - $766.72
   Adjusted bank balance                              $52,312.62
   QuickBooks register balance                        $52,312.62
----------------------------------------------------------------
...
REVIEW NOTES & WARNINGS:
  * [Stale] 04/21/2026 Check #1073 Julius A Alcantara -$350.00 has been uncleared for 162 days; follow up with the payee, or void and reissue.
  * [Cash] Cash withdrawal on 09/11/2026 for -$986.00; confirm it was approved and is supported by receipts or a cash advance form.
  * The quarter balances, but the items above need the treasurer's attention.
================================================================
```

---

## Notes

- **Model:** the Gemini model is set by `GEMINI_MODEL` near the top of `pta_common.py`.
  For the Cash Verification Agent, `--provider` / `LLM_PROVIDER` selects `gemini`
  (default) or `ollama`, and `--model` / `OLLAMA_MODEL` picks the model
  (Ollama default `qwen2.5vl:7b`).
- **API limits (Gemini):** the Reimbursement Agent makes one request for the form
  and one per receipt file; the Cash Verification Agent makes two (form and deposit
  slip); the Quarterly Financial Review Agent makes one per statement and
  reconciliation report (six for a quarter). Gemini's free tier allows only a few
  requests per minute per model, so a run may fail with `429 RESOURCE_EXHAUSTED`. Wait a minute and retry, or enable
  billing on your Google AI project.
- **Review the results:** the AI can misread a scan or handwriting. Check the
  extracted amounts against the original documents before approving a
  reimbursement or signing off on a deposit.
