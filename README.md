# PTA Cash Verification Agent

A command-line tool that checks PTA cash deposits. It reads a completed
California State PTA **Cash Verification Form** and the bank **deposit slip or
receipt** (PDF scans or images), extracts the key details with Google Gemini,
and checks that the amount deposited matches the amount counted on the form.

## What it does

The agent is a small [LangGraph](https://github.com/langchain-ai/langgraph) workflow with three steps:

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

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Poppler](https://poppler.freedesktop.org/), used to convert PDFs to images
  (macOS: `brew install poppler`, Debian/Ubuntu: `apt install poppler-utils`)
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

```bash
cd pta-cash-verification-agent
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
uv run --env-file .env pta-cash-agent.py \
  -f 5th_grade_popsicle_sales_20261002.pdf \
  -d 5th_grade_popsicle_sales_20261002.png \
  -o 5th_grade_popsicle_sales_20261002
```

| Option | Description |
|---|---|
| `-f`, `--cash_verification_form` | The completed Cash Verification Form (PDF or image). Required. |
| `-d`, `--deposit_slip` | The bank deposit slip or teller receipt (PDF, PNG, JPG, JPEG, or WEBP). Required. |
| `-o`, `--output` | Optional output prefix. The report is also written to `<prefix>_cash_verification.pdf`. |

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

## Notes

- **Model:** the Gemini model is set by `GEMINI_MODEL` near the top of `pta-cash-agent.py`.
- **API limits:** each run makes two Gemini requests, one for the form and one
  for the deposit slip. If you hit `429 RESOURCE_EXHAUSTED` on the free tier, wait
  a minute and retry.
- **Review the results:** the AI can misread handwriting. Check the extracted
  counts against the form before signing off on a deposit.
