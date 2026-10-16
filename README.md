# PTA Cash Verification Agent

A command-line tool that checks PTA cash deposits. It reads a completed
California State PTA **Cash Verification Form** and the bank **deposit slip or
receipt** (PDF scans or images), extracts the key details with a vision LLM (Google Gemini by default, or a local Ollama model),
and checks that the amount deposited matches the amount counted on the form.

## What it does

The agent is a small [LangGraph](https://github.com/langchain-ai/langgraph) workflow with three steps:

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

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [Poppler](https://poppler.freedesktop.org/), used to convert PDFs to images
  (macOS: `brew install poppler`, Debian/Ubuntu: `apt install poppler-utils`)
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
  (the default), **or** [Ollama](https://ollama.com/) with a vision model to run locally

## Setup

```bash
cd pta-cash-verification-agent
uv sync
```

### Option A: Use Google Gemini (default)

Store your API key in a `.env` file in the project folder (it is already in
`.gitignore`, so it won't be committed):

```bash
echo 'GOOGLE_API_KEY=your-key-here' > .env
chmod 600 .env
```

### Option B: Run locally with Ollama

Pass `--provider ollama`. No API key is needed, and your scans never leave your
machine, but small local models misread handwritten tables more often than Gemini.

```bash
brew install ollama          # or download the app from https://ollama.com/download
ollama serve                 # not needed if the Ollama menu-bar app is running
ollama pull qwen2.5vl:7b     # vision model, ~6 GB
```

Use a different model with `--model`, for example
`uv run pta-cash-agent.py --provider ollama --model gemma3:12b ...`. The model must support
images.

## Usage

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
| `-o`, `--output` | Optional output prefix. The report is also written to `<prefix>_cash_verification.pdf`. |
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

## Notes

- **Provider:** `--provider` (or `LLM_PROVIDER`) selects `gemini` (default) or `ollama`.
- **Model:** the Ollama model comes from `OLLAMA_MODEL` (default `qwen2.5vl:7b`).
  The Gemini model is set by `GEMINI_MODEL` near the top of `pta-cash-agent.py`.
- **Local accuracy:** small local models misread handwriting more often than
  Gemini. If the counts look wrong, try a larger vision model.
- **API limits (Gemini):** each run makes two Gemini requests, one for the form and one
  for the deposit slip. If you hit `429 RESOURCE_EXHAUSTED` on the free tier, wait
  a minute and retry.
- **Review the results:** the AI can misread handwriting. Check the extracted
  counts against the form before signing off on a deposit.
