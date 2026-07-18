# Retrieval Coverage Gaps

Found while building the gold evaluation set (from `labeling_worksheet.md`,
sampled from the real 200-row ticket dataset). For each of these, the
ticket's actual `resolution_text` describes something that **no chunk in
the current 33-document knowledge base actually covers**. This is a data
coverage gap, not a retrieval-quality problem — no ranking algorithm can
retrieve a chunk that doesn't exist.

Worth reporting as a finding on its own: **16 of 32 sampled real customer
queries (50%)** had no matching source content.

## ⚠️ Flagged as a content-accuracy risk, not just a gap

**"I want to foreclose my car loan"** — the closest retrieved chunk
(`qa_QA008`) states floating-rate **home** loans have no foreclosure
charges. This ticket's actual resolution says a **car loan** foreclosure
carries a 2% charge. If `qa_QA008` were retrieved for a real car-loan
foreclosure question, the bot would confidently state the wrong policy.
This needs either a dedicated car-loan foreclosure QA/policy chunk, or a
guard so `qa_QA008` isn't surfaced for non-home-loan foreclosure queries.

## Gaps by intent

**Account Access**
- "I forgot my ATM PIN and can't reset it online" — no ATM PIN reset content exists
- "My account is showing negative balance but I didn't overdraw" — no fee-reversal content
- "I can't see my recent transactions in the app" — no app/technical-issue content
- "My fixed deposit matured but amount not credited" — no FD maturity content
- "I need to close my account but there are pending charges" — no account-closure content
- "Why was my NEFT transfer delayed?" — no NEFT-specific content (closest analog: qa_QA016, but that's about salary/payroll delay, a different cause)

**Fraud/Unauthorized**
- "There's an ATM withdrawal from [city] but I'm in [city]" (×3 sampled) — no location-mismatch-specific content; closest analog is generic qa_QA001
- "I never signed up for this subscription but ₹X is charged monthly" — no recurring-mandate/subscription content
- "I got a message that ₹X was debited but I didn't shop" — same subscription/mandate gap

**KYC**
- "Why is my account marked as dormant?" — no dormancy-specific content
- "My PAN verification is failing on net banking" — no PAN/Income-Tax-database content
- "I need to update my mobile number for KYC" — no mobile-number-update content
- "Why is my KYC showing as incomplete even after Aadhaar update?" — no UIDAI-linkage-pending content

**Loan**
- "Can I get an education loan for my child?" — closest analog (qa_QA019) is about overseas study loans; this ticket is specifically about domestic courses
- "My personal loan was approved but I haven't received the amount" — no disbursement-delay/NACH-mandate content
- "I want to foreclose my car loan" — see risk flag above
- "Interest rate on my loan seems higher than agreed" — no MCLR-reset/rate-variation content

## Recommendation

Rather than treating this purely as an eval-scope problem, it's worth
feeding back into the RAG data itself: add QA pairs or policy chunks for
the highest-frequency gaps above (NEFT delays, subscription/mandate
disputes, and KYC mobile-number updates look like they'd recur often in
a real ticket volume). Re-run `sample_gold_candidates.py` afterward to see
if coverage improves.


## Intent Classifier — Account Access Template Widening (2026-07-16)

**Finding:** The "Account Access" eval samples (n=10) were not all genuine
access/login issues. Of the 10, only 5 were actually about logging in,
passwords, PINs, or OTP (e.g. "unable to login to net banking", "forgot
my ATM PIN"). The other 5 were general account-servicing queries that
happened to be bucketed under the same ground-truth category label:
salary credit delay, NEFT transfer delay, adding a joint account holder,
a returned cheque, and account closure with pending charges. None of
these describe an access/login problem in any normal reading of the
phrase — the ground-truth category is broader/vaguer than its name
suggests, not the model failing to generalize.

**Baseline** (narrow template — "unable to log in, forgotten password,
account locked, OTP verification issue or account access problem"):

```
Account Access    precision 0.50  recall 0.50  f1 0.50  (n=10)
Fraud/Unauthorized precision 0.94  recall 0.91  f1 0.92  (n=32)
KYC               precision 0.89  recall 0.80  f1 0.84  (n=10)
Loan              precision 0.88  recall 1.00  f1 0.93  (n=14)
Accuracy: 0.85 | Macro-F1: 0.80
```

**Change tried:** widened the Account Access hypothesis label to also
cover general account servicing ("...or a general account servicing
request such as balance, credits, transfers, or account maintenance"),
on the hypothesis that this might catch some of the 5 mislabeled queries
without hurting the other three categories. Pre-registered revert
criteria: keep only if Account Access improved AND Fraud F1 stayed ≥0.90
AND Loan F1 stayed ≥0.90 AND KYC F1 stayed ≥0.80.

**Result** — kept, all criteria passed:

```
Account Access    precision 0.83  recall 0.50  f1 0.62  (n=10)
Fraud/Unauthorized precision 0.94  recall 0.97  f1 0.95  (n=32)
KYC               precision 0.90  recall 0.90  f1 0.90  (n=10)
Loan              precision 0.82  recall 1.00  f1 0.90  (n=14)
Accuracy: 0.89 | Macro-F1: 0.85
```

**What actually improved, precisely:** Account Access's *recall* stayed
flat at 0.50 — the same 5 queries (salary credit, NEFT delay, joint
holder, cheque return, account closure) are still misrouted, and
reasonably so, since none of them describe account servicing in a way
the widened template obviously captures either. What improved is
*precision* on Account Access (0.50 → 0.83) plus recall gains on
Fraud/Unauthorized (0.91 → 0.97) and KYC (0.80 → 0.90) — the widened
template stopped incorrectly absorbing borderline Fraud/KYC queries into
Account Access, which is why those two categories' recall jumped. The
net effect is a genuine overall improvement (accuracy 0.85 → 0.89,
macro-F1 0.80 → 0.85), but it is not "Account Access's blind spot got
fixed" — it's "Account Access stopped stealing traffic that belonged to
other categories." The 5 originally-miscategorized queries remain
unsolved and are a ground-truth labeling-granularity issue, not
something a template wording change can reasonably resolve.

**Not pursued further:** attempting to also catch the remaining 5 would
require broadening the template enough that it risks re-absorbing
Fraud/KYC traffic incorrectly — a real regression risk for a marginal
gain on a category whose true definition ("access") doesn't naturally
include salary credits or cheque returns. Documented here as a known,
accepted limitation rather than force-fit into the taxonomy.