# Common TPU Labeling Mistakes

Avoid these frequent errors.

---

## MISTAKE #1: Confusing trade volume with trade policy

**WRONG:** "Exports rose 5%" → 1
**RIGHT:** → **0** (trade volume statistics, not policy uncertainty)

**Trade volume** (imports, exports, trade balance, shipments) ≠ **Trade policy** (tariffs, agreements, regulations)

Only label 1 if the article discusses uncertainty about POLICY RULES, not trade flows.

---

## MISTAKE #2: Labeling implemented policies as automatically uncertain

**WRONG:** "Tariffs collected $5B" → 1 (assumes tariffs = uncertainty)
**RIGHT:** → **0** (past statistics, no future uncertainty mentioned)

Implementation status alone doesn't create uncertainty. Ask: Is there debate, negotiation, or uncertainty about the FUTURE of this policy?

---

## MISTAKE #3: Missing the causal link requirement

**WRONG:** Article mentions "trade negotiations" AND "market volatility" separately → 1
**RIGHT:** → **0** (if volatility is caused by something OTHER than trade policy)

Both trade policy AND uncertainty must be present AND linked. If they're about different things, label 0.

---

## MISTAKE #4: Ignoring RISK language

**WRONG:** "Companies face trade war risks" → 0 (not explicit enough)
**RIGHT:** → **1** (RISK IS UNCERTAINTY — this is a clear TPU signal)

Any mention of:
- "tariff risks"
- "trade policy risks"
- "trade war risks"
- "risks from potential tariffs"

...is almost certainly Label 1. Risk language is a PRIMARY indicator.

---

## MISTAKE #5: Missing meetings/discussions as uncertainty signals

**WRONG:** "Officials meeting to discuss tariffs" → 0 (just a meeting)
**RIGHT:** → **1** (meeting implies policy not settled, potential for change)

If there's a meeting, talk, discussion, or summit about trade policy → likely Label 1. Why meet if policy is settled?

---

## MISTAKE #6: Treating operational issues as policy uncertainty

**WRONG:** "Shipment delayed at border due to paperwork error" → 1
**RIGHT:** → **0** (operational/logistical, not policy uncertainty)

Delays, errors, friction in trade execution ≠ uncertainty about trade policy rules.

---

## MISTAKE #7: Labeling past settled events

**WRONG:** "Agreement signed last year" → 1 (it's a trade agreement)
**RIGHT:** → **0** (completed action, no uncertainty about future mentioned)

Unless the article mentions debate/renegotiation/questions about a signed agreement, it's settled.

---

## MISTAKE #8: Missing "implemented BUT contested" cases

**WRONG:** "Tariffs in effect but under review by Congress" → 0 (implemented = settled)
**RIGHT:** → **1** (under review = uncertainty about future even if currently implemented)

Implementation + ongoing debate/negotiation/review = STILL creates uncertainty.

---

## MISTAKE #9: Confusing certainty about bad outcomes with uncertainty

**WRONG:** "Trade war WILL damage economy" → 1 (trade policy + bad outcome)
**RIGHT:** → **0** (certain prediction of damage, not uncertainty about policy)

If the article is CERTAIN about an outcome (even a bad one), that's not uncertainty. Look for doubt, risk, possibility, unclear outcomes.

---

## MISTAKE #10: Over-interpreting vague mentions

**WRONG:** "International business environment remains challenging" → 1
**RIGHT:** → **0** (vague, no specific trade policy or clear uncertainty link)

Be specific. Need explicit trade policy + explicit uncertainty language + causal link.

---

## CRITICAL REMINDERS

1. **RISK = UNCERTAINTY**: "tariff risk", "trade policy risk" → almost always Label 1
2. **Meetings signal uncertainty**: Discussions/talks about trade policy → likely Label 1
3. **Three criteria ALL required**: Trade policy + Uncertainty + Causal link
4. **Implemented ≠ settled**: If policy is being debated/renegotiated → Label 1
5. **Past statistics without future uncertainty** → Label 0
6. **Trade volume ≠ trade policy**: Imports/exports stats → Label 0
7. **Operational issues ≠ policy uncertainty**: Border delays, paperwork errors → Label 0
8. **Causal link matters**: Trade + uncertainty in same article but different topics → Label 0
9. **Settled agreements without debate** → Label 0
10. **When in doubt about the link, label 0**: Uncertainty must be ABOUT the trade policy
