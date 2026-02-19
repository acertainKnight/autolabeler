# TPU Classification Examples

Study these examples to understand the classification boundaries.

---

## POSITIVE EXAMPLES (Label 1 - Trade Policy Uncertainty EXISTS)

### Example 1: Future Policy Uncertainty
**Text:** "Investors are sitting on the sidelines, fearing that the administration might impose new tariffs on imported steel next month."

**Label:** 1

**Reasoning:** Connects a specific trade policy (tariffs) with uncertainty sentiment (fearing, might impose). Future policy not yet settled.

---

### Example 2: Negotiation Uncertainty
**Text:** "The outcome of the ongoing trade negotiations remains unclear, casting a shadow over the manufacturing sector's outlook."

**Label:** 1

**Reasoning:** Discusses lack of clarity (unclear) regarding a trade policy event (negotiations). Uncertainty about what rules will emerge.

---

### Example 3: Risk Language
**Text:** "Companies face tariff risks as trade negotiations between the US and China remain stalled."

**Label:** 1

**Reasoning:** "Tariff risks" is direct uncertainty language. RISK = UNCERTAINTY. Trade policy + risk language = certain Label 1.

---

### Example 4: Meeting/Discussion Signal
**Text:** "Trade ministers from both countries will meet next week to discuss potential changes to existing tariff agreements."

**Label:** 1

**Reasoning:** Meetings/discussions about trade policy imply the policy is not settled and could change, creating uncertainty.

---

### Example 5: Implemented BUT Under Debate
**Text:** "The steel tariffs implemented last year remain in effect, but manufacturers face uncertainty as lawmakers debate whether to repeal or modify them before the next election."

**Label:** 1

**Reasoning:** Even though tariffs are implemented, ongoing debate creates uncertainty about the FUTURE of the policy. Implementation status ≠ settled.

---

### Example 6: Speculation About Outcomes
**Text:** "Brexit trade deal talks at risk of collapse, with businesses hesitant to invest amid the uncertainty."

**Label:** 1

**Reasoning:** Speculation about success/failure of trade agreement + explicit "uncertainty" + business hesitation = clear TPU signal.

---

## NEGATIVE EXAMPLES (Label 0 - NO Trade Policy Uncertainty)

### Example 1: Settled Policy (Past Statistics)
**Text:** "The government collected $8.2 billion in tariff revenue last quarter, up 15% from the previous year."

**Label:** 0

**Reasoning:** Just reporting statistics on implemented policy. No mention of debate, negotiation, or uncertainty about future.

---

### Example 2: Uncertainty Unrelated to Trade
**Text:** "Global markets faced high volatility and risk this week following the release of the unexpected inflation report."

**Label:** 0

**Reasoning:** Mentions "risk" and "volatility" (uncertainty), but cause is inflation, NOT trade policy. Not causally linked.

---

### Example 3: Trade Volume (Not Policy)
**Text:** "Exports of soybeans rose by 5% last quarter, reaching a record high."

**Label:** 0

**Reasoning:** This is trade volume/activity, not trade policy rules. No policy uncertainty discussed.

---

### Example 4: Signed and Settled Agreement
**Text:** "The United States officially signed the new trade agreement yesterday, setting the tax rate at 5% for all signatories."

**Label:** 0

**Reasoning:** Trade policy mentioned, but it's a completed/settled event with no indication of future uncertainty.

---

### Example 5: Operational/Logistical Issue
**Text:** "A shipment of 500 cars was delayed at the border due to a logistical error in the paperwork."

**Label:** 0

**Reasoning:** Trade friction, but operational/logistical, not policy uncertainty. The rules are clear; execution failed.

---

### Example 6: Companies Adapted (No Future Uncertainty)
**Text:** "Companies have fully adapted to the new tariff regime implemented two years ago."

**Label:** 0

**Reasoning:** Past implementation with no mention of future changes, debates, or uncertainty. Policy is treated as stable.

---

## EDGE CASES AND KEY DISTINCTIONS

### Edge Case 1: "Tariff Revenue Collected" vs "Tariff Risks"

**"Government collected $5B in tariffs"** → **0** (statistics, settled)
vs.
**"Companies face tariff risks"** → **1** (risk language = uncertainty)

### Edge Case 2: Implemented + Stable vs Implemented + Contested

**"Tariffs collected, companies adapted"** → **0** (implemented and stable)
vs.
**"Tariffs in effect, but under congressional review"** → **1** (implemented but contested)

### Edge Case 3: Trade Activity vs Trade Policy

**"Trade deficit widened in Q3"** → **0** (trade statistics, not policy)
vs.
**"Trade deficit concerns prompt tariff discussions"** → **1** (policy uncertainty triggered by concerns)

### Edge Case 4: Meetings Signal Uncertainty

**"Leaders met to discuss trade"** → **1** (meeting implies policy not settled)
vs.
**"Leaders signed trade agreement"** → **0** (completed action, settled)

---

## THREE-CRITERIA CHECKLIST

Before marking Label 1, verify ALL THREE:

✓ **CRITERION 1:** Is a TRADE POLICY mentioned?
  - Trade policies: tariffs, import duties, export restrictions, trade agreements, trade wars, anti-dumping, quotas, trade negotiations, customs rules, sanctions, preferential trade terms

✓ **CRITERION 2:** Is there UNCERTAINTY language?
  - Risk language: risk, risks, risky, at risk, tariff risk, trade war risk
  - Emotion: worry, fear, concern, anxiety, nervous, hesitation
  - Possibility: might, could, may, possible, potentially, likelihood
  - Clarity: unclear, ambiguous, unknown, unpredictable, uncertain
  - Action: waiting to see, holding off, delaying, paused
  - Volatility: volatile, unstable
  - Speculation: speculation, debate, question, doubt
  - Meetings: meeting, talks, discussion, negotiate, review

✓ **CRITERION 3:** Are they CAUSALLY LINKED?
  - Is the uncertainty ABOUT or CAUSED BY the trade policy?
  - If trade + uncertainty are mentioned but about DIFFERENT things → Label 0
