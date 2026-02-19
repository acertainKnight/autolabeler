# Common Mistakes to Avoid

These are the most frequent labeling errors. Review them carefully.

---

## MISTAKE #1: Treating PLAIN economic descriptions as directional

**WRONG:** "GDP DATA SHOWING STRONG ECONOMY" → 1
**RIGHT:** → **0** (plain "strong", no intensity modifier)

**WRONG:** "PLOSSER: PAYROLL GROWTH SHOWS A CLEAR POSITIVE TREND" → 1
**RIGHT:** → **0** (reporting trend, no rate pressure signal)

vs. **CORRECT directional:** "LABOR MARKET IS INCREDIBLY STRONG" → 1 ✓

**ASK:** Is there an intensity word (incredibly, remarkably, booming, persistent, sticky, notably, sharply, "more than expected")? If not → 0.

---

## MISTAKE #2: Treating satisfaction as directional

**WRONG:** "FED ON GOLDEN PATH; EFFORT WORKING" → -1
**RIGHT:** → **0** (satisfied, no change signal)

**WRONG:** "WE'RE IN A PRETTY GOOD SPOT" → 1
**RIGHT:** → **0** (status quo satisfaction)

---

## MISTAKE #3: Labeling speaker reputation instead of headline text

**WRONG:** "*FISHER SPEAKS IN RADIO INTERVIEW" → 1 (Fisher = hawk)
**RIGHT:** → **-99** (announcement, zero content)

Always label ONLY what the headline says, not what you know about the speaker.

---

## MISTAKE #4: Confusing pace with direction

**WRONG:** "SHALLOW RATE-HIKE PATH" → -1 (shallow sounds dovish)
**RIGHT:** → **1** (HIKING = positive, pace irrelevant)

**WRONG:** "GRADUAL APPROACH TO CUTS" → 0 (gradual sounds neutral)
**RIGHT:** → **-1** (CUTTING = negative, pace irrelevant)

The direction of the action (hiking vs cutting) determines the label. How fast or slow is irrelevant.

---

## MISTAKE #5: Continuation forecasts as directional

**WRONG:** "ECONOMY LIKELY TO EXPAND" → 1
**RIGHT:** → **0** (continuation, not heating up)

vs. **CORRECT:** "EXPECT BELOW-POTENTIAL GROWTH" → -1 ✓ (downturn)

Ask: Is this predicting a CHANGE in trend, or continuation of current trajectory?

---

## MISTAKE #6: Treating "policy is working" as directional

**WRONG:** "POLICY IS HAVING INTENDED EFFECT" → -1
**RIGHT:** → **0** (satisfied that policy transmission is working)

**WRONG:** "SEEING PROGRESS ON INFLATION" → -1
**RIGHT:** → **0** (on track, no change needed)

---

## MISTAKE #7: Missing negations that flip meaning

**WRONG:** "RATES ARE NOT RESTRICTIVE ENOUGH" → 0
**RIGHT:** → **1** (negation creates hawkish signal)

**WRONG:** "NOT WORRIED ABOUT INFLATION" → -1
**RIGHT:** → **0** (negation neutralizes concern)

Always parse the full sentence. Negation can either neutralize or create signal.

---

## MISTAKE #8: Treating "slowing the pace" as dovish

**WRONG:** "SLOWING PACE OF HIKES" → -1
**RIGHT:** → **0** (still hiking, just more gradually — pace not direction)

**WRONG:** "FEWER HIKES AHEAD" → -1
**RIGHT:** → **0** (still hiking, just fewer times)

---

## MISTAKE #9: Forcing directional labels on ambiguous headlines

**WRONG:** Seeing "ECONOMY CONTINUES TO GROW" and feeling pressure to choose 1 or -1
**RIGHT:** → **0** (if there's no clear signal, default to neutral)

Many headlines are genuinely neutral. Don't force a directional interpretation.

---

## MISTAKE #10: Treating "data dependent" as hawkish or dovish

**WRONG:** "WILL BE GUIDED BY DATA" → 1 or -1
**RIGHT:** → **0** (wait-and-see, no bias)

Data dependence means "we'll decide later" — not a signal for any specific direction.

---

## CRITICAL REMINDERS

1. **Intensity matters:** "strong" (neutral) vs "incredibly strong" (hawkish)
2. **Satisfaction = neutral:** "policy working" means no change needed
3. **Pace ≠ direction:** "gradual hikes" is still hawkish, "gradual cuts" is still dovish
4. **Continuation = neutral:** "economy will expand" vs "economy will boom"
5. **Default to 0:** When in doubt, choose neutral
6. **Most headlines are neutral:** Don't over-interpret
7. **Parse negations:** "not worried" (neutral) vs "not restrictive enough" (hawkish)
8. **Label the text, not the speaker:** Ignore reputation
9. **Direction > everything:** The action (hiking/cutting) matters, not the speed
10. **45-55% should be neutral:** If you're labeling <30% neutral, you're likely over-interpreting
