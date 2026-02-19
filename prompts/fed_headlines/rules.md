# Classification Rules for Fed Headlines

## THE SCALE

- **-2 EXPLICITLY DOVISH**: Speaker directly advocates for easing. Calls for rate cuts, more accommodation, extends emergency programs, or commits to specific easing actions with timelines.

- **-1 IMPLICITLY DOVISH**: Speaker describes conditions that imply downward pressure on rates — economic weakness with intensity/surprise, rising slack, cooling inflation below target, downside risks, financial stress. Or expresses concern about overtightening.

- **0 NEUTRAL**: No meaningful directional signal. Includes:
  - Plain economic descriptions without intensity/surprise modifiers
  - Satisfaction with current policy ("on track", "working", "good place")
  - Data dependence and patience
  - Continuation forecasts (not predicting heating up or downturn)
  - Descriptions of current policy stance without advocacy
  - Balanced risk assessments
  - Negated concerns that cancel to no signal
  - **THIS IS THE DEFAULT. When in doubt, label 0.**

- **+1 IMPLICITLY HAWKISH**: Speaker describes conditions that imply upward pressure on rates — economic overheating, tight labor markets with intensity/surprise, persistent/sticky inflation, upside risks, financial froth. Or expresses concern that policy isn't restrictive enough.

- **+2 EXPLICITLY HAWKISH**: Speaker directly advocates for tightening. Calls for rate hikes, faster normalization, or commits to specific tightening actions with magnitude/timeline.

- **-99 NOT RELEVANT**: Procedural announcements, non-monetary-policy topics.

---

## RULE 1: THE INTENSITY/SURPRISE TEST FOR ECONOMIC OBSERVATIONS

Fed speakers constantly describe the economy. The question is whether their description carries SIGNAL about rates.

**An economic observation is DIRECTIONAL (±1) if it describes conditions with INTENSITY, SURPRISE, or IMPLICATION that creates pressure on the rate path.**

**An economic observation is NEUTRAL (0) if it is a plain description without intensity/surprise modifiers — just reporting how things are.**

### Directional Modifiers (trigger ±1):
incredibly, remarkably, surprisingly, notably, significantly, sharply, dramatically, strikingly, unexpectedly, booming, overheated, persistent, sticky, stubborn, "a lot of", "more than expected", "stronger/weaker than anticipated", deteriorating, "lack of progress"

### Neutral Descriptions (stay at 0):
strong, solid, good, healthy, weak, slow, soft — **WITHOUT modifiers above**
"steady trend", "gradually", "continues to"

### Examples:
- "LABOR MARKET IS INCREDIBLY STRONG" → **1** (intensity modifier)
- "GDP DATA SHOWING STRONG ECONOMY" → **0** (plain "strong")
- "ECONOMY IS REALLY BOOMING" → **1** ("booming" = overheating)
- "ECONOMY REMAINS QUITE WEAK" → **-1** (intensity "quite weak")
- "SOFTER INFLATION READINGS, SOFTER JOBS REPORT" → **0** (plain reporting)
- "A LOT OF SLACK IN THE SYSTEM" → **-1** (intensity "a lot of")
- "DOWNWARD TREND IN LABOR PARTICIPATION" → **0** (structural observation)

**THE KEY QUESTION:** Does the modifier or framing imply the CURRENT RATE PATH may be insufficient? If yes → ±1. If just description → 0.

---

## RULE 2: CONCERN = DIRECTIONAL

When a speaker expresses CONCERN, WORRY, or RISK about conditions, that IS directional — it signals the speaker sees a problem the current policy path may not be addressing.

### Examples:
- "CONCERNED ABOUT OVERTIGHTENING" → **-1**
- "RISKS TILTED TO THE DOWNSIDE" → **-1**
- "WORRIED ABOUT INFLATION PERSISTENCE" → **1**
- "NOT CONVINCED WE'VE DONE ENOUGH" → **1**
- "RISK OF SHARPER-THAN-EXPECTED SLOWDOWN" → **-1**
- "INFLATION COULD PROVE STICKIER" → **1**
- "'ULTRA-EASY MONETARY POLICY' RISKS INFLATION" → **1**
- "LACK OF FURTHER PROGRESS ON INFLATION" → **1**

### Balanced Concern is Neutral:
- "RISKS ARE ROUGHLY BALANCED" → **0**
- "BOTH UPSIDE AND DOWNSIDE RISKS" → **0**

---

## RULE 3: SATISFACTION / "GOOD PLACE" = NEUTRAL

Satisfaction that current policy is working = ALWAYS NEUTRAL, unless explicitly tied to continued directional action.

### Examples:
- "WE'RE IN A PRETTY GOOD SPOT" → **0**
- "FED ON GOLDEN PATH; EFFORT WORKING" → **0**
- "MAKING REAL PROGRESS ON INFLATION" → **0**
- "FINANCIAL CONDITIONS HAVING INTENDED EFFECT" → **0**
- "OUR POLICY IS VERY ACCOMMODATIVE" → **0** (describing, not advocating)

### BUT tied to action:
- "GOOD POSITION TO KEEP HIKING" → **1**

---

## RULE 4: PACE IS IRRELEVANT — ONLY DIRECTION MATTERS

Ignore pace. Label based on DIRECTION of the action only.

### Talking about HIKING (any pace) → positive:
- "RATE-HIKE PATH SHOULD BE SHALLOW" → **1**
- "MEASURED RATE-HIKE MOVES USEFUL" → **1**
- "GRADUAL RATE INCREASES AHEAD" → **1**
- "FAVOR A 50BP HIKE" → **2**

### Talking about CUTTING (any pace) → negative:
- "GRADUAL APPROACH TO RATE CUTS" → **-1**
- "METHODICAL AND CAREFUL CUTS" → **-1**
- "FAVOR A 50BP CUT" → **-2**

### Ambiguous without direction → 0:
- "FED MAKING SLOW GRADUAL PROGRESS" → **0**

---

## RULE 5: FORECASTS — HEATING/COOLING vs CONTINUATION

### Continuation of current trends → 0:
- "ECONOMY LIKELY TO EXPAND" → **0**
- "GDP GROWTH BIT ABOVE 2%" → **0**
- "EXPECT INFLATION TO GRADUALLY RETURN TO TARGET" → **0**

### Heating up or overheating → 1:
- "GDP OVER 4%, ECONOMY IS REALLY BOOMING" → **1**
- "GROWTH COULD ACCELERATE BEYOND EXPECTATIONS" → **1**

### Downturn or deterioration → -1:
- "EXPECT PERIOD OF BELOW-POTENTIAL GROWTH" → **-1**
- "RECESSION RISK HAS INCREASED" → **-1**

---

## RULE 6: DATA DEPENDENCE AND PATIENCE = NEUTRAL

- "WILL BE GUIDED BY THE DATA" → **0**
- "NO MEETING IS OFF THE TABLE" → **0**
- "PATIENT APPROACH" → **0**
- "DECISIONS DEPEND ON TOTALITY OF DATA" → **0**

---

## RULE 7: NEGATION — PARSE THE FULL SENTENCE

Negation commonly NEUTRALIZES a statement:
- "NOT WORRIED ABOUT INFLATION" → **0**
- "DON'T SEE NEED FOR RATE CUT" → **0**
- "SHOULDN'T RUSH TO RAISE RATES" → **0**

BUT negation can CREATE signal:
- "RATES ARE NOT RESTRICTIVE ENOUGH" → **1**
- "NOT SEEING THE PROGRESS WE NEED" → **1**

---

## RULE 8: SPEAKER INDEPENDENCE

Label ONLY what the headline text says. DO NOT use knowledge of the speaker's historical views. A hawk can make a dovish statement.

---

## RULE 9: NOT RELEVANT (-99)

- Procedural (speech/event/interview notices)
- Non-monetary-policy topics
- Pure data releases without analytical commentary
- Dissent notices without substantive content

---

## RULE 10: RESOLVING AMBIGUITY

- When in doubt between 0 and ±1: **choose 0**
- When in doubt between ±1 and ±2: **choose ±1**
- Mixed hawk/dove signals: label the NET, or **0 if unclear**

---

## CRITICAL PATTERNS TO WATCH FOR

### 1. NEGATIONS
Pay extreme attention to negation words that flip the meaning:
- "TOO FAR ABOVE neutral" = **DOVISH** (rates are too high, need to come down)
- "NOT YET neutral" = context-dependent
- "NOT WORRIED about inflation" = **NEUTRAL**
- "rates are NOT restrictive" = **NEUTRAL**

### 2. CONTEXTUAL PHRASES about neutral rate:
- "getting to neutral" in LOW rate environment = **HAWKISH** (need to raise)
- "getting to neutral" in HIGH rate environment = **DOVISH** (need to cut)
- "above neutral" = typically **DOVISH** (rates too high)
- "below neutral" = typically **HAWKISH** (rates too low)
- **IF DIRECTION IS AMBIGUOUS AND ECONOMIC CONTEXT IS MISSING, MARK AS NEUTRAL**

### 3. COMPARATIVE LANGUAGE:
- "higher than needed" = **DOVISH**
- "lower than needed" = **HAWKISH**
- "more restrictive than necessary" = **DOVISH**
- "insufficiently restrictive" = **HAWKISH**

---

## FED-SPECIFIC EXPERTISE

### A) Inflation Context:
- "inflation at 2%" or "near target" = mission accomplished (**DOVISH** lean)
- "inflation well above 2%" = concern, need restrictive policy (**HAWKISH**)
- "core vs headline" - core inflation matters more for policy
- "inflation expectations anchored" = **NEUTRAL/DOVISH** (no urgency)
- "inflation expectations rising" = **HAWKISH** (credibility at stake)

### B) Employment/Labor Market:
- "unemployment rising" = **DOVISH** (dual mandate concern)
- "tight labor market" or "wage pressures" = **HAWKISH** (inflation risk)
- "labor market in balance" = **NEUTRAL**

### C) Policy Stance & Path:
- "sufficiently restrictive" = rates are high enough (**DOVISH** lean - no more hikes)
- "not yet restrictive enough" = **HAWKISH** (more tightening needed)
- "expeditiously" or "quickly" = urgency (**HAWKISH** if raising, **DOVISH** if cutting)
- "gradual" or "patient" = slower pace (less urgent)
- "data-dependent" alone = typically **NEUTRAL** (waiting for more info)
- "risks are balanced" = **NEUTRAL** (no urgency either direction)

### D) Forward Guidance:
- "likely to raise rates" = **EXPLICITLY HAWKISH** (pre-commitment)
- "expect rates to remain elevated" = **HAWKISH** (higher for longer)
- "rate cuts could come soon" = **EXPLICITLY DOVISH** (signaling ease)
- "premature to discuss cuts" = **HAWKISH** (don't expect ease)

### E) Financial Conditions:
- "financial conditions have tightened" = can substitute for rate hikes (**NEUTRAL/DOVISH**)
- "financial conditions too loose" = **HAWKISH** (need policy action)

---

## LABEL CATEGORIES DETAIL

### EXPLICITLY DOVISH (-2):
- Direct calls for/commitment to easing
- Examples: "should cut rates", "will ease policy", "need stimulus"
- Extreme risk language implying forceful response: "depression", "crisis", "horrible", "whatever it takes"
- High certainty about cuts: discussing magnitude/number of cuts, declaring specific meeting "live" for cuts
- Current policy "too tight", "restrictive", specific timeline against tightening

### IMPLICITLY DOVISH (-1):
- Economic weakness suggesting need for cuts: slowing growth/inflation, weakening labor market, downside risks
- Forward guidance suggesting CUTS may be appropriate: "cuts possible if data confirms"
- Inflation "is falling", "progress made", "moving toward target" COMBINED with concern about overtightening
- Discussing fiscal stimulus/support as necessary

**CRITICAL:** Statements about rates being restrictive must be evaluated carefully:
- "Rates are restrictive" + "policy is working" = **NEUTRAL (0)** - mission accomplished
- "Rates are TOO restrictive" or "overtightening risk" = **DOVISH (-1/-2)** - need to cut
- "Rates are appropriately restrictive" = **NEUTRAL (0)** - just right
- "Rates are not yet restrictive enough" = **HAWKISH (1/2)** - need more hikes

**CRITICAL:** "Patient approach" context matters:
- "Patient about WHEN to cut" = **DOVISH (-1)** - assumes cuts coming
- "Patient approach" (no direction specified) = **NEUTRAL (0)** - wait-and-see
- "Patient about WHEN to hike" = **HAWKISH (1)** - assumes hikes coming

### NEUTRAL (0):
- No directional signal, balanced risks, "wait and see", data-dependent without bias
- Economy "doing fine", no action needed
- Maintaining current stance, rejecting policy changes without calling for opposite action
- Dismissing pressures as "transitory"/"contained"
- Any statement not clearly fitting dovish or hawkish categories
- Discussion of INACTION or STATUS QUO from either direction
- Saying current policy is "appropriate" or "suitable"
- Headlines that are ambiguous, vague, or could be interpreted multiple ways
- Statements about "slowing the pace", "moderating", "gradual approach" - these describe PACE not DIRECTION
- "Policy is working", "seeing progress", "on track"
- "Will assess", "evaluate data", "monitor conditions"
- Any discussion of HOW policy will proceed (pace, timing, sequencing) without indicating WHETHER it will continue

### IMPLICITLY HAWKISH (1):
- Economic strength suggesting need for tightening: rising inflation, tight labor markets, strong growth, upside risks
- Inflation "remains above target", "more work to do", warning against "falling behind curve"
- Economy strong enough for restrictive policy
- Discussing fiscal austerity/tightening as beneficial

**CRITICAL:** PACE vs DIRECTION:
- "Slowing pace of hikes" = **NEUTRAL (0)** - Still hiking (tightening), just more gradually
- "Fewer hikes ahead" = **NEUTRAL (0)** - Still hiking, just fewer times
- "May pause to assess effects" = **NEUTRAL (0)** - Pausing to evaluate
- "Taking time to see data" = **NEUTRAL (0)** - Wait-and-see approach
- "Patient approach" = **NEUTRAL (0)** - Maintaining current stance

**KEY DISTINCTION:**
- "Will slow hikes" (**NEUTRAL**) vs "Should stop hiking" (**DOVISH**) vs "Should start cutting" (**EXPLICITLY DOVISH**)
- Discussing the PACE of ongoing policy ≠ signaling a CHANGE in policy direction

### EXPLICITLY HAWKISH (2):
- Direct calls for/commitment to tightening: "should raise rates", "will tighten policy"
- Discussing specific rate hike path/pace/magnitude, achieving/maintaining "restrictive stance"
- Current policy "not restrictive enough", specific timeline against easing
- Extreme risk language implying forceful tightening needed: "overheating", "runaway inflation"

---

## REMEMBER

- Economic observation (-1/1) vs policy prescription (-2/2): "inflation is falling" (-1) vs "we should cut rates" (-2)
- Possibility (1/-1) vs commitment (2/-2): "cuts possible if..." (-1) vs "we will cut rates" (-2)
- General resolve (1/-1) vs specific action (2/-2): "committed to goals" (1) vs "will take steps to raise rates" (2)
- Suggesting patience/status quo from either direction = **Neutral (0)**
- Discussion about strength/weakness of industries or sectors is implicitly hawkish/dovish based on context
- Normalizing policy could be hawkish or dovish depending on context (low rates = hawkish, high rates = dovish)
- Comments that every meeting is "live" indicate implicit hawkishness or dovishness depending on context
