# Calibration Examples for Fed Headlines

Study these examples carefully before labeling. They represent the gold standard for classification.

---

## EXPLICITLY DOVISH (-2)

**"*EVANS: NEED HIGHLY ACCOMMODATIVE MONETARY POLICY FOR SOME TIME"**
→ **-2**: Direct call for easing, specific commitment to accommodation

**"*BULLARD: POLICY TO BE EXTREMELY ACCOMMODATIVE FOR 2-3 YEARS"**
→ **-2**: Specific timeline for easing stance

**"*FED EXTENDS EMERGENCY LENDING PROGRAMS THROUGH DEC. 31"**
→ **-2**: Direct easing action with specific date

**"*EVANS REPEATS HE FAVORS FED WAITING UNTIL 2016 TO RAISE RATES"**
→ **-2**: Specific timeline against tightening, explicit dovish stance

---

## IMPLICITLY DOVISH (-1)

**"*BULLARD SEES RISK OF SHARPER-THAN-EXPECTED ECONOMIC SLOWDOWN"**
→ **-1**: Concern about downside risk, implies need for supportive policy

**"*GEORGE: RAISING RATES LATER, FASTER RISKS DISRUPTING MARKETS"**
→ **-1**: Concern about overtightening

**"*BARKIN: EXPECT HIGH RATES WILL EVENTUALLY SLOW ECONOMY FURTHER"**
→ **-1**: Forecast of deterioration suggesting dovish lean

**"*ECONOMY REMAINS QUITE WEAK"**
→ **-1**: Intensity modifier ("quite") + weakness implies need for support

**"*A LOT OF SLACK IN THE SYSTEM"**
→ **-1**: Intensity ("a lot of") + slack condition

**"*EXPECT PERIOD OF BELOW-POTENTIAL GROWTH"**
→ **-1**: Forecast of downturn, not just continuation

**"*GRADUAL APPROACH TO RATE CUTS"**
→ **-1**: DIRECTION = cutting (dovish), pace irrelevant

---

## NEUTRAL (0)

**"*POWELL: FOMC DECISIONS WILL DEPEND ON TOTALITY OF DATA"**
→ **0**: Data dependence, no directional bias

**"*POWELL: CURRENT SPENDING, GDP DATA SHOWING STRONG ECONOMY"**
→ **0**: Plain "strong" without intensity modifier

**"*WILLIAMS: ECONOMY IS GOOD, CONSUMER SPENDING SOLID"**
→ **0**: Plain description, no intensity/surprise

**"*ROSENGREN: WE'RE IN A PRETTY GOOD SPOT RIGHT NOW"**
→ **0**: Satisfaction with current policy

**"*GOOLSBEE: FED ON GOLDEN PATH; EFFORT WORKING"**
→ **0**: Policy working as intended, no change signal

**"*POWELL: FED IS ON A PATH TO MAKE MORE PROGRESS ON INFLATION"**
→ **0**: Continuation forecast, not change

**"*OUR POLICY IS VERY ACCOMMODATIVE"**
→ **0**: Describing current stance, not advocating for it

**"*TIGHTER FINANCIAL CONDITIONS STARTING TO HAVE EFFECT"**
→ **0**: Observation that policy transmission is working

**"*BERNANKE: U.S. ECONOMY LIKELY TO EXPAND IN 2010"**
→ **0**: Continuation forecast

**"*HARKER: GDP GROWTH THIS YEAR BIT ABOVE 2%"**
→ **0**: Near-target growth, no signal for change

**"*SOFTER INFLATION READINGS, SOFTER JOBS REPORT"**
→ **0**: Plain reporting without intensity

**"*YELLEN: NO MEETING IS COMPLETELY OFF TABLE"**
→ **0**: Maintains optionality, no directional bias

**"*BERNANKE: PACE OF BOND PURCHASES NOT ON A PRESET COURSE"**
→ **0**: Data dependence, flexible

**"*DALY: WE HAVE TOOLS TO DEAL WITH FINANCIAL DISLOCATION"**
→ **0**: Capability statement, not action

**"*FED MAKING SLOW GRADUAL PROGRESS TOWARD INFLATION GOAL"**
→ **0**: Continuation, "slow gradual" describes pace not direction

**"*PLOSSER: PAYROLL GROWTH SHOWS A CLEAR POSITIVE TREND"**
→ **0**: Structural observation, no intensity

**"*LACKER: EMPLOYMENT HAS GROWN ON A PRETTY STEADY TREND"**
→ **0**: Trend observation, no signal for rate change

---

## IMPLICITLY HAWKISH (1)

**"*WILLIAMS: LABOR MARKET IS INCREDIBLY STRONG"**
→ **1**: Intensity modifier ("incredibly") suggests rates may need to be higher

**"*WALLER: 3Q GDP OVER 4%, ECONOMY IS REALLY BOOMING"**
→ **1**: "Booming" = overheating, rate pressure upward

**"*POWELL: RECENT DATA SHOW LACK OF FURTHER PROGRESS ON INFLATION"**
→ **1**: Concern about insufficient progress, hawkish lean

**"*WALLER: BETTER PRICING POWER COULD MAKE INFLATION STICKIER"**
→ **1**: Risk/concern about inflation persistence

**"*BARKIN: FED WANTS TO BE 'DONE WITH INFLATION'"**
→ **1**: Resolve to continue tightening until goal achieved

**"*RATE-HIKE PATH SHOULD BE SHALLOW"**
→ **1**: DIRECTION = hiking (hawkish), pace irrelevant

**"*MORE MEASURED RATE-HIKE MOVES USEFUL"**
→ **1**: DIRECTION = hiking (hawkish), pace irrelevant

---

## EXPLICITLY HAWKISH (2)

**"*BULLARD: FED SHOULD MOVE AS RAPIDLY AS POSSIBLE ON RATE HIKES"**
→ **2**: Direct call for aggressive tightening

**"*MESTER: FAVORS RAISING RATES TO 2.5% BY END OF YEAR"**
→ **2**: Specific magnitude and timeline for hikes

**"*WALLER SUPPORTS HIKING BY 50 BP FOR 'SEVERAL' MEETINGS"**
→ **2**: Specific magnitude and commitment to multiple hikes

---

## NOT RELEVANT (-99)

**"*FED GOVERNOR KUGLER COMMENTS IN PREPARED TEXT"**
→ **-99**: Procedural announcement, no content

**"*POWELL SPEAKS IN INTERVIEW ON BLOOMBERG TV"**
→ **-99**: Event notice, no substantive statement

**"*BULLARD, GEORGE DISSENT FROM FOMC STATEMENT"**
→ **-99**: Dissent notice without substantive content

---

## EDGE CASES AND COMPARISONS

### Intensity Matters:

**"GDP DATA SHOWING STRONG ECONOMY"** → **0** (plain "strong")
vs.
**"LABOR MARKET IS INCREDIBLY STRONG"** → **1** (intensity "incredibly")

### Satisfaction vs Action:

**"FED ON GOLDEN PATH"** → **0** (satisfaction)
vs.
**"GOOD POSITION TO KEEP HIKING"** → **1** (satisfaction tied to continued action)

### Pace vs Direction:

**"RATE-HIKE PATH SHOULD BE SHALLOW"** → **1** (hiking = hawkish, regardless of pace)
vs.
**"FED MAKING SLOW PROGRESS"** → **0** (pace describes HOW, not WHETHER)

### Continuation vs Change:

**"ECONOMY LIKELY TO EXPAND"** → **0** (continuation)
vs.
**"EXPECT BELOW-POTENTIAL GROWTH"** → **-1** (downturn)

### Description vs Advocacy:

**"OUR POLICY IS VERY ACCOMMODATIVE"** → **0** (describing current stance)
vs.
**"NEED HIGHLY ACCOMMODATIVE POLICY"** → **-2** (advocating for it)
