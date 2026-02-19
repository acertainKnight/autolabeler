# Annotation Task: Trade Policy Uncertainty (TPU)

**Goal:** Determine if a news article snippet discusses **uncertainty** related to **trade policy**.

### 1. The Concept: What is Trade Policy Uncertainty?

We are looking for articles that express doubt, risk, lack of clarity, or unpredictability regarding the rules of international trade. It is not enough for an article to mention "trade" or "uncertainty" separately; they must be linked.

**Trade Policy Uncertainty exists when:**

* There is a discussion of **future** trade rules that are not yet settled (e.g., "threat of new tariffs," "stalled trade talks").
* Economic actors (businesses, governments, investors) express **worry, fear, or hesitation** because they do not know what trade policies will be in place soon.
* There is speculation about the potential success or failure of trade agreements (e.g., NAFTA, WTO, Brexit trade deals).

### 2. Labeling Instructions

Please read the provided text snippet and assign one of the following labels:

* **LABEL: 1 (Positive)** – The text explicitly discusses uncertainty, risk, or unpredictability regarding trade policy.
* **LABEL: 0 (Negative)** – The text DOES NOT discuss trade policy uncertainty.

**Important Nuance for Label "0" (Negative):**
Assign **0** if:

* The text discusses Trade Policy *as a settled fact* (e.g., "The government collected $5B in tariffs last year" — this is certain, not uncertain).
* The text discusses Uncertainty *unrelated to trade* (e.g., "Stock markets are volatile due to interest rate fears" — no trade connection).
* The text discusses Trade, but not policy rules (e.g., "Exports of soybeans rose by 5%" — this is trade volume, not policy uncertainty).

### 3. Examples

| Text Snippet | Label | Reasoning |
| --- | --- | --- |
| "Investors are sitting on the sidelines, **fearing** that the administration might impose **new tariffs** on imported steel next month." | **1** | Connects a specific trade policy (tariffs) with an uncertainty sentiment (fearing, might impose). |
| "The **outcome** of the ongoing **trade negotiations** remains **unclear**, casting a shadow over the manufacturing sector's outlook." | **1** | Discusses the lack of clarity (unclear) regarding a trade policy event (negotiations). |
| "The United States officially **signed** the new **trade agreement** yesterday, setting the tax rate at 5% for all signatories." | **0** | This is a trade policy, but it is a settled event. There is no uncertainty described here. |
| "Global markets faced high **volatility** and **risk** this week following the release of the unexpected inflation report." | **0** | Mentions "risk" and "volatility" (uncertainty), but the cause is inflation, not trade policy. |
| "A shipment of 500 cars was delayed at the border due to a logistical error in the paperwork." | **0** | This is a trade friction, but it is operational/logistical, not a policy uncertainty. |

### 4. Annotation Checklist

Before marking **1 (Positive)**, ask yourself:

1. Is a **Trade Policy** mentioned? (Tariffs, import duties, trade wars, trade agreements, WTO, dumping, etc.)
2. Is there an expression of **Uncertainty**? (Risk, threat, worry, possibility, unclear, concern, volatile, etc.)
3. Are they **causally linked**? (Is the uncertainty *about* the trade policy?)