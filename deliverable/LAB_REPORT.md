# Lab 7 Report: AutoGen vs CrewAI

## Setup Notes

This lab ran successfully with Groq. Because the AutoGen dependency range in this repository is not compatible with Python 3.13, I used Python 3.10 for the final runs.

Example `.env`:

```bash
GROQ_API_KEY=your-groq-key-here
GROQ_MODEL=llama-3.3-70b-versatile
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=2000
VERBOSE=True
DEBUG=False
```

After creating `.env`, run:

```bash
/opt/homebrew/bin/pip3.10 install -r requirements.txt
/opt/homebrew/bin/python3.10 shared_config.py
/opt/homebrew/bin/python3.10 autogen/autogen_simple_demo.py
/opt/homebrew/bin/python3.10 crewai/crewai_demo.py
```

## Completed Runs

### Shared Configuration

- Provider: Groq
- Model: `llama-3.3-70b-versatile`
- Validation passed successfully with `shared_config.py`

### AutoGen Output

- Output file: `autogen/groupchat_output_20260424_110016.txt`
- Product concept created: `HireSphere`
- Main recommendations: prioritize the actionable insights dashboard and candidate portal, phase the virtual interview studio later, and keep diversity/inclusivity as a core differentiator.

### CrewAI Output

- Output file: `crewai/crewai_output_iceland.txt`
- Final result: a 5-day Iceland travel plan with budget ranges
- Final cost summary:
  - Budget: `$1,450-$2,000`
  - Mid-range: `$2,500-$3,500`
  - Luxury: `$4,000-$5,500`

## Exercise 2 Results

### AutoGen Behavior Change

I changed the `ResearchAgent` from AI interview platforms to AI-powered employee onboarding tools and reran the workflow.

- Updated output file: `autogen/groupchat_output_20260424_113214.txt`
- The conversation shifted away from recruiting and toward onboarding pain points such as personalized learning paths, real-time coaching, HR system integration, and employee engagement.
- Downstream agents adapted without me rewriting their role logic for a new product category.

### CrewAI Behavior Change

I changed the `FlightAgent` persona so it explicitly prioritized budget airlines and direct flights where reasonable.

- Updated output file: `crewai/crewai_output_iceland.txt`
- The recommended flight changed to `PLAY Airlines` at `$349` round-trip.
- The final budget output reflected that cheaper recommendation and produced lower total estimates:
  - Budget: `$1,451-$1,901`
  - Mid-range: `$2,240-$2,840`
  - Luxury: `$3,440-$4,240`

### Exercise 2 Answers

- One agent's changed behavior rippled through both systems, but differently.
- In AutoGen, the changed research focus altered the discussion topic itself, so later agents shifted from interview-platform features to onboarding-platform features.
- In CrewAI, the cheaper flight preference propagated into the final budget math and lowered the trip totals.
- In AutoGen, the `GroupChatManager` did not select speakers in the same order. The onboarding rerun was even less linear than the original run.
- In CrewAI, yes, the budget agent reflected the flight agent's new priorities because the final report used `PLAY Airlines` as the recommended flight and reduced the cost ranges.

## Framework Comparison

### AutoGen

- Uses a shared `GroupChat` where agents can see the full conversation history.
- The `GroupChatManager` chooses the next speaker dynamically.
- This makes the workflow more conversational, flexible, and emergent.

### CrewAI

- Uses explicit `Task` objects assigned to specialized agents.
- Tasks run in sequence, so outputs flow from one agent to the next.
- This makes the workflow easier to follow and more structured.

## What I Observed

AutoGen is better when the problem benefits from discussion, iteration, and agents reacting to each other in real time. CrewAI is better when the workflow is predictable and each step has a clear responsibility.

In the AutoGen run, the speaker order did not follow the intended sequence exactly: `AnalysisAgent` spoke before `ResearchAgent`. This is a good example of how `GroupChatManager` dynamically selects speakers. In CrewAI, the order stayed fixed because tasks are predefined and sequential.

## Exercise 2 Reflection

Changing one agent's persona affects downstream results in both frameworks, but the effect looks different:

- In AutoGen, the change ripples through the conversation because later agents react directly to earlier messages.
- In CrewAI, the change appears in later task outputs because each task inherits prior context.

If the `ResearchAgent` becomes focused on a different market, the `AnalysisAgent` and `BlueprintAgent` will naturally respond to that new context. If the `FlightAgent` prioritizes budget or direct flights, the final budget report should also reflect those priorities.

## Exercise 3 Changes Completed

### AutoGen

Added a fifth agent: `CostAnalyst`

- Reviews the product blueprint
- Estimates implementation effort and timeline
- Ranks features by cost-benefit tradeoff
- Passes cost context to the reviewer

### CrewAI

Added a fifth agent: `LocalExpert`

- Provides local customs and etiquette tips
- Adds safety and packing guidance
- Contributes destination-specific money-saving advice
- Gives the `BudgetAgent` better practical context
- Was implemented in a Groq-safe way by passing curated research packets into task prompts instead of relying on live CrewAI tool calls

## Exercise 4 Custom Problem

I chose **conference planning** as the custom domain and ran both frameworks on the same problem: planning a two-day event called `AI in Higher Education Summit`.

### AutoGen Conference Run

- File: `autogen/autogen_conference_demo.py`
- Output file: `autogen/conference_groupchat_output_20260424_114517.txt`
- Result: a conversational conference strategy covering audience, venue setup, agenda, sponsorship tiers, and a final `GO` recommendation

### CrewAI Conference Run

- File: `crewai/crewai_conference_demo.py`
- Output file: `crewai/crewai_conference_output.txt`
- Result: a structured conference budget and launch recommendation
- Key numbers:
  - Estimated total budget: `$150,000-$250,000`
  - Suggested pre-launch condition: secure at least `$50,000` in revenue before confirming the event

### Which Framework Was More Useful?

For conference planning, **CrewAI produced the more useful final result**.

- AutoGen was better for brainstorming, positioning, and collaborative idea generation.
- CrewAI was better for turning the plan into an actionable launch recommendation with clearer budget structure, staged decisions, and operational next steps.
- If the goal is ideation, AutoGen feels more natural. If the goal is a decision-ready planning document, CrewAI is stronger.

## Requirement Check

The lab requirements are completed:

- Both original demos were run successfully
- The AutoGen and CrewAI workflows were compared
- Exercise 2 was completed with actual reruns and observed behavior changes
- Exercise 3 was completed by adding a fifth agent to both systems
- Exercise 4 was completed with a shared conference-planning problem across both frameworks
- A written report was prepared with results and comparisons

## Conclusion

AutoGen demonstrates collaborative reasoning through conversation, while CrewAI demonstrates structured orchestration through explicit tasks. In this lab, AutoGen was strongest for open-ended ideation and dynamic interaction, while CrewAI was strongest for producing structured, decision-ready outputs. Together, the two frameworks gave a clear comparison of conversational versus task-based multi-agent design.
