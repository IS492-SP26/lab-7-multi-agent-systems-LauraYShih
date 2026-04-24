"""
CrewAI Multi-Agent Demo: Conference Planning System
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from crewai import Agent, Task, Crew

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared_config import Config, validate_config


def get_audience_packet() -> str:
    return """Audience & Market Packet
============================================================
Conference Theme: AI in Higher Education Summit
Audience size target: 300 attendees
Duration: 2 days

Target segments:
1. University administrators exploring AI policy, procurement, and governance
2. Faculty leaders interested in classroom adoption and curriculum design
3. Instructional designers and edtech teams responsible for implementation
4. Student-success and operations teams using AI for support workflows

Market trends:
- Institutions want practical AI adoption frameworks, not only inspiration
- Attendees prefer case-study sessions with implementation details
- Responsible AI, privacy, and academic integrity remain top concerns
- Buyers value peer examples from other universities

Positioning gap:
- Many education conferences are broad and not execution-focused
- There is room for a conference that combines strategy, tooling, governance, and operations
"""


def get_venue_packet() -> str:
    return """Venue & Operations Packet
============================================================
Recommended setup assumptions:
- 300 in-person attendees
- One large ballroom for keynote sessions
- Three breakout rooms for parallel tracks
- One workshop room with flexible tables
- Dedicated sponsor expo area with coffee nearby
- Networking lounge and registration/check-in zone

Operational priorities:
- Strong Wi-Fi and AV support for live demos
- Accessible entrances, elevators, restrooms, and reserved seating
- Good catering flow for breakfast, lunch, and afternoon breaks
- Clear signage and enough space for sponsor booths
- Nearby hotel block or easy public transit access
"""


def get_program_packet() -> str:
    return """Program Packet
============================================================
Program goals:
- Blend strategic leadership content with practical implementation sessions
- Balance inspiration, peer case studies, workshops, and networking
- Serve administrators, faculty leaders, and implementation teams

Strong session types:
- Opening keynote on AI strategy in higher education
- Panel on AI governance, privacy, and academic integrity
- Case studies from universities with measurable outcomes
- Hands-on workshops for faculty and instructional design teams
- Vendor-neutral implementation roundtables
- Closing session with action planning and next steps
"""


def get_sponsorship_packet() -> str:
    return """Sponsorship Packet
============================================================
Likely sponsor categories:
- Learning management systems
- AI productivity and tutoring platforms
- Student-success software vendors
- Cloud providers and data platforms
- Consulting firms focused on digital transformation

Potential sponsorship structure:
- Platinum: keynote visibility, premium booth, speaking panel consideration
- Gold: breakout sponsorship, booth, branded networking session
- Silver: expo booth, logo placement, attendee list opt-in leads
- Startup/demo tier: small booth and lightning demo slot

Primary budget drivers:
- Venue rental
- AV and production
- Catering
- Speaker travel/honoraria
- Event staffing and marketing
"""


def get_budget_packet() -> str:
    return """Budget Packet
============================================================
Planning assumptions for 300 attendees:
- Venue + AV: $22,000-$32,000
- Catering (2 days): $18,000-$26,000
- Speaker travel/honoraria: $8,000-$18,000
- Staffing + registration tools: $6,000-$10,000
- Marketing + creative: $5,000-$9,000
- Miscellaneous contingency: 10-15%

Revenue assumptions:
- Early-bird ticket: $249
- Standard ticket: $349
- Team/institution bundle available
- Sponsorship revenue target: $35,000-$70,000
"""


def create_research_agent():
    return Agent(
        role="Audience Strategist",
        goal="Define the target audience, market positioning, and attendee value proposition for the conference.",
        backstory="You are an event strategist who turns market research into a clear event positioning strategy.",
        verbose=True,
        allow_delegation=False,
    )


def create_venue_agent():
    return Agent(
        role="Venue and Operations Planner",
        goal="Recommend the best event setup, room flow, and operational priorities for a successful conference.",
        backstory="You specialize in conference logistics, attendee flow, accessibility, and operational execution.",
        verbose=True,
        allow_delegation=False,
    )


def create_program_agent():
    return Agent(
        role="Program Designer",
        goal="Create a compelling two-day conference agenda for multiple stakeholder groups.",
        backstory="You design conference programs that balance strategic thinking, practical sessions, and strong attendee engagement.",
        verbose=True,
        allow_delegation=False,
    )


def create_sponsorship_agent():
    return Agent(
        role="Sponsorship Strategist",
        goal="Design sponsorship tiers and a revenue approach that supports the conference financially.",
        backstory="You build sponsor packages and monetization plans that match event goals without hurting attendee experience.",
        verbose=True,
        allow_delegation=False,
    )


def create_budget_agent():
    return Agent(
        role="Conference Finance Advisor",
        goal="Combine prior outputs into a practical launch budget and go/no-go recommendation.",
        backstory="You specialize in event budgeting, pricing, and financial risk management for conferences.",
        verbose=True,
        allow_delegation=False,
    )


def create_research_task(agent):
    return Task(
        description=f"Use the packet below to define the target audience, key event positioning, and the top attendee needs for the AI in Higher Education Summit. Do not invent outside research.\n\n{get_audience_packet()}",
        agent=agent,
        expected_output="A concise audience strategy with target segments, event positioning, and top attendee needs.",
    )


def create_venue_task(agent):
    return Task(
        description=f"Use the packet below to recommend venue requirements, room mix, and operational priorities for the conference. Build on the audience strategy from the previous task.\n\n{get_venue_packet()}",
        agent=agent,
        expected_output="A venue and operations recommendation with room layout, logistics priorities, and attendee flow guidance.",
    )


def create_program_task(agent):
    return Task(
        description=f"Use the packet below to design a two-day program for the AI in Higher Education Summit. Build on the audience and venue recommendations already created.\n\n{get_program_packet()}",
        agent=agent,
        expected_output="A two-day conference agenda with keynote, breakouts, workshops, and networking elements.",
    )


def create_sponsorship_task(agent):
    return Task(
        description=f"Use the packet below to recommend sponsorship tiers, sponsor categories, and revenue logic for the event. Build on the conference positioning and agenda from the prior tasks.\n\n{get_sponsorship_packet()}",
        agent=agent,
        expected_output="A sponsorship strategy with tier structure, target sponsors, and monetization rationale.",
    )


def create_budget_task(agent):
    return Task(
        description=f"Use the packet below plus all previous task outputs to create a realistic budget summary, pricing logic, and final go/no-go recommendation for launching the conference.\n\n{get_budget_packet()}",
        agent=agent,
        expected_output="A final conference budget summary with cost ranges, revenue logic, launch risks, and a go/no-go recommendation.",
    )


def main():
    print("=" * 80)
    print("CrewAI Multi-Agent Conference Planning System")
    print("Planning: AI in Higher Education Summit")
    print("=" * 80)
    print()

    print("🔍 Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed.")
        exit(1)

    os.environ["OPENAI_API_KEY"] = Config.API_KEY
    os.environ["OPENAI_API_BASE"] = Config.API_BASE
    if Config.USE_GROQ:
        os.environ["OPENAI_MODEL_NAME"] = Config.OPENAI_MODEL

    print("✅ Configuration validated successfully!")
    print()
    Config.print_summary()
    print()

    print("Creating conference planning agents...")
    research_agent = create_research_agent()
    venue_agent = create_venue_agent()
    program_agent = create_program_agent()
    sponsorship_agent = create_sponsorship_agent()
    budget_agent = create_budget_agent()
    print("Agents created successfully!")
    print()

    print("Creating tasks...")
    research_task = create_research_task(research_agent)
    venue_task = create_venue_task(venue_agent)
    program_task = create_program_task(program_agent)
    sponsorship_task = create_sponsorship_task(sponsorship_agent)
    budget_task = create_budget_task(budget_agent)
    print("Tasks created successfully!")
    print()

    crew = Crew(
        agents=[research_agent, venue_agent, program_agent, sponsorship_agent, budget_agent],
        tasks=[research_task, venue_task, program_task, sponsorship_task, budget_task],
        verbose=True,
        process="sequential",
    )

    print("=" * 80)
    print("Starting Crew Execution...")
    print("=" * 80)
    print()

    result = crew.kickoff()

    print()
    print("=" * 80)
    print("✅ Crew Execution Completed Successfully!")
    print("=" * 80)
    print(result)

    output_path = Path(__file__).parent / "crewai_conference_output.txt"
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CrewAI Conference Planning System - Execution Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Provider: {'Groq' if Config.USE_GROQ else 'OpenAI'}\n")
        f.write(f"Model: {Config.OPENAI_MODEL}\n\n")
        f.write(str(result))
        f.write("\n")

    print(f"\n✅ Output saved to {output_path.name}")


if __name__ == "__main__":
    main()
