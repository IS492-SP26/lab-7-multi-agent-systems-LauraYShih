"""
AutoGen GroupChat Demo - Conference Planning Workflow
"""

import os
from datetime import datetime
from config import Config

try:
    import autogen
except ImportError:
    print("ERROR: AutoGen is not installed!")
    print("Please run: pip install -r ../requirements.txt")
    exit(1)


class GroupChatConferencePlanner:
    """Multi-agent GroupChat workflow for conference planning using AutoGen."""

    def __init__(self):
        if not Config.validate_setup():
            print("ERROR: Configuration validation failed!")
            exit(1)

        self.config_list = Config.get_config_list()
        self.llm_config = {
            "config_list": self.config_list,
            "temperature": Config.AGENT_TEMPERATURE,
        }

        self._create_agents()
        self._setup_groupchat()

        print("All conference planning agents created and GroupChat initialized.")

    def _create_agents(self):
        self.user_proxy = autogen.UserProxyAgent(
            name="ConferenceDirector",
            system_message="A conference director who kicks off the planning discussion and keeps the team aligned.",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

        self.research_agent = autogen.AssistantAgent(
            name="AudienceResearchAgent",
            system_message="""You are a conference market researcher specializing in technology events.
Your role is to START the discussion with audience needs, event trends, and competitive positioning.

Your responsibilities:
- Identify the target audience for a two-day AI in Higher Education conference
- Summarize 3 relevant conference trends
- Note what competing events usually do well and where they leave gaps
- Recommend a positioning angle that would make this event stand out

After presenting your research, invite the VenuePlannerAgent to propose the right event setup.
Keep your response under 350 words.""",
            llm_config=self.llm_config,
            description="Researches attendee needs, conference trends, and market positioning.",
        )

        self.venue_agent = autogen.AssistantAgent(
            name="VenuePlannerAgent",
            system_message="""You are an event operations strategist.
Your role is to BUILD on the research and recommend the best venue and on-site setup.

Your responsibilities:
- Recommend venue requirements for a two-day conference with 300 attendees
- Propose room layout, session format mix, and logistics priorities
- Highlight accessibility, AV, catering, and networking space needs
- Explain how the setup supports the target audience and event positioning

After presenting your venue plan, invite the ProgramDesignerAgent to create the agenda.
Keep your response under 350 words.""",
            llm_config=self.llm_config,
            description="Plans venue requirements, event operations, and attendee flow.",
        )

        self.program_agent = autogen.AssistantAgent(
            name="ProgramDesignerAgent",
            system_message="""You are a conference program designer.
Your role is to DESIGN the agenda and attendee journey.

Your responsibilities:
- Create a two-day conference agenda with keynote, breakouts, workshops, and networking
- Recommend speaker profile types rather than specific people
- Balance strategic content, practical implementation sessions, and community-building
- Show how the agenda serves faculty leaders, administrators, and edtech teams

After presenting the program plan, invite the SponsorshipAgent to propose the revenue model.
Keep your response under 350 words.""",
            llm_config=self.llm_config,
            description="Designs the conference agenda, session mix, and attendee journey.",
        )

        self.sponsorship_agent = autogen.AssistantAgent(
            name="SponsorshipAgent",
            system_message="""You are a conference sponsorship and partnerships strategist.
Your role is to evaluate the plan from a sponsorship, monetization, and budget feasibility perspective.

Your responsibilities:
- Recommend sponsorship tiers and likely sponsor categories
- Suggest ticket pricing logic and revenue streams
- Highlight the most important budget drivers and cost controls
- Rank the conference elements by ROI and sponsor appeal

After presenting your analysis, invite the ReviewerAgent to provide the final conference recommendation.
Keep your response under 350 words.""",
            llm_config=self.llm_config,
            description="Creates sponsorship tiers, revenue strategy, and budget tradeoff guidance.",
        )

        self.reviewer_agent = autogen.AssistantAgent(
            name="ReviewerAgent",
            system_message="""You are an executive event advisor.
Your role is to REVIEW the full conference plan and provide final recommendations.

Your responsibilities:
- Evaluate the market fit and feasibility of the proposed conference
- Recommend the top 3 launch priorities
- Identify 3 major risks with mitigation steps
- Suggest a concise go/no-go recommendation

Reference the audience research, venue setup, program design, and sponsorship strategy.
End your final message with TERMINATE.""",
            llm_config=self.llm_config,
            description="Reviews the full conference strategy and provides the final go/no-go recommendation.",
        )

    def _setup_groupchat(self):
        self.groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy,
                self.research_agent,
                self.venue_agent,
                self.program_agent,
                self.sponsorship_agent,
                self.reviewer_agent,
            ],
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
            send_introductions=True,
        )

        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

    def run(self):
        print("\n" + "=" * 80)
        print("AUTOGEN GROUPCHAT - CONFERENCE PLANNING")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {Config.OPENAI_MODEL}")
        print(f"Max Rounds: {self.groupchat.max_round}")
        print(f"Speaker Selection: {self.groupchat.speaker_selection_method}")
        print("\nAgents in GroupChat:")
        for agent in self.groupchat.agents:
            print(f"  - {agent.name}")
        print("\n" + "=" * 80)
        print("MULTI-AGENT CONVERSATION BEGINS")
        print("=" * 80 + "\n")

        initial_message = """Team, we need to design a two-day conference called AI in Higher Education Summit.

Let's collaborate on this:
1. AudienceResearchAgent: define audience needs, market trends, and positioning
2. VenuePlannerAgent: recommend venue and operational setup
3. ProgramDesignerAgent: build the agenda and session mix
4. SponsorshipAgent: propose sponsorship tiers and revenue strategy
5. ReviewerAgent: provide final recommendations and go/no-go advice

AudienceResearchAgent, please begin with your market analysis."""

        chat_result = self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message,
            summary_method="last_msg",
        )

        self._print_summary(chat_result)
        output_file = self._save_results(chat_result)
        print(f"\nFull results saved to: {output_file}")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def _print_summary(self, chat_result):
        print("\n" + "=" * 80)
        print("CONVERSATION COMPLETE")
        print("=" * 80)

        print(f"\nTotal conversation rounds: {len(self.groupchat.messages)}")
        print("\nSpeaker order (as selected by GroupChatManager):")
        for i, msg in enumerate(self.groupchat.messages, 1):
            speaker = msg.get("name", "Unknown")
            content = msg.get("content", "")
            preview = content[:80].replace("\n", " ") + "..." if len(content) > 80 else content.replace("\n", " ")
            print(f"  {i}. [{speaker}]: {preview}")

        if chat_result.summary:
            print("\n" + "-" * 80)
            print("EXECUTIVE SUMMARY")
            print("-" * 80)
            print(chat_result.summary)

    def _save_results(self, chat_result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, f"conference_groupchat_output_{timestamp}.txt")

        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOGEN GROUPCHAT - CONFERENCE PLANNING\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {Config.OPENAI_MODEL}\n")
            f.write(f"Conversation Rounds: {len(self.groupchat.messages)}\n\n")

            f.write("=" * 80 + "\n")
            f.write("MULTI-AGENT CONVERSATION\n")
            f.write("=" * 80 + "\n\n")

            for i, msg in enumerate(self.groupchat.messages, 1):
                speaker = msg.get("name", "Unknown")
                content = msg.get("content", "")
                f.write(f"--- Turn {i}: {speaker} ---\n")
                f.write(content + "\n\n")

            if chat_result.summary:
                f.write("=" * 80 + "\n")
                f.write("EXECUTIVE SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(chat_result.summary + "\n")

        return output_file


if __name__ == "__main__":
    try:
        workflow = GroupChatConferencePlanner()
        workflow.run()
        print("\nConference planning workflow completed successfully!")
    except Exception as e:
        print(f"\nError during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()
