"""Integration tests for BattleSubagent that invoke a real LLM.

These tests are skipped automatically if no LLM backend is reachable.
"""
import copy
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rs.llm.battle_subagent import BattleSubagent, BattleSubagentConfig
from rs.llm.providers.battle_llm_provider import _SYSTEM_PROMPT
from rs.machine.state import GameState
from rs.utils.config import config as llm_runtime_config
from rs.utils.llm_utils import run_llm_preflight_check
from test_helpers.resources import load_resource_state


class FakeLangMemService:
    def __init__(self):
        self.recorded = []
        self.custom_memories = []

    def build_context_memory(self, context):
        return {
            "retrieved_episodic_memories": "none",
            "retrieved_semantic_memories": "none",
            "langmem_status": "ready",
        }

    def record_accepted_decision(self, context, decision):
        self.recorded.append((context, decision))

    def record_custom_memory(self, context, content, tags=(), reflect=False):
        self.custom_memories.append((context, content, tags, reflect))

    def pause_reflections(self):
        pass

    def resume_reflections(self):
        pass

    def status(self):
        return "ready"


class FakeBattleRuntime:
    def __init__(self, initial_state, next_states):
        self._current_state = initial_state
        self._next_states = list(next_states)
        self.command_batches = []

    def current_state(self):
        return self._current_state

    def execute(self, commands):
        self.command_batches.append(list(commands))
        if self._next_states:
            self._current_state = self._next_states.pop(0)
        return self._current_state


def _build_chat_model() -> ChatOpenAI:
    base_url = llm_runtime_config.llm_base_url or llm_runtime_config.openai_base_url or None
    api_key = llm_runtime_config.llm_api_key or llm_runtime_config.openai_key or "test"
    return ChatOpenAI(
        model=llm_runtime_config.fast_llm_model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.0,
    )


class _InstrumentedChatModel:
    """Wraps a real ChatOpenAI and records every invoke call with its messages."""

    def __init__(self, inner: ChatOpenAI):
        self._inner = inner
        self._inner_with_tools = inner
        self.invoke_log: list[list] = []

    def bind_tools(self, tools):
        self._inner_with_tools = self._inner.bind_tools(tools)
        return _BoundInstrumented(self._inner_with_tools, self.invoke_log)

    def invoke(self, messages):
        self.invoke_log.append(list(messages))
        return self._inner.invoke(messages)


class _BoundInstrumented:
    """Bound-tools wrapper that shares the invoke_log."""

    def __init__(self, bound, invoke_log: list):
        self._bound = bound
        self.invoke_log = invoke_log

    def invoke(self, messages):
        self.invoke_log.append(list(messages))
        return self._bound.invoke(messages)


class TestBattleSubagentReal(unittest.TestCase):

    def setUp(self):
        preflight = run_llm_preflight_check(model=llm_runtime_config.fast_llm_model)
        if not preflight.available:
            self.skipTest(f"LLM backend not reachable: {preflight.error}")

    def test_messages_accumulate_across_card_plays(self):
        """Verify that the conversation persists: think reasoning from step 1 is visible in step 2."""
        inner_model = _build_chat_model()
        instrumented = _InstrumentedChatModel(inner_model)

        langmem = FakeLangMemService()
        subagent = BattleSubagent(
            chat_model=instrumented,
            langmem_service=langmem,
            config=BattleSubagentConfig(
                max_decision_loops=16,
                max_tool_calls=16,
                fallback_max_path_count=100,
            ),
        )

        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        mid_state = load_resource_state("battles/general/another_simple.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [mid_state, final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertGreaterEqual(result.steps, 1)

        # Check that messages accumulate: later invoke calls should have MORE messages
        # than earlier ones, proving the conversation persists.
        call_lengths = [len(msgs) for msgs in instrumented.invoke_log]
        self.assertGreater(len(call_lengths), 2, "Expected at least 3 invoke calls (think + agent + ...)")

        # The first call (think_node) should have the fewest messages (system + state)
        # Later calls should have strictly more messages (accumulated reasoning + tool results)
        first_call_len = call_lengths[0]
        max_call_len = max(call_lengths)
        self.assertGreater(
            max_call_len, first_call_len,
            f"Messages should accumulate: first call had {first_call_len} msgs, "
            f"max had {max_call_len} msgs. All call lengths: {call_lengths}"
        )

    def test_system_message_set_once(self):
        """Verify SystemMessage appears only once, at the start of the conversation."""
        inner_model = _build_chat_model()
        instrumented = _InstrumentedChatModel(inner_model)

        langmem = FakeLangMemService()
        subagent = BattleSubagent(
            chat_model=instrumented,
            langmem_service=langmem,
            config=BattleSubagentConfig(
                max_decision_loops=16,
                max_tool_calls=16,
                fallback_max_path_count=100,
            ),
        )

        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)
        self.assertTrue(result.handled)

        # Every invoke call should have exactly 1 SystemMessage (the first message)
        for i, call_messages in enumerate(instrumented.invoke_log):
            system_msgs = [m for m in call_messages if isinstance(m, SystemMessage)]
            self.assertEqual(
                1, len(system_msgs),
                f"Invoke call {i} should have exactly 1 SystemMessage, got {len(system_msgs)}"
            )
            # SystemMessage should always be first
            self.assertIsInstance(
                call_messages[0], SystemMessage,
                f"Invoke call {i}: first message should be SystemMessage, got {type(call_messages[0]).__name__}"
            )

    def test_think_reasoning_visible_to_agent(self):
        """Verify that the think_node's AIMessage is in the agent_node's input."""
        inner_model = _build_chat_model()
        instrumented = _InstrumentedChatModel(inner_model)

        langmem = FakeLangMemService()
        subagent = BattleSubagent(
            chat_model=instrumented,
            langmem_service=langmem,
            config=BattleSubagentConfig(
                max_decision_loops=16,
                max_tool_calls=16,
                fallback_max_path_count=100,
            ),
        )

        initial_state = load_resource_state("battles/general/battle_simple_state.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)
        self.assertTrue(result.handled)

        # invoke_log[0] = think_node (bare model, gets SystemMessage + HumanMessage + HumanMessage("Analyse..."))
        # invoke_log[1] = agent_node (model+tools, gets SystemMessage + HumanMessage + HumanMessage("Battle analysis:..."))
        self.assertGreaterEqual(len(instrumented.invoke_log), 2)

        think_call = instrumented.invoke_log[0]
        agent_call = instrumented.invoke_log[1]

        # Agent call should have more HumanMessages than think call (the analysis was injected)
        think_human = [m for m in think_call if isinstance(m, HumanMessage)]
        agent_human = [m for m in agent_call if isinstance(m, HumanMessage)]
        self.assertGreater(
            len(agent_human), len(think_human),
            "Agent call should contain think_node's analysis as an additional HumanMessage"
        )

        # The last HumanMessage in agent call should contain the battle analysis
        analysis_msg = agent_human[-1]
        self.assertIn(
            "Battle analysis:", analysis_msg.content,
            "Agent call should contain 'Battle analysis:' HumanMessage from think_node"
        )
        self.assertIn(
            "submit_battle_commands", analysis_msg.content,
            "Analysis message should prompt the agent to execute via submit_battle_commands"
        )


class TestStsdbDescriptionsInPipeline(unittest.TestCase):
    """Verify that stsdb descriptions for cards, potions, relics, and powers
    flow through the context building into the actual LLM prompt."""

    def _build_rich_state(self):
        """Build a GameState with hand cards, monsters with powers, potions, and relics."""
        import json
        base = json.load(open("tests/res/battles/general/breaks_when_no_monsters_alive_but_not_won.json"))
        # This fixture already has relics and a potion; it also has a real combat_state
        return load_resource_state("battles/general/breaks_when_no_monsters_alive_but_not_won.json")

    def test_hand_cards_have_descriptions(self):
        from rs.llm.integration.battle_context import build_battle_agent_context
        state = self._build_rich_state()
        context = build_battle_agent_context(state, "BattleHandler")
        hand = context.extras.get("hand_cards", [])
        if not hand:
            self.skipTest("Fixture has no hand cards")
        cards_with_desc = [c for c in hand if c.get("description")]
        self.assertGreater(
            len(cards_with_desc), 0,
            f"At least one hand card should have a stsdb description, got: {hand}"
        )

    def test_relic_summaries_have_descriptions(self):
        from rs.llm.integration.battle_context import build_battle_agent_context
        state = self._build_rich_state()
        context = build_battle_agent_context(state, "BattleHandler")
        relics = context.extras.get("relic_summaries", [])
        self.assertGreater(len(relics), 0, "Fixture should have relics")
        relics_with_desc = [r for r in relics if r.get("description")]
        self.assertGreater(
            len(relics_with_desc), 0,
            f"At least one relic should have a stsdb description, got: {relics}"
        )
        # Spot check a known relic
        pure_water = [r for r in relics if r["name"] == "Pure Water"]
        if pure_water:
            self.assertIn("Miracle", pure_water[0]["description"])

    def test_potion_summaries_have_descriptions(self):
        from rs.llm.integration.battle_context import build_battle_agent_context
        state = self._build_rich_state()
        context = build_battle_agent_context(state, "BattleHandler")
        potions = context.extras.get("potion_summaries", [])
        if not potions:
            self.skipTest("Fixture has no potions")
        potions_with_desc = [p for p in potions if p.get("description")]
        self.assertGreater(
            len(potions_with_desc), 0,
            f"At least one potion should have a stsdb description, got: {potions}"
        )

    def test_player_powers_have_descriptions(self):
        from rs.llm.integration.battle_context import build_battle_agent_context
        state = self._build_rich_state()
        context = build_battle_agent_context(state, "BattleHandler")
        powers = context.extras.get("player_powers", [])
        if not powers:
            self.skipTest("Fixture has no player powers")
        powers_with_desc = [p for p in powers if p.get("description")]
        self.assertGreater(
            len(powers_with_desc), 0,
            f"At least one player power should have a stsdb description, got: {powers}"
        )

    def test_monster_powers_have_descriptions(self):
        from rs.llm.integration.battle_context import build_battle_agent_context
        state = self._build_rich_state()
        context = build_battle_agent_context(state, "BattleHandler")
        monsters = context.extras.get("monster_summaries", [])
        if not monsters:
            self.skipTest("Fixture has no monsters")
        all_powers = [p for m in monsters for p in m.get("powers", [])]
        if not all_powers:
            self.skipTest("Fixture monsters have no powers")
        powers_with_desc = [p for p in all_powers if p.get("description")]
        self.assertGreater(
            len(powers_with_desc), 0,
            f"At least one monster power should have a stsdb description, got: {all_powers}"
        )

    def test_descriptions_reach_system_prompt_in_real_pipeline(self):
        """Verify relic descriptions appear in the SystemMessage sent to the LLM."""
        preflight = run_llm_preflight_check(model=llm_runtime_config.fast_llm_model)
        if not preflight.available:
            self.skipTest(f"LLM backend not reachable: {preflight.error}")

        inner_model = _build_chat_model()
        instrumented = _InstrumentedChatModel(inner_model)

        langmem = FakeLangMemService()
        subagent = BattleSubagent(
            chat_model=instrumented,
            langmem_service=langmem,
            config=BattleSubagentConfig(
                max_decision_loops=16,
                max_tool_calls=16,
                fallback_max_path_count=100,
            ),
        )

        initial_state = load_resource_state("battles/general/breaks_when_no_monsters_alive_but_not_won.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        # Check the SystemMessage in the first invoke call for relic descriptions
        self.assertGreater(len(instrumented.invoke_log), 0)
        first_call = instrumented.invoke_log[0]
        system_msgs = [m for m in first_call if isinstance(m, SystemMessage)]
        self.assertEqual(1, len(system_msgs))
        system_content = system_msgs[0].content

        # The fixture has "Pure Water" relic — its description should be in the system prompt
        self.assertIn("Pure Water", system_content, "Relic name should appear in system prompt")
        self.assertIn("Miracle", system_content, "Pure Water's description mentions Miracle")

    def test_descriptions_reach_state_update_in_real_pipeline(self):
        """Verify card/power descriptions appear in HumanMessage state updates."""
        preflight = run_llm_preflight_check(model=llm_runtime_config.fast_llm_model)
        if not preflight.available:
            self.skipTest(f"LLM backend not reachable: {preflight.error}")

        inner_model = _build_chat_model()
        instrumented = _InstrumentedChatModel(inner_model)

        langmem = FakeLangMemService()
        subagent = BattleSubagent(
            chat_model=instrumented,
            langmem_service=langmem,
            config=BattleSubagentConfig(
                max_decision_loops=16,
                max_tool_calls=16,
                fallback_max_path_count=100,
            ),
        )

        initial_state = load_resource_state("battles/general/breaks_when_no_monsters_alive_but_not_won.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeBattleRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        # Find HumanMessages containing state updates
        self.assertGreater(len(instrumented.invoke_log), 0)
        first_call = instrumented.invoke_log[0]
        human_msgs = [m for m in first_call if isinstance(m, HumanMessage)]
        self.assertGreater(len(human_msgs), 0)

        # The state update should contain "description" fields from stsdb
        state_content = human_msgs[0].content
        self.assertIn("description", state_content,
                       "State update should contain stsdb description fields for cards/powers")


if __name__ == "__main__":
    unittest.main()
