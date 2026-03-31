import copy
import unittest

from rs.llm.providers.reward_llm_provider import RewardCommandProposal
from rs.llm.reward_subagent import AstrolabeTransformSubagent, BossRewardSubagent, CombatRewardSubagent
from test_helpers.resources import load_resource_state


class FakeLangMemService:
    def __init__(self):
        self.recorded = []

    def build_context_memory(self, context):
        return {
            "retrieved_episodic_memories": "EP reward memory",
            "retrieved_semantic_memories": "SEM reward memory",
            "langmem_status": "ready",
        }

    def record_accepted_decision(self, context, decision):
        self.recorded.append((context, decision))

    def status(self):
        return "ready"


class FakeRewardRuntime:
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


class ScriptedRewardProvider:
    def __init__(self, proposals):
        self._proposals = list(proposals)
        self.seen_contexts = []

    def propose(self, context, working_memory, validation_feedback=None):
        self.seen_contexts.append((context, dict(working_memory), dict(validation_feedback or {})))
        if not self._proposals:
            return RewardCommandProposal(None, 0.0, "empty_script")
        return self._proposals.pop(0)


class TestRewardSubagents(unittest.TestCase):
    def test_combat_reward_subagent_can_loop_until_scope_ends(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("choose 0", 0.8, "take_gold_first"),
        ])
        subagent = CombatRewardSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/combat_reward/combat_reward_gold.json")
        mid_state = load_resource_state("/combat_reward/combat_reward_card.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeRewardRuntime(initial_state, [mid_state, final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"], ["choose 0"]], runtime.command_batches)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual(2, len(langmem_service.recorded))
        self.assertEqual(1, len(provider.seen_contexts))

    def test_combat_reward_subagent_delegates_card_pick_to_card_reward_handler(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("choose 0", 0.8, "take_gold_first"),
        ])
        subagent = CombatRewardSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/combat_reward/combat_reward_several_rewards.json")
        mid_state = self._combat_reward_only_card(initial_state)
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeRewardRuntime(initial_state, [mid_state, final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"], ["choose 0"]], runtime.command_batches)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())
        self.assertEqual(1, len(provider.seen_contexts))
        first_context, _, _ = provider.seen_contexts[0]
        self.assertEqual("COMBAT_REWARD", first_context.screen_type)
        self.assertEqual([], [row for row in first_context.extras["reward_summaries"] if row["reward_type"] == "CARD"])

    def test_boss_reward_subagent_executes_skip_mapping(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("skip", 0.9, "skip_all_relics"),
        ])
        subagent = BossRewardSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/relics/boss_reward_nothing_to_take.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeRewardRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["skip", "proceed"]], runtime.command_batches)
        self.assertEqual(1, len(langmem_service.recorded))

    def test_astrolabe_transform_subagent_can_make_three_selections(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("choose 0", 0.9, "first_transform"),
            RewardCommandProposal("choose 1", 0.9, "second_transform"),
            RewardCommandProposal("choose 2", 0.9, "third_transform"),
        ])
        subagent = AstrolabeTransformSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/relics/boss_reward_astrolabe.json")
        second_state = self._astrolabe_state_with_selected_count(initial_state, 2)
        third_state = self._astrolabe_state_with_selected_count(initial_state, 3)
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeRewardRuntime(initial_state, [second_state, third_state, final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"], ["choose 1"], ["choose 2"]], runtime.command_batches)
        self.assertEqual(3, result.steps)
        self.assertEqual("CARD_REWARD", result.final_state.screen_type())

    def test_combat_reward_subagent_rejects_ambiguous_choose_token(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("choose relic", 0.8, "ambiguous_bad_command"),
        ])
        subagent = CombatRewardSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/combat_reward/combat_reward_relics_2_take_second.json")
        runtime = FakeRewardRuntime(initial_state, [])

        result = subagent.run(initial_state, runtime)

        self.assertFalse(result.handled)
        self.assertEqual([], runtime.command_batches)
        self.assertEqual(0, len(langmem_service.recorded))

    def test_combat_reward_subagent_refines_invalid_choose_token_to_index(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("choose relic", 0.8, "bad_token_form"),
            RewardCommandProposal("choose 1", 0.9, "correct_index"),
        ])
        subagent = CombatRewardSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/combat_reward/combat_reward_relics_2_take_second.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeRewardRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 1"]], runtime.command_batches)
        self.assertEqual(2, len(provider.seen_contexts))
        self.assertEqual("choose_requires_index", provider.seen_contexts[1][2].get("code"))

    def test_combat_reward_subagent_stops_after_validation_retry_limit(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedRewardProvider([
            RewardCommandProposal("choose relic", 0.8, "bad_token_form"),
            RewardCommandProposal("choose relic", 0.8, "bad_token_form_again"),
            RewardCommandProposal("choose 1", 0.9, "should_not_be_used"),
        ])
        subagent = CombatRewardSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/combat_reward/combat_reward_relics_2_take_second.json")
        runtime = FakeRewardRuntime(initial_state, [])

        result = subagent.run(initial_state, runtime)

        self.assertFalse(result.handled)
        self.assertEqual([], runtime.command_batches)
        self.assertEqual(2, len(provider.seen_contexts))

    def _astrolabe_state_with_selected_count(self, source_state, selected_count):
        payload = copy.deepcopy(source_state.json)
        payload["game_state"]["screen_state"]["selected_cards"] = payload["game_state"]["screen_state"]["cards"][:selected_count]
        from rs.machine.state import GameState
        return GameState(payload)

    def _combat_reward_only_card(self, source_state):
        payload = copy.deepcopy(source_state.json)
        rewards = payload["game_state"]["screen_state"]["rewards"]
        choices = payload["game_state"]["choice_list"]
        payload["game_state"]["screen_state"]["rewards"] = rewards[-1:]
        payload["game_state"]["choice_list"] = choices[-1:]
        from rs.machine.state import GameState
        return GameState(payload)

if __name__ == "__main__":
    unittest.main()
