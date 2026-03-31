import unittest

from rs.llm.campfire_subagent import CampfireSubagent
from rs.llm.providers.campfire_llm_provider import CampfireCommandProposal
from test_helpers.resources import load_resource_state


class FakeLangMemService:
    def __init__(self):
        self.recorded = []

    def build_context_memory(self, context):
        return {
            "retrieved_episodic_memories": "EP campfire memory",
            "retrieved_semantic_memories": "SEM campfire memory",
            "langmem_status": "ready",
        }

    def record_accepted_decision(self, context, decision):
        self.recorded.append((context, decision))

    def status(self):
        return "ready"


class FakeCampfireRuntime:
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


class ScriptedCampfireProvider:
    def __init__(self, proposals):
        self._proposals = list(proposals)
        self.validation_feedbacks = []

    def propose(self, context, working_memory, validation_feedback=None):
        self.validation_feedbacks.append(dict(validation_feedback or {}))
        if not self._proposals:
            return CampfireCommandProposal(None, 0.0, "empty_script")
        return self._proposals.pop(0)


class TestCampfireSubagent(unittest.TestCase):
    def test_subagent_executes_rest_choice(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedCampfireProvider([
            CampfireCommandProposal("choose 0", 0.8, "rest_now"),
        ])
        subagent = CampfireSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/campfire/campfire_rest.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeCampfireRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"]], runtime.command_batches)
        self.assertEqual(1, len(langmem_service.recorded))

    def test_subagent_executes_single_option_recall(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedCampfireProvider([
            CampfireCommandProposal("choose 0", 0.9, "only_valid_choice"),
        ])
        subagent = CampfireSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/campfire/campfire_default_because_options_blocked.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeCampfireRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"]], runtime.command_batches)

    def test_subagent_rejects_invalid_option(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedCampfireProvider([
            CampfireCommandProposal("choose impossible", 0.5, "bad_command"),
        ])
        subagent = CampfireSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/campfire/campfire_rest.json")
        runtime = FakeCampfireRuntime(initial_state, [])

        result = subagent.run(initial_state, runtime)

        self.assertFalse(result.handled)
        self.assertEqual([], runtime.command_batches)
        self.assertEqual(0, len(langmem_service.recorded))

    def test_subagent_refines_choose_token_to_index(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedCampfireProvider([
            CampfireCommandProposal("choose rest", 0.8, "bad_token_form"),
            CampfireCommandProposal("choose 0", 0.9, "fixed_index"),
        ])
        subagent = CampfireSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/campfire/campfire_rest.json")
        final_state = load_resource_state("/card_reward/card_reward_take.json")
        runtime = FakeCampfireRuntime(initial_state, [final_state])

        result = subagent.run(initial_state, runtime)

        self.assertTrue(result.handled)
        self.assertEqual([["choose 0"]], runtime.command_batches)
        self.assertEqual("choose_requires_index", provider.validation_feedbacks[1].get("code"))

    def test_subagent_stops_after_choose_validation_retry_limit(self):
        langmem_service = FakeLangMemService()
        provider = ScriptedCampfireProvider([
            CampfireCommandProposal("choose rest", 0.8, "bad_token_form"),
            CampfireCommandProposal("choose rest", 0.8, "bad_token_form_again"),
            CampfireCommandProposal("choose 0", 0.9, "should_not_be_used"),
        ])
        subagent = CampfireSubagent(provider=provider, langmem_service=langmem_service)
        initial_state = load_resource_state("/campfire/campfire_rest.json")
        runtime = FakeCampfireRuntime(initial_state, [])

        result = subagent.run(initial_state, runtime)

        self.assertFalse(result.handled)
        self.assertEqual([], runtime.command_batches)
        self.assertEqual(2, len(provider.validation_feedbacks))


if __name__ == "__main__":
    unittest.main()
