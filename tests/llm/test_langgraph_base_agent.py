import unittest

from rs.llm.agents.base_agent import AgentContext
from rs.llm.agents.langgraph_base_agent import LangGraphBaseAgent


class FakeCompiledGraph:
    def __init__(self):
        self.invocations = 0

    def invoke(self, graph_input):
        self.invocations += 1
        return {
            "command": "choose 0",
            "confidence": graph_input.get("confidence", 0.0),
            "note": "ok",
        }


class FakeLangGraphAgent(LangGraphBaseAgent):
    def __init__(self):
        super().__init__(name="fake_langgraph")
        self.build_calls = 0
        self.compile_calls = 0
        self.compiled = FakeCompiledGraph()

    def build_graph(self):
        self.build_calls += 1
        return {"graph": "fake"}

    def compile_graph(self, graph):
        self.compile_calls += 1
        return self.compiled

    def build_graph_input(self, context: AgentContext):
        return {
            "handler": context.handler_name,
            "confidence": 0.6,
        }

    def parse_graph_output(self, graph_output):
        return {
            "proposed_command": graph_output["command"],
            "confidence": graph_output["confidence"],
            "explanation": graph_output["note"],
            "required_tools_used": ["graph"],
        }


class TestLangGraphBaseAgent(unittest.TestCase):
    def test_decide_uses_graph_and_parses_output(self):
        agent = FakeLangGraphAgent()
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        decision = agent.decide(context)

        self.assertEqual("choose 0", decision.proposed_command)
        self.assertEqual(0.6, decision.confidence)
        self.assertEqual(["graph"], decision.required_tools_used)
        self.assertFalse(decision.fallback_recommended)

    def test_graph_is_compiled_lazily_once(self):
        agent = FakeLangGraphAgent()
        context = AgentContext(
            handler_name="EventHandler",
            screen_type="EVENT",
            available_commands=["choose"],
            choice_list=["a"],
        )

        agent.decide(context)
        agent.decide(context)

        self.assertEqual(1, agent.build_calls)
        self.assertEqual(1, agent.compile_calls)
        self.assertEqual(2, agent.compiled.invocations)


if __name__ == "__main__":
    unittest.main()
