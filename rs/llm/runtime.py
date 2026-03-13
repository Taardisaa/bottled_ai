from __future__ import annotations

from rs.llm.agents.battle_meta_advisor_agent import BattleMetaAdvisorAgent
from rs.llm.agents.card_reward_advisor_agent import CardRewardAdvisorAgent
from rs.llm.agents.event_advisor_agent import EventAdvisorAgent
from rs.llm.agents.map_advisor_agent import MapAdvisorAgent
from rs.llm.agents.shop_purchase_advisor_agent import ShopPurchaseAdvisorAgent
from rs.llm.config import load_llm_config
from rs.llm.orchestrator import AIPlayerAgent

_event_orchestrator: AIPlayerAgent | None = None
_battle_meta_advisor: BattleMetaAdvisorAgent | None = None


def get_event_orchestrator() -> AIPlayerAgent:
    """Return a singleton orchestrator for event advisor decisions."""
    global _event_orchestrator
    if _event_orchestrator is None:
        orchestrator = AIPlayerAgent(config=load_llm_config())
        orchestrator.register_agent("EventHandler", EventAdvisorAgent())
        orchestrator.register_agent("ShopPurchaseHandler", ShopPurchaseAdvisorAgent())
        orchestrator.register_agent("CardRewardHandler", CardRewardAdvisorAgent())
        orchestrator.register_agent("MapHandler", MapAdvisorAgent())
        _event_orchestrator = orchestrator
    return _event_orchestrator


def get_battle_meta_advisor() -> BattleMetaAdvisorAgent:
    global _battle_meta_advisor
    if _battle_meta_advisor is None:
        _battle_meta_advisor = BattleMetaAdvisorAgent()
    return _battle_meta_advisor
