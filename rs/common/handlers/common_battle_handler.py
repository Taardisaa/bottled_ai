from dataclasses import dataclass, field

from rs.calculator.executor import get_best_battle_action
from rs.calculator.interfaces.comparator_interface import ComparatorInterface
from rs.common.comparators.big_fight_comparator import BigFightComparator
from rs.common.comparators.common_general_comparator import CommonGeneralComparator
from rs.common.comparators.gremlin_nob_comparator import GremlinNobComparator
from rs.common.comparators.three_sentry_comparator import ThreeSentriesComparator
from rs.common.comparators.three_sentry_turn_1_comparator import ThreeSentriesTurn1Comparator
from rs.common.comparators.transient_comparator import TransientComparator
from rs.common.comparators.waiting_lagavulin_comparator import WaitingLagavulinComparator
from rs.game.card import CardType
from rs.machine.command import Command
from rs.machine.handlers.handler import Handler
from rs.machine.handlers.handler_action import HandlerAction
from rs.machine.state import GameState


@dataclass
class BattleHandlerConfig:
    big_fight_floors: list[int] = field(default_factory=lambda: [33, 50])
    big_fight_comparator: ComparatorInterface = BigFightComparator
    gremlin_nob_comparator: ComparatorInterface = GremlinNobComparator
    three_sentries_comparator: ComparatorInterface = ThreeSentriesComparator
    three_sentries_turn_1_comparator: ComparatorInterface = ThreeSentriesTurn1Comparator
    transient_comparator: ComparatorInterface = TransientComparator
    waiting_lagavulin_comparator: ComparatorInterface = WaitingLagavulinComparator
    general_comparator: ComparatorInterface = CommonGeneralComparator


class CommonBattleHandler(Handler):

    def __init__(
            self,
            config: BattleHandlerConfig = BattleHandlerConfig(),
            max_path_count: int = 11_000,
    ):
        self.config: BattleHandlerConfig = config
        self.max_path_count: int = max_path_count

    def can_handle(self, state: GameState) -> bool:
        return state.has_command(Command.PLAY) \
               or state.current_action() == "DiscardAction" \
               or state.current_action() == "ExhaustAction"

    def select_comparator_profile_key(self, state: GameState) -> str:
        alive_monsters = len(list(filter(lambda m: not m["is_gone"], state.get_monsters())))

        big_fight = state.floor() in self.config.big_fight_floors

        gremlin_nob_is_present = state.has_monster("Gremlin Nob")

        three_sentries_are_alive_turn_1 = state.has_monster("Sentry") \
                                   and alive_monsters == 3 \
                                   and state.combat_state()['turn'] == 1

        three_sentries_are_alive = state.has_monster("Sentry") \
                                          and alive_monsters == 3

        lagavulin_is_sleeping = state.has_monster("Lagavulin") \
                                and state.combat_state()['turn'] <= 2 \
                                and not state.game_state()['room_type'] == "EventRoom"

        lagavulin_is_worth_delaying = state.deck.contains_type(CardType.POWER) \
                                      or state.deck.contains_cards(["Terror", "Terror+"]) \
                                      or state.deck.contains_cards(["Talk To The Hand", "Talk To The Hand+"]) \
                                      or state.has_relic("Warped Tongs") \
                                      or state.has_relic("Ice Cream")

        transient_is_present = state.has_monster("Transient") and alive_monsters == 1

        if big_fight:
            return "big_fight"
        if gremlin_nob_is_present:
            return "gremlin_nob"
        if three_sentries_are_alive_turn_1:
            return "three_sentries_turn_1"
        if three_sentries_are_alive:
            return "three_sentries"
        if lagavulin_is_sleeping and lagavulin_is_worth_delaying:
            return "waiting_lagavulin"
        if transient_is_present:
            return "transient"
        return "general"

    def get_available_comparator_profile_keys(self, state: GameState) -> list[str]:
        deterministic_profile = self.select_comparator_profile_key(state)
        if deterministic_profile == "big_fight":
            return ["big_fight", "general"]
        if deterministic_profile == "gremlin_nob":
            return ["gremlin_nob", "general"]
        if deterministic_profile == "three_sentries_turn_1":
            return ["three_sentries_turn_1", "three_sentries", "general"]
        if deterministic_profile == "three_sentries":
            return ["three_sentries", "general"]
        if deterministic_profile == "waiting_lagavulin":
            return ["waiting_lagavulin", "general"]
        if deterministic_profile == "transient":
            return ["transient", "general"]
        return ["general"]

    def _instantiate_comparator(self, profile_key: str) -> ComparatorInterface:
        profile_factories = {
            "big_fight": self.config.big_fight_comparator,
            "gremlin_nob": self.config.gremlin_nob_comparator,
            "three_sentries": self.config.three_sentries_comparator,
            "three_sentries_turn_1": self.config.three_sentries_turn_1_comparator,
            "transient": self.config.transient_comparator,
            "waiting_lagavulin": self.config.waiting_lagavulin_comparator,
            "general": self.config.general_comparator,
        }
        comparator_factory = profile_factories.get(profile_key, self.config.general_comparator)
        return comparator_factory()

    def select_comparator(self, state: GameState) -> ComparatorInterface:
        profile = self.select_comparator_profile_key(state)
        return self._instantiate_comparator(profile)

    def handle(self, state: GameState) -> HandlerAction:
        actions = get_best_battle_action(state, self.select_comparator(state), self.max_path_count)
        if actions:
            return actions
        if state.has_command(Command.END):
            return HandlerAction(commands=["end"])
        return HandlerAction(commands=[])
