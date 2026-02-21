from typing import Any, Dict, List, Tuple

from rs.calculator.interfaces.memory_items import MemoryItem
from rs.game.deck import Deck
from rs.game.event import Event
from rs.machine.command import Command
from rs.machine.orb import Orb
from rs.machine.the_bots_memory_book import TheBotsMemoryBook


class GameState:
    def __init__(self, 
            json_state: Dict[str, Any], 
            the_bots_memory_book: TheBotsMemoryBook
        ):
        self.the_bots_memory_book: TheBotsMemoryBook = the_bots_memory_book
        self.json: Dict[str, Any] = json_state
        if "game_state" in json_state:
            if "combat_state" in json_state["game_state"]:
                self.hand: Deck = Deck(json_state["game_state"]["combat_state"]["hand"])
                self.draw_pile: Deck = Deck(json_state["game_state"]["combat_state"]["draw_pile"])
                self.discard_pile: Deck = Deck(json_state["game_state"]["combat_state"]["discard_pile"])
                self.exhaust_pile: Deck = Deck(json_state["game_state"]["combat_state"]["exhaust_pile"])

                current_turn = json_state["game_state"]["combat_state"]["turn"]
                if self.the_bots_memory_book.memory_general[MemoryItem.LAST_KNOWN_TURN] != current_turn:
                    self.the_bots_memory_book.set_new_turn_state()
                self.the_bots_memory_book.memory_general[MemoryItem.LAST_KNOWN_TURN] = current_turn

            else:
                self.the_bots_memory_book.set_new_battle_state()

            self.deck: Deck = Deck(json_state["game_state"]["deck"])
            self.memory_by_card = self.the_bots_memory_book.memory_by_card.copy()
            self.memory_general = self.the_bots_memory_book.memory_general.copy()

    def is_game_running(self) -> bool:
        return self.json["in_game"]

    def game_state(self) -> Dict[str, Any]:
        return self.json["game_state"]

    def combat_state(self) -> Dict[str, Any] | None:
        if 'combat_state' in self.game_state():
            return self.game_state()["combat_state"]
        else:
            return None

    def has_command(self, command: Command) -> bool:
        available_commands = self.json.get("available_commands")
        if not isinstance(available_commands, list):
            return False
        return command.value in available_commands

    def get_player_combat(self) -> Dict[str, Any]:
        return self.game_state()["combat_state"]["player"]

    def get_player_health_percentage(self) -> float:
        return self.game_state()["current_hp"] / self.game_state()["max_hp"]

    def get_monsters(self) -> List[Dict[str, Any]]:
        if "combat_state" not in self.game_state():
            return []
        return self.game_state()["combat_state"]["monsters"]

    def get_choice_list(self) -> List[str]:
        return self.game_state()["choice_list"]

    def get_choice_list_upgrade_stripped_from_choice(self) -> List[str]:
        choice_list_modified = self.get_choice_list().copy()
        for idx, choice in enumerate(choice_list_modified):
            choice_list_modified[idx] = choice.replace("+", "")
        return choice_list_modified

    def get_relics(self) -> List[Dict[str, Any]]:
        return self.game_state()["relics"]

    def has_relic(self, relic_name: str) -> bool:
        for relic in self.get_relics():
            if relic['name'].lower() == relic_name.lower():
                return True
        return False

    def get_relic_counter(self, relic_name: str) -> int:
        for relic in self.get_relics():
            if relic['name'] == relic_name:
                return relic['counter']
        return 0

    def get_potions(self) -> List[Dict[str, Any]]:
        return self.game_state()["potions"]

    def get_held_potion_names(self) -> List[str]:
        potion_names: List[str] = []
        for pot in self.game_state()["potions"]:
            potion_names.append(pot["name"])
        potion_names = [potion_name.lower() for potion_name in potion_names]
        return potion_names

    def get_reward_potion_names(self) -> List[str]:
        potion_names: List[str] = []
        for reward in self.game_state()["screen_state"]["rewards"]:
            if reward["reward_type"] == "POTION":
                potion_names.append(reward["potion"]["name"])
        potion_names = [potion_name.lower() for potion_name in potion_names]
        return potion_names

    def are_potions_full(self) -> bool:
        for pot in self.get_potions():
            if pot['id'] == "Potion Slot":
                return False
        return True

    def screen_type(self) -> str:
        return self.game_state()["screen_type"]

    def screen_state(self) -> Dict[str, Any]:
        return self.game_state()["screen_state"]

    def screen_state_max_cards(self) -> int:
        state = self.screen_state()
        return 0 if not state else state["max_cards"]

    def screen_state_must_pick_card(self) -> bool:
        state = self.screen_state()
        return False if not state else state["can_pick_zero"]

    def screen_state_exhaust_cards(self) -> int:
        return 0 if not self.current_action() == "ExhaustAction" else self.screen_state_max_cards()

    def screen_state_discard_cards(self) -> int:
        return 0 if not self.current_action() == "DiscardAction" else self.screen_state_max_cards()

    def current_action(self) -> str | None:
        if self.game_state()["screen_type"] == "HAND_SELECT" or \
                (self.combat_state() is not None and self.game_state()["screen_type"] == "GRID"):
            return self.game_state()["current_action"]
        return None

    def get_cards_discarded_this_turn(self) -> int:
        state = self.combat_state()
        return 0 if not state else state["cards_discarded_this_turn"]

    def floor(self) -> int:
        return self.game_state()["floor"]

    def player_entangled(self) -> bool:
        return bool(next((p for p in self.get_player_combat()["powers"] if p["id"] == "Entangled"), None))

    def get_deck_card_list_by_id(self) -> dict[str, int]:
        cards = {}
        for card in self.deck.cards:
            card_id = card.id.lower()
            if card_id in cards:
                cards[card_id] += 1
            else:
                cards[card_id] = 1
        return cards

    def get_deck_card_list_by_name_with_upgrade_stripped(self) -> dict[str, int]:
        cards = {}
        for card in self.deck.cards:
            name = card.name.replace("+", "")
            name = name.lower()
            if name in cards:
                cards[name] += 1
            else:
                cards[name] = 1
        return cards

    def get_map(self) -> List[Dict[str, Any]]:
        return self.game_state()["map"]

    def has_monster(self, name: str) -> bool:
        for monster in self.get_monsters():
            if monster['name'] == name:
                return True
        return False

    def get_player_block(self) -> int:
        return self.get_player_combat()['block']

    def get_player_orbs(self) -> List[Tuple[Orb, int]]:
        orbs = self.get_player_combat()['orbs']
        if not orbs:
            return []
        return [
            (Orb(o['id']), int(o['evoke_amount']))
            for o in orbs
            if 'id' in o and o['id'] != 'Empty' and 'evoke_amount' in o
        ]

    def get_player_orb_slots(self) -> int:
        orbs = self.get_player_combat()['orbs']
        if not orbs:
            return 0
        return len(orbs)

    def get_falling_event_options(self) -> List[str]:
        options: List[str] = []

        def extract_card_from_text(text: str) -> str | None:
            keyword = "Lose"
            if keyword in text:
                return text.split(keyword, 1)[1].strip()
            return None

        for option in self.screen_state()["options"]:
            if not option["disabled"]:
                text = str(option["text"])
                extracted_card = extract_card_from_text(text)
                if extracted_card is not None:
                    options.append(extracted_card.lower())
        for idx, choice in enumerate(options):
            options[idx] = choice.replace("+", "")
        return options

    def get_event(self) -> Event | str:
        event_name = self.game_state()['screen_state']['event_name']
        possible_events = set(item.value for item in Event)

        if event_name not in possible_events:
            return event_name
        return Event(event_name)
