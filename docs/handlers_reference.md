# Handler Reference

This document explains what each handler does in Bottled AI.

## How handlers are used

- A handler implements `can_handle(state)` and `handle(state) -> HandlerAction` (`rs/machine/handlers/handler.py`).
- The game loop checks handlers in order; first match wins (`rs/machine/game.py:69`).
- Runtime order is: strategy handlers first, then default handlers (`rs/machine/game.py:21`).

## Machine-level fallback handlers

- `DefaultGameOverHandler` (`rs/machine/default_game_over.py`): handles game-over screen, logs run/missing enum stats, sends `proceed`.
- `DefaultLeaveHandler` (`rs/machine/handlers/default_leave.py`): fallback for `leave` command; returns `return`.
- `DefaultShopHandler` (`rs/machine/handlers/default_shop.py`): enters/leaves basic shop-room flow with `proceed`.
- `DefaultChooseHandler` (`rs/machine/handlers/default_choose.py`): generic fallback choice (`choose 0`), with potion discard fallback if slots are full.
- `DefaultConfirmHandler` (`rs/machine/handlers/default_confirm.py`): generic `confirm/proceed` fallback.
- `DefaultPlayHandler` (`rs/machine/handlers/default_play.py`): generic in-combat card play (first affordable card, first alive target if needed).
- `DefaultEndHandler` (`rs/machine/handlers/default_end.py`): generic combat turn-end (`end`).
- `DefaultCancelHandler` (`rs/machine/handlers/default_cancel.py`): generic cancel/return action.
- `DefaultWaitHandler` (`rs/machine/handlers/default_wait.py`): generic wait fallback (`wait 30`).

## Common handlers (shared strategy building blocks)

- `CommonBattleHandler` (`rs/common/handlers/common_battle_handler.py`): battle decision engine entrypoint; calls calculator and chooses comparator per fight context.
- `CommonMapHandler` (`rs/common/handlers/common_map_handler.py`): map pathing using reward/survivability scoring.
- `CommonEventHandler` (`rs/common/handlers/common_event_handler.py`): large event decision policy table by event name.
- `CommonCombatRewardHandler` (`rs/common/handlers/common_combat_reward_handler.py`): reward pickup logic (gold/relic/potion/card), potion juggling, and proceed behavior.
- `CommonBossRelicHandler` (`rs/common/handlers/common_boss_relic_handler.py`): boss relic prioritization with anti-synergy guards (for example, Snecko + Pyramid).
- `CommonCampfireHandler` (`rs/common/handlers/common_campfire_handler.py`): rest/smith/toke/dig/lift decision logic based on hp/relics/deck.
- `CommonChestHandler` (`rs/common/handlers/common_chest_handler.py`): chest open/skip logic, including Cursed Key + Omamori handling.
- `CommonNeowHandler` (`rs/common/handlers/common_neow_handler.py`): Neow bonus selection priority list.
- `CommonPurgeHandler` (`rs/common/handlers/common_purge_handler.py`): card removal target selection in purge grids.
- `CommonTransformHandler` (`rs/common/handlers/common_transform_handler.py`): card transform selection in transform grids.
- `CommonUpgradeHandler` (`rs/common/handlers/common_upgrade_handler.py`): upgrade-grid selection with strategy hook for dynamic priorities.
- `CommonAstrolabeHandler` (`rs/common/handlers/common_astrolabe_handler.py`): handles 3-card transform selection (Astrolabe-like flows).
- `CommonMassDiscardHandler` (`rs/common/handlers/common_mass_discard_handler.py`): hand-select mass discard flows (Gambling Chip/Gambler's Brew style).
- `CommonScryHandler` (`rs/common/handlers/common_scry_handler.py`): scry filtering and confirm sequence.
- `CommonShopEntranceHandler` (`rs/common/handlers/common_shop_entrance_handler.py`): enters shop screen from shop room.
- `CommonGridSelectHandler` (`rs/common/handlers/common_grid_select_handler.py`): generic GRID multi-select + confirm handler (for screens not covered by transform/purge handlers).

## Common card-reward handler variants

- `CommonCardRewardHandler` (`rs/common/handlers/card_reward/common_card_reward_handler.py`): per-card desired-map logic with max-count caps.
- `CommonGroupedCardRewardHandler` (`rs/common/handlers/card_reward/common_grouped_card_reward_handler.py`): grouped card-priority logic with group-level capacity.
- `CommonCardRewardTakeFirstCardHandler` (`rs/common/handlers/card_reward/common_card_reward_take_first_card_handler.py`): always picks first card (debug/experimentation helper).

## Strategy-specific handlers by strategy

These usually extend common handlers and customize preferences, edge cases, or comparator wiring.

### `_example`

- `EventHandler` (`rs/ai/_example/handlers/event_handler.py`): sample custom event override on top of `CommonEventHandler`.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/_example/handlers/potions_handler.py`): sample potion-use policy by fight type.
- `ShopPurchaseHandler` (`rs/ai/_example/handlers/shop_purchase_handler.py`): sample shop buy/purge priority logic.
- `UpgradeHandler` (`rs/ai/_example/handlers/upgrade_handler.py`): sample upgrade priority customization.

### `claw_is_law`

- `BossRelicHandler` (`rs/ai/claw_is_law/handlers/boss_relic_handler.py`): Claw strategy boss relic preferences.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/claw_is_law/handlers/potions_handler.py`): Claw potion timing policy.
- `ShopPurchaseHandler` (`rs/ai/claw_is_law/handlers/shop_purchase_handler.py`): Claw shop buying priorities.
- `UpgradeHandler` (`rs/ai/claw_is_law/handlers/upgrade_handler.py`): Claw upgrade order tuning.

### `peaceful_pummeling`

- `BossRelicHandler` (`rs/ai/peaceful_pummeling/handlers/boss_relic_handler.py`): Watcher-specific boss relic tuning.
- `CardRewardHandler` (`rs/ai/peaceful_pummeling/handlers/card_reward_handler.py`): Watcher card reward preferences with state-based adjustments.
- `EventHandler` (`rs/ai/peaceful_pummeling/handlers/event_handler.py`): Watcher event overrides.
- `NeowHandler` (`rs/ai/peaceful_pummeling/handlers/neow_handler.py`): Watcher Neow preference override.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/peaceful_pummeling/handlers/potions_handler.py`): Watcher potion policy.
- `ShopPurchaseHandler` (`rs/ai/peaceful_pummeling/handlers/shop_purchase_handler.py`): Watcher shop purchase strategy.
- `UpgradeHandler` (`rs/ai/peaceful_pummeling/handlers/upgrade_handler.py`): Watcher upgrade priorities.

### `pwnder_my_orbs`

- `get_pmo_battle_handler` (`rs/ai/pwnder_my_orbs/handlers/battle_handler.py`): factory that builds `CommonBattleHandler` with Defect-specific comparators/floor config.
- `BossRelicHandler` (`rs/ai/pwnder_my_orbs/handlers/boss_relic_handler.py`): Defect boss relic tuning.
- `CardRewardHandler` (`rs/ai/pwnder_my_orbs/handlers/card_reward_handler.py`): grouped Defect card-reward priorities.
- `EventHandler` (`rs/ai/pwnder_my_orbs/handlers/event_handler.py`): Defect event overrides.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/pwnder_my_orbs/handlers/potions_handler.py`): Defect potion policy.
- `ShopPurchaseHandler` (`rs/ai/pwnder_my_orbs/handlers/shop_purchase_handler.py`): Defect shop strategy.
- `UpgradeHandler` (`rs/ai/pwnder_my_orbs/handlers/upgrade_handler.py`): Defect upgrade priorities.

### `requested_strike`

- `BossRelicHandler` (`rs/ai/requested_strike/handlers/boss_relic_handler.py`): Ironclad boss relic tuning.
- `EventHandler` (`rs/ai/requested_strike/handlers/event_handler.py`): Ironclad event overrides.
- `NeowHandler` (`rs/ai/requested_strike/handlers/neow_handler.py`): Ironclad Neow override.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/requested_strike/handlers/potions_handler.py`): Ironclad potion policy.
- `ShopPurchaseHandler` (`rs/ai/requested_strike/handlers/shop_purchase_handler.py`): Ironclad shop strategy.
- `UpgradeHandler` (`rs/ai/requested_strike/handlers/upgrade_handler.py`): Ironclad upgrade priorities.

### `shivs_and_giggles`

- `BossRelicHandler` (`rs/ai/shivs_and_giggles/handlers/boss_relic_handler.py`): Silent boss relic tuning.
- `CardRewardHandler` (`rs/ai/shivs_and_giggles/handlers/card_reward_handler.py`): Silent card reward adjustments.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/shivs_and_giggles/handlers/potions_handler.py`): Silent potion policy.
- `ShopPurchaseHandler` (`rs/ai/shivs_and_giggles/handlers/shop_purchase_handler.py`): Silent shop strategy.
- `SmartPathHandler` (`rs/ai/shivs_and_giggles/handlers/smart_path_handler.py`): strategy-specific map pathing variant.
- `UpgradeHandler` (`rs/ai/shivs_and_giggles/handlers/upgrade_handler.py`): Silent upgrade priorities.

### `smart_agent`

- `EventHandler` (`rs/ai/smart_agent/handlers/event_handler.py`): event override skeleton copied from example.
- `PotionsBaseHandler`, `PotionsEliteHandler`, `PotionsEventFightHandler`, `PotionsBossHandler` (`rs/ai/smart_agent/handlers/potions_handler.py`): potion policy skeleton.
- `ShopPurchaseHandler` (`rs/ai/smart_agent/handlers/shop_purchase_handler.py`): shop-policy skeleton.
- `UpgradeHandler` (`rs/ai/smart_agent/handlers/upgrade_handler.py`): upgrade-priority skeleton.

## Practical note

- If you are adding a new strategy, start by composing common handlers first.
- Add strategy-specific handlers only when common behavior is insufficient.
- Keep in mind ordering is behavior: earlier handlers preempt later ones.
