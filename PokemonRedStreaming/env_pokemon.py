import os
from typing import Dict, Iterable, Optional, Set, Tuple

# Ensure SDL never attempts to output audio (disables in-game music/SFX).
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "PyBoy is required to run PokemonRedEnv. Install it with `pip install pyboy`."
    ) from exc


# Core WRAM addresses documented in the pret/pokered project.
# These values let us peek at the player's state without pausing emulation.
MAP_ID_ADDR = 0xD35E
PLAYER_Y_ADDR = 0xD361
PLAYER_X_ADDR = 0xD362
BATTLE_TYPE_ADDR = 0xD057  # Non-zero while in battle
BATTLE_RESULT_ADDR = 0xD059  # Encodes outcome once a battle ends
ENCOUNTER_SPECIES_ADDR = 0xD0B5
CHAMPION_FLAG_ADDR = 0xD163
BATTLE_TYPE_DETAIL_ADDR = 0xD057
ENEMY_HP_CURRENT_ADDR = 0xCFE6  # little-endian (lo, hi)
ENEMY_HP_MAX_ADDR = 0xCFE8  # little-endian (lo, hi)

# Bag and Pokédex metadata.
BAG_ITEMS_ADDR = 0xD31D  # alternating item id / quantity, terminated by 0xFF
MAX_BAG_SLOTS = 20
POKEDEX_OWNED_ADDR = 0xD2F7  # 19 bytes support all 151 species
POKEDEX_OWNED_BYTES = 19

# Lead Pokémon status/HP.
LEAD_STATUS_ADDR = 0xD16A
LEAD_HP_CURRENT_ADDR = 0xD16B  # little-endian (lo, hi)
LEAD_HP_MAX_ADDR = 0xD16D  # little-endian (lo, hi)

# Known overworld map ids the reward modules care about.
MAP_NAMES: Dict[int, str] = {
    0x00: "pallet town",
    0x01: "pallet player house 1f",
    0x02: "pallet player house 2f",
    0x03: "pallet rival house",
    0x04: "pallet oaks lab",
    0x05: "viridian city",
    0x06: "viridian gym",
    0x07: "viridian mart",
    0x08: "viridian school",
    0x09: "viridian house",
    0x0A: "route 1",
    0x0B: "route 2",
    0x0C: "viridian forest",
}

# Gym badge bitfield is stored at 0xD355 in Pokémon Red/Blue.
BADGE_FLAGS_ADDR = 0xD355
BADGE_NAMES = (
    "boulder",
    "cascade",
    "thunder",
    "rainbow",
    "soul",
    "marsh",
    "volcano",
    "earth",
)

# Mapping battle result values → readable labels.
BATTLE_RESULT_LABELS: Dict[int, str] = {
    0x00: "ongoing",
    0x01: "won",
    0x02: "lost",
    0x03: "blackout",
    0x04: "escaped",
    0x07: "caught",
}


def _downsample_frame(frame: np.ndarray) -> np.ndarray:
    """Reduce the native 160x144 RGB frame to 72x80 grayscale for lightweight networks."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    resized = cv2.resize(gray, (80, 72), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)[..., None]


class PokemonRedEnv(gym.Env):
    """Minimal PyBoy-powered environment focused on Route 1 catch attempts."""

    metadata = {"render_modes": ["human"]}

    BUTTON_EVENT_MAP = {
        "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
        "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
        "select": (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
        "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    }

    def __init__(
        self,
        rom_path: Optional[str] = None,
        *,
        show_display: bool = False,
        frame_skip: int = 4,
        boot_steps: int = 120,
        max_no_input_frames: int = 60,
        state_path: Optional[str] = None,
        story_flag_defs: Optional[Dict[str, Dict[str, object]]] = None,
        track_visit_stats: bool = True,
        delete_sav_on_reset: bool = True,
        input_spacing_frames: int = 0,
        emulation_speed: float | int = 1,
    ) -> None:
        super().__init__()
        self.rom_path = rom_path or os.path.join(os.path.dirname(__file__), "pokemon_red.gb")
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(
                f"Pokemon ROM not found at {self.rom_path}. "
                "Place pokemon_red.gb inside epsilon/pokemon_rl or pass rom_path explicitly."
            )

        self.frame_skip = max(1, int(frame_skip))
        self.show_display = bool(show_display)
        self.boot_steps = max(0, int(boot_steps))
        self.max_no_input_frames = max(1, int(max_no_input_frames))
        self.state_path = state_path
        self.story_flag_defs = self._normalise_story_defs(story_flag_defs)
        self.track_visit_stats = bool(track_visit_stats)
        self.delete_sav_on_reset = bool(delete_sav_on_reset)
        self.input_spacing_frames = max(0, int(input_spacing_frames))
        self.emulation_speed = max(0, int(round(emulation_speed)))

        self._pyboy: Optional[PyBoy] = None
        self._screen = None
        self._steps = 0
        self._frames_since_input = 0
        self._prev_in_battle = False
        self._last_battle_result: Optional[str] = None
        self._caught_flag = False
        self._last_raw_frame: Optional[np.ndarray] = None
        self._story_cache = {}
        self._global_visit_counts: Dict[Tuple[int, int], int] = {}
        self._episode_visits: Set[Tuple[int, int]] = set()
        self._episode_revisit_steps = 0

        self._button_groups: Tuple[Tuple[str, ...], ...] = (
            tuple(),  # NOOP
            ("a",),
            ("b",),
            ("start",),
            ("select",),
            ("up",),
            ("down",),
            ("left",),
            ("right",),
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(72, 80, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self._button_groups))

        self._initialize_emulator()

    @property
    def pyboy(self) -> PyBoy:
        assert self._pyboy is not None, "PyBoy has not been initialised."
        return self._pyboy

    def _initialize_emulator(self) -> None:
        """Create the PyBoy instance, optionally load a state, and warm up the ROM."""
        if self.delete_sav_on_reset:
            self._purge_save_file()
        window = "SDL2" if self.show_display else "null"
        try:
            self._pyboy = PyBoy(self.rom_path, window=window, sound=False)
        except TypeError:
            # Older PyBoy builds use window_type.
            self._pyboy = PyBoy(self.rom_path, window_type=window, sound=False)  # type: ignore[arg-type]  # pragma: no cover
        if not self.show_display:
            speed = 0
        else:
            speed = max(1, self.emulation_speed)
        self._pyboy.set_emulation_speed(speed)
        try:
            self._screen = self.pyboy.screen
        except AttributeError:
            # PyBoy >=1.6 exposes the screen via the botsupport manager instead.
            self._screen = self.pyboy.botsupport_manager().screen()

        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as fh:
                self.pyboy.load_state(fh)
        else:
            # Fast-forward the intro sequence to land on the overworld.
            for _ in range(self.boot_steps):
                self.pyboy.tick()
            self._press_and_tick(("start",), hold_frames=30)

        self._steps = 0
        self._frames_since_input = 0
        self._prev_in_battle = bool(self._read_u8(BATTLE_TYPE_ADDR))
        self._last_battle_result = None
        self._caught_flag = False
        self._last_raw_frame = None
        self._last_bag_snapshot: list[tuple[int, int]] = []
        self._reset_episode_visit_stats()

    def _press_and_tick(
        self,
        buttons: Iterable[str],
        *,
        hold_frames: int = 1,
    ) -> None:
        """Helper to press a group of buttons, tick the emulator, then release."""
        pressed = tuple(buttons)
        if pressed:
            for button in pressed:
                self._press_button(button)
        for _ in range(max(1, hold_frames)):
            self.pyboy.tick()
        if pressed:
            for button in pressed[::-1]:
                self._release_button(button)

    def _press_button(self, button: str) -> None:
        if hasattr(self.pyboy, "button_press"):
            self.pyboy.button_press(button)
            return
        press_event, _ = self.BUTTON_EVENT_MAP[button]
        self.pyboy.send_input(press_event)

    def _release_button(self, button: str) -> None:
        if hasattr(self.pyboy, "button_release"):
            self.pyboy.button_release(button)
            return
        _, release_event = self.BUTTON_EVENT_MAP[button]
        self.pyboy.send_input(release_event)

    def _read_u8(self, address: int) -> int:
        try:
            return int(self.pyboy.memory[address])
        except Exception:
            return 0

    @staticmethod
    def _normalise_story_defs(defs: Optional[Dict[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
        if not defs:
            return {}
        normalised: Dict[str, Dict[str, object]] = {}
        for name, cfg in defs.items():
            if not isinstance(cfg, dict):
                continue
            addr = cfg.get("address")
            bit = cfg.get("bit", 0)
            reward = cfg.get("reward") if isinstance(cfg.get("reward"), (int, float)) else None
            try:
                if isinstance(addr, str):
                    addr_int = int(addr, 16) if addr.lower().startswith("0x") else int(addr)
                elif isinstance(addr, (int, float)):
                    addr_int = int(addr)
                else:
                    continue
                bit_int = max(0, int(bit))
            except Exception:
                continue
            normalised[name] = {
                "address": addr_int,
                "bit": bit_int,
            }
            if reward is not None:
                normalised[name]["reward"] = float(reward)
        return normalised

    def _purge_save_file(self) -> None:
        sav_path = os.path.splitext(self.rom_path)[0] + ".sav"
        if os.path.exists(sav_path):
            try:
                os.remove(sav_path)
            except OSError:
                pass

    def _reset_episode_visit_stats(self) -> None:
        self._episode_visits.clear()
        self._episode_revisit_steps = 0
        if not self.track_visit_stats:
            self._global_visit_counts.clear()

    def _record_visit_info(self, coord: Tuple[int, int]) -> Dict[str, object]:
        if not self.track_visit_stats:
            return {
                "episode_unique_tiles": 0,
                "total_unique_tiles": 0,
                "tile_visit_count": 0,
                "episode_revisit_ratio": 0.0,
            }

        was_seen = coord in self._episode_visits
        if not was_seen:
            self._episode_visits.add(coord)
        else:
            self._episode_revisit_steps += 1

        visit_total = self._global_visit_counts.get(coord, 0) + 1
        self._global_visit_counts[coord] = visit_total

        episode_unique = len(self._episode_visits)
        total_unique = len(self._global_visit_counts)
        revisit_ratio = self._episode_revisit_steps / max(1, self._steps)
        return {
            "episode_unique_tiles": episode_unique,
            "total_unique_tiles": total_unique,
            "tile_visit_count": visit_total,
            "episode_revisit_ratio": revisit_ratio,
        }

    def _gather_info(self) -> Dict[str, object]:
        map_id = self._read_u8(MAP_ID_ADDR)
        map_name = MAP_NAMES.get(map_id)
        if map_name is None:
            map_name = f"map_{map_id:02X}"

        info: Dict[str, object] = {
            "map_id": map_id,
            "map_name": map_name,
            "agent_coords": (
                self._read_u8(PLAYER_X_ADDR),
                self._read_u8(PLAYER_Y_ADDR),
            ),
            "steps": self._steps,
        }

        badge_mask = self._read_u8(BADGE_FLAGS_ADDR)
        badges = {name: bool(badge_mask & (1 << idx)) for idx, name in enumerate(BADGE_NAMES)}
        info["badge_bits"] = badge_mask
        info["badge_count"] = sum(badges.values())
        info["badges"] = badges

        champion_defeated = bool(self._read_u8(CHAMPION_FLAG_ADDR))
        info["champion_defeated"] = champion_defeated

        if self.story_flag_defs:
            story_flags = {}
            for name, cfg in self.story_flag_defs.items():
                addr = cfg["address"]
                bit = cfg["bit"]
                value = self._read_u8(addr)
                story_flags[name] = bool((value >> bit) & 1)
            info["story_flags"] = story_flags

        in_battle = self._read_u8(BATTLE_TYPE_ADDR) != 0
        info["in_battle"] = in_battle

        battle_code = self._read_u8(BATTLE_RESULT_ADDR)
        info["battle_result_code"] = battle_code
        battle_label = BATTLE_RESULT_LABELS.get(battle_code)
        info["battle_result"] = battle_label
        info["encounter_species"] = self._read_u8(ENCOUNTER_SPECIES_ADDR)
        info["battle_type"] = self._read_u8(BATTLE_TYPE_DETAIL_ADDR)

        caught_now = False
        battle_transition = self._prev_in_battle and not in_battle
        if battle_transition:
            if battle_label == "caught":
                caught_now = True
                self._caught_flag = True
            self._last_battle_result = battle_label
        else:
            # Reset catch flag once we leave the battle resolution screen.
            if not in_battle and self._caught_flag:
                self._caught_flag = False
        info["caught_pokemon"] = caught_now
        info["last_battle_result"] = self._last_battle_result
        info["raw_frame"] = (
            np.array(self._last_raw_frame, copy=True) if self._last_raw_frame is not None else None
        )
        bag_items = self._read_bag_items()
        info["bag_items"] = bag_items
        info["bag_item_ids"] = [item_id for item_id, _ in bag_items]
        info["key_item_ids"] = self._infer_key_items(bag_items)
        pokedex_bits = self._read_pokedex_owned_bits()
        info["pokedex_owned_bits"] = pokedex_bits
        info["pokedex_owned_count"] = self._count_bits(pokedex_bits)
        info["first_pokemon_hp"] = self._read_lead_pokemon_hp()
        info["enemy_hp"] = self._read_enemy_hp()
        coords_int = (int(info["agent_coords"][0]), int(info["agent_coords"][1]))
        info.update(self._record_visit_info(coords_int))
        self._prev_in_battle = in_battle
        return info

    def _obs(self) -> np.ndarray:
        assert self._screen is not None
        screen = self._screen
        if hasattr(screen, "ndarray"):
            frame = screen.ndarray
        elif hasattr(screen, "screen_ndarray"):
            frame = screen.screen_ndarray()
        elif hasattr(screen, "buffer"):
            frame = screen.buffer
        elif hasattr(screen, "screen_image"):
            frame = np.array(screen.screen_image())
        else:  # pragma: no cover - defensive fallback for unknown PyBoy builds
            raise AttributeError("PyBoy screen does not expose ndarray-compatible data")
        self._last_raw_frame = np.array(frame, copy=True)
        return _downsample_frame(self._last_raw_frame)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        # Always perform a hard reset since reset_game() does not exist in PyBoy
        self.close()
        self._initialize_emulator()

        self._steps = 0
        self._frames_since_input = 0
        self._prev_in_battle = False
        self._last_battle_result = None
        self._caught_flag = False
        self._last_raw_frame = None
        obs = self._obs()
        info = self._gather_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action index {action}"
        buttons = self._button_groups[action]
        hold_frames = self.frame_skip
        self._press_and_tick(buttons, hold_frames=hold_frames)

        self._steps += 1
        self._frames_since_input = 0 if buttons else self._frames_since_input + hold_frames

        if self.input_spacing_frames > 0:
            self._press_and_tick(tuple(), hold_frames=self.input_spacing_frames)
            self._frames_since_input += self.input_spacing_frames

        obs = self._obs()
        info = self._gather_info()

        # End the episode only when the first Pokemon is caught,
        # as requested. Do not terminate on escape/loss.
        terminated = bool(info.get("caught_pokemon"))

        truncated = self._frames_since_input >= self.max_no_input_frames

        # Environment reward is always neutral; auxiliary modules supply shaped reward.
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def render(self):
        # PyBoy handles window updates internally when show_display=True.
        return None

    def close(self):
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None
        self._screen = None
        self._last_bag_snapshot = []

    def _read_bag_items(self) -> list[tuple[int, int]]:
        items: list[tuple[int, int]] = []
        try:
            for slot in range(MAX_BAG_SLOTS):
                item_addr = BAG_ITEMS_ADDR + slot * 2
                item_id = self._read_u8(item_addr)
                if item_id == 0xFF:
                    break
                quantity = self._read_u8(item_addr + 1)
                if item_id == 0:
                    continue
                items.append((item_id, quantity))
        except Exception:
            pass
        return items

    @staticmethod
    def _infer_key_items(bag_items: list[tuple[int, int]]) -> list[int]:
        key_items: list[int] = []
        for item_id, quantity in bag_items:
            if quantity == 1 and item_id not in key_items:
                key_items.append(item_id)
        return key_items

    def _read_pokedex_owned_bits(self) -> list[int]:
        data: list[int] = []
        try:
            for offset in range(POKEDEX_OWNED_BYTES):
                data.append(self._read_u8(POKEDEX_OWNED_ADDR + offset))
        except Exception:
            pass
        return data

    @staticmethod
    def _count_bits(bytes_list: list[int]) -> int:
        return sum(bin(byte & 0xFF).count("1") for byte in bytes_list)

    def _read_lead_pokemon_hp(self) -> Dict[str, int]:
        try:
            current = self._read_u8(LEAD_HP_CURRENT_ADDR) | (self._read_u8(LEAD_HP_CURRENT_ADDR + 1) << 8)
            maximum = self._read_u8(LEAD_HP_MAX_ADDR) | (self._read_u8(LEAD_HP_MAX_ADDR + 1) << 8)
            status = self._read_u8(LEAD_STATUS_ADDR)
            return {"current": current, "max": maximum, "status": status}
        except Exception:
            return {"current": 0, "max": 0, "status": 0}
        self._last_raw_frame = None

    def _read_enemy_hp(self) -> Dict[str, int]:
        try:
            current = self._read_u8(ENEMY_HP_CURRENT_ADDR) | (self._read_u8(ENEMY_HP_CURRENT_ADDR + 1) << 8)
            maximum = self._read_u8(ENEMY_HP_MAX_ADDR) | (self._read_u8(ENEMY_HP_MAX_ADDR + 1) << 8)
            return {"current": current, "max": maximum}
        except Exception:
            return {"current": 0, "max": 0}
