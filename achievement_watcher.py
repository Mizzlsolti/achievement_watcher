# -*- coding: utf-8 -*-

from __future__ import annotations


import random
import subprocess
import hashlib
import os, sys, time, json, re, glob, threading, zipfile
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter


# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QTextBrowser, QSystemTrayIcon, QMenu, QFileDialog, QMessageBox, QTabWidget,
    QCheckBox, QSlider, QComboBox, QDialog, QGroupBox, QColorDialog, QLineEdit,
    QFontComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import (Qt, pyqtSignal, QEvent, QTimer, QRect,
                          QAbstractNativeEventFilter, QCoreApplication, QObject, QPoint)
from PyQt6.QtGui import (QIcon, QColor, QFont, QTransform, QPixmap,
                         QPainter, QImage, QPen)


try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
try:
    import requests
except Exception:
    requests = None
try:
    import win32gui
except Exception:
    win32gui = None
try:
    import olefile
except Exception:
    olefile = None

# RawInput / Joystick (Windows)
import ctypes
from ctypes import wintypes
_winmm = ctypes.WinDLL("winmm", use_last_error=True)
_user2 = ctypes.WinDLL("user32", use_last_error=True)


# Füge diese Imports nach den PyQt-/ctypes-Imports ein:
import ssl
from urllib.request import Request, urlopen

# … oben im File (nach Imports) einfügen:
def resource_path(rel: str) -> str:
    """
    Robust path resolver für PyInstaller onefile:
    - Im Bundle (sys._MEIPASS) suchen
    - Sonst im App-Ordner (APP_DIR)
    """
    base = getattr(sys, "_MEIPASS", None)
    if base and os.path.isdir(base):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return os.path.join(APP_DIR, rel)

def _fetch_json_url(url: str, timeout: int = 25) -> dict:
    """
    Fetch JSON from URL using requests if available, otherwise urllib.
    Raises on HTTP or parse errors.
    """
    ua = "AchievementWatcher/1.0 (+https://github.com/Mizzlsolti)"
    if requests:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": ua})
        r.raise_for_status()
        return r.json()
    req = Request(url, headers={"User-Agent": ua})
    ctx = ssl.create_default_context()
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return json.loads(raw)

def _fetch_bytes_url(url: str, timeout: int = 25) -> bytes:
    """
    Fetch bytes from URL using requests if available, otherwise urllib.
    Raises on HTTP errors.
    """
    ua = "AchievementWatcher/1.0 (+https://github.com/Mizzlsolti)"
    if requests:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": ua})
        r.raise_for_status()
        return r.content
    req = Request(url, headers={"User-Agent": ua})
    ctx = ssl.create_default_context()
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        return resp.read()

class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("dwXpos", wintypes.DWORD), ("dwYpos", wintypes.DWORD), ("dwZpos", wintypes.DWORD),
        ("dwRpos", wintypes.DWORD), ("dwUpos", wintypes.DWORD), ("dwVpos", wintypes.DWORD),
        ("dwButtons", wintypes.DWORD), ("dwButtonNumber", wintypes.DWORD),
        ("dwPOV", wintypes.DWORD), ("dwReserved1", wintypes.DWORD), ("dwReserved2", wintypes.DWORD),
    ]
JOY_RETURNALL = 0x000000FF
JOYERR_NOERROR = 0
_joyGetPosEx = _winmm.joyGetPosEx
_joyGetPosEx.argtypes = [wintypes.UINT, ctypes.POINTER(JOYINFOEX)]
_joyGetPosEx.restype = wintypes.UINT


RIDEV_INPUTSINK = 0x00000100
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_HOTKEY = 0x0312 

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.c_ushort),
        ("usUsage", ctypes.c_ushort),
        ("dwFlags", ctypes.c_uint),
        ("hwndTarget", wintypes.HWND),
    ]
_RegisterRawInputDevices = _user2.RegisterRawInputDevices
_RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), ctypes.c_uint, ctypes.c_uint]
_RegisterRawInputDevices.restype = wintypes.BOOL

_MapVirtualKeyW = _user2.MapVirtualKeyW
_MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
_MapVirtualKeyW.restype = wintypes.UINT
_GetKeyNameTextW = _user2.GetKeyNameTextW
_GetKeyNameTextW.argtypes = [wintypes.LONG, ctypes.c_wchar_p, ctypes.c_int]
_GetKeyNameTextW.restype = ctypes.c_int

def vk_to_name(vk: int) -> str:
    try:
        sc = _MapVirtualKeyW(vk, 0)
        lparam = (sc << 16)
        buf = ctypes.create_unicode_buffer(64)
        if _GetKeyNameTextW(lparam, buf, 64) > 0:
            return buf.value
    except Exception:
        pass
    return f"VK 0x{vk:02X}"

def register_raw_input_for_window(hwnd: int) -> bool:
    devices = (RAWINPUTDEVICE * 3)(
        RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK, hwnd),
        RAWINPUTDEVICE(0x01, 0x04, RIDEV_INPUTSINK, hwnd),
        RAWINPUTDEVICE(0x01, 0x05, RIDEV_INPUTSINK, hwnd),
    )
    ok = _RegisterRawInputDevices(devices, 3, ctypes.sizeof(RAWINPUTDEVICE))
    return bool(ok)


APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
CONFIG_FILE = os.path.join(APP_DIR, "config.json")


DEFAULT_OVERLAY = {
    "scale_pct": 100,
    "background": "auto",
    "portrait_mode": False,
    "portrait_rotate_ccw": False,
    "lines_per_category": 5,
    "toggle_input_source": "keyboard",
    "toggle_vk": 120,
    "toggle_joy_button": 2,
    "title_color": "#FFFFFF",
    "highlight_color": "#FFFFFF",
    "player1_color": "#00B050",
    "player2_color": "#00B050",
    "player3_color": "#00B050",
    "player4_color": "#00B050",
    "font_family": "Segoe UI",
    "base_title_size": 36,
    "base_body_size": 20,
    "base_hint_size": 16,
    "use_xy": False,
    "pos_x": 100,
    "pos_y": 100,

    "prefer_ascii_icons": False,
    "auto_show_on_end": True,
    "live_updates": False,

    # CPU‑Sim Settings
    "cpu_sim_active": True,
    "cpu_sim_ai": True,
    "cpu_sim_correlated": True,

    # NEU: CPU-Simulationsmodus ("live" | "postgame")
    "cpu_sim_mode": "postgame",
    # NEU: CPU Postgame Monte-Carlo Rollouts
    "cpu_postgame_rollouts": 120,
}


DEFAULT_OVERLAY.setdefault("automatic_creation", True)  # opt-out switch (default ON)

EXCLUDED_FIELDS = {
    "Last Game Start", "Last Printout", "Last Replay", "Champion Reset", "Clock Last Set", "Coins Cleared",
    "Factory Setting", "Recent Paid Cred", "Recent Serv. Cred", "Burn-in Time", "Totals Cleared", "Audits Cleared",
    "Play Time", "R. Universe Won", "Last Serv. Cred"
}
# NEU: lowercased Spiegelmenge
EXCLUDED_FIELDS_LC = {s.lower() for s in EXCLUDED_FIELDS}

def is_excluded_field(label: str) -> bool:
    """
    Case-insensitive Exclude-Filter für offensichtliche Nicht-Gameplay-/Verwaltungsfelder.
    Schließt 'Last Printout'/'Last Replay' zuverlässig aus (egal in welcher Schreibweise).
    """
    ll = str(label or "").strip().lower()
    return (
        ll in EXCLUDED_FIELDS_LC or
        "reset" in ll or
        "cleared" in ll or
        "factory" in ll or
        "timestamp" in ll or
        # 'last' Felder, die kein Gameplay sind (Printout/Replay) immer raus:
        ("last" in ll and ("printout" in ll or "replay" in ll)) or
        # Alte Heuristik: 'Last' + 'Game'
        ("last" in ll and "game" in ll)
    )
 

@dataclass
class AppConfig:
    BASE: str = r"C:\vPinball\Achievements"
    NVRAM_DIR: str = r"C:\vPinball\VisualPinball\VPinMAME\nvram"
    TABLES_DIR: str = r"C:\vPinball\VisualPinball\Tables"
    OVERLAY: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OVERLAY))
    FIRST_RUN: bool = True
    SIMPLE_END_SNAPSHOT_ONLY: bool = True
    HOOK_AUTO_SETUP: bool = True
    HOOK_BIN_URL_BASE: str = "https://github.com/Mizzlsolti/WatcherInjector/releases/latest/download"
    BOOTSTRAP_USE_CB: bool = False
    LOG_CTRL: bool = False

    # Injector/DLL on/off
    HOOK_ENABLE: bool = True

    # NEU: Muster für Log-Unterdrückung (Substrings), leere Liste => Defaults aus DEFAULT_LOG_SUPPRESS
    LOG_SUPPRESS: List[str] = field(default_factory=list)

    @staticmethod
    def load(path: str = CONFIG_FILE) -> "AppConfig":
        """
        Load configuration from disk.
        - Keeps sane defaults when keys are missing.
        - IMPORTANT: preserves the class default for HOOK_BIN_URL_BASE when not present in the file.
        - If the JSON cannot be read, falls back to defaults (FIRST_RUN=True).
        """
        if not os.path.exists(path):
            return AppConfig(FIRST_RUN=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Merge overlay with defaults
            ov = dict(DEFAULT_OVERLAY)
            ov.update(data.get("OVERLAY", {}))

            return AppConfig(
                BASE=data.get("BASE", AppConfig.BASE),
                NVRAM_DIR=data.get("NVRAM_DIR", AppConfig.NVRAM_DIR),
                TABLES_DIR=data.get("TABLES_DIR", AppConfig.TABLES_DIR),
                OVERLAY=ov,
                FIRST_RUN=bool(data.get("FIRST_RUN", False)),
                SIMPLE_END_SNAPSHOT_ONLY=bool(data.get("SIMPLE_END_SNAPSHOT_ONLY", False)),
                HOOK_AUTO_SETUP=bool(data.get("HOOK_AUTO_SETUP", True)),
                # Preserve class default if not present in file
                HOOK_BIN_URL_BASE=str(data.get("HOOK_BIN_URL_BASE", AppConfig.HOOK_BIN_URL_BASE)),
                BOOTSTRAP_USE_CB=bool(data.get("BOOTSTRAP_USE_CB", False)),
                LOG_CTRL=bool(data.get("LOG_CTRL", False)),
                HOOK_ENABLE=bool(data.get("HOOK_ENABLE", True)),
                # If not stored, fallback to DEFAULT_LOG_SUPPRESS
                LOG_SUPPRESS=list(data.get("LOG_SUPPRESS", DEFAULT_LOG_SUPPRESS)),
            )
        except Exception:
            # Safe fallback to defaults (incl. DEFAULT_LOG_SUPPRESS)
            cfg = AppConfig(FIRST_RUN=True)
            cfg.LOG_SUPPRESS = list(DEFAULT_LOG_SUPPRESS)
            return cfg

    def save(self, path: str = CONFIG_FILE) -> None:
        try:
            to_dump = {
                "BASE": self.BASE,
                "NVRAM_DIR": self.NVRAM_DIR,
                "TABLES_DIR": self.TABLES_DIR,
                "OVERLAY": self.OVERLAY,
                "FIRST_RUN": self.FIRST_RUN,
                "SIMPLE_END_SNAPSHOT_ONLY": self.SIMPLE_END_SNAPSHOT_ONLY,
                "HOOK_AUTO_SETUP": self.HOOK_AUTO_SETUP,
                "HOOK_BIN_URL_BASE": self.HOOK_BIN_URL_BASE,
                "BOOTSTRAP_USE_CB": self.BOOTSTRAP_USE_CB,
                "LOG_CTRL": self.LOG_CTRL,
                "HOOK_ENABLE": self.HOOK_ENABLE,
                # Persistiere die aktuelle Suppress-Liste
                "LOG_SUPPRESS": self.LOG_SUPPRESS if self.LOG_SUPPRESS else DEFAULT_LOG_SUPPRESS,
            }
            ensure_dir(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(to_dump, f, indent=2)
        except Exception:
            pass

def p_maps(cfg):         return os.path.join(cfg.BASE, "NVRAM_Maps")
def p_local_maps(cfg):   return os.path.join(p_maps(cfg), "maps")
def p_overrides(cfg):    return os.path.join(p_maps(cfg), "overrides")
def p_session(cfg):      return os.path.join(cfg.BASE, "session_stats")
def p_highlights(cfg):   return os.path.join(p_session(cfg), "Highlights")
def p_rom_spec(cfg):     return os.path.join(cfg.BASE, "rom_specific_achievements")
def p_custom(cfg):       return os.path.join(cfg.BASE, "custom_achievements")
def p_bin(cfg):          return os.path.join(cfg.BASE, "bin")  # Ablage für DLL/Injector
def p_ai(cfg):          return os.path.join(cfg.BASE, "AI")  # globaler KI-Ordner unterhalb von BASE

def f_global_ach(cfg):   return os.path.join(cfg.BASE, "global_achievements.json")
def f_achievements_state(cfg: "AppConfig") -> str:
    return os.path.join(cfg.BASE, "achievements_state.json")
def f_log(cfg):          return os.path.join(cfg.BASE, "watcher.log")
def f_index(cfg):        return os.path.join(p_maps(cfg), "index.json")
def f_romnames(cfg):     return os.path.join(p_maps(cfg), "romnames.json")
def p_field_whitelists(cfg): return os.path.join(p_session(cfg), "whitelists")

GITHUB_BASE = "https://raw.githubusercontent.com/tomlogic/pinmame-nvram-maps/475fa3619134f5aa732ccd80244e1613e7e6e9a1"
INDEX_URL = f"{GITHUB_BASE}/index.json"
ROMNAMES_URL = f"{GITHUB_BASE}/romnames.json"
PREFETCH_MODE = "background"
PREFETCH_LOG_EVERY = 50
ROLLING_HISTORY_PER_ROM = 10

def ensure_dir(path): os.makedirs(path, exist_ok=True)
def _ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


DEFAULT_LOG_SUPPRESS = [
    "[SNAP] pregame player_count detected",
    "[FW] Loaded whitelist",
    "[HOOK] Global keyboard hook installed",
    "[HOOK] toggle fired",                 
    "[HOTKEY] Registered WM_HOTKEY",       
]
quiet_prefixes: tuple[str, ...] = ()

def log(cfg: AppConfig, msg: str, level: str = "INFO"):
    # Vollständige Unterdrückung definierter Muster (Datei UND Konsole)
    try:
        suppress_list = (getattr(cfg, "LOG_SUPPRESS", None) or DEFAULT_LOG_SUPPRESS) if cfg else DEFAULT_LOG_SUPPRESS
        for pat in suppress_list:
            if pat and pat in str(msg):
                return  # nichts schreiben
    except Exception:
        pass

    line = f"[{_ts()}] [{level}] {msg}"

    # Nur Konsole unterdrücken? (bestehende Mechanik über quiet_prefixes)
    suppress_console = any(str(msg).startswith(p) for p in quiet_prefixes) if quiet_prefixes else False

    # Datei-Log
    try:
        ensure_dir(os.path.dirname(f_log(cfg)))
        with open(f_log(cfg), "a", encoding="utf-8") as fp:
            fp.write(line + "\n")
    except Exception:
        pass

    # Konsole (optional)
    if not suppress_console:
        print(line)

def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    """
    Atomar JSON speichern:
      1) in .tmp-Datei schreiben
      2) flush + fsync
      3) os.replace(.tmp -> final) = atomarer Swap
    Rückgabe: True bei Erfolg, sonst False (räumt .tmp auf).
    """
    tmp = None
    try:
        ensure_dir(os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            os.replace(tmp, path)
        except Exception:
            os.rename(tmp, path)
        return True
    except Exception:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False

def write_text(path, text):
    """
    Atomar Text speichern (analog zu save_json).
    """
    tmp = None
    try:
        ensure_dir(os.path.dirname(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        try:
            os.replace(tmp, path)
        except Exception:
            os.rename(tmp, path)
        return True
    except Exception:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False

def _is_weird_value(x: int) -> bool:
    """
    Markiert offensichtlich falsche/ausreißende Werte als 'weird'.
    Schwelle so gesetzt, dass 433_483_904 herausgefiltert wird.
    """
    try:
        return abs(int(x)) >= 400_000_000
    except Exception:
        return False


def apply_tooltips(owner, tips: dict):
    """
    Safely set .setToolTip(text) on attributes of 'owner' if they exist.
    tips: { 'attr_name': 'tooltip text', ... }
    """
    for name, text in (tips or {}).items():
        try:
            w = getattr(owner, name, None)
            if w:
                w.setToolTip(text)
        except Exception:
            pass

def sanitize_filename(s):
    s = re.sub(r"[^\w\-. ]+", "_", str(s))
    return s.strip().replace(" ", "_")

_RE_CGAMENAME = re.compile(r'Const\s+cGameName\s*=\s*"([^"\r\n]+)"', re.IGNORECASE)
_VPX_CACHE_LOCK = threading.Lock()
_VPX_CACHE: dict[str, tuple[float, str, bool]] = {}

def extract_cgamename_from_vpx(vpx_path: str) -> str | None:
    if not os.path.isfile(vpx_path):
        return None
    try:
        mtime = os.path.getmtime(vpx_path)
    except OSError:
        return None
    with _VPX_CACHE_LOCK:
        cached = _VPX_CACHE.get(vpx_path)
        if cached and cached[0] == mtime:
            return cached[1] if cached[2] else None
    rom = None
    found = False
    try:
        if zipfile.is_zipfile(vpx_path):
            with zipfile.ZipFile(vpx_path, "r") as zf:
                candidates = [n for n in zf.namelist() if "gamedata" in n.lower() or "gamestg" in n.lower()]
                if not candidates:
                    candidates = zf.namelist()[:12]
                for name in candidates:
                    try:
                        data = zf.read(name)
                        text = data.decode("latin1", errors="ignore")
                        m = _RE_CGAMENAME.search(text)
                        if m:
                            rom = m.group(1).strip()
                            found = True
                            break
                    except Exception:
                        continue
    except Exception:
        pass
    if not found and olefile:
        try:
            if olefile.isOleFile(vpx_path):
                ole = olefile.OleFileIO(vpx_path)
                try:
                    for entry in ole.listdir():
                        if any(part.lower() == "gamedata" for part in entry):
                            try:
                                data = ole.openstream(entry).read()
                                text = data.decode("latin1", errors="ignore")
                                m = _RE_CGAMENAME.search(text)
                                if m:
                                    rom = m.group(1).strip()
                                    found = True
                                    break
                            except Exception:
                                continue
                finally:
                    ole.close()
        except Exception:
            pass
    with _VPX_CACHE_LOCK:
        _VPX_CACHE[vpx_path] = (mtime, rom or "", found)
    return rom if found and rom else None

def fallback_rom_from_table_name(table_name: str) -> str:
    base = table_name
    if base.lower().endswith(".vpx"):
        base = base[:-4]
    base = re.sub(r'\([^)]*\)', '', base)
    base = re.sub(r'\[[^]]*\]', '', base)
    base = re.sub(r'[^a-zA-Z0-9]+', '_', base).strip('_').lower()
    if len(base) > 20:
        base = base[:20]
    return base

class Watcher:
    MIN_SEGMENTS_FOR_CLASSIFICATION = 1
    SUMMARY_FILENAME = "session_latest.summary.json"

    def __init__(self, cfg: AppConfig, bridge: "Bridge"):
        self.cfg = cfg
        self.bridge = bridge
        self._stop = threading.Event()
        self._flush_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self._last_logged_rom = None

        self.current_table: Optional[str] = None
        self.current_rom: Optional[str] = None
        self.start_time: Optional[float] = None
        self.game_active: bool = False
        self.start_audits: Dict[str, Any] = {}
        self.current_player = 1
        self.players: Dict[int, Dict[str, Any]] = {}
        self.ball_track = {
            "active": False, "index": 0, "start_time": None,
            "score_base": 0, "last_balls_played": None, "balls": []
        }
        self._last_audits_global: Dict[str, Any] = {}
        self.INDEX: Dict[str, Any] = {}
        self.ROMNAMES: Dict[str, Any] = {}

        # Hier fest einschalten: Segment-/Snapshot-Modus aktiv
        self.snapshot_mode = True

        self.field_stats: Dict[str, dict] = {}
        self.active_field_whitelist: set[str] = set()
        self.snap_player = 1
        self.snap_start_audits: Dict[str, Any] = {}
        self.snap_initialized = False
        self.snap_players_in_game = 1
        self.snap_players_locked = False
        self.snap_last_balls_played = None
        self.snap_segment_index = 0
        self.snap_segment_start_time = None
        self.SNAP_PREV_SEGMENT_FIELDS: Dict[str, int] = {}
        self.SNAP_DELAY_WINDOW_SEC = 1.5
        self.bootstrap_phase = False

        self._field_layout_cache: Dict[str, Dict[str, Any]] = {}
        self.current_segment_provisional_diff: Dict[str, int] = {}
        self.include_current_segment_in_overlay = True
        self._pending_detector_switch: Optional[int] = None
        # Debounce/State für Rotationen
        self._last_rotate_ts = 0.0
        self.CP_MIN_ROTATE_INTERVAL = 0.8  # sec debounce für Current-Player-Rotation
        self._control_fields_cache: Dict[str, List[dict]] = {}
        self._last_injected_pid = 0
        self._injector_procs: List[subprocess.Popen] = []
        self._injector_started: bool = False
        self._last_ini_sig = ""
        self.cpu = {}  # noch kein Laufzeitzustand – beim ersten Init Config-Werte übernehmen
 
 
    def _oneball_check_and_schedule(self, audits_ctl: dict) -> None:
        """
        Robuste One-Ball-Erkennung:
        - Löst aus, wenn current_ball oder Balls Played um +1 steigt (rising edge),
          zusätzlich zu evtl. vorhandenen Baselines (baseline_cb / baseline_bp).
        - Setzt einen verzögerten Kill (pending_kill_at), damit Logs/Snapshots sauber sind.
        """
        import time

        try:
            ch = getattr(self, "challenge", {}) or {}
            if not ch.get("active") or ch.get("kind") != "oneball":
                # Auch Prev-Zähler zurücksetzen, wenn One-Ball nicht aktiv ist
                self._prev_cb_ob = None
                self._prev_bp_ob = None
                return

            if not ch.get("one_ball_active", False):
                # Bereits beendet/geschedult
                return

            audits_ctl = audits_ctl or {}

            def _to_int(v):
                try:
                    return int(v)
                except Exception:
                    return None

            cur_cb = _to_int(audits_ctl.get("current_ball"))
            cur_bp = _to_int(audits_ctl.get("Balls Played"))

            prev_cb = getattr(self, "_prev_cb_ob", None)
            prev_bp = getattr(self, "_prev_bp_ob", None)

            base_cb = _to_int(ch.get("baseline_cb"))
            base_bp = _to_int(ch.get("baseline_bp"))

            trigger = False
            reason = None

            # 1) Rising edges (am robustesten, unabhängig von Baselines/Maps)
            if prev_cb is not None and cur_cb is not None and cur_cb == prev_cb + 1:
                trigger = True
                reason = f"current_ball rising-edge {prev_cb}->{cur_cb}"
            if not trigger and prev_bp is not None and cur_bp is not None and cur_bp == prev_bp + 1:
                trigger = True
                reason = f"Balls Played rising-edge {prev_bp}->{cur_bp}"

            # 2) Baseline-Deltas (fallback / zusätzlich)
            if not trigger and base_cb is not None and cur_cb is not None and cur_cb >= base_cb + 1:
                trigger = True
                reason = f"current_ball baseline {base_cb}->{cur_cb}"
            if not trigger and base_bp is not None and cur_bp is not None and cur_bp >= base_bp + 1:
                trigger = True
                reason = f"Balls Played baseline {base_bp}->{cur_bp}"

            # Always update prevs for next loop
            self._prev_cb_ob = cur_cb
            self._prev_bp_ob = cur_bp

            if not trigger:
                return

            # Einmalig kill schedulen
            if not ch.get("pending_kill_at"):
                delay_s = float(ch.get("one_ball_kill_delay", 3.0))  # 0.5–3.0 empfehlenswert
                try:
                    # Snapshot vom Zustand direkt vorm Kill (optional)
                    ch["prekill_end"] = dict(audits_ctl or {})
                except Exception:
                    pass
                ch["one_ball_active"] = False
                ch["pending_kill_at"] = time.time() + delay_s
                self.challenge = ch

                try:
                    log(self.cfg, f"[CHALLENGE] 1-ball finished ({reason}) – will kill VPX in {delay_s:.1f}s")
                except Exception:
                    try:
                        print(f"[CHALLENGE] 1-ball finished ({reason}) – will kill VPX in {delay_s:.1f}s")
                    except Exception:
                        pass

        except Exception as e:
            try:
                log(self.cfg, f"[CHALLENGE] _oneball_check_and_schedule error: {e}", "WARN")
            except Exception:
                try:
                    print(f"[CHALLENGE] _oneball_check_and_schedule error: {e}")
                except Exception:
                    pass
 
 
 
 
     # In class Watcher ergänzen
    def _alt_f4_visual_pinball_player(self, wait_ms: int = 0) -> bool:
        """
        Sendet ALT+F4 an alle sichtbaren 'Visual Pinball Player'-Fenster.
        Optional kann nach dem Senden kurz gewartet werden (wait_ms).
        """
        try:
            import time
            import win32con
            import win32gui
            import win32api

            def _match_vpx_title(title: str) -> bool:
                t = (title or "").strip().lower()
                return (
                    ("pinball" in t and "player" in t)
                    or t.startswith("visual pinball player")
                    or t.startswith("vpinballx player")
                    or t.startswith("visual pinball x")
                )

            def _enum_handler(hwnd, acc):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd) or ""
                if _match_vpx_title(title):
                    acc.append(hwnd)

            hwnds = []
            win32gui.EnumWindows(_enum_handler, hwnds)

            if not hwnds:
                try:
                    log(self.cfg, "[ALT+F4] No VPX Player windows found", "WARN")
                except Exception:
                    pass
                return False

            # ALT+F4 senden
            for hwnd in hwnds:
                try:
                    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                    win32api.keybd_event(win32con.VK_F4, 0, 0, 0)
                    win32api.keybd_event(win32con.VK_F4, 0, win32con.KEYEVENTF_KEYUP, 0)
                    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                except Exception:
                    pass

            # optional kurze Wartezeit
            if int(wait_ms or 0) > 0:
                try:
                    time.sleep(max(0.0, float(wait_ms) / 1000.0))
                except Exception:
                    pass

            try:
                log(self.cfg, f"[ALT+F4] Sent ALT+F4 to {len(hwnds)} VPX window(s)")
            except Exception:
                pass
            return True

        except Exception as e:
            try:
                log(self.cfg, f"[ALT+F4] error: {e}", "WARN")
            except Exception:
                try:
                    print(f"[ALT+F4] error: {e}")
                except Exception:
                    pass
            return False
 
    def _auto_map_enabled(self) -> bool:
        """
        Auto-mapping is ON by default. It can be turned off only via config.json:
          OVERLAY.automatic_creation = false
        Legacy opt-out aliases (also disable when set to false):
          OVERLAY.automatischanlegen = false
          OVERLAY.auto_generate_maps = false
        """
        try:
            ov = getattr(self.cfg, "OVERLAY", {}) or {}
            # New primary switch
            if ov.get("automatic_creation") is False:
                return False
            # Legacy aliases (still honored for opt-out)
            if ov.get("automatischanlegen") is False:
                return False
            if ov.get("auto_generate_maps") is False:
                return False
            return True
        except Exception:
            return True

    def _base_map_exists(self, rom: str) -> bool:
        """
        Returns True if a base map for the ROM exists strictly under BASE/NVRAM_Maps/maps.
        Accepts either <rom>.json or <rom>.map.json.
        """
        if not rom:
            return False
        maps_dir = p_local_maps(self.cfg)
        return (
            os.path.isfile(os.path.join(maps_dir, f"{rom}.json")) or
            os.path.isfile(os.path.join(maps_dir, f"{rom}.map.json"))
        )
 
    def _nvram_sampler_start(self, rom: str):
        """
        Start lightweight NVRAM sampling for ROMs without a base map.
        Captures consecutive raw .nv snapshots (bytes) for simple change analysis.
        """
        try:
            if not rom or not self._auto_map_enabled() or self._base_map_exists(rom):
                self._nv_samp = None
                return
            nv_path = os.path.join(self.cfg.NVRAM_DIR, rom + ".nv")
            if not os.path.isfile(nv_path):
                self._nv_samp = None
                return
            # Keep ~60s at 4 Hz by default (max 240 samples)
            self._nv_samp = {
                "rom": rom,
                "path": nv_path,
                "samples": [],
                "interval": 0.25,
                "last": 0.0,
                "max": 240
            }
            log(self.cfg, f"[AUTOMAP] sampler started for {rom}")
        except Exception:
            self._nv_samp = None 
 
    def _nvram_sampler_tick(self):
        """
        Append a new NVRAM sample (bytes) at the configured interval while the game is active.
        Runs only if a sampler session is active for the current ROM.
        """
        try:
            s = getattr(self, "_nv_samp", None)
            if not s or not self.game_active or not self.current_rom or self.current_rom != s.get("rom"):
                return
            now = time.time()
            if now - float(s.get("last", 0.0)) < float(s.get("interval", 0.25)):
                return
            s["last"] = now
            with open(s["path"], "rb") as f:
                data = f.read()
            if not data:
                return
            samples = s["samples"]
            samples.append(data)
            if len(samples) > int(s.get("max", 240)):
                samples.pop(0)
            self._nv_samp = s
        except Exception:
            pass  
            
    def _looks_bcd(self, bs: bytes) -> bool:
        """
        Heuristic: return True if all nibbles are <= 9 and not all bytes are zero.
        Minimum length = 2 bytes.
        """
        if not bs or len(bs) < 2:
            return False
        any_nonzero = False
        for b in bs:
            if (b >> 4) > 9 or (b & 0xF) > 9:
                return False
            if b != 0:
                any_nonzero = True
        return any_nonzero            
 
    def _nvram_autogen_map(self, rom: str):
        """
        Generate a simple but usable NVRAM map for ROMs without a base map.
        Writes BASE/NVRAM_Maps/maps/<rom>.json with:
          - fields[]: candidates for game_state (player_count, current_player, Balls Played),
                      BCD score slots (P1..P4 Score),
                      and a handful of counters (Counter 1..n).
        Guardrails:
          - Active by default; can be disabled via OVERLAY.automatic_creation = false
          - Does NOT overwrite official/community maps or overrides.
        """
        try:
            if not rom or not self._auto_map_enabled() or self._base_map_exists(rom):
                return

            # Prefer the sampled tail; otherwise, read a single end-state snapshot.
            s = getattr(self, "_nv_samp", None)
            samples: list[bytes] = []
            data: bytes | None = None
            if s and s.get("samples"):
                tail = s["samples"][-20:]
                samples = tail
                data = tail[-1] if tail else None
            else:
                nv = os.path.join(self.cfg.NVRAM_DIR, rom + ".nv")
                if not os.path.isfile(nv):
                    return
                with open(nv, "rb") as f:
                    data = f.read()
                samples = [data] if data else []

            if not data or len(data) < 64:
                return

            L = len(data)

            # 1) Change profile: count how often each offset changed between successive samples
            deltas = [0] * L
            try:
                for i in range(1, len(samples)):
                    a, b = samples[i - 1], samples[i]
                    m = min(len(a), len(b), L)
                    for off in range(m):
                        if a[off] != b[off]:
                            deltas[off] += 1
            except Exception:
                pass

            # 2) Build candidate windows (1/2/3/4 bytes, be/le, BCD)
            cands = []
            def add_cand(off: int, sz: int, enc: str | None, endian: str | None, score: int):
                cands.append({"offset": off, "size": sz, "encoding": enc, "endian": endian, "score": score})

            for off in range(L):
                if deltas[off] <= 0:
                    continue
                # 1 byte
                add_cand(off, 1, None, "be", deltas[off])
                # 2 bytes (be/le)
                if off + 1 < L:
                    sc2 = deltas[off] + deltas[off + 1]
                    add_cand(off, 2, None, "be", sc2)
                    add_cand(off, 2, None, "le", sc2)
                # BCD 3/4 bytes
                if off + 2 < L and self._looks_bcd(data[off:off + 3]):
                    add_cand(off, 3, "bcd", None, deltas[off] + deltas[off + 1] + deltas[off + 2] + 2)
                if off + 3 < L and self._looks_bcd(data[off:off + 4]):
                    add_cand(off, 4, "bcd", None,
                             deltas[off] + deltas[off + 1] + deltas[off + 2] + deltas[off + 3] + 3)
                # uint32 (be/le)
                if off + 3 < L:
                    sc4 = sum(deltas[off:off + 4])
                    add_cand(off, 4, None, "be", sc4)
                    add_cand(off, 4, None, "le", sc4)

            # 3) Pick top non-overlapping candidates (up to ~40)
            cands.sort(key=lambda c: c["score"], reverse=True)
            picked = []
            def overlaps(a, b):
                return not (a["offset"] + a["size"] <= b["offset"] or b["offset"] + b["size"] <= a["offset"])
            for c in cands:
                if any(overlaps(c, p) for p in picked):
                    continue
                picked.append(c)
                if len(picked) >= 40:
                    break

            # 4) Labeling heuristics
            fields = []
            def push(label, spec):
                fields.append({
                    "name": label,
                    "label": label,
                    "offset": int(spec["offset"]),
                    "size": int(spec["size"]),
                    "encoding": spec.get("encoding"),
                    "endian": spec.get("endian"),
                    "scale": 1.0,
                    "signed": False
                })

            def series_value(spec):
                vals = []
                sz = int(spec["size"]); off = int(spec["offset"])
                for snap in samples[-20:]:
                    if off + sz <= len(snap):
                        window = snap[off:off + sz]
                        if (spec.get("encoding") or "") == "bcd":
                            v = self.decode_bcd(window) or 0
                        else:
                            v = self.decode_uint(window, spec.get("endian") or "be", False)
                        vals.append(int(v))
                return vals

            # Game-state: look for 1-byte candidates ranging in 0..4 to guess player_count/current_player
            gs_specs = [c for c in picked if c["size"] == 1]
            one_to_four = []
            for s in gs_specs:
                vals = series_value(s)
                if not vals:
                    continue
                lo, hi = min(vals), max(vals)
                if 0 <= lo <= 4 and 0 <= hi <= 4 and any(1 <= v <= 4 for v in vals):
                    one_to_four.append((s, hi))
            one_to_four.sort(key=lambda t: t[1], reverse=True)
            if one_to_four:
                push("player_count", one_to_four[0][0])
            if len(one_to_four) > 1:
                push("current_player", one_to_four[1][0])

            # Balls Played: small, slowly increasing counter (1 or 2 bytes, non-BCD)
            bp_cand = None
            for s in picked:
                if s["size"] in (1, 2) and (s.get("encoding") or None) != "bcd":
                    vals = series_value(s)
                    if len(vals) >= 4:
                        diffs = [max(0, vals[i] - vals[i - 1]) for i in range(1, len(vals))]
                        if sum(1 for d in diffs if d in (0, 1)) >= int(0.7 * len(diffs)) and (max(vals) - min(vals)) <= 40:
                            bp_cand = s
                            break
            if bp_cand:
                push("Balls Played", bp_cand)

            # Scores: BCD 3–4 bytes for up to 4 players
            bcds = [c for c in picked if (c.get("encoding") or None) == "bcd" and c["size"] in (3, 4)]
            for idx, spec in enumerate(bcds[:4], start=1):
                push(f"P{idx} Score", spec)

            # Additional neutral counters (few)
            rest = 0
            for s in picked:
                if any(abs(s["offset"] - f["offset"]) < 1 and s["size"] == f["size"] for f in fields):
                    continue
                if (s.get("encoding") or None) == "bcd":
                    continue
                if s["size"] in (1, 2, 4) and rest < 10:
                    rest += 1
                    push(f"Counter {rest}", s)

            if not fields:
                log(self.cfg, f"[AUTOMAP] no fields detected for {rom}", "WARN")
                return

            # 5) Write maps/<rom>.json
            maps_dir = p_local_maps(self.cfg)
            ensure_dir(maps_dir)
            out_path = os.path.join(maps_dir, f"{rom}.json")
            payload = {"generated": True, "fields": fields}
            if save_json(out_path, payload):
                log(self.cfg, f"[AUTOMAP] generated map: {out_path} ({len(fields)} fields)")
        except Exception as e:
            log(self.cfg, f"[AUTOMAP] failed: {e}", "WARN") 
 
 
    def _start_detector_http(self, host: str = "127.0.0.1", port: int = 8765):
        # Deaktiviert: kein HTTP-Detector mehr
        return

    def _stop_detector_http(self):
        # Deaktiviert: kein HTTP-Server zu stoppen
        return

    def _kill_b2s_process_if_enabled(self):
        # Deaktiviert: keinen B2S-Prozess beenden
        return

    def _plausible_counter(self, label: str) -> bool:
        if not label:
            return False
        l = label.lower()
        keys = [
            "games", "balls", "ramp", "bumper", "spinner", "extra",
            "bonus", "hits", "made", "served", "targets", "loops",
            "lane", "kicks", "multiball", "jackpot", "mode"
        ]
        return any(k in l for k in keys)

    def _session_milestones_for_field(self, field_label: str) -> list[int]:
        """
        Realistische Session-Meilensteine (nvram_delta) pro Feld.
        WICHTIG: 'Extra Balls' maximal 3 oder 5 in einer Session.
        """
        f = (field_label or "").lower()
        if "extra ball" in f:
            return [3, 5]
        if "ball save" in f:
            return [3, 5, 10]
        if "jackpot" in f:
            return [1, 3, 5, 10, 15]
        if "multiball" in f:
            return [1, 3, 5]
        if "ramp" in f:
            return [5, 10, 15, 20, 25]
        if "loop" in f or "orbit" in f:
            return [3, 5, 10, 15]
        if "spinner" in f:
            return [10, 20, 30, 50]
        if "target" in f:
            return [10, 20, 30, 50]
        if "mode" in f:
            return [1, 3, 5, 10]
        # Default
        return [1, 3, 5, 10, 15, 20, 25, 30]
        
    def _overall_milestones_for_field(self, field_label: str) -> list[int]:
        """
        Realistische Overall-Meilensteine (nvram_overall) pro Feld.
        Konservativer als Session-Ziele.
        """
        f = (field_label or "").lower()
        if "games started" in f:
            return [50, 100, 250, 500]
        if "balls played" in f:
            return [100, 250, 500]
        if "extra ball" in f:
            return [10, 20, 30]
        if "ball save" in f:
            return [20, 50, 100]
        if "jackpot" in f:
            return [25, 50, 100, 150]
        if "multiball" in f:
            return [10, 25, 50]
        if "ramp" in f:
            return [100, 200, 300, 500]
        if "loop" in f or "orbit" in f:
            return [100, 200, 500]
        if "spinner" in f:
            return [100, 200, 500]
        if "target" in f:
            return [200, 400, 800]
        if "modes completed" in f or ("mode" in f and "complete" in f):
            return [10, 25, 50]
        if "modes started" in f or ("mode" in f and "start" in f):
            return [25, 50, 100]
        # Default
        return [50, 100, 250, 500]     


    def _generate_default_global_rules(self) -> list[dict]:
        """
        Erzeugt ca. 50 realistische globale Regeln:
          - session_time inkl. 15/35/45 Minuten
          - feldspezifische nvram_overall-Meilensteine
        """
        rules: list[dict] = []
        seen: set[str] = set()

        # 1) Zeitbasierte Global-Regeln (immer gültig)
        for mins in [10, 15, 20, 30, 35, 45, 60]:
            title = self._unique_title(f"Global – {mins} Minutes", seen)
            rules.append({
                "title": title,
                "scope": "global",
                "condition": {"type": "session_time", "min_seconds": int(mins * 60)}
            })

        # 2) Feldbasierte Overall-Regeln (realistisch)
        candidate_fields = [
            "Games Started", "Balls Played", "Ramps Made", "Jackpots",
            "Total Multiballs", "Loops", "Spinner", "Drop Targets",
            "Orbits", "Combos", "Extra Balls", "Ball Saves",
            "Modes Started", "Modes Completed"
        ]

        total_target = 50
        ci = 0
        while len(rules) < total_target and candidate_fields:
            fld = candidate_fields[ci % len(candidate_fields)]
            for m in self._overall_milestones_for_field(fld):
                if len(rules) >= total_target:
                    break
                title = self._unique_title(f"Global – {fld} {m}", seen)
                rules.append({
                    "title": title,
                    "scope": "global",
                    "condition": {"type": "nvram_overall", "field": fld, "min": int(m)}
                })
            ci += 1

        return rules[:total_target]        
        

    def _ensure_rom_specific(self, rom: str, audits: dict):
        """
        Create ROM-specific session-only achievements (no global rules).
        Target:
          - ~36 session rules:
            * 6x session_time: 5/10/15/20/30/45 minutes
            * ~30x nvram_delta with caps per field (max 2 milestones per field)
        Field selection:
          - Prefer hot stats -> whitelist -> plausible counters
          - Mix categories (power/precision/progress/meta/other) round-robin
        """
        if not rom or not audits:
            return
        path = os.path.join(p_rom_spec(self.cfg), f"{rom}.ach.json")
        if os.path.exists(path):
            return

        target_session_total = 36
        session_time_minutes = [5, 10, 15, 20, 30, 45]
        max_session_milestones_per_field = 2
        max_session_uses_per_field = 2

        def ok_label(lbl: str) -> bool:
            if not isinstance(lbl, str) or not lbl.strip():
                return False
            ll = lbl.lower()
            if "score" in ll:
                return False
            if is_excluded_field(lbl) or self.NOISE_REGEX.search(lbl):
                return False
            if ll in {"current_player", "player_count", "current_ball", "balls played", "credits", "tilted", "game over", "tilt warnings"}:
                return False
            return True

        def category(lbl: str) -> str:
            ll = (lbl or "").lower()
            if any(k in ll for k in ["extra ball", "ball save", "multiball", "jackpot", "wizard"]):
                return "power"
            if any(k in ll for k in ["ramp", "loop", "orbit", "spinner", "target", "combo"]):
                return "precision"
            if any(k in ll for k in ["mode", "lock", "locks lit", "balls locked"]):
                return "progress"
            if any(k in ll for k in ["games started", "balls played"]):
                return "meta"
            return "other"

        def uniq(seq):
            seen = set(); out = []
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out

        def session_cap_for(lbl: str) -> int:
            l = (lbl or "").lower()
            if "extra ball" in l: return 5
            if "ball save" in l:  return 10
            if "jackpot" in l:    return 10
            if "multiball" in l:  return 3
            if "ramp" in l:       return 20
            if "loop" in l or "orbit" in l: return 10
            if "spinner" in l:    return 30
            if "target" in l:     return 30
            if "mode" in l:       return 5
            return 15

        def pick_session_milestones(lbl: str) -> list[int]:
            mils = self._session_milestones_for_field(lbl) or []
            cap = session_cap_for(lbl)
            mils = [m for m in mils if m <= cap]
            if not mils:
                return []
            if len(mils) == 1:
                return [mils[0]]
            low = mils[0]
            mid = mils[max(1, len(mils)//2 - 1)]
            return uniq([low, mid])[:max_session_milestones_per_field]

        int_fields = [k for k, v in audits.items() if isinstance(v, int)]
        plausible = [k for k in int_fields if self._plausible_counter(k) and ok_label(k)]

        hot = []
        try:
            if self.field_stats:
                hot = sorted(
                    [k for k in self.field_stats.keys() if ok_label(k)],
                    key=lambda k: (self.field_stats[k].get("increments_sum", 0),
                                   self.field_stats[k].get("segments_with_delta", 0)),
                    reverse=True
                )
        except Exception:
            hot = []

        wl = [k for k in (list(getattr(self, "active_field_whitelist", []) or [])) if ok_label(k)]
        ordered = uniq([*hot, *wl, *plausible]) or plausible or int_fields

        cats = {"power": [], "precision": [], "progress": [], "meta": [], "other": []}
        for f in ordered:
            cats[category(f)].append(f)

        rr = [("power", cats["power"]), ("precision", cats["precision"]),
              ("progress", cats["progress"]), ("meta", cats["meta"]), ("other", cats["other"])]

        session_fields = []
        target_session_unique_fields = 15
        idxs = {k: 0 for k, _ in rr}
        while len(session_fields) < target_session_unique_fields:
            progressed = False
            for key, arr in rr:
                i = idxs[key]
                while i < len(arr) and arr[i] in session_fields:
                    i += 1
                idxs[key] = i
                if i < len(arr):
                    session_fields.append(arr[i])
                    idxs[key] = i + 1
                    progressed = True
            if not progressed:
                break
        if not session_fields:
            session_fields = ordered[:target_session_unique_fields]

        rules: list[dict] = []
        seen_titles: set[str] = set()

        # Session time targets
        for mins in session_time_minutes:
            secs = int(mins * 60)
            title = self._unique_title(f"{rom} – {mins} Minutes (Session)", seen_titles)
            rules.append({
                "title": title,
                "condition": {"type": "session_time", "min_seconds": secs},
                "scope": "session"
            })

        # Session nvram_delta
        remaining_session = max(0, target_session_total - len(rules))
        used_session_per_field: dict[str, int] = {}

        for fld in session_fields:
            if remaining_session <= 0:
                break
            fl = (fld or "").lower()
            if "games started" in fl:
                continue
            picks = pick_session_milestones(fld)
            if not picks:
                continue
            for m in picks:
                if remaining_session <= 0:
                    break
                cnt = used_session_per_field.get(fld, 0)
                if cnt >= max_session_uses_per_field:
                    break
                title = self._unique_title(f"{rom} – {fld} {int(m)} (Session)", seen_titles)
                rules.append({
                    "title": title,
                    "condition": {"type": "nvram_delta", "field": fld, "min": int(m)},
                    "scope": "session"
                })
                used_session_per_field[fld] = cnt + 1
                remaining_session -= 1

        if save_json(path, {"rules": rules}):
            log(self.cfg, f"[ROM_SPEC] created {path} with {len(rules)} session-only rules")

    def _ach_persist_after_session(self, end_audits: dict, duration_sec: int, nplayers: int):
        """
        Persist one-time unlocks after a session:
          - Global: record only titles coming from global_achievements.json
          - Session: record only for 1-player sessions (skip when nplayers >= 2)
        """
        try:
            awarded, _all_global, awarded_meta = self._evaluate_achievements(self.current_rom, self.start_audits, end_audits, duration_sec)
        except Exception as e:
            log(self.cfg, f"[ACH] eval failed: {e}", "WARN")
            awarded, awarded_meta = [], []

        # Global: restrict to origin=global_achievements
        try:
            from_ga = [m for m in (awarded_meta or []) if (m.get("origin") == "global_achievements")]
            if from_ga:
                self._ach_record_unlocks("global", self.current_rom, from_ga)
        except Exception as e:
            log(self.cfg, f"[ACH] persist global failed: {e}", "WARN")

        # Session: only for 1-player sessions
        try:
            if int(nplayers or 1) == 1:
                sess_achs_p1 = self._evaluate_player_session_achievements(1, self.current_rom) or []
                if sess_achs_p1:
                    self._ach_record_unlocks("session", self.current_rom, list(sess_achs_p1))
            else:
                log(self.cfg, "[ACH] session achievements skipped (multi-player session)")
        except Exception as e:
            log(self.cfg, f"[ACH] persist session failed: {e}", "WARN")
        try:
            if self.cfg.OVERLAY.get("auto_show_on_end", True):
                from PyQt6.QtCore import QTimer
                from PyQt6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    log(self.cfg, "[OVERLAY] auto-show triggered (postgame)")
                    QTimer.singleShot(500, lambda: self.bridge.show_overlay_summary())
        except Exception as e:
            log(self.cfg, f"[OVERLAY] auto-show failed: {e}", "WARN")
    

    def _unique_title(self, title: str, seen: set[str]) -> str:
        """
        Sorgt dafür, dass Titel eindeutig sind (Suffix bei Kollision).
        """
        base = title.strip()
        if base not in seen:
            seen.add(base)
            return base
        i = 2
        while True:
            cand = f"{base} #{i}"
            if cand not in seen:
                seen.add(cand)
                return cand
            i += 1

    def _milestones(self, kind: str) -> list[int]:
        """
        Liefert passende Meilensteine:
        kind: 'session' | 'overall' | 'time'
        """
        if kind == "session":
            # feingranular für Session-Deltas
            return [1, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50]
        if kind == "overall":
            # robustere Overall-Stufen
            return [25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
        if kind == "time":
            # Sekunden
            return [180, 300, 480, 600, 720, 900, 1200, 1500, 1800, 2400, 3000]
        return []


    def bootstrap(self):
        for d in [
            self.cfg.BASE,
            p_maps(self.cfg),
            p_local_maps(self.cfg),
            p_overrides(self.cfg),
            p_session(self.cfg),
            p_highlights(self.cfg),
            p_rom_spec(self.cfg),
            p_custom(self.cfg),
            p_field_whitelists(self.cfg),
            p_bin(self.cfg),  # Ablage für DLL/Injector
        ]:
            ensure_dir(d)

        # Globaler KI-Ordner + Default-Dateien (BASE\AI)
        try:
            self._ai_global_bootstrap()
        except Exception as e:
            log(self.cfg, f"[AI] bootstrap call failed: {e}", "WARN")

        # Profiling-Bootstrap (history + profile.json)
        try:
            self._ai_profile_bootstrap()
        except Exception as e:
            log(self.cfg, f"[AI-PROFILE] bootstrap call failed: {e}", "WARN")

        # Hook/Injector-Binaries automatisch bereitstellen (falls konfiguriert)
        try:
            if bool(self.cfg.HOOK_AUTO_SETUP):
                self._ensure_hook_binaries()
        except Exception as e:
            log(self.cfg, f"[HOOK] auto-setup failed: {e}", "WARN")

        def ensure_file(path, url):
            if os.path.exists(path):
                return
            try:
                data = _fetch_bytes_url(url, timeout=25)
                ensure_dir(os.path.dirname(path))
                with open(path, "wb") as f:
                    f.write(data)
                log(self.cfg, f"Downloaded {url} -> {path}")
            except Exception as e:
                log(self.cfg, f"Could not download {url}: {e}", "ERROR")

        ensure_file(f_index(self.cfg), INDEX_URL)
        ensure_file(f_romnames(self.cfg), ROMNAMES_URL)
        self.INDEX = load_json(f_index(self.cfg), {}) or {}
        self.ROMNAMES = load_json(f_romnames(self.cfg), {}) or {}



    def _ensure_hook_binaries(self):
        try:
            ensure_dir(p_bin(self.cfg))
            targets = [
                "WatcherInjector64.dll",
                "WatcherInjector64.exe",
            ]
            base_url = (self.cfg.HOOK_BIN_URL_BASE or "").strip().rstrip("/")
            for fn in targets:
                local = os.path.join(p_bin(self.cfg), fn)
                if os.path.exists(local):
                    continue
                if not base_url:
                    log(self.cfg, f"[HOOK] Missing {fn}. Place it in {p_bin(self.cfg)} or set HOOK_BIN_URL_BASE.", "WARN")
                    continue
                try:
                    url = f"{base_url}/{fn}"
                    data = _fetch_bytes_url(url, timeout=25)
                    with open(local, "wb") as f:
                        f.write(data)
                    log(self.cfg, f"[HOOK] Downloaded {fn} -> {local}")
                except Exception as e:
                    log(self.cfg, f"[HOOK] Could not fetch {fn}: {e}", "WARN")
        except Exception as e:
            log(self.cfg, f"[HOOK] ensure binaries failed: {e}", "WARN")


    def _prefetch_worker(self):
        # INDEX laden
        if not self.INDEX:
            log(self.cfg, "Prefetch: INDEX empty, attempting reload...", "WARN")
            try:
                self.INDEX = load_json(f_index(self.cfg), {}) or {}
                if not self.INDEX:
                    mj = _fetch_json_url(INDEX_URL, timeout=25)
                    save_json(f_index(self.cfg), mj)
                    self.INDEX = mj or {}
            except Exception as e:
                log(self.cfg, f"Prefetch aborted: cannot load INDEX: {e}", "ERROR")
                return

        # Einzigartige Map-Pfade aus INDEX sammeln
        unique_rels = set()
        total_roms = 0
        for rom, entry in self.INDEX.items():
            if str(rom).startswith("_"):
                continue
            total_roms += 1
            rel = entry if isinstance(entry, str) else (entry.get("path") or entry.get("file"))
            if not rel:
                continue
            # Normalisieren: "maps/" vorne abstrippen, damit lokale Ablage unter BASE/NVRAM_Maps/maps/<rel>
            if rel.startswith("maps/"):
                rel = rel[len("maps/"):]
            unique_rels.add(rel)

        # Eindeutige Dateien laden
        downloaded = 0
        for rel in sorted(unique_rels):
            local = os.path.join(p_local_maps(self.cfg), rel.replace("/", os.sep))
            if os.path.exists(local):
                continue
            try:
                url = f"{GITHUB_BASE}/maps/{rel.lstrip('/')}"
                mj = _fetch_json_url(url, timeout=25)
                if save_json(local, mj):
                    downloaded += 1
                    if downloaded % PREFETCH_LOG_EVERY == 0:
                        log(self.cfg, f"Prefetch progress: downloaded {downloaded} unique maps...")
            except Exception as e:
                log(self.cfg, f"Prefetch miss {rel}: {e}", "WARN")

        log(self.cfg, f"Prefetch finished. ROMs in index: {total_roms}, unique map files: {len(unique_rels)}, newly downloaded: {downloaded}")

    def start_prefetch_background(self):
        if PREFETCH_MODE != "background":
            log(self.cfg, "Prefetch disabled (mode != background)")
            return
        threading.Thread(target=self._prefetch_worker, daemon=True).start()



    @staticmethod
    def _to_int(v, default=2):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            try:
                if s.startswith("0x"):
                    return int(s, 16)
                return int(s)
            except Exception:
                return default
        return default

    def parse_map(self, mj):
        """
        Unterstützt:
          - maps mit "fields"
          - "offsets"
          - "audits" (verschachtelt)
          - "game_state" (scores, current_player, player_count, current_ball, ball_count, credits, Flags)
        Gibt eine Liste von Feldspezifikationen zurück:
          {"name","label","offset","size","encoding","endian","scale","signed", optional: "mask","value_offset"}
        """
        fields: List[Dict[str, Any]] = []
        if not isinstance(mj, dict):
            return fields

        # 1) Direktes "fields"-Schema
        if isinstance(mj.get("fields"), list):
            for f in mj["fields"]:
                if not isinstance(f, dict):
                    continue
                fields.append({
                    "name": f.get("name") or f.get("label") or "field",
                    "label": f.get("label") or f.get("name") or "field",
                    "offset": self._to_int(f.get("offset", f.get("start", 0)), 0),
                    "size": self._to_int(f.get("size", f.get("length", 2)), 2),
                    "encoding": f.get("encoding") or None,
                    "endian": f.get("endian") or None,
                    "scale": float(f.get("scale") or 1.0),
                    "signed": bool(f.get("signed", False)),
                    "mask": self._to_int(f.get("mask", 0), 0),
                    "value_offset": self._to_int(f.get("value_offset", f.get("offset_adjust", f.get("valueoffset", 0))), 0)
                })
            return fields

        # 2) "offsets"-Kurzform
        if isinstance(mj.get("offsets"), dict):
            d_enc = mj.get("encoding")
            d_end = mj.get("endian")
            d_size = self._to_int(mj.get("size", 2), 2)
            for name, off in mj["offsets"].items():
                if isinstance(off, (int, str)):
                    fields.append({
                        "name": name, "label": name, "offset": self._to_int(off, 0),
                        "size": d_size, "encoding": d_enc, "endian": d_end,
                        "scale": 1.0, "signed": False
                    })
            return fields

        # 3) Verschachtelte "audits"-Struktur
        if isinstance(mj.get("audits"), (dict, list)):
            def walk(c):
                if isinstance(c, list):
                    for it in c:
                        if isinstance(it, dict) and ("label" in it and ("start" in it or "offset" in it)):
                            fields.append({
                                "name": it.get("name") or it.get("label") or "field",
                                "label": it.get("label") or it.get("name") or "field",
                                "offset": self._to_int(it.get("start", it.get("offset", 0)), 0),
                                "size": self._to_int(it.get("length", it.get("size", 2)), 2),
                                "encoding": it.get("encoding"), "endian": it.get("endian"),
                                "scale": float(it.get("scale") or 1.0), "signed": bool(it.get("signed", False)),
                                "mask": self._to_int(it.get("mask", 0), 0),
                                "value_offset": self._to_int(it.get("value_offset", it.get("offset_adjust", it.get("valueoffset", 0))), 0)
                            })
                elif isinstance(c, dict):
                    for k, v in c.items():
                        if isinstance(v, dict) and ("label" in v and ("start" in v or "offset" in v)):
                            fields.append({
                                "name": v.get("name") or k,
                                "label": v.get("label") or k,
                                "offset": self._to_int(v.get("start", v.get("offset", 0)), 0),
                                "size": self._to_int(v.get("length", v.get("size", 2)), 2),
                                "encoding": v.get("encoding"), "endian": v.get("endian"),
                                "scale": float(v.get("scale") or 1.0), "signed": bool(v.get("signed", False)),
                                "mask": self._to_int(v.get("mask", 0), 0),
                                "value_offset": self._to_int(v.get("value_offset", v.get("offset_adjust", v.get("valueoffset", 0))), 0)
                            })
                        else:
                            walk(v)
            walk(mj["audits"])
            # kein sofortiges return – evtl. zusätzlich game_state

        # 4) game_state-Schema
        if isinstance(mj.get("game_state"), dict):
            gs = mj["game_state"]

            # Scores → "P{n} Score"
            scores = gs.get("scores")
            if isinstance(scores, list):
                for idx, sc in enumerate(scores, start=1):
                    if not isinstance(sc, dict) or "start" not in sc:
                        continue
                    pid = idx
                    lab_in = str(sc.get("label", "")).strip()
                    m = re.search(r"(\d+)$", lab_in)
                    if m:
                        try:
                            pid = int(m.group(1))
                        except Exception:
                            pid = idx
                    fields.append({
                        "name": f"P{pid} Score",
                        "label": f"P{pid} Score",
                        "offset": self._to_int(sc.get("start", 0), 0),
                        "size": self._to_int(sc.get("length", 2), 2),
                        "encoding": sc.get("encoding") or "bcd",
                        "endian": sc.get("endian") or None,
                        "scale": float(sc.get("scale") or 1.0),
                        "signed": bool(sc.get("signed", False)),
                        "mask": self._to_int(sc.get("mask", 0), 0),
                        "value_offset": self._to_int(sc.get("value_offset", sc.get("offset", 0)), 0)
                    })

            def add_gs(name_in: str, label_out: str | None = None):
                ent = gs.get(name_in)
                if not isinstance(ent, dict) or "start" not in ent:
                    return
                lab = label_out or str(ent.get("label") or name_in)
                fields.append({
                    "name": label_out or name_in,
                    "label": lab,
                    "offset": self._to_int(ent.get("start", 0), 0),
                    "size": self._to_int(ent.get("length", ent.get("size", 1)), 1),
                    "encoding": ent.get("encoding") or None,
                    "endian": ent.get("endian") or None,
                    "scale": float(ent.get("scale") or 1.0),
                    "signed": bool(ent.get("signed", False)),
                    "mask": self._to_int(ent.get("mask", 0), 0),
                    "value_offset": self._to_int(ent.get("value_offset", ent.get("offset", 0)), 0)
                })

            add_gs("credits", "Credits")
            add_gs("current_player", "current_player")
            add_gs("player_count", "player_count")
            add_gs("current_ball", "current_ball")
            if "ball_count" in gs:
                add_gs("ball_count", "Balls Played")
            add_gs("tilted", "Tilted")
            add_gs("game_over", "Game Over")
            add_gs("extra_balls", "Extra Balls")
            add_gs("tilt_warnings", "Tilt Warnings")

        # 5) Letzter Fallback: flache ints/strings
        if not fields:
            for k, v in mj.items():
                if isinstance(v, (int, str)):
                    fields.append({
                        "name": k, "label": k, "offset": self._to_int(v, 0), "size": 2,
                        "encoding": None, "endian": "be", "scale": 1.0, "signed": False
                    })
        return fields

    def _build_control_field_specs_for_ini(self, rom: str) -> List[dict]:
        """
        Build INI fields for the DLL using the normalized control specs from _control_fields_for().
        Ensures current_player has a safe mask/value_offset, and 'Balls Played' label is normalized.
        """
        fields_ini: List[dict] = []
        if not rom or not self.cfg.HOOK_ENABLE:
            return fields_ini

        src = self._control_fields_for(rom)  # already normalized: cp mask=3, value_offset=1 (-> 1..4)
        if not src:
            return fields_ini

        def push(label, off, size=1, mask=0, voff=0):
            fields_ini.append({
                "label": label, "offset": int(off), "size": int(size),
                "mask": int(mask), "value_offset": int(voff)
            })

        bp_spec = None
        cb_spec = None
        for f in src:
            lbl = str(f.get("label") or f.get("name") or "").strip()
            ll = lbl.lower()
            if ll == "current_player":
                push("current_player",
                     f.get("offset", 0), 1,
                     f.get("mask", 0) or 0,
                     f.get("value_offset", 0) or 0)
            elif ll == "player_count":
                push("player_count",
                     f.get("offset", 0), 1,
                     f.get("mask", 0) or 0,
                     f.get("value_offset", 0) or 0)
            elif ll == "current_ball":
                cb_spec = f
                push("current_ball",
                     f.get("offset", 0), 1,
                     f.get("mask", 0) or 0,
                     f.get("value_offset", 0) or 0)
            elif ll == "balls played" or ll == "ball count" or ("balls" in ll and "played" in ll):
                bp_spec = f

        # Prefer explicit Balls Played, else Ball Count, else fallback to current_ball
        if bp_spec:
            push("Balls Played",
                 bp_spec.get("offset", 0), 1,
                 bp_spec.get("mask", 0) or 0,
                 bp_spec.get("value_offset", 0) or 0)
        elif cb_spec:
            push("Balls Played",
                 cb_spec.get("offset", 0), 1,
                 cb_spec.get("mask", 0) or 0,
                 cb_spec.get("value_offset", 0) or 0)

        return fields_ini


    def _load_base_map_for_rom(self, rom: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Lädt die Basis-Map NUR aus NVRAM_Maps/maps (keine Overrides) und gibt parse_map(fields) zurück.
        Nutzt INDEX/romnames/Family-Präfix wie load_map_for_rom – aber ohne Schritt (1) Override.
        """
        if not rom:
            return None, None
        # 1) Lokale Aliasse (json + map.json)
        alias = os.path.join(p_local_maps(self.cfg), rom + ".json")
        if os.path.exists(alias):
            return self.parse_map(load_json(alias, {}) or {}), alias
        alias2 = os.path.join(p_local_maps(self.cfg), rom + ".map.json")
        if os.path.exists(alias2):
            return self.parse_map(load_json(alias2, {}) or {}), alias2
        # 2) INDEX
        entry = self.INDEX.get(rom)
        if entry:
            rel = entry if isinstance(entry, str) else (entry.get("path") or entry.get("file"))
            if rel:
                f, p = self._load_map_from_local_rel(rel)
                if f:
                    return f, p
        # 3) romnames-Umleitung
        base_rom = self.ROMNAMES.get(rom)
        if base_rom and base_rom != rom:
            f, p = self._load_base_map_for_rom(base_rom)
            if f and p:
                save_json(alias, load_json(p, {}) or {})
                log(self.cfg, f"[CTRL] Saved alias base map via romnames -> {alias}")
                return f, alias
        # 4) Familien-Prefix
        prefix = str(rom).split("_")[0].lower()
        for cand in list(self.INDEX.keys()):
            if not cand.lower().startswith(prefix) or cand == rom:
                continue
            e2 = self.INDEX.get(cand)
            rel2 = e2 if isinstance(e2, str) else (e2.get("path") or e2.get("file"))
            if not rel2:
                continue
            f2, p2 = self._load_map_from_local_rel(rel2)
            if f2:
                save_json(alias, load_json(p2, {}) or {})
                log(self.cfg, f"[CTRL] Saved alias base map via family prefix -> {alias}")
                return f2, alias
        return None, None
        
    def _write_watcher_hook_ini(self, rom: str):
        """
        Schreibt BASE\\bin\\watcher_hook.ini für die DLL.
        Guard: HOOK_ENABLE muss True sein. Felder stammen aus Basis-Maps.
        """
        try:
            if not rom or not self.cfg.HOOK_ENABLE:
                return
            nv_path = os.path.join(self.cfg.NVRAM_DIR, rom + ".nv")
            base = self.cfg.BASE
            fields = self._build_control_field_specs_for_ini(rom)
            if not os.path.isfile(nv_path) or not fields:
                return

            lines = [f"base={base}", f"rom={rom}", f"nvram={nv_path}"]
            for f in fields:
                label = str(f.get("label") or f.get("name") or "").strip()
                # Normiere Balls Played für die DLL
                if "balls played" in label.lower() or label.lower() == "ball count":
                    label = "Balls Played"
                lines.append(
                    "field="
                    f"label={label},"
                    f"offset={int(f['offset'])},"
                    f"size={int(f['size'])},"
                    f"mask={int(f.get('mask', 0))},"
                    f"value_offset={int(f.get('value_offset', 0))}"
                )

            ini_path = os.path.join(p_bin(self.cfg), "watcher_hook.ini")
            ensure_dir(os.path.dirname(ini_path))
            with open(ini_path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(lines) + "\n")
            log(self.cfg, f"[HOOK] INI written: {ini_path}")
        except Exception as e:
            log(self.cfg, f"[HOOK] INI write failed: {e}", "WARN")




    def _ensure_hook_ini_once(self):
        """
        Schreibt watcher_hook.ini nur, wenn sich die relevanten Inhalte (rom, nvram, Felder)
        geändert haben. Verhindert 'INI written'-Spam.
        """
        try:
            if not self.cfg.HOOK_ENABLE:
                return
            rom = (self.current_rom or "").strip()
            if not rom:
                return
            nv_path = os.path.join(self.cfg.NVRAM_DIR, rom + ".nv")
            if not os.path.isfile(nv_path):
                return
            fields = self._build_control_field_specs_for_ini(rom) or []
            sig = f"{rom}|{nv_path}|{json.dumps(fields, sort_keys=True)}"
            if sig == getattr(self, "_last_ini_sig", ""):
                return
            self._write_watcher_hook_ini(rom)
            self._last_ini_sig = sig
        except Exception as e:
            log(self.cfg, f"[HOOK] ensure-ini-once failed: {e}", "WARN")


    def _try_start_injectors(self):
        """
        Startet NUR den 64-bit Injector. Start-CWD auf BASE\\bin setzen.
        Hinweis: Diese Methode wird im normalen Flow nicht benötigt, bleibt aber
        als Hilfsstarter rein für x64 erhalten.
        """
        try:
            if getattr(self, "_injector_started", False):
                return
            bin_dir = p_bin(self.cfg)
            exe = os.path.join(bin_dir, "WatcherInjector64.exe")
            if not os.path.isfile(exe):
                log(self.cfg, "[HOOK] Injector executable not found in BASE\\bin (WatcherInjector64.exe)", "WARN")
                return
            proc = subprocess.Popen([exe], creationflags=0x08000000, cwd=bin_dir)
            self._injector_procs = [proc]
            self._injector_started = True
            log(self.cfg, "[HOOK] Injector launched (x64 only)")
        except Exception as e:
            log(self.cfg, f"[HOOK] Injector start failed (x64): {e}", "WARN")
        # ACHTUNG: hier KEINE verschachtelten def _pnames_injectors/_tasklist_has mehr!

    def _pnames_injectors(self) -> List[str]:
        """
        Nur noch x64-Injector ist bekannt (32-bit vollständig entfernt).
        """
        return ["WatcherInjector64.exe"]

    def _tasklist_has(self, imagename: str) -> bool:
        """
        Prüft, ob ein Prozessname in tasklist sichtbar ist.
        """
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {imagename}"],
                creationflags=0x08000000
            ).decode(errors="ignore")
            return imagename.lower() in out.lower()
        except Exception:
            return False

    def _get_process_architecture(self, pid: int) -> Optional[str]:
        """
        Liefert 'x64' oder 'x86' für den Zielprozess, None bei Fehler.
        Nutzt bevorzugt IsWow64Process2 (Win10+), fällt zurück auf IsWow64Process.
        """
        try:
            k32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_QUERY_INFORMATION = 0x0400
            access = PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_QUERY_INFORMATION
            h = k32.OpenProcess(access, False, int(pid))
            if not h:
                return None
            arch = None
            try:
                # IsWow64Process2(HANDLE, USHORT* pProcessMachine, USHORT* pNativeMachine)
                try:
                    IsWow64Process2 = k32.IsWow64Process2
                    IsWow64Process2.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
                    IsWow64Process2.restype = wintypes.BOOL
                    pm = ctypes.c_ushort(0)
                    nm = ctypes.c_ushort(0)
                    if IsWow64Process2(h, ctypes.byref(pm), ctypes.byref(nm)):
                        IMAGE_FILE_MACHINE_I386 = 0x014c
                        IMAGE_FILE_MACHINE_AMD64 = 0x8664
                        if pm.value == 0:
                            # Kein WOW – native Bitness aus nm
                            arch = "x64" if nm.value == IMAGE_FILE_MACHINE_AMD64 else "x86"
                        else:
                            # WOW64 -> 32-Bit Prozess auf 64-Bit OS
                            arch = "x86" if pm.value == IMAGE_FILE_MACHINE_I386 else "x86"
                    else:
                        arch = None
                except Exception:
                    # Fallback: IsWow64Process(HANDLE, PBOOL) – True => 32-bit Prozess auf 64-bit OS
                    try:
                        IsWow64Process = k32.IsWow64Process
                        IsWow64Process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
                        IsWow64Process.restype = wintypes.BOOL
                        wow = wintypes.BOOL()
                        if IsWow64Process(h, ctypes.byref(wow)):
                            arch = "x86" if bool(wow.value) else ("x64" if sys.maxsize > 2**32 else "x86")
                    except Exception:
                        arch = None
            finally:
                try:
                    k32.CloseHandle(h)
                except Exception:
                    pass
            return arch
        except Exception:
            return None

    def _start_injector(self, exe_name: str) -> bool:
        """
        Startet genau EINE Injector-EXE (x86 ODER x64). Setzt CWD auf BASE\\bin.
        """
        try:
            bin_dir = p_bin(self.cfg)
            exe = os.path.join(bin_dir, exe_name)
            if not os.path.isfile(exe):
                log(self.cfg, f"[HOOK] Injector not found: {exe_name} in {bin_dir}", "WARN")
                return False
            proc = subprocess.Popen([exe], creationflags=0x08000000, cwd=bin_dir)
            self._injector_procs = [proc]
            self._injector_started = True
            log(self.cfg, f"[HOOK] Injector launched ({exe_name})")
            return True
        except Exception as e:
            log(self.cfg, f"[HOOK] Injector start failed ({exe_name}): {e}", "WARN")
            return False



    def _try_start_injectors_for_vpx(self):
        """
        Startet den 64‑Bit Injector (WatcherInjector64.exe), sobald VPX läuft.
        Schreibt watcher_hook.ini nur bei Änderungen.
        Guard: HOOK_ENABLE muss True sein.
        """
        if not self.cfg.HOOK_ENABLE:
            return
        try:
            pid = self._find_vpx_pid()
        except Exception:
            pid = None

        if not pid:
            self._last_injected_pid = 0
            return
        if not self.current_rom:
            return

        # INI nur bei Änderung
        try:
            self._ensure_hook_ini_once()
        except Exception:
            pass

        # Nur einmal pro VPX‑Prozess starten
        if int(pid) == int(getattr(self, "_last_injected_pid", 0) or 0):
            return

        try:
            self._ensure_hook_binaries()
            exe = "WatcherInjector64.exe"  # nur x64
            if self._start_injector(exe):
                self._last_injected_pid = int(pid)
                log(self.cfg, f"[HOOK] injector started for PID {pid} (x64 only)")
            else:
                log(self.cfg, f"[HOOK] injector start failed for PID {pid} (x64)", "WARN")
        except Exception as e:
            log(self.cfg, f"[HOOK] injector start (x64) failed for PID {pid}: {e}", "WARN")


    def _kill_injectors(self, force: bool = True, verify: bool = True, timeout_verify: float = 3.0):
        """
        Beendet alle bekannten Injector-Prozesse. Fallback per taskkill, danach Verifikation via tasklist.
        """
        # 1) Gemerkte Popen-Handles terminieren/killen
        try:
            for proc in list(getattr(self, "_injector_procs", []) or []):
                try:
                    if proc and (proc.poll() is None):
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.5)
                        except Exception:
                            pass
                        if force and (proc.poll() is None):
                            proc.kill()
                except Exception:
                    continue
        except Exception:
            pass
        self._injector_procs = []

        # 2) Fallback: taskkill nach Image-Namen
        try:
            for img in self._pnames_injectors():
                try:
                    subprocess.run(
                        ["taskkill", "/IM", img, "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=0x08000000
                    )
                except Exception:
                    continue
        except Exception:
            pass

        # 3) Status zurücksetzen
        self._injector_started = False

        # 4) Verifikation (optional)
        if verify:
            deadline = time.time() + max(0.2, float(timeout_verify))
            still = []
            while time.time() < deadline:
                still = [img for img in self._pnames_injectors() if self._tasklist_has(img)]
                if not still:
                    break
                time.sleep(0.2)
            if still:
                log(self.cfg, f"[HOOK] WARNING: injector(s) still running after kill: {', '.join(still)}", "WARN")
            else:
                log(self.cfg, "[HOOK] injectors terminated")


    def _control_fields_for(self, rom: str) -> List[dict]:
        """
        Liefert nur die Felder für Steuerung (current_player, player_count, current_ball, Balls Played/ball_count)
        aus der BASIS-MAP (maps/). Inklusive sicherem Mask/Offset für current_player.
        Cached pro ROM.
        """
        if not rom:
            return []
        cached = self._control_fields_cache.get(rom)
        if cached is not None:
            return cached

        fields, _ = self._load_base_map_for_rom(rom)
        out: List[dict] = []
        want = {"current_player", "player_count", "current_ball", "balls played", "ball count"}
        for f in (fields or []):
            lbl = str(f.get("label") or f.get("name") or "").strip()
            ll = lbl.lower()
            if ll in want or ("balls" in ll and "played" in ll):
                ff = dict(f)  # Kopie
                if ll == "current_player":
                    try:
                        sz = int(ff.get("size", 1))
                    except Exception:
                        sz = 1
                    m = int(ff.get("mask", 0) or 0)
                    vo = int(ff.get("value_offset", 0) or 0)
                    if sz == 1 and m == 0:
                        ff["mask"] = 3  # 2 Bits (0..3)
                    if vo == 0:
                        ff["value_offset"] = 1  # -> 1..4
                out.append(ff)

        self._control_fields_cache[rom] = out
        return out

  
    def _read_control_signals(self, rom: str) -> Dict[str, int]:
        """
        Liest Control-Signale ausschließlich aus BASE\\session_stats\\live.session.json (vom Injector).
        Kein Löschen/Migrieren. Fallback: direkte .nv-Auswertung aus Basis-Map-Control-Feldern.
        Erwartetes JSON der DLL (Beispiel):
          { "rom":"afm_113b", "cp":1, "pc":2, "cb":1, "bp":0 }
        """
        out: Dict[str, int] = {}
        if not rom:
            return out

        live_path = os.path.join(self.cfg.BASE, "session_stats", "live.session.json")

        # 1) Frische live.session.json der DLL
        try:
            if os.path.isfile(live_path):
                st = os.stat(live_path)
                if (time.time() - float(st.st_mtime)) <= 3.0:
                    data = load_json(live_path, {}) or {}
                    # Falls 'rom' im JSON vorhanden ist, muss sie passen
                    if not data or (data.get("rom") and data.get("rom") != rom):
                        pass
                    else:
                        mapping = (("cp", "current_player"), ("pc", "player_count"),
                                   ("cb", "current_ball"), ("bp", "Balls Played"))
                        for key_in, key_out in mapping:
                            if data.get(key_in) is not None:
                                try:
                                    out[key_out] = int(data.get(key_in))
                                except Exception:
                                    pass
                        if out:
                            return out
        except Exception:
            pass

        # 2) Fallback: NVRAM direkt auslesen (Basis-Map-Control-Felder)
        out2: Dict[str, int] = {}
        nv_path = os.path.join(self.cfg.NVRAM_DIR, rom + ".nv")
        if not os.path.exists(nv_path):
            return out
        try:
            with open(nv_path, "rb") as f:
                raw = f.read()
        except Exception:
            return out
        for f in self._control_fields_for(rom):
            try:
                lbl = str(f.get("label") or f.get("name") or "").strip()
                val = self._decode_field_value(raw, f)
                if val is None:
                    continue
                out2[lbl] = int(val)
            except Exception:
                continue
        # Falls nur "Ball Count" existiert, auf "Balls Played" mappen
        kl = {k.lower(): k for k in out2.keys()}
        if "balls played" not in kl and "ball count" in kl:
            try:
                out2["Balls Played"] = int(out2[kl["ball count"]])
            except Exception:
                pass
        # Zusätzlicher Fallback: wenn es nur 'current_ball' gibt, nutze diesen für 'Balls Played'
        if "Balls Played" not in out2 and "current_ball" in out2:
            try:
                out2["Balls Played"] = int(out2["current_ball"])
            except Exception:
                pass
        return out2

    # --- PATCH: _load_map_from_local_rel auf Fallback-Downloader umstellen ---
    def _load_map_from_local_rel(self, rel) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        if rel.startswith("maps/"):
            rel = rel[len("maps/"):]
        local = os.path.join(p_local_maps(self.cfg), rel.replace("/", os.sep))
        if not os.path.exists(local):
            try:
                url = f"{GITHUB_BASE}/maps/{rel.lstrip('/')}"
                mj = _fetch_json_url(url, timeout=25)
                save_json(local, mj)
            except Exception as e:
                log(self.cfg, f"Map fetch failed {rel}: {e}", "WARN")
                return None, None
        mj = load_json(local, {}) or {}
        return self.parse_map(mj), local

    def _emit_mini_info_if_missing_map(self, rom: str, seconds: int = 5):
        """
        Show small info overlay for N seconds if no base map exists strictly in BASE\\NVRAM_Maps\\maps.
        """
        try:
            if not rom:
                return
            maps_dir = p_local_maps(self.cfg)
            cand1 = os.path.join(maps_dir, f"{rom}.json")
            cand2 = os.path.join(maps_dir, f"{rom}.map.json")
            if not (os.path.isfile(cand1) or os.path.isfile(cand2)):
                if getattr(self, "bridge", None):
                    self.bridge.mini_info_show.emit(rom, int(seconds))
        except Exception:
            pass


    def load_map_for_rom(self, rom):
        if not rom:
            return None, None
        # 1) Override
        override = os.path.join(p_overrides(self.cfg), rom + ".json")
        if os.path.exists(override):
            return self.parse_map(load_json(override, {}) or {}), override
        # 2) Lokale Aliasse (json + map.json)
        alias = os.path.join(p_local_maps(self.cfg), rom + ".json")
        if os.path.exists(alias):
            return self.parse_map(load_json(alias, {}) or {}), alias
        alias2 = os.path.join(p_local_maps(self.cfg), rom + ".map.json")
        if os.path.exists(alias2):
            return self.parse_map(load_json(alias2, {}) or {}), alias2
        # 3) INDEX
        entry = self.INDEX.get(rom)
        if entry:
            rel = entry if isinstance(entry, str) else (entry.get("path") or entry.get("file"))
            if rel:
                f, p = self._load_map_from_local_rel(rel)
                if f:
                    return f, p
        # 4) romnames-Umleitung
        base_rom = self.ROMNAMES.get(rom)
        if base_rom and base_rom != rom:
            f, p = self.load_map_for_rom(base_rom)
            if f and p:
                save_json(alias, load_json(p, {}) or {})
                log(self.cfg, f"Saved alias map via romnames -> {alias}")
                return f, alias
        # 5) Familien-Prefix
        prefix = str(rom).split("_")[0].lower()
        for cand in list(self.INDEX.keys()):
            if not cand.lower().startswith(prefix) or cand == rom:
                continue
            e2 = self.INDEX.get(cand)
            rel2 = e2 if isinstance(e2, str) else (e2.get("path") or e2.get("file"))
            if not rel2:
                continue
            f2, p2 = self._load_map_from_local_rel(rel2)
            if f2:
                save_json(alias, load_json(p2, {}) or {})
                log(self.cfg, f"Saved alias map via family prefix -> {alias}")
                return f2, alias
        return None, None

    @staticmethod
    def decode_bcd(raw: bytes) -> Optional[int]:
        digits = []
        for b in raw:
            hi, lo = (b >> 4) & 0xF, b & 0xF
            if hi > 9 or lo > 9:
                return None
            digits.append(str(hi))
            digits.append(str(lo))
        s = "".join(digits).lstrip("0")
        return int(s) if s else 0

    @staticmethod
    def decode_uint(raw: bytes, endian: Optional[str], signed: bool) -> int:
        e = "big" if (endian or "be") in ("be", "big") else "little"
        return int.from_bytes(raw, e, signed=bool(signed))

    @staticmethod
    def _plausibility_score(label, value):
        if value is None:
            return 1e12
        if value < 0:
            return 1e9
        caps = {"bumper": 200000, "spinner": 500000, "ramp": 200000, "ball": 100000, "extra": 10000}
        v = str(label).lower()
        cap = 500000
        for k, c in caps.items():
            if k in v:
                cap = c
                break
        penalty = (value - cap) * 10 if value > cap else 0
        return value + penalty

    def _decode_field_value(self, raw: bytes, fld: dict):
        """
        Unterstützt encoding: None (uint), 'bcd', 'int', 'bool'
        Optional: 'mask' (Bitmaske), 'value_offset' (Wertversatz) nach Dekodierung.
        """
        offset = int(fld["offset"])
        size = int(fld["size"])
        enc = (fld.get("encoding") or "").lower() or None
        endian = fld.get("endian") or "be"
        signed = bool(fld.get("signed", False))
        scale = float(fld.get("scale", 1.0))
        if offset < 0 or offset + size > len(raw):
            return None
        window = raw[offset: offset + size]

        if enc == "bcd":
            val = self.decode_bcd(window)
        elif enc in ("int", "uint", "sint"):
            val = self.decode_uint(window, endian, signed)
        elif enc == "bool":
            val = self.decode_uint(window, endian, False)
            val = 1 if int(val or 0) != 0 else 0
        else:
            val = self.decode_uint(window, endian, signed)

        if val is None:
            return None

        try:
            mask = int(fld.get("mask", 0) or 0)
            if mask:
                val = int(val) & mask
        except Exception:
            pass

        if scale != 1.0:
            try:
                val = int(int(val) * scale)
            except Exception:
                val = int(val)

        try:
            vo = int(fld.get("value_offset", 0) or 0)
            if vo:
                val = int(val) + vo
        except Exception:
            pass

        return int(val)

    def auto_fix_field(self, raw: bytes, base_enc, base_end, base_size, signed, label):
        sizes = sorted({int(base_size or 2), int(base_size or 2) + 1, int(base_size or 2) + 2})
        candidates = []
        for sz in sizes:
            if sz > len(raw):
                continue
            chunk = raw[:sz]
            encs = [base_enc] if base_enc else [None, "bcd"]
            if "bcd" not in encs:
                encs.append("bcd")
            for enc in encs:
                if enc == "bcd":
                    val = self.decode_bcd(chunk)
                    if val is not None:
                        candidates.append((val, {"encoding": "bcd", "endian": None, "size": sz}))
                else:
                    for e in ("be", "le"):
                        val = self.decode_uint(chunk, e, signed)
                        candidates.append((val, {"encoding": None, "endian": e, "size": sz}))
        best, cfg, best_score = None, None, 1e18
        for val, c in candidates:
            sc = self._plausibility_score(label, int(val))
            if sc < best_score:
                best, cfg, best_score = int(val), c, sc
        return best, cfg

    def _load_cached_layout(self, rom: str):
        return self._field_layout_cache.get(rom)

    def _store_cached_layout(self, rom: str, layout_fields: List[dict]):
        self._field_layout_cache[rom] = {
            "fields": layout_fields,
            "cache_time": time.time()
        }

    def read_nvram_audits_with_autofix(self, rom: str) -> Tuple[Dict[str, Any], List[str], bool]:
        if not rom:
            return {}, [], False
        nv_path = os.path.join(self.cfg.NVRAM_DIR, rom + ".nv")
        if not os.path.exists(nv_path):
            return {}, [], False
        try:
            with open(nv_path, "rb") as f:
                raw = f.read()
        except Exception:
            return {}, [], False

        cached = self._load_cached_layout(rom)
        if cached:
            audits = {}
            notes: List[str] = []
            for fld in cached["fields"]:
                try:
                    label = fld["label"]
                    val = self._decode_field_value(raw, fld)
                    if val is None:
                        continue
                    audits[label] = val
                except Exception:
                    continue
            try:
                self._ensure_rom_specific(rom, audits)
            except Exception as e:
                log(self.cfg, f"[ROM_SPEC] ensure failed (cached path): {e}", "WARN")
            return audits, notes, False

        fields, _ = self.load_map_for_rom(rom)
        if not fields:
            return {}, [], False
        audits, notes, need_override, fixed_fields = {}, [], False, []
        for fld in fields:
            try:
                label = (fld.get("label") or fld.get("name") or "field")
                offset = int(fld.get("offset", 0))
                size = int(fld.get("size", 2))
                enc = (fld.get("encoding") or "").lower() or None
                endian = (fld.get("endian") or "").lower() or None
                scale = float(fld.get("scale") or 1.0)
                signed = bool(fld.get("signed", False))
                if offset < 0 or offset + size > len(raw):
                    continue
                win_len = max(4, min(6, size + 2))
                window = raw[offset: min(len(raw), offset + win_len)]
                best, cfg = self.auto_fix_field(window, enc, endian, size, signed, label)
                val = int(best or 0)
                if scale != 1.0:
                    val = int(val * scale)
                audits[label] = val
                enc_new = (cfg or {}).get("encoding")
                end_new = (cfg or {}).get("endian")
                size_new = int((cfg or {}).get("size") or size)
                spec = {
                    "name": fld.get("name") or label,
                    "label": label,
                    "offset": offset,
                    "size": size_new,
                    "encoding": enc_new,
                    "endian": end_new,
                    "scale": scale,
                    "signed": signed,
                    # Masken/Offsets aus Map mit übernehmen
                    "mask": self._to_int(fld.get("mask", 0), 0),
                    "value_offset": self._to_int(fld.get("value_offset", 0), 0),
                }
                # Entfernt: kein erzwungenes mask/+1 mehr für current_player
                # (Wir übernehmen Mask/Offset ausschließlich aus der Map/Auto-Fix-Erkennung.)
                fixed_fields.append(spec)
                if (enc_new or None) != (enc or None) or (end_new or None) != (endian or None) or size_new != size:
                    need_override = True
                    notes.append(f"[AUTO-FIX] {label}: enc {enc}->{enc_new}, endian {endian}->{end_new}, size {size}->{size_new}")
            except Exception as e:
                notes.append(f"[READ-WARN] {fld} -> {e}")

        if need_override:
            try:
                override_path = os.path.join(p_overrides(self.cfg), f"{rom}.json")
                if save_json(override_path, {"fields": fixed_fields}):
                    log(self.cfg, f"Override written: {override_path}")
            except Exception as e:
                notes.append(f"[OVERRIDE-WARN] {e}")

        self._store_cached_layout(rom, fixed_fields)
        try:
            self._ensure_rom_specific(rom, audits)
        except Exception as e:
            log(self.cfg, f"[ROM_SPEC] ensure failed: {e}", "WARN")
        return audits, notes, need_override

    HIGHLIGHT_RULES = {
        "multiball": {"cat": "Power", "emoji": "💥", "label": "Multiball Frenzy", "type": "count"},
        "jackpot": {"cat": "Power", "emoji": "🎯", "label": "Jackpot Hunter", "type": "count"},
        "super_jackpot": {"cat": "Power", "emoji": "💎", "label": "Super Jackpot", "type": "count"},
        "triple_jackpot": {"cat": "Power", "emoji": "👑", "label": "Triple Jackpot", "type": "count"},
        "ball_save": {"cat": "Power", "emoji": "🛡️", "label": "Ball Saves", "type": "count"},
        "extra_ball": {"cat": "Power", "emoji": "➕", "label": "Extra Balls", "type": "count"},
        "special_award": {"cat": "Power", "emoji": "🎁", "label": "Special Awards", "type": "count"},
        "mode_completed": {"cat": "Power", "emoji": "🏆", "label": "Modes Completed", "type": "count"},
        "best_ball": {"cat": "Power", "emoji": "🔥", "label": "Best Ball", "type": "always"},
        "wizard_mode": {"cat": "Power", "emoji": "🧙", "label": "Wizard Mode", "type": "flag"},
        "loops": {"cat": "Precision", "emoji": "🔁", "label": "Loop Machine", "type": "count"},
        "spinner": {"cat": "Precision", "emoji": "🌀", "label": "Spinner Madness", "type": "count"},
        "combo": {"cat": "Precision", "emoji": "🎯", "label": "Combo King", "type": "count"},
        "drop_targets": {"cat": "Precision", "emoji": "🎯", "label": "Target Slayer", "type": "count"},
        "ramps": {"cat": "Precision", "emoji": "🏹", "label": "Rampage", "type": "count"},
        "orbit": {"cat": "Precision", "emoji": "🌌", "label": "Orbit Runner", "type": "count"},
        "skillshot": {"cat": "Precision", "emoji": "🎯", "label": "Skill Shot", "type": "count"},
        "super_skillshot": {"cat": "Precision", "emoji": "💥", "label": "Super Skill Shot", "type": "count"},
        "mode_starts": {"cat": "Precision", "emoji": "🎬", "label": "Modes Started", "type": "count"},
        "tilt_warnings": {"cat": "Fun", "emoji": "🛡️", "label": "Tilt Warnings", "type": "count"},
        "tilt": {"cat": "Fun", "emoji": "💀", "label": "Tilted", "type": "count"},
        "devils_number": {"cat": "Fun", "emoji": "👹", "label": "Devil’s Number", "type": "flag"},
        "match": {"cat": "Fun", "emoji": "🎲", "label": "Match Lucky", "type": "count"},
        "initials": {"cat": "Fun", "emoji": "😂", "label": "Initials", "type": "flag"},
    }

    EVENT_KEYWORDS = {
        "jackpot": ["jackpot"],
        "super_jackpot": ["super jackpot", "super-jackpot"],
        "triple_jackpot": ["triple jackpot", "triple-jackpot"],
        "multiball": ["multiball", "multi-ball", "multi ball", "multiballs", "mb"],
        "ball_save": ["ball save", "ball saves"],
        "extra_ball": ["extra ball", "extra balls"],
        "special_award": ["special"],
        "loops": ["loop", "loops"],
        "spinner": ["spinner"],
        "combo": ["combo", "combos"],
        "drop_targets": ["drop target", "targets"],
        "ramps": ["ramp", "ramps"],
        "orbit": ["orbit", "orbits"],
        "skillshot": ["skill shot", "skillshot"],
        "super_skillshot": ["super skill", "super skillshot"],
        "mode_starts": ["mode start", "modes started"],
        "mode_completed": ["mode complete", "modes completed"],
        "tilt_warnings": ["tilt warn", "tilt warning", "tilt warnings"],
        "tilt": [" tilt "],
        "match": ["match awards", "match"],
    }

    NOISE_REGEX = re.compile(r"(minutes on|play time|recent|total .*slot|paid cred|serv|factory|reset|cleared|burn|clock|coins|h\.s\.t\.d)", re.I)
    KEYWORD_FALLBACK = [
        "jackpot", "multiball", "skill", "mode", "lock", "locks", "extra", "ball save", "save", "wave",
        "combo", "martian", "video", "hurry", "random", "tilt", "wizard",
        "games started", "balls locked", "locks lit", "extra balls", "ball saves", "bonus", "mode start",
        "mode compl", "annihil", "martn.", "strobe"
    ]


    def _find_vpx_pid(self) -> Optional[int]:
        """
        PID von Visual Pinball über Fenster-Titel holen.
        """
        if not win32gui:
            return None
        hwnd = {"h": None}
        def _cb(h, _):
            if win32gui.IsWindowVisible(h):
                title = win32gui.GetWindowText(h)
                if title.startswith("Visual Pinball - ["):
                    hwnd["h"] = h
                    return False
            return True
        try:
            win32gui.EnumWindows(_cb, None)
        except Exception:
            return None
        if not hwnd["h"]:
            return None
        pid = wintypes.DWORD(0)
        try:
            ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd["h"]), ctypes.byref(pid))
            return int(pid.value or 0) or None
        except Exception:
            return None

    # In class Watcher einfügen (z. B. direkt unter _find_vpx_pid)
    def _graceful_close_visual_pinball_player(self, timeout_s: float = 5.0) -> bool:
        """
        Schließt den Visual Pinball Player möglichst "sanft":
        - Erst SC_CLOSE/WM_CLOSE
        - Dann ALT+F4 als Fallback
        - Toleranter Fenstertitel-Match (VPX, DX9, etc.)
        Gibt True zurück, wenn mindestens ein Fenster angesprochen wurde.
        """
        try:
            import time
            import win32con
            import win32gui
            import win32api
            import win32process

            matched = []

            def _match_vpx_title(title: str) -> bool:
                t = (title or "").strip().lower()
                return (
                    ("pinball" in t and "player" in t)
                    or t.startswith("visual pinball player")
                    or t.startswith("vpinballx player")
                    or t.startswith("visual pinball x")
                )

            def _enum_handler(hwnd, acc):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd) or ""
                if _match_vpx_title(title):
                    acc.append(hwnd)

            hwnds = []
            win32gui.EnumWindows(_enum_handler, hwnds)

            if not hwnds:
                try:
                    log(self.cfg, "[CLOSE] No VPX Player windows found", "WARN")
                except Exception:
                    pass
                return False

            # 1) Versuche SC_CLOSE/WM_CLOSE
            for hwnd in hwnds:
                try:
                    win32gui.PostMessage(hwnd, win32con.WM_SYSCOMMAND, win32con.SC_CLOSE, 0)
                    matched.append(hwnd)
                except Exception:
                    pass

            time.sleep(min(0.5, timeout_s))

            # 2) Fallback ALT+F4 falls noch offen
            for hwnd in hwnds:
                try:
                    if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd):
                        # ALT down + F4 down/up + ALT up
                        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                        win32api.keybd_event(win32con.VK_F4, 0, 0, 0)
                        win32api.keybd_event(win32con.VK_F4, 0, win32con.KEYEVENTF_KEYUP, 0)
                        win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                except Exception:
                    pass

            try:
                log(self.cfg, f"[CLOSE] Attempted to close {len(hwnds)} VPX window(s)")
            except Exception:
                pass
            return True

        except Exception as e:
            try:
                log(self.cfg, f"[CLOSE] _graceful_close_visual_pinball_player error: {e}", "WARN")
            except Exception:
                try:
                    print(f"[CLOSE] _graceful_close_visual_pinball_player error: {e}")
                except Exception:
                    pass
            return False
        
    def _kill_vpx_process(self):
        """
        Beende NUR den Visual Pinball Player:
        - 1) "Sanft" via WM_CLOSE (graceful)
        - 2) Fallback ALT+F4 (mit kurzer Wartezeit)
        - 3) Letzter Fallback erneut WM_CLOSE auf exakte Titel
        Kein taskkill.
        """
        # 1) Graceful
        try:
            if hasattr(self, "_graceful_close_visual_pinball_player"):
                if self._graceful_close_visual_pinball_player(timeout_s=3.0):
                    log(self.cfg, "[CHALLENGE] VP Player closed (graceful)")
                    return
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] graceful close failed: {e}", "WARN")

        # 2) ALT+F4 Fallback
        try:
            if self._alt_f4_visual_pinball_player(wait_ms=3000):
                log(self.cfg, "[CHALLENGE] VP Player closed via Alt+F4")
                return
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] Alt+F4 failed: {e}", "WARN")

        # 3) Letzter Fallback: WM_CLOSE auf exakten Titel
        try:
            import win32gui, win32con
            def _cb(hwnd, _):
                try:
                    if not win32gui.IsWindowVisible(hwnd):
                        return True
                    title = (win32gui.GetWindowText(hwnd) or "").strip()
                    if title.startswith("Visual Pinball Player"):
                        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                except Exception:
                    pass
                return True

            win32gui.EnumWindows(_cb, None)
            log(self.cfg, "[CHALLENGE] VP Player requested close via WM_CLOSE (fallback)")
        except Exception:
            log(self.cfg, "[CHALLENGE] WARNING: could not send WM_CLOSE to Visual Pinball Player", "WARN")

    def _challenge_mark_pending_kill(self, delay_sec: int = 5):
        """
        Mark VPX termination after N seconds and suppress big overlay once.
        """
        try:
            ch = getattr(self, "challenge", {})
            ch["pending_kill_at"] = time.time() + max(0, int(delay_sec))
            ch["suppress_big_overlay_once"] = True
            self.challenge = ch
        except Exception:
            pass

    def start_timed_challenge(self, total_seconds: int = 330):
        """
        Start 'Timed Challenge'.
        Default: 330s total = 30s warm-up + 300s countdown.
        - Emits warm-up banner immediately
        - Schedules bottom-left countdown via GUI after warmup (handled in MainWindow)
        - Arms end_at for automatic kill at the end of the total window
        Duplicate guard: ignores if a timed challenge is already active.
        """
        try:
            if not self.game_active or not self.current_rom:
                log(self.cfg, "[CHALLENGE] timed: ignored (no active game)", "WARN")
                return

            ch = getattr(self, "challenge", {}) or {}
            if ch.get("active") and ch.get("kind") == "timed":
                log(self.cfg, "[CHALLENGE] timed already active – ignored duplicate")
                return

            warmup = 30
            total = max(1, int(total_seconds))
            countdown = max(1, total - warmup)

            ch.clear()
            ch.update({
                "active": True,
                "kind": "timed",
                "rom": self.current_rom,
                "table": self.current_table,
                "started_at": time.time(),
                "end_at": time.time() + total,          # Ende = warmup + countdown
                "pending_kill_at": None,
                "suppress_big_overlay_once": True,
            })
            self.challenge = ch

            # GUI: Warm-up Banner + (verzögert) Countdown starten
            try:
                self.bridge.challenge_warmup_show.emit(warmup, "Timed challenge – warm-up")
                # Wichtig: wir geben weiterhin 'total' an; der GUI-Handler rechnet warmup ab
                self.bridge.challenge_timer_start.emit(total)
            except Exception:
                pass

            log(self.cfg, f"[CHALLENGE] timed armed – warmup={warmup}s, countdown={countdown}s (total={total}s)")
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] timed start failed: {e}", "WARN")

    def stop_timed_challenge(self):
        """
        Stop the timed challenge (UI timer will be closed via signal). Does not kill VPX.
        """
        ch = getattr(self, "challenge", {})
        if ch.get("kind") == "timed":
            ch["active"] = False
            self.challenge = ch
            try:
                self.bridge.challenge_timer_stop.emit()
            except Exception:
                pass
            log(self.cfg, "[CHALLENGE] timed stopped")

    def start_one_ball_challenge(self):
        try:
            if not self.game_active or not self.current_rom:
                log(self.cfg, "[CHALLENGE] one-ball: ignored (no active game)", "WARN")
                return
            ch = getattr(self, "challenge", {}) or {}
            if ch.get("active") and ch.get("kind") == "oneball" and ch.get("one_ball_active", False):
                log(self.cfg, "[CHALLENGE] one-ball already armed – ignored duplicate")
                return
            ch.clear()
            baseline = self._get_balls_played(self._last_audits_global or self.start_audits) or 0
            ch.update({
                "active": True,
                "kind": "oneball",
                "rom": self.current_rom,
                "table": self.current_table,
                "started_at": time.time(),
                "baseline_bp": int(baseline),
                "one_ball_active": True,
                "one_ball_done": False,
                "end_at": None,
                "pending_kill_at": None,
                "suppress_big_overlay_once": True
            })
            self.challenge = ch
            try:
                self.bridge.challenge_info_show.emit("One-Ball challenge – good luck!", 5, "#34C759")
                self.bridge.challenge_speak.emit("One ball challenge started")
            except Exception:
                pass
            log(self.cfg, f"[CHALLENGE] one-ball armed (baseline Balls Played={baseline})")
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] one-ball start failed: {e}", "WARN")


    def stop_one_ball_challenge(self):
        ch = getattr(self, "challenge", {})
        if ch.get("kind") == "oneball":
            ch["one_ball_active"] = False
            ch["active"] = False
            self.challenge = ch
            log(self.cfg, "[CHALLENGE] one-ball disarmed")

# KLEINER FIX: vor Pre-Kill-Snapshot bei Timed-Challenge kurz warten
    def _challenge_tick(self, audits: dict):
        """
        Called every loop iteration while VPX is active.
        """
        try:
            ch = getattr(self, "challenge", {}) or {}
            if not ch or not ch.get("active"):
                return
            now = time.time()

            if ch.get("kind") == "timed":
                end_at = float(ch.get("end_at", 0.0) or 0.0)
                if now >= end_at and not ch.get("pending_kill_at"):
                    # Kurze Pause, damit Komponenten flushen können
                    try:
                        time.sleep(0.5)
                    except Exception:
                        pass

                    # Pre-kill snapshot
                    try:
                        audits_now, _, _ = self.read_nvram_audits_with_autofix(self.current_rom)
                        if audits_now:
                            self._last_audits_global = dict(audits_now)
                            try:
                                duration_now = int(now - (self.start_time or now))
                                self.export_overlay_snapshot(audits_now, duration_now, on_demand=True)
                            except Exception:
                                pass
                            ch["prekill_end"] = dict(audits_now)
                    except Exception:
                        pass

                    ch["pending_kill_at"] = time.time() + 3.0
                    self.challenge = ch
                    log(self.cfg, "[CHALLENGE] timed finished – will kill VP Player in 3s")
                    return

            if ch.get("pending_kill_at") and now >= float(ch["pending_kill_at"]):
                ch["pending_kill_at"] = None
                ch["active"] = False
                self.challenge = ch
                self._kill_vpx_process()
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] tick failed: {e}", "WARN")


    def _challenge_best_final_score(self, end_audits: dict, pid: int = 1) -> int:
        """
        Liefert den bestmöglichen finalen Score für Spieler pid speziell für Challenges.
        Reihenfolge:
          1) Wert aus end_audits
          2) Wert aus Live-Cache (_last_audits_global)
          3) Beste Schätzung aus Ball-Tracker
        """
        try:
            # 1) End-Audits
            key = f"P{pid} Score"
            v = int((end_audits or {}).get(key, 0) or 0)
            if v > 0:
                return v

            # 2) Live-Cache
            cache = getattr(self, "_last_audits_global", {}) or {}
            cv = int(cache.get(key, 0) or 0)
            if cv > 0:
                return cv

            # 3) Ball-Tracker (nur P1 sinnvoll, falls vorhanden)
            if pid == 1:
                try:
                    balls = (self.ball_track or {}).get("balls", []) or []
                    if balls:
                        best_ball = max(balls, key=lambda b: (int(b.get("score", 0)), int(b.get("duration", 0))))
                        bv = int(best_ball.get("score", 0) or 0)
                        if bv > 0:
                            return bv
                except Exception:
                    pass
        except Exception:
            pass
        return 0


    def _inject_best_score_for_timed(self, end_audits: dict) -> dict:
        """
        Gibt end_audits zurück, bei Timed Challenge mit robustem P1-Score überschrieben.
        Macht nichts für nicht-timed Challenges.
        """
        try:
            ch = getattr(self, "challenge", {}) or {}
            if str(ch.get("kind", "")).lower() != "timed":
                return dict(end_audits or {})

            ea = dict(end_audits or {})
            best = self._challenge_best_final_score(ea, pid=1)
            if best > 0:
                ea["P1 Score"] = best
                try:
                    log(self.cfg, f"[CHALLENGE] timed: injected best P1 Score={best}")
                except Exception:
                    pass
            return ea
        except Exception:
            return dict(end_audits or {})



    def _challenge_record_result(self, kind: str, end_audits: dict, duration_sec: int):
        """
        Persist a challenge result under BASE/challenges/history/<rom>.json.
        Also show a small centered result banner for 10s (white text).
        """
        try:
            ch = getattr(self, "challenge", {}) or {}
            rom = ch.get("rom") or self.current_rom or ""
            if not rom:
                return
            table = ch.get("table") or self.current_table or ""
            try:
                score = int(self._find_score_from_audits(end_audits) or 0)
            except Exception:
                score = 0
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": str(kind or ""),
                "rom": rom,
                "table": table,
                "duration_sec": int(duration_sec or 0),
                "score": int(score)
            }
            # Write
            out_dir = os.path.join(self.cfg.BASE, "challenges", "history")
            ensure_dir(out_dir)
            path = os.path.join(out_dir, f"{sanitize_filename(rom)}.json")
            hist = load_json(path, {"results": []}) or {"results": []}
            hist.setdefault("results", []).append(payload)
            save_json(path, hist)

            # Result banner
            try:
                msg = f"{'Timed' if kind=='timed' else 'One-Ball'} Challenge – Score: {score:,d}".replace(",", ".")
                self.bridge.challenge_info_show.emit(msg, 10, "#FFFFFF")
            except Exception:
                pass
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] record result failed: {e}", "WARN")




    def _fw_path(self, rom: str) -> str:
        return os.path.join(p_field_whitelists(self.cfg), f"{rom}.fields.json")

    def _fw_load(self, rom: str) -> dict:
        data = load_json(self._fw_path(rom), None)
        if not isinstance(data, dict):
            return {"rom": rom, "fields": {}}
        if "fields" not in data or not isinstance(data["fields"], dict):
            data["fields"] = {}
        return data

    # DeprecationWarning fix in _fw_save
    def _fw_save(self, rom: str, data: dict):
        data["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ensure_dir(p_field_whitelists(self.cfg))
        save_json(self._fw_path(rom), data)

    def _fw_load_active_whitelist(self, rom: str):
        data = self._fw_load(rom)
        wl = {f for f, meta in data["fields"].items() if meta.get("classification") == "whitelisted"}
        self.active_field_whitelist = wl
        log(self.cfg, f"[FW] Loaded whitelist for {rom}: {len(wl)} fields")

    def _fw_record_field_stats(self, pid: int, diff: dict, segment_duration: float):
        now = time.time()
        for label, delta in diff.items():
            fs = self.field_stats.setdefault(label, {
                "segments_with_delta": 0,
                "increments_sum": 0,
                "segments_touched": 0,
                "players_set": set(),
                "max_delta_single_segment": 0,
                "delay_hits": 0,
                "first_seen_ts": now,
                "last_seen_ts": now,
            })
            fs["segments_with_delta"] += 1
            fs["increments_sum"] += int(delta)
            fs["segments_touched"] += 1
            fs["players_set"].add(pid)
            if int(delta) > fs["max_delta_single_segment"]:
                fs["max_delta_single_segment"] = int(delta)
            fs["last_seen_ts"] = now
            last_seg = self.SNAP_PREV_SEGMENT_FIELDS.get(label)
            if last_seg is not None and last_seg == (self.snap_segment_index - 1):
                if segment_duration <= self.SNAP_DELAY_WINDOW_SEC:
                    fs["delay_hits"] += 1
            self.SNAP_PREV_SEGMENT_FIELDS[label] = self.snap_segment_index

    def _fw_classify_fields(self, rom: str):
        if not self.field_stats:
            log(self.cfg, "[FW] No field stats collected – deferring classification", "INFO")
            return
        if self.snap_segment_index < self.MIN_SEGMENTS_FOR_CLASSIFICATION:
            log(self.cfg, f"[FW] Too few segments (< {self.MIN_SEGMENTS_FOR_CLASSIFICATION + 1}) – deferring", "INFO")
            return

        data = self._fw_load(rom)
        fldmap = data["fields"]
        total_segments = max(1, self.snap_segment_index)
        promotions_needed = False

        for label, st in self.field_stats.items():
            seg_with_delta = st["segments_with_delta"]
            inc_sum = st["increments_sum"]
            segs = st["segments_touched"]
            players = len(st["players_set"])
            avg_delta = inc_sum / max(1, seg_with_delta)
            ssr = segs / total_segments
            max_seg = st["max_delta_single_segment"]
            delay_rate = st["delay_hits"] / max(1, seg_with_delta)
            noise_match = bool(self.NOISE_REGEX.search(label))

            auto_score = 0
            auto_score += 30 if players > 0 else 0
            auto_score += 25 if segs >= 2 else 0
            auto_score += 15 * min(1.0, avg_delta / 3.0)
            auto_score += 20 if max_seg >= 2 else 0
            if ssr > 0.9 and avg_delta < 2:
                auto_score -= 25
            if delay_rate > 0.3:
                auto_score -= 40
            if noise_match:
                auto_score -= 100

            if auto_score >= 40:
                cls = "whitelisted"
            elif auto_score >= 10:
                cls = "candidate"
            else:
                cls = "noise"

            prev = fldmap.get(label, {})
            manual = prev.get("manual_override")

            # Niemals whitelisted zurückstufen (außer force_noise)
            if prev.get("classification") in ("whitelisted", "stale") and manual != "force_noise":
                cls = "whitelisted"

            if manual == "force_whitelist":
                cls = "whitelisted"
            elif manual == "force_noise":
                cls = "noise"

            fldmap[label] = {
                "classification": cls,
                "auto_score": round(auto_score, 2),
                "segments_with_delta": seg_with_delta,
                "increments_sum": inc_sum,
                "segments_touched": segs,
                "players_touched": players,
                "max_delta_single_segment": max_seg,
                "delay_hits": st["delay_hits"],
                "first_seen_ts": prev.get("first_seen_ts", st["first_seen_ts"]),
                "last_seen_ts": st["last_seen_ts"]
            }

        # KEINE Demotion mehr auf "stale" – Whitelist soll nur wachsen.
        # (Optional: Wenn du "stale" behalten willst, kommentiere die nächste Schleife einfach aus.)
        # cutoff = time.time() - 14 * 24 * 3600
        # for lbl, meta in fldmap.items():
        #     if meta.get("classification") == "whitelisted" and meta.get("last_seen_ts", time.time()) < cutoff:
        #         meta["classification"] = "stale"

        # Falls noch keine whitelisted vorhanden: beste Kandidaten befördern
        if not any(meta.get("classification") == "whitelisted" for meta in fldmap.values()):
            candidates = [(lbl, m) for lbl, m in fldmap.items() if m.get("classification") == "candidate"]
            candidates.sort(key=lambda x: x[1].get("auto_score", 0), reverse=True)
            for lbl, _ in candidates[:8]:
                fldmap[lbl]["classification"] = "whitelisted"
                promotions_needed = True

        self._fw_save(rom, data)

        # Aktiv: Whitelist = 'whitelisted' ODER 'stale'
        wl = {f for f, meta in fldmap.items() if meta.get("classification") in ("whitelisted", "stale")}
        self.active_field_whitelist = wl
        if wl:
            self.bootstrap_phase = False

        log(self.cfg, f"[FW] Classified {len(fldmap)} fields -> whitelist={len(wl)} promoted={promotions_needed} bootstrap={self.bootstrap_phase}")

    def _snap_reset(self, audits: dict):
        """
        Initialisiert den Snapshot-/Segmentmodus. Setzt die Baseline und
        merkt sich den aktuellen 'Balls Played'-Stand case-insensitive.
        """
        self.snap_player = 1
        self.snap_start_audits = dict(audits)
        self.snap_initialized = True
        self.snap_players_in_game = 1
        self.snap_players_locked = False
        try:
            bp = self._get_balls_played(audits)
            self.snap_last_balls_played = int(bp) if bp is not None else None
        except Exception:
            self.snap_last_balls_played = None
        self.snap_segment_index = 0
        self.snap_segment_start_time = time.time()
        self.SNAP_PREV_SEGMENT_FIELDS.clear()
        self.field_stats.clear()
        self.current_segment_provisional_diff = {}


    def _snap_detect_players(self, audits: dict):
        # 1) Direkt aus game_state.player_count (via Map/Injector), case-insensitive
        try:
            pc = int(self._nv_get_int_ci(audits, "player_count", 0))
            if 1 <= pc <= 4:
                if pc != int(self.snap_players_in_game or 1):
                    self.snap_players_in_game = pc
                if pc >= 2:
                    self.snap_players_locked = True
                return
        except Exception:
            pass

        # Wenn schon sicher gelockt auf >=2 Spieler, nicht weiter raten
        if self.snap_players_locked and self.snap_players_in_game >= 2:
            return

        # 2) Fallback: Audit-Deltas seit Session-Start
        def d(lbl_ci: str) -> int:
            try:
                s = int(self._nv_get_int_ci(self.start_audits, lbl_ci, 0))
                e = int(self._nv_get_int_ci(audits, lbl_ci, 0))
                return max(0, e - s)
            except Exception:
                return 0

        if d("4 Player Games") > 0:
            self.snap_players_in_game = 4
            self.snap_players_locked = True
            return
        if d("3 Player Games") > 0 and self.snap_players_in_game < 3:
            self.snap_players_in_game = 3
        if d("2 Player Games") > 0 and self.snap_players_in_game < 2:
            self.snap_players_in_game = 2

        # 3) Letzter Fallback: Präsenz von Pn Score im Snapshot
        for pid in (4, 3, 2):
            key = f"P{pid} Score"
            try:
                if int(audits.get(key, 0) or 0) > 0 and self.snap_players_in_game < pid:
                    self.snap_players_in_game = pid
            except Exception:
                pass
  
    def _snap_field_allowed(self, label: str) -> bool:
           # Harte Filterung ‘BONUS X’
        if self._exclude_bonus_x_field(label):
            return False
        # Grundfilter
        if not isinstance(label, str):
            return False

        # Harte Excludes (spezielle Last-/Reset-/Timestamp-/Overall-Felder)
        if is_excluded_field(label):
            return False

        ll = label.lower()

        # Scores generell raus
        if "score" in ll:
            return False

        # Generisches Rauschen
        if self.NOISE_REGEX.search(label):
            return False

        # Wenn eine Whitelist existiert, erlauben wir:
        # 1) Whitelisted Felder
        # 2) Zusätzlich plausible Felder (Exploration), damit die Whitelist wachsen kann
        if self.active_field_whitelist:
            if label in self.active_field_whitelist:
                return True

            # Exploration-Heuristik: Keywords/Generics/kurze plausible Labels
            if any(k in ll for k in self.KEYWORD_FALLBACK):
                return True

            GENERIC_OK = [
                "games started", "balls played", "extra balls", "locks lit", "balls locked",
                "ball saves", "total multiballs", "bonus x", "annihil", "martian", "strobe",
                "video", "hurry up"
            ]
            if any(g in ll for g in GENERIC_OK):
                return True

            # sehr kurze, plausible Labels ohne typische Rausch-Wörter
            if len(label) <= 24 and not any(bad in ll for bad in ["total ", "recent", "minutes", "play time", "reset", "coin", "last "]):
                return True

            return False

        # Fallback-Heuristiken (bis Whitelist gelernt ist)
        if any(k in ll for k in self.KEYWORD_FALLBACK):
            return True

        GENERIC_OK = [
            "games started", "balls played", "extra balls", "locks lit", "balls locked",
            "ball saves", "total multiballs", "bonus x", "annihil", "martian", "strobe",
            "video", "hurry up"
        ]
        if any(g in ll for g in GENERIC_OK):
            return True

        # In ganz frühen Sessions: kurze, plausible Labels zulassen (ohne typische Rauschwörter)
        if not self.field_stats:
            if len(label) <= 24 and not any(bad in ll for bad in ["total ", "recent", "minutes", "play time", "reset", "coin", "last "]):
                return True

        return False


    def _exclude_bonus_x_field(self, label: str) -> bool:
        """
        Returns True if the given audit/field label should be excluded from Session-Deltas/Highlights,
        specifically filters out 'BONUS X' (case-insensitive, tolerates leading/trailing spaces).
        """
        try:
            ll = str(label).strip().lower()
        except Exception:
            return False
        # Exakt 'bonus x' oder Varianten wie 'bonus x ....' blocken
        return ll == "bonus x" or ll.startswith("bonus x ")
        
    def _snap_diff(self, start: dict, end: dict) -> dict:
        """
        Berechnet die Feldänderungen (Deltas) zwischen zwei Snapshots,
        korrigiert um globale Felder und mehrfach gezählte Werte.
        Nur whitelisted bzw. plausible Felder werden berücksichtigt.
        """
        diff = {}
        for label, v_end in end.items():
            if not isinstance(label, str):
                continue
            if label.startswith("P"):  # spielerspezifische Felder separat
                continue
            if not self._snap_field_allowed(label):
                continue
            try:
                s = int(start.get(label, 0) or 0)
                e = int(v_end or 0)
            except Exception:
                continue
            d = e - s

            # --- Korrektur: negative Deltas bei Wraparound abfangen ---
            if d < 0:
                d = (65536 - s) + e

            # --- Filter gegen globale Felder ---
            ll = label.lower()
            if "total" in ll or "overall" in ll or "recent" in ll or "cleared" in ll:
                continue

            # --- Nur plausible Zuwächse ---
            if 0 < d < 1000000:
                # Games Started darf nur im ersten Segment gezählt werden
                if "games started" in ll and self.snap_segment_index > 0:
                    continue
                diff[label] = d
        return diff


    def _snap_finalize_segment(self, audits: dict, note: str = ""):
        # Nur wenn SNAP aktiv und initialisiert
        if not self.snap_initialized:
            return
        # WICHTIG: Vor dem ersten echten Spiel KEINE Segmente finalisieren
        if not getattr(self, "_snap_bootstrap_done", False):
            return

        # Valider Spieler (1..4) zwingend
        try:
            pid = int(self.snap_player or 0)
        except Exception:
            pid = 0
        if pid <= 0 or pid > 4:
            log(self.cfg, f"[SNAP] finalize ignored (invalid pid={pid})", "WARN")
            return

        try:
            # Segmentdauer
            seg_start = self.snap_segment_start_time or time.time()
            segment_duration = float(time.time() - seg_start)

            # Deltas dieses Segments
            diff = self._snap_diff(self.snap_start_audits, audits)

            # Feld-Statistiken sammeln (für Auto-Whitelist)
            if diff:
                self._fw_record_field_stats(pid, diff, segment_duration)

                # Laufende Spieler-Session-Deltas auffüllen (nur Nicht-Score-Felder)
                try:
                    rec = self.players.setdefault(pid, {
                        "start_audits": self._player_field_filter(self.start_audits, pid) or {f"P{pid} Score": 0},
                        "last_audits": self._player_field_filter(self.start_audits, pid) or {f"P{pid} Score": 0},
                        "active_play_seconds": 0.0,
                        "start_time": time.time(),
                        "session_deltas": {},
                        "event_counts": {},
                    })
                    for k, v in diff.items():
                        if "score" in str(k).lower():
                            continue
                        rec["session_deltas"][k] = rec["session_deltas"].get(k, 0) + int(v)
                except Exception:
                    pass

            # Periodisch klassifizieren (z. B. alle 3 Segmente)
            try:
                if self.snap_segment_index >= self.MIN_SEGMENTS_FOR_CLASSIFICATION and (self.snap_segment_index % 3 == 0):
                    self._fw_classify_fields(self.current_rom)
            except Exception as e:
                log(self.cfg, f"[FW] live classification failed: {e}", "WARN")

        except Exception as e:
            log(self.cfg, f"[SNAP] finalize segment failed: {e}", "WARN")


    # In class Watcher: _snap_rotate – am Ende Debounce setzen
    def _snap_rotate(self, audits: dict, steps: int = 1):
        if not getattr(self, "_snap_bootstrap_done", False):
            return
        try:
            self._snap_finalize_segment(audits, note=f"(bp+{steps})")
        except Exception as e:
            log(self.cfg, f"[SNAP] finalize on rotate failed: {e}", "WARN")
        for _ in range(max(1, int(steps))):
            current = int(self.snap_player or 1)
            next_player = current + 1
            if next_player > max(1, int(self.snap_players_in_game or 1)):
                next_player = 1
            self.players.setdefault(current, {})
            self.players.setdefault(next_player, {})
            self.players[current]["end_audits"] = dict(audits)
            self.players[next_player]["start_audits"] = dict(audits)
            self.snap_player = next_player
            self.snap_segment_index += 1
        self.snap_start_audits = dict(audits)
        self.snap_segment_start_time = time.time()
        self._last_rotate_ts = self.snap_segment_start_time  # wichtig: eine Rotation pro Tick
        log(self.cfg, f"[SNAP] rotate -> active player {self.snap_player}", "INFO")

    def _maybe_rotate_on_current_player(self, audits: dict):
        """
        Rotiert auf Basis eines Current-Player-Felds aus den Audits (aus der Map normalisiert).
        Direkt nach Bootstrap optional gesperrt, um Pre-Game-Peaks zu ignorieren.
        """
        if not (self.snapshot_mode and self.snap_initialized and getattr(self, "_snap_bootstrap_done", False)):
            return

        # Sperrfenster nach Bootstrap: CP-Rotation kurz ignorieren
        try:
            if hasattr(self, "_cp_rotate_lock_until") and time.time() < float(self._cp_rotate_lock_until or 0):
                return
        except Exception:
            pass

        now = time.time()
        if now - getattr(self, "_last_rotate_ts", 0.0) < float(self.CP_MIN_ROTATE_INTERVAL):
            return
        try:
            cp_val = int(audits.get("current_player", 0) or 0)
        except Exception:
            cp_val = 0
        if not (1 <= cp_val <= max(1, int(self.snap_players_in_game or 1))):
            return
        try:
            current = int(self.snap_player or 1)
        except Exception:
            current = 1
        if cp_val == current:
            return
        try:
            self._snap_finalize_segment(audits, note="(cp-switch)")
        except Exception as e:
            log(self.cfg, f"[SNAP] finalize(cp-switch) failed: {e}", "WARN")
        self.players.setdefault(current, {})
        self.players.setdefault(cp_val, {})
        self.players[current]["end_audits"] = dict(audits)
        self.players[cp_val]["start_audits"] = dict(audits)
        self.snap_player = cp_val
        self.snap_start_audits = dict(audits)
        self.snap_segment_index += 1
        self.snap_segment_start_time = now
        self._last_rotate_ts = now
        log(self.cfg, f"[SNAP] current-player -> switched to P{self.snap_player}")
    
    def _maybe_rotate_on_score_delta(self, audits: dict):
        """
        Fallback-Rotation auf Basis von Pn Score-Deltas (ingame, live).
        Wenn genau EIN Spieler in diesem Tick Punkte gemacht hat, rotiert auf diesen Spieler.
        - Debounced über CP_MIN_ROTATE_INTERVAL
        - Ignoriert Cases mit mehreren gleichzeitigen Scorern im selben Tick
        """
        if not (self.snapshot_mode and self.snap_initialized and getattr(self, "_snap_bootstrap_done", False)):
            return
        now = time.time()
        if now - getattr(self, "_last_rotate_ts", 0.0) < float(self.CP_MIN_ROTATE_INTERVAL):
            return

        # Hole aktuelle Scores aus Audits
        deltas: Dict[int, int] = {}
        for pid in range(1, max(1, int(self.snap_players_in_game or 4)) + 1):
            key = f"P{pid} Score"
            try:
                cur = int(audits.get(key, 0) or 0)
            except Exception:
                cur = 0
            # Vorheriger Stand aus last_audits (falls leer -> start_audits -> 0)
            prev = 0
            try:
                prev = int(self.players.get(pid, {}).get("last_audits", {}).get(key, 0) or 0)
            except Exception:
                try:
                    prev = int(self.players.get(pid, {}).get("start_audits", {}).get(key, 0) or 0)
                except Exception:
                    prev = 0
            d = cur - prev
            if d > 0:
                deltas[pid] = d

        if not deltas:
            return
        # Eindeutigen Scorer ermitteln (genau einer mit maximalem Delta)
        sorted_p = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
        top_pid, top_delta = sorted_p[0]
        if len(sorted_p) > 1 and sorted_p[1][1] == top_delta:
            # unklar – mehrere Scorer gleicher Höhe im selben Tick -> ignorieren
            return

        try:
            current = int(self.snap_player or 1)
        except Exception:
            current = 1
        if top_pid == current:
            return

        # Segment finalisieren und rotieren
        try:
            self._snap_finalize_segment(audits, note="(score-switch)")
        except Exception as e:
            log(self.cfg, f"[SNAP] finalize(score-switch) failed: {e}", "WARN")
        self.players.setdefault(current, {})
        self.players.setdefault(top_pid, {})
        self.players[current]["end_audits"] = dict(audits)
        self.players[top_pid]["start_audits"] = dict(audits)
        self.snap_player = top_pid
        self.snap_start_audits = dict(audits)
        self.snap_segment_index += 1
        self.snap_segment_start_time = now
        self._last_rotate_ts = now
        log(self.cfg, f"[SNAP] score-delta -> switched to P{self.snap_player} (+{top_delta})", "INFO")    
    
    @staticmethod
    def _is_number(x):
        try:
            int(x)
            return True
        except Exception:
            return False

    @staticmethod
    def _extract_numeric(value):
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return 0

    def _compute_player_deltas(self, audits_end: dict):
        """
        Berechnet am Session-Ende pro Spieler die Deltas aus gespeicherten Audit-Ständen.
        Nutzt globale NVRAM-Audits, aber getrennt nach Spielerwechseln.
        - Läuft nur, wenn ein echtes Spiel gebootstrapped wurde.
        - Akzeptiert nur Spieler-IDs 1..4.
        - Weist jedem Spieler diff(start_audits, end_audits) zu.
        """
        # Falls nie ein echtes Spiel erkannt wurde -> keine Spieler-Deltas
        if not getattr(self, "_snap_bootstrap_done", False):
            return {}

        deltas_by_player = {}

        # Letzten aktiven Spieler abschließen (nur wenn pid 1..4 und vorhanden)
        try:
            last_pid = int(self.snap_player or 0)
        except Exception:
            last_pid = 0
        if 1 <= last_pid <= 4 and last_pid in self.players:
            self.players[last_pid]["end_audits"] = dict(audits_end)

        for pid, pdata in list(self.players.items()):
            # sichere int-ID und Bereich prüfen
            try:
                ipid = int(pid)
            except Exception:
                ipid = 0
            if ipid < 1 or ipid > 4:
                continue

            start = pdata.get("start_audits")
            end = pdata.get("end_audits")
            if not start or not end:
                continue

            diff = self._snap_diff(start, end)
            pdata["session_deltas"] = diff
            deltas_by_player[ipid] = diff

            log(self.cfg, f"[SNAP] Computed delta for P{ipid}: {len(diff)} fields")

        # Globale Summe aus allen Spieler-Deltas bilden (nur Information)
        self.global_session_deltas = {}
        for _p, diff in deltas_by_player.items():
            for k, v in diff.items():
                self.global_session_deltas[k] = self.global_session_deltas.get(k, 0) + v

        return deltas_by_player


    def _compute_player_deltas_end_only(self, start: dict, end: dict) -> dict[int, dict]:
        """
        Mehrspieler-Fallback OHNE Live-Segmente:
        - Globale Deltas Start->Ende (negatives => 0).
        - Korrigiert 'Games Started' und 'Balls Played' relativ zur Bootstrap-Basis.
        - Per-Player-Felder (Pn ...) direkt.
        - Globale Felder proportional verteilt – aber mit robusteren Gewichten:
            1) Score-Deltas (wenn vorhanden)
            2) sonst End-Scores (falls > 0)
            3) sonst Gleichverteilung (min. Gewicht 1 je Spieler)
        - Meta-Felder werden NICHT leistungsgewichtet, sondern sinnvoll verteilt:
            * 'Games Started', '[2-4] Player Games', 'Balls Played' -> kopiere zu ALLEN Spielern (1:1)
            * 'Game Time ...' -> gleichmäßig verteilen
        """
        # Spielerzahl sicher bestimmen (max 4, mind. 2 wenn Multi vermutet)
        nplayers = self._infer_players_in_game_from_audits(start, end)
        nplayers = max(2, min(int(nplayers or 2), 4))

        # Score-Deltas und End-Scores einsammeln
        score_deltas: dict[int, int] = self._score_deltas_from_audits(start, end, nplayers)
        end_scores: dict[int, int] = {}
        for pid in range(1, nplayers + 1):
            key = f"P{pid} Score"
            try:
                end_scores[pid] = int(end.get(key, 0) or 0)
            except Exception:
                end_scores[pid] = 0

        tot_delta = sum(max(0, int(v)) for v in score_deltas.values())
        tot_end = sum(max(0, int(v)) for v in end_scores.values())

        # Gewichte bestimmen
        weights: dict[int, float] = {}
        if tot_delta > 0:
            # 1) Score-Deltas
            for pid in range(1, nplayers + 1):
                weights[pid] = float(max(0, int(score_deltas.get(pid, 0))))
        elif tot_end > 0:
            # 2) End-Scores
            for pid in range(1, nplayers + 1):
                weights[pid] = float(max(0, int(end_scores.get(pid, 0))))
        else:
            # 3) Gleichverteilung (min. 1 je Spieler)
            for pid in range(1, nplayers + 1):
                weights[pid] = 1.0

        # Korrekturen (Games Started, Balls Played) relativ zur Bootstrap-Basis
        try:
            end_g = self._nv_get_int_ci(end, "Games Started", 0)
            base_g = int(getattr(self, "_snap_bootstrap_games",
                                 self._nv_get_int_ci(start, "Games Started", end_g)) or 0)
            corr_g_delta = max(0, int(end_g) - int(base_g))
        except Exception:
            corr_g_delta = None

        corr_bp_delta = None
        corr_bp_lbl = None
        try:
            kl_end = {str(k).lower(): k for k in end.keys()}
            kl_start = {str(k).lower(): k for k in start.keys()}
            for cand in ("balls played", "games balls played", "total balls played"):
                if cand in kl_end:
                    k_end = kl_end[cand]
                    k_sta = kl_start.get(cand, k_end)
                    end_bp = int(end.get(k_end, 0) or 0)
                    base_bp = int(getattr(self, "_snap_bootstrap_balls",
                                          int(start.get(k_sta, end_bp) or 0)) or 0)
                    corr_bp_delta = max(0, end_bp - base_bp)
                    corr_bp_lbl = k_end
                    break
        except Exception:
            corr_bp_delta = None
            corr_bp_lbl = None

        # Globale Deltas (mit strikterem Filter)
        CONTROL_FIELDS = {
            "current_player", "player_count", "current_ball", "balls played",
            "credits", "tilted", "game over", "tilt warnings"
        }
        global_deltas = {}
        for k, ve in end.items():
            if not isinstance(k, str) or k.startswith("P"):
                continue
            ll = k.lower()
            if ll in CONTROL_FIELDS or is_excluded_field(k) or self.NOISE_REGEX.search(k):
                continue
            try:
                s = int(start.get(k, 0) or 0)
                e = int(ve or 0)
            except Exception:
                continue
            d = e - s
            if d < 0:
                d = 0
            global_deltas[k] = d

        # Korrekturen in globale Deltas zurückschreiben
        if corr_g_delta is not None:
            km = {k.lower(): k for k in global_deltas.keys()}
            if "games started" in km:
                global_deltas[km["games started"]] = int(corr_g_delta)
        if corr_bp_delta is not None and corr_bp_lbl:
            global_deltas[corr_bp_lbl] = int(corr_bp_delta)

        # Per-Player-Felder direkt
        per_player_direct: dict[int, dict] = {pid: {} for pid in range(1, nplayers + 1)}
        for pid in range(1, nplayers + 1):
            sP = self._player_field_filter(start, pid)
            eP = self._player_field_filter(end, pid)
            for k, ve in eP.items():
                if "score" in str(k).lower():
                    continue
                try:
                    s = int(sP.get(k, 0) or 0)
                    e = int(ve or 0)
                    d = e - s
                except Exception:
                    d = 0
                if d < 0:
                    d = 0
                if d > 0:
                    per_player_direct[pid][k] = d

        # Ergebnis vorbereiten
        assigned_by_pid: dict[int, dict] = {pid: dict(per_player_direct[pid]) for pid in range(1, nplayers + 1)}

        # Regex für Meta-Felder
        re_copy_all = [
            re.compile(r"^\s*games\s+started\s*$", re.I),
            re.compile(r"^\s*[234]\s*player\s+games\s*$", re.I),
            re.compile(r"^\s*balls\s+played\s*$", re.I),
        ]
        re_equal_share = [
            re.compile(r"^\s*game\s+time\b", re.I),
        ]

        # Verteile globale Deltas
        for k, d in global_deltas.items():
            if d <= 0:
                continue
            kl = k.lower()

            # 1) Felder, die zu allen Spielern 1:1 kopiert werden
            if any(rx.match(k) for rx in re_copy_all):
                for pid in range(1, nplayers + 1):
                    assigned_by_pid[pid][k] = assigned_by_pid[pid].get(k, 0) + int(d)
                continue

            # 2) Felder, die gleichmäßig geteilt werden
            if any(rx.match(k) for rx in re_equal_share):
                equal_weights = {pid: 1.0 for pid in range(1, nplayers + 1)}
                alloc = self._allocate_int_proportional(int(d), equal_weights)
                for pid, share in alloc.items():
                    if share > 0:
                        assigned_by_pid[pid][k] = assigned_by_pid[pid].get(k, 0) + int(share)
                continue

            # 3) Standard: gewichtete Verteilung (robuste Gewichte)
            alloc = self._allocate_int_proportional(int(d), weights)
            for pid, share in alloc.items():
                if share > 0:
                    assigned_by_pid[pid][k] = assigned_by_pid[pid].get(k, 0) + int(share)

        # Debug: Gewichte einmal loggen
        try:
            w_str = ", ".join([f"P{pid}={int(weights.get(pid, 0))}" for pid in range(1, nplayers + 1)])
            self.cfg and log(self.cfg, f"[SNAP] end-only weights: {w_str}")
        except Exception:
            pass

        return assigned_by_pid


    @staticmethod
    def _player_field_filter(audits: dict, pid: int) -> dict:
        prefix = f"P{pid} "
        return {k: v for k, v in audits.items() if isinstance(k, str) and k.startswith(prefix)}

    def _find_score_from_audits(self, audits: dict, pid: Optional[int] = None) -> int:
        def _is_num(x):
            try:
                int(x)
                return True
            except Exception:
                return False
        if pid is not None:
            key = f"P{pid} Score"
            val = audits.get(key)
            return int(val) if _is_num(val) else 0
        scores = []
        for i in range(1, 5):
            key = f"P{i} Score"
            val = audits.get(key)
            if _is_num(val):
                scores.append(int(val))
        if scores:
            return max(scores)
        val = audits.get("Score")
        return int(val) if _is_num(val) else 0

    def _build_events_from_deltas(self, deltas: dict) -> dict:
        events = {k: 0 for k in self.HIGHLIGHT_RULES.keys()}
        for label, val in deltas.items():
            l = str(label).lower()
            for key, words in self.EVENT_KEYWORDS.items():
                if any(w in l for w in words):
                    events[key] = events.get(key, 0) + int(val or 0)
        return events

    def _attribute_events(self, audits: dict) -> bool:
        """
        Attribuiert Zuwächse aus globalen audits in die aktuellen Spieler-Records.
        Rückgabe: True, wenn mindestens ein Event_count inkrementiert wurde.
        - Ignoriert Pn-Felder und Score-Felder.
        - Aktualisiert self._last_global_for_player_attr für die nächste Tick-Vergleichsbasis.
        """
        if not audits or not getattr(self, "current_player", None) or (self.current_player not in self.players):
            # Keine Spieler aktiv / kein Grund zu attribuieren
            # Speichere trotzdem den letzten globalen Snapshot sauber für nächste Runde
            try:
                self._last_global_for_player_attr = {
                    k: v for k, v in audits.items() if isinstance(k, str) and not k.startswith("P")
                }
            except Exception:
                pass
            return False

        try:
            prev = getattr(self, "_last_global_for_player_attr", {}) or {}
            cur_player = int(self.current_player or 1)
            player_rec = self.players.setdefault(cur_player, {
                "start_audits": self._player_field_filter(self.start_audits, cur_player) or {f"P{cur_player} Score": 0},
                "last_audits": self._player_field_filter(self.start_audits, cur_player) or {f"P{cur_player} Score": 0},
                "active_play_seconds": 0.0,
                "start_time": time.time(),
                "session_deltas": {},
                "event_counts": {},
            })
        except Exception:
            return False

        changed = False
        for label, val_now in (audits or {}).items():
            if not isinstance(label, str):
                continue
            if label.startswith("P"):
                continue
            ll = label.lower()
            if "score" in ll:
                continue
            try:
                now_i = int(val_now or 0)
            except Exception:
                continue
            try:
                old_i = int(prev.get(label, 0) or 0)
            except Exception:
                old_i = 0
            diff = now_i - old_i
            if diff <= 0:
                continue

            # sichere Session-Deltas
            try:
                sd = player_rec.setdefault("session_deltas", {})
                sd[label] = sd.get(label, 0) + int(diff)
            except Exception:
                pass

            # Event-Schlüssel erkennen und zählen
            for ev_key, words in (self.EVENT_KEYWORDS or {}).items():
                if any(w in ll for w in words):
                    try:
                        ec = player_rec.setdefault("event_counts", {})
                        ec[ev_key] = ec.get(ev_key, 0) + int(diff)
                        changed = True
                    except Exception:
                        pass
                    break

        # Update last_global baseline (nur Nicht-Pn Felder)
        try:
            self._last_global_for_player_attr = {
                k: v for k, v in audits.items() if isinstance(k, str) and not k.startswith("P")
            }
        except Exception:
            pass

        return bool(changed)

    def _icon(self, key: str, prefer_ascii: bool | None = None) -> str:
        """
        Liefert ein Symbol pro Highlight-Schlüssel.
        Fallback: ASCII-Icons, damit im Overlay kein '??' erscheint, wenn Emoji-Fonts fehlen.
        Du kannst mit cfg.OVERLAY['prefer_ascii_icons'] = False Emojis erzwingen.
        """
        ov = getattr(self.cfg, "OVERLAY", {}) or {}
        use_ascii = ov.get("prefer_ascii_icons", True) if prefer_ascii is None else bool(prefer_ascii)

        if use_ascii:
            ascii_map = {
                "best_ball": "[BB]",
                "wizard_mode": "[WZ]",
                "multiball": "[MB]",
                "jackpot": "[JP]",
                "super_jackpot": "[SJP]",
                "triple_jackpot": "[TJP]",
                "ball_save": "[BS]",
                "extra_ball": "[EB]",
                "special_award": "[SPC]",
                "mode_completed": "[MODE✓]",
                "loops": "[LOOP]",
                "spinner": "[SPIN]",
                "combo": "[COMBO]",
                "drop_targets": "[DT]",
                "ramps": "[RAMP]",
                "orbit": "[ORBIT]",
                "skillshot": "[SS]",
                "super_skillshot": "[SS+]",
                "mode_starts": "[MODE]",
                "tilt_warnings": "[TILT!]",
                "tilt": "[TILT]",
                "devils_number": "[666]",
                "match": "[MATCH]",
                "initials": "[INIT]",
            }
            return ascii_map.get(key, "[*]")
        else:
            # Emoji-Map (falls Fonts vorhanden)
            emoji_map = {
                "best_ball": "🔥",
                "wizard_mode": "🧙",
                "multiball": "💥",
                "jackpot": "🎯",
                "super_jackpot": "💎",
                "triple_jackpot": "👑",
                "ball_save": "🛡️",
                "extra_ball": "➕",
                "special_award": "🎁",
                "mode_completed": "🏆",
                "loops": "🔁",
                "spinner": "🌀",
                "combo": "🎯",
                "drop_targets": "🎯",
                "ramps": "🏹",
                "orbit": "🌌",
                "skillshot": "🎯",
                "super_skillshot": "💥",
                "mode_starts": "🎬",
                "tilt_warnings": "🛡️",
                "tilt": "💀",
                "devils_number": "👹",
                "match": "🎲",
                "initials": "✍️",
            }
            return emoji_map.get(key, "•")



    def analyze_session(self, stats: dict) -> dict:
        """
        Build highlight lines per category from stats['events'].
        - Uses 'events' (even if playtime_sec==0), duration_sec is only for optional rates.
        - ASCII icon fallback so no '??' appear in the overlay when emoji fonts are missing.
        - NOTE: Best Ball is intentionally NOT rendered anymore.
        """
        events = stats.get("events", {}) or {}
        duration_sec = int(stats.get("duration_sec", 0) or 0)
        lines_per_cat = int((self.cfg.OVERLAY or {}).get("lines_per_category", 5))

        out = {"Power": [], "Precision": [], "Fun": []}
        buckets = {"Power": [], "Precision": [], "Fun": []}

        # Werte auf Kategorien verteilen (Best Ball wird NICHT mehr hinzugefügt)
        for key, rule in (self.HIGHLIGHT_RULES or {}).items():
            if key == "best_ball":
                continue  # entfernt
            cat = rule.get("cat", "Fun")
            typ = rule.get("type", "count")
            icon = self._icon(key)
            if typ == "count":
                val = int(events.get(key, 0) or 0)
                if val > 0:
                    base_w = 100
                    weight = base_w + val
                    label = f"{icon} {rule.get('label','')}".strip()
                    buckets[cat].append((weight, f"{label} – {val}"))
            elif typ == "flag":
                v = events.get(key, False)
                if isinstance(v, str):
                    v = v.strip()
                    if v:
                        buckets[cat].append((150, f"{icon} {rule.get('label','')} – {v}"))
                elif bool(v):
                    buckets[cat].append((150, f"{icon} {rule.get('label','')} – Yes"))
            # 'always' wird ignoriert (z. B. best_ball)

        # Top N je Kategorie
        for cat in ["Power", "Precision", "Fun"]:
            arr = sorted(buckets[cat], key=lambda x: x[0], reverse=True)[:max(1, lines_per_cat)]
            out[cat] = [s for _, s in arr]

        return out

    def _get_balls_played(self, audits: dict) -> Optional[int]:
        """
        Primär: 'Balls Played' (case-insensitive).
        Sekundär: 'Ball Count' aus game_state als 'Balls Played'.
        """
        kl = {str(k).lower(): k for k in audits.keys()}
        for key in ["balls played", "games balls played", "total balls played"]:
            if key in kl:
                try:
                    return int(audits[kl[key]])
                except Exception:
                    pass
        for lk, orig in kl.items():
            if lk == "ball count" or ("ball" in lk and "count" in lk):
                try:
                    return int(audits[orig])
                except Exception:
                    continue
        for lk, orig in kl.items():
            if "balls" in lk and "played" in lk:
                try:
                    return int(audits[orig])
                except Exception:
                    continue
        return None

    def _nv_get_int_ci(self, audits: dict, label: str, default: int = 0) -> int:
        """
        Hole einen Audit-Wert case-insensitive als int.
        """
        try:
            kl = {str(k).lower(): k for k in audits.keys()}
            key = kl.get(label.lower())
            if key is None:
                return int(default)
            return int(audits.get(key, default) or default)
        except Exception:
            return int(default)

    def _ball_reset(self, start_audits: dict):
        self.ball_track.update({
            "active": True,
            "index": 1,
            "start_time": time.time(),
            "current_pid": int(getattr(self, "current_player", 1) or 1),
            "score_base": self._find_score_from_audits(start_audits, pid=int(getattr(self, "current_player", 1) or 1)),
            "last_balls_played": self._get_balls_played(start_audits),
            "balls": []
        })

    def _ball_finalize_current(self, current_audits: dict, force: bool = False):
        if not self.ball_track.get("active"):
            return
        now = time.time()
        pid = int(self.ball_track.get("current_pid") or getattr(self, "current_player", 1) or 1)
        cur_score = self._find_score_from_audits(current_audits, pid=pid)
        base_score = int(self.ball_track.get("score_base", 0))
        ball_score = max(0, int(cur_score) - base_score)
        if ball_score == 0 and cur_score > 0 and base_score == 0:
            ball_score = cur_score
        duration = int(now - (self.ball_track.get("start_time") or now))
        if force or ball_score > 0 or duration > 0:
            entry = {
                "pid": pid,
                "num": self.ball_track.get("index", 1),
                "score": int(ball_score),
                "score_abs": int(cur_score),
                "duration": duration
            }
            self.ball_track["balls"].append(entry)
            self.ball_track["index"] = int(self.ball_track.get("index", 1)) + 1
            self.ball_track["start_time"] = now
            self.ball_track["score_base"] = cur_score

    def _ball_update(self, current_audits: dict):
        if not self.ball_track.get("active"):
            return
        cp = int(getattr(self, "current_player", 1) or 1)
        if cp != int(self.ball_track.get("current_pid") or cp):
            self._ball_finalize_current(current_audits, force=False)
            self.ball_track["current_pid"] = cp
            self.ball_track["start_time"] = time.time()
            self.ball_track["score_base"] = self._find_score_from_audits(current_audits, pid=cp)
        bp = self._get_balls_played(current_audits)
        if bp is None:
            return
        if self.ball_track.get("last_balls_played") is None:
            self.ball_track["last_balls_played"] = bp
            return
        if bp > int(self.ball_track.get("last_balls_played", 0)):
            self._ball_finalize_current(current_audits, force=False)
            self.ball_track["last_balls_played"] = bp
            cp = int(getattr(self, "current_player", 1) or 1)
            self.ball_track["current_pid"] = cp
            self.ball_track["start_time"] = time.time()
            self.ball_track["score_base"] = self._find_score_from_audits(current_audits, pid=cp)

            # One-Ball Challenge: end as soon as Balls Played increases once
            try:
                ch = getattr(self, "challenge", {})
                if ch.get("active") and ch.get("kind") == "oneball" and ch.get("one_ball_active", False):
                    base = int(ch.get("baseline_bp", 0))
                    if int(bp) >= base + 1 and not ch.get("pending_kill_at"):
                        # Pre-kill Snapshot sichern (aktueller Audit-Stand)
                        try:
                            ch["prekill_end"] = dict(current_audits or {})
                        except Exception:
                            pass
                        # 3s delayed kill (gleich wie Timed)
                        ch["one_ball_active"] = False
                        ch["pending_kill_at"] = time.time() + 3.0
                        self.challenge = ch
                        log(self.cfg, "[CHALLENGE] 1-ball finished – will kill VPX in 3s")
            except Exception as e:
                log(self.cfg, f"[CHALLENGE] one-ball hook failed: {e}", "WARN")

  
    def _best_ball_for_player(self, pid: int):
        try:
            balls = [b for b in self.ball_track.get("balls", []) if int(b.get("pid", 0)) == pid]
            if not balls:
                return None
            return max(balls, key=lambda b: (int(b.get("score", 0)), int(b.get("duration", 0))))
        except Exception:
            return None

    def _init_player_snaps(self, start_audits: dict):
        self.players.clear()
        now = time.time()
        for pid in range(1, 5):
            snap = self._player_field_filter(start_audits, pid)
            if not snap:
                snap = {f"P{pid} Score": 0}
            self.players[pid] = {
                "start_audits": dict(snap),
                "last_audits": dict(snap),
                "active_play_seconds": 0.0,
                "start_time": now,
                "session_deltas": {},
                "event_counts": {},
            }
        self._last_tick_time = time.time()
        self._last_global_for_player_attr = {
            k: v for k, v in start_audits.items() if isinstance(k, str) and not k.startswith("P")
        }

    def _compute_session_deltas(self, start: dict, end: dict) -> dict:
        """
        Rohe globale Deltas Start->Ende für Nicht-Spielerfelder.
        Negative Deltas werden auf 0 geklemmt (Wraps/Resets).
        """
        out = {}
        if not isinstance(end, dict):
            return out
        start = start or {}
        for k, ve in end.items():
            if not isinstance(k, str) or k.startswith("P"):
                continue
            try:
                s = int(start.get(k, 0) or 0)
                e = int(ve or 0)
            except Exception:
                continue
            d = e - s
            if d < 0:
                d = 0
            out[k] = d
        return out

    def _infer_players_in_game_from_audits(self, start: dict, end: dict) -> int:
        """
        Bestimmt die Spieleranzahl am Ende der Session (case-insensitive).
        Priorität: game_state.player_count -> Audit-Deltas -> Pn Score Präsenz.
        """
        try:
            pc = int(self._nv_get_int_ci(end, "player_count", 0))
            if 1 <= pc <= 6:
                return pc
        except Exception:
            pass

        def d(lbl_ci: str) -> int:
            try:
                s = int(self._nv_get_int_ci(start, lbl_ci, 0))
                e = int(self._nv_get_int_ci(end, lbl_ci, 0))
                return max(0, e - s)
            except Exception:
                return 0

        if d("4 Player Games") > 0: return 4
        if d("3 Player Games") > 0: return 3
        if d("2 Player Games") > 0: return 2

        max_pid = 1
        for pid in (2, 3, 4, 5, 6):
            key = f"P{pid} Score"
            try:
                if int(end.get(key, 0) or 0) > 0:
                    max_pid = max(max_pid, pid)
            except Exception:
                pass
        return max(1, min(max_pid, 6))


    def _score_deltas_from_audits(self, start: dict, end: dict, players: int) -> dict[int, int]:
        """
        Score-Deltas je Spieler (>=0).
        """
        out = {}
        for pid in range(1, max(1, int(players)) + 1):
            key = f"P{pid} Score"
            try:
                s = int(start.get(key, 0) or 0)
                e = int(end.get(key, 0) or 0)
                d = e - s
                out[pid] = d if d > 0 else 0
            except Exception:
                out[pid] = 0
        return out


    def _allocate_int_proportional(self, total: int, weights: dict[int, float]) -> dict[int, int]:
        """
        Verteilt 'total' als ganze Zahlen nach Gewichten auf Keys aus 'weights'.
        Summe der Rückgaben == total. Stabil mit Resteverteilung.
        """
        if total <= 0 or not weights:
            return {k: 0 for k in weights.keys()}
        s = sum(max(0.0, float(w)) for w in weights.values())
        if s <= 0:
            n = len(weights)
            base = total // n
            rem = total - base * n
            out = {k: base for k in weights.keys()}
            for k in list(weights.keys())[:rem]:
                out[k] += 1
            return out
        fractional = {}
        out = {}
        assigned = 0
        for k, w in weights.items():
            share = (float(w) / s) * total
            base = int(share)
            out[k] = base
            assigned += base
            fractional[k] = share - base
        rem = total - assigned
        if rem > 0:
            for k, _frac in sorted(fractional.items(), key=lambda x: x[1], reverse=True):
                if rem <= 0:
                    break
                out[k] += 1
                rem -= 1
        return out


    # --- REPLACE: realistischere Fallback-Parameter pro Difficulty (ohne globalen Score-Scaler) ---
    def _cpu_params_for(self, difficulty: str) -> dict:
        """
        Events/Minute + Fallback-Score je Event.
        Hinweis: Wenn das globale Modell (w) gelernt ist, hat das Priorität und
        diese Fallbacks greifen nur selten. Pro ist moderat höher, nicht absurd.
        """
        d = (difficulty or "").lower()
        if d == "leicht":
            return {"events_per_min": 6,  "score_min": 30_000,   "score_max": 120_000}
        if d == "mittel":
            return {"events_per_min": 10, "score_min": 80_000,   "score_max": 400_000}
        if d == "schwer":
            return {"events_per_min": 14, "score_min": 250_000,  "score_max": 1_200_000}
        # pro – leicht höherer Durchsatz + Millionenbereich als Fallback
        return {"events_per_min": 20, "score_min": 1_000_000, "score_max": 6_000_000}

    def _cpu_params_pg(self, difficulty: str) -> dict:
        """
        Postgame-Sim: gleiches Kaliber wie live, damit die Einschätzungen konsistent wirken.
        """
        d = (difficulty or "").lower()
        if d == "leicht":
            return {"events_per_min": 6,  "score_min": 30_000,   "score_max": 120_000}
        if d == "mittel":
            return {"events_per_min": 10, "score_min": 80_000,   "score_max": 400_000}
        if d == "schwer":
            return {"events_per_min": 14, "score_min": 250_000,  "score_max": 1_200_000}
        # pro
        return {"events_per_min": 20, "score_min": 1_000_000, "score_max": 6_000_000}

    def _cpu_label_caps_for(self, difficulty: str) -> dict:
        """
        Harte Session-Caps für 'überdrehende' Labels pro Schwierigkeit.
        """
        d = (difficulty or "mittel").lower()
        if d == "leicht":
            return {"Extra Balls": 1, "Ball Saves": 8,  "Total Multiballs": 1, "Jackpots": 3, "Modes Completed": 1}
        if d == "mittel":
            return {"Extra Balls": 2, "Ball Saves": 12, "Total Multiballs": 2, "Jackpots": 4, "Modes Completed": 2}
        if d == "schwer":
            return {"Extra Balls": 3, "Ball Saves": 18, "Total Multiballs": 3, "Jackpots": 6, "Modes Completed": 3}
        # pro
        return {"Extra Balls": 4, "Ball Saves": 22, "Total Multiballs": 4, "Jackpots": 8, "Modes Completed": 4}

    def _cpu_ai_get_model(self) -> dict:
        """
        Lädt das globale Lernmodell (mit Cache, Reload alle 30s).
        Struktur wie in _ai_load_global_model().
        """
        try:
            now = time.time()
            nxt = float(getattr(self, "_cpu_ai_next_reload", 0.0) or 0.0)
            if getattr(self, "_cpu_ai_cache", None) and now < nxt:
                return self._cpu_ai_cache
            model = self._ai_load_global_model()
            self._cpu_ai_cache = model
            self._cpu_ai_next_reload = now + 30.0
            return model
        except Exception:
            return {"events": {}, "meta": {}}

    def _cpu_label_for_event(self, key: str) -> str:
        """
        Mappt einen Event-Key (jackpot, multiball, ...) auf ein plausibles
        Audit-Label, das im Overlay als Session-Delta erscheint und von
        _build_events_from_deltas erkannt wird (Keywords!).
        """
        mapping = {
            "jackpot": "Jackpots",
            "super_jackpot": "Super Jackpot",
            "triple_jackpot": "Triple Jackpot",
            "multiball": "Total Multiballs",
            "ball_save": "Ball Saves",
            "extra_ball": "Extra Balls",
            "special_award": "Special Awards",
            "mode_completed": "Modes Completed",
            "mode_starts": "Modes Started",
            "loops": "Loops",
            "spinner": "Spinner",
            "combo": "Combos",
            "drop_targets": "Drop Targets",
            "ramps": "Ramps Made",
            "orbit": "Orbits",
            "skillshot": "Skill Shot",
            "super_skillshot": "Super Skill Shot",
            "tilt_warnings": "Tilt Warnings",
            "tilt": "Tilt",
            "match": "Match",
            # Fallback
            "best_ball": "Best Ball",
            "wizard_mode": "Wizard Mode",
        }
        return mapping.get(key, key.capitalize())

    def _cpu_ai_pick_event_keys(self, k: int, model: dict) -> list[str]:
        """
        Wählt k Event-Keys:
          - Basis: gelernte Werte (value_per_event) aus model['events']
          - plus Difficulty-Multiplikatoren (z. B. EB stark dämpfen, Jackpots boosten)
        """
        base_pool = [
            "jackpot", "multiball", "ramps", "loops", "spinner", "orbit",
            "combo", "drop_targets", "skillshot", "super_skillshot",
            "ball_save", "extra_ball", "mode_starts", "mode_completed",
        ]
        events = (model or {}).get("events", {}) or {}

        diff = str((self.cpu or {}).get("difficulty", "mittel")).lower()
        # Multiplikatoren je Stufe – Extra Balls stark runter, Jackpots leicht rauf
        diff_weights_map = {
            "leicht": {"extra_ball": 0.08, "ball_save": 0.7,  "jackpot": 1.10, "multiball": 1.00,
                       "ramps": 1.05, "loops": 1.05, "spinner": 1.0, "orbit": 1.0, "combo": 1.05,
                       "skillshot": 1.0, "super_skillshot": 1.0, "mode_starts": 1.0, "mode_completed": 0.9},
            "mittel": {"extra_ball": 0.06, "ball_save": 0.6,  "jackpot": 1.20, "multiball": 1.05,
                       "ramps": 1.10, "loops": 1.10, "spinner": 1.0, "orbit": 1.05, "combo": 1.10,
                       "skillshot": 1.0, "super_skillshot": 1.0, "mode_starts": 1.0, "mode_completed": 1.0},
            "schwer": {"extra_ball": 0.04, "ball_save": 0.5,  "jackpot": 1.30, "multiball": 1.10,
                       "ramps": 1.15, "loops": 1.15, "spinner": 1.05, "orbit": 1.05, "combo": 1.15,
                       "skillshot": 1.05, "super_skillshot": 1.10, "mode_starts": 1.05, "mode_completed": 1.05},
            "pro":    {"extra_ball": 0.03, "ball_save": 0.4,  "jackpot": 1.35, "multiball": 1.15,
                       "ramps": 1.20, "loops": 1.20, "spinner": 1.10, "orbit": 1.10, "combo": 1.20,
                       "skillshot": 1.10, "super_skillshot": 1.15, "mode_starts": 1.10, "mode_completed": 1.10},
        }
        diff_weights = diff_weights_map.get(diff, diff_weights_map["mittel"])

        weighted = []
        for key in base_pool:
            slot = events.get(key) or {}
            w = float(slot.get("w", 0.0) or 0.0)
            n = int(slot.get("n", 0) or 0)
            base_w = w if n >= 2 and w > 0 else 1.0
            mult = float(diff_weights.get(key, 1.0))
            weight = max(0.001, base_w * mult)
            weighted.append((key, weight))

        total = sum(w for _, w in weighted) or 1.0
        picks = []
        for _ in range(max(1, int(k))):
            r = random.random() * total
            acc = 0.0
            choice = weighted[0][0]
            for key, w in weighted:
                acc += w
                if r <= acc:
                    choice = key
                    break
            picks.append(choice)
        return picks

    # --- REPLACE: Score pro Event – human-like Multiplikatoren + MB-Boost ---
    def _cpu_score_for_event(self, key: str, params: dict, model: dict) -> int:
        """
        Punktebeitrag für ein einzelnes Event:
        1) Bevorzugt: normalverteilt um das gelernte w (value_per_event)
        2) Fallback: Difficulty-Range + Event-Multiplikator (realistischer gewichtet)
        3) Multiball-Bonus: Jackpots sind in MB deutlich wertvoller; Combos leicht erhöht
        """
        try:
            slot = (model or {}).get("events", {}).get(key) or {}
            w = float(slot.get("w", 0.0) or 0.0)
            n = int(slot.get("n", 0) or 0)
        except Exception:
            w, n = 0.0, 0

        if n >= 2 and w > 0:
            mu = w
            sigma = 0.25 * w
            try:
                val = int(max(0, random.gauss(mu, sigma)))
            except Exception:
                val = int(max(0, mu))
        else:
            smin = int(params.get("score_min", 100_000))
            smax = int(params.get("score_max", 300_000))
            base = random.randint(smin, smax)
            # Realistischere Relationen: Super/Triple Jackpot >> Jackpot > MB > Combos > Ramps/Loops
            mult = {
                "super_jackpot": 6.0,
                "triple_jackpot": 8.0,
                "jackpot": 3.0,
                "multiball": 1.4,
                "combo": 1.5,
                "ramps": 1.1,
                "loops": 1.1,
                "spinner": 0.9,
                "orbit": 1.0,
                "drop_targets": 0.8,
                "skillshot": 1.0,
                "super_skillshot": 2.0,
                "mode_starts": 0.7,
                "mode_completed": 2.4,
                "ball_save": 0.2,
                "extra_ball": 0.4,
                "tilt_warnings": 0.0,
                "tilt": -0.8,
                "match": 0.0,
            }.get(key, 1.0)
            val = int(base * max(-0.8, mult))

        # Multiball-Bonus (Jackpots/Combos profitieren, wenn MB-Fenster aktiv)
        try:
            st = (self.cpu or {}).get("state", {}) or {}
            now = time.time()
            mb_until = float(st.get("multiball_until", 0.0))
            if now < mb_until and key in ("jackpot", "super_jackpot", "triple_jackpot", "combo"):
                val = int(val * 1.5)
        except Exception:
            pass

        return max(0, val)  


    def _cpu_diff_multipliers(self, difficulty: str) -> dict:
        """
        Difficulty-Gewichte pro Eventtyp (dämpft z. B. Extra Balls) – nur Postgame.
        """
        d = (difficulty or "mittel").lower()
        table = {
            "leicht": {"extra_ball": 0.08, "ball_save": 0.7,  "jackpot": 1.10, "multiball": 1.00,
                       "ramps": 1.05, "loops": 1.05, "spinner": 1.0, "orbit": 1.0, "combo": 1.05,
                       "skillshot": 1.0, "super_skillshot": 1.0, "mode_starts": 1.0, "mode_completed": 0.9},
            "mittel": {"extra_ball": 0.06, "ball_save": 0.6,  "jackpot": 1.20, "multiball": 1.05,
                       "ramps": 1.10, "loops": 1.10, "spinner": 1.0, "orbit": 1.05, "combo": 1.10,
                       "skillshot": 1.0, "super_skillshot": 1.0, "mode_starts": 1.0, "mode_completed": 1.0},
            "schwer": {"extra_ball": 0.04, "ball_save": 0.5,  "jackpot": 1.30, "multiball": 1.10,
                       "ramps": 1.15, "loops": 1.15, "spinner": 1.05, "orbit": 1.05, "combo": 1.15,
                       "skillshot": 1.05, "super_skillshot": 1.10, "mode_starts": 1.05, "mode_completed": 1.05},
            "pro":    {"extra_ball": 0.03, "ball_save": 0.4,  "jackpot": 1.35, "multiball": 1.15,
                       "ramps": 1.20, "loops": 1.20, "spinner": 1.10, "orbit": 1.10, "combo": 1.20,
                       "skillshot": 1.10, "super_skillshot": 1.15, "mode_starts": 1.10, "mode_completed": 1.10},
        }
        return table.get(d, table["mittel"])

    def _cpu_build_weighted_pool(self, model: dict, diff_mult: dict) -> list[tuple[str, float]]:
        """
        Gewichteter Pool (Key -> Gewicht) aus globalem Modell + Difficulty – nur Postgame.
        """
        base_pool = [
            "jackpot", "multiball", "ramps", "loops", "spinner", "orbit",
            "combo", "drop_targets", "skillshot", "super_skillshot",
            "ball_save", "extra_ball", "mode_starts", "mode_completed",
        ]
        weighted = []
        evm = (model or {}).get("events", {}) or {}
        for key in base_pool:
            slot = evm.get(key) or {}
            w = float(slot.get("w", 0.0) or 0.0)
            n = int(slot.get("n", 0) or 0)
            base_w = w if n >= 2 and w > 0 else 1.0
            mult = float(diff_mult.get(key, 1.0))
            weighted.append((key, max(0.001, base_w * mult)))
        return weighted

    def _cpu_weighted_choice(self, weighted: list[tuple[str, float]]) -> str:
        total = sum(w for _, w in weighted) or 1.0
        r = random.random() * total
        acc = 0.0
        for key, w in weighted:
            acc += w
            if r <= acc:
                return key
        return weighted[0][0]

    # --- REPLACE: Postgame-Sim – gleicher “menschlicher” Stil wie live (mit Korrelation) ---
    def _cpu_postgame_simulate(self, rom: str, duration_sec: int):
        """
        Monte-Carlo Postgame-Simulation:
          - Budget = Events/Minute * Dauer
          - Berücksichtigt Korrelationen/State (Modes, MB, Combos)
          - Utility = Score + Diversität (wie gehabt)
          - Ergebnis -> Player 5 (nur Overlay/Stats)
        """
        if not bool((self.cfg.OVERLAY or {}).get("cpu_sim_active", True)):
            return

        sim_head = getattr(self, "cpu", {}) or {}
        diff = str(sim_head.get("difficulty", self.cfg.OVERLAY.get("cpu_sim_difficulty", "mittel"))).lower()

        model = self._ai_load_global_model() or {"events": {}}
        params = self._cpu_params_pg(diff)
        ev_budget = int(round(float(params["events_per_min"]) * max(1, int(duration_sec)) / 60.0))
        if ev_budget <= 0:
            ev_budget = 1

        diff_mult = self._cpu_diff_multipliers(diff)
        weighted_pool = self._cpu_build_weighted_pool(model, diff_mult)
        caps = self._cpu_label_caps_for(diff)
        corp = self._cpu_cor_params_for(diff)

        R = int(self.cfg.OVERLAY.get("cpu_postgame_rollouts", 120))
        best = None
        best_util = -1e18

        for _ in range(max(10, R)):
            sd: dict[str, int] = {}
            ev: dict[str, int] = {}
            score = 0
            kinds = set()

            # lokaler State wie live
            st = {"modes_in_progress": [], "multiball_until": 0.0, "balls_next_ts": 0.0}
            def pick_weighted():
                total = sum(w for _, w in weighted_pool) or 1.0
                r = random.random() * total
                acc = 0.0
                for key, w in weighted_pool:
                    acc += w
                    if r <= acc:
                        return key
                return weighted_pool[0][0]

            for _i in range(ev_budget):
                # 1) Budgetiertes Event
                key = pick_weighted()
                lab = self._cpu_label_for_event(key)
                cmax = caps.get(lab)
                if cmax is None or int(sd.get(lab, 0)) < int(cmax):
                    sd[lab] = int(sd.get(lab, 0)) + 1
                    ev[key] = int(ev.get(key, 0)) + 1
                    kinds.add(key)
                    # Score inkl. MB-Bonus (nutzt self.cpu.state normalerweise – hier simulieren wir inline)
                    # Temporär simuliere MB-Bonus mit einfacher Abfrage auf Fenster
                    mb_active = (time.time() < float(st.get("multiball_until", 0.0)))
                    val = self._cpu_score_for_event(key, params, model)
                    if mb_active and key in ("jackpot", "super_jackpot", "triple_jackpot", "combo"):
                        val = int(val * 1.5)
                    score += int(max(0, val))

                    # State: MB-Fenster, Combos, Modes
                    if key == "multiball":
                        lo, hi = corp["mb_window_s"]
                        st["multiball_until"] = time.time() + random.uniform(lo, hi)
                    if key in ("ramps", "loops", "orbit"):
                        if random.random() < float(corp["p_combo_after_ramp_loop"]):
                            lab_c = self._cpu_label_for_event("combo")
                            if caps.get(lab_c, 1_000_000) > int(sd.get(lab_c, 0)):
                                sd[lab_c] = int(sd.get(lab_c, 0)) + 1
                                ev["combo"] = int(ev.get("combo", 0)) + 1
                                kinds.add("combo")
                                valc = self._cpu_score_for_event("combo", params, model)
                                if (time.time() < float(st.get("multiball_until", 0.0))):
                                    valc = int(valc * 1.5)
                                score += int(max(0, valc))
                    if key == "mode_starts":
                        plo, phi = corp["p_mode_complete"]
                        if random.random() < random.uniform(plo, phi):
                            dlo, dhi = corp["mode_delay_s"]
                            st["modes_in_progress"].append({"due": time.time() + random.uniform(dlo, dhi)})

                # 2) Korrelierte “kostenlose” Effekte pro Schritt
                # A) Mode-Completion fällig?
                keep = []
                for slot in st.get("modes_in_progress", []):
                    if time.time() >= float(slot.get("due", 0.0)):
                        lmc = self._cpu_label_for_event("mode_completed")
                        if caps.get(lmc, 1_000_000) > int(sd.get(lmc, 0)):
                            sd[lmc] = int(sd.get(lmc, 0)) + 1
                            ev["mode_completed"] = int(ev.get("mode_completed", 0)) + 1
                            kinds.add("mode_completed")
                            score += int(max(0, self._cpu_score_for_event("mode_completed", params, model)))
                    else:
                        keep.append(slot)
                st["modes_in_progress"] = keep

                # B) Während MB zusätzliche Jackpots (rollend)
                if time.time() < float(st.get("multiball_until", 0.0)):
                    lo, hi = corp["jackpot_extra_per_tick"]
                    n_extra = random.randint(lo, hi)
                    for _x in range(n_extra):
                        labx = self._cpu_label_for_event("jackpot")
                        if caps.get(labx, 1_000_000) > int(sd.get(labx, 0)):
                            sd[labx] = int(sd.get(labx, 0)) + 1
                            ev["jackpot"] = int(ev.get("jackpot", 0)) + 1
                            kinds.add("jackpot")
                            score += int(max(0, int(self._cpu_score_for_event("jackpot", params, model) * 1.5)))

            util = float(score) + 5_000.0 * len(kinds)
            if util > best_util:
                best_util = util
                best = (sd, ev, score)

        if not best:
            return
        sd, ev, score = best
        self.cpu = {
            "active": True,
            "difficulty": diff,
            "params": params,
            "active_play_seconds": int(duration_sec),
            "session_deltas": sd,
            "event_counts": ev,
            "score": int(score),
        }

    # --- REPLACE: Korrelation-Parameter je Difficulty (Pro = “spielerisch schlau”) ---
    def _cpu_cor_params_for(self, difficulty: str) -> dict:
        """
        Skill-/Korrelationstuning:
          - p_mode_complete: Chance, dass ein gestarteter Mode später abgeschlossen wird.
          - mode_delay_s: Zeitfenster bis Mode-Completion.
          - mb_window_s: Dauerfenster, in dem Multiball-Bonus/Jackpots häufiger fallen.
          - jackpot_extra_per_tick: zusätzliche Jackpots pro Tick während MB (0..2).
          - p_combo_after_ramp_loop: nach Ramp/Loop eine Combo hinterher.
          - balls_interval_s: Zeit bis zum nächsten 'Balls Played' Tick (kosmetisch für Session-Deltas).
        """
        d = (difficulty or "mittel").lower()
        if d == "leicht":
            return {
                "p_mode_complete": (0.30, 0.50),
                "mode_delay_s": (16, 40),
                "mb_window_s": (7, 11),
                "jackpot_extra_per_tick": (0, 1),
                "p_combo_after_ramp_loop": 0.15,
                "balls_interval_s": (32, 52),
            }
        if d == "schwer":
            return {
                "p_mode_complete": (0.55, 0.75),
                "mode_delay_s": (10, 26),
                "mb_window_s": (10, 16),
                "jackpot_extra_per_tick": (1, 2),
                "p_combo_after_ramp_loop": 0.40,
                "balls_interval_s": (20, 34),
            }
        if d == "pro":
            return {
                "p_mode_complete": (0.65, 0.85),
                "mode_delay_s": (8, 20),
                "mb_window_s": (12, 18),
                "jackpot_extra_per_tick": (1, 3),
                "p_combo_after_ramp_loop": 0.55,
                "balls_interval_s": (16, 28),
            }
        # mittel
        return {
            "p_mode_complete": (0.45, 0.65),
            "mode_delay_s": (12, 30),
            "mb_window_s": (9, 14),
            "jackpot_extra_per_tick": (0, 2),
            "p_combo_after_ramp_loop": 0.30,
            "balls_interval_s": (24, 40),
        }

    def _cpu_state_init(self, sim: dict, now: float, diff: str):
        """
        Initialisiert den Korrelations-Zustand einmalig im sim['state'].
        """
        st = sim.setdefault("state", {})
        st.setdefault("modes_in_progress", [])      # Liste von {'due': ts}
        st.setdefault("multiball_until", 0.0)       # ts bis Multiball-Bonus gilt
        st.setdefault("balls_next_ts", now + random.uniform(*self._cpu_cor_params_for(diff)["balls_interval_s"]))

    def _cpu_generate_correlated_events(self, sim: dict, now: float, diff: str) -> dict:
        """
        Erzeugt zusätzliche Event-Keys aus Korrelationen.
        Rückgabe:
          {
            "event_keys": [ "mode_completed", "jackpot", ... ],
            "inc_labels": { "Balls Played": +1, ... }
          }
        """
        out = {"event_keys": [], "inc_labels": {}}
        st = sim.get("state") or {}
        params = self._cpu_cor_params_for(diff)

        # A) Fällige Mode-Completions
        modes = st.get("modes_in_progress", [])
        if modes:
            keep = []
            for slot in modes:
                if now >= float(slot.get("due", 0.0)):
                    out["event_keys"].append("mode_completed")
                else:
                    keep.append(slot)
            st["modes_in_progress"] = keep

        # B) Balls Played zeitbasiert
        try:
            if now >= float(st.get("balls_next_ts", 0.0)):
                out["inc_labels"]["Balls Played"] = out["inc_labels"].get("Balls Played", 0) + 1
                st["balls_next_ts"] = now + random.uniform(*params["balls_interval_s"])
        except Exception:
            pass

        # C) Während aktivem Multiball hin und wieder Extra-Jackpots
        try:
            mb_until = float(st.get("multiball_until", 0.0))
            if now < mb_until:
                lo, hi = params["jackpot_extra_per_tick"]
                n_extra = random.randint(lo, hi)
                for _ in range(n_extra):
                    out["event_keys"].append("jackpot")
        except Exception:
            pass

        return out
  
        
    # --- REPLACE: Live-Tick – nutzt Korrelationen + “pro” spielt wie ein Mensch (stackt, comboed, finished modes) ---
    def _cpu_sim_tick(self, dt: float):
        """
        KI-CPU Tick (realistisch):
          - Läuft NUR im Modus 'live'
          - Nur wenn Spiel aktiv UND Bootstrap abgeschlossen
          - Ereignis-Budget: Events/Minute (sanft)
          - Korrelationen: MB-Fenster, Mode-Start->Completion, Ramp/Loop -> Combo
          - Pro: höhere Abschlusswahrscheinlichkeit, längere MB-Fenster, mehr Jackpot-Ketten
        """
        # Nur im Live-Modus
        mode = str((self.cfg.OVERLAY or {}).get("cpu_sim_mode", "postgame")).lower()
        if mode != "live":
            return
        sim = getattr(self, "cpu", {}) or {}
        if (not sim.get("active")) or (not self.game_active) or (not getattr(self, "_snap_bootstrap_done", False)):
            return

        try:
            # Zeit/State
            sim["active_play_seconds"] = float(sim.get("active_play_seconds", 0.0)) + float(dt or 0.0)
            diff = str(sim.get("difficulty", "mittel")).lower()

            params = sim.get("params") or self._cpu_params_for(diff)
            ev_per_min = float(params.get("events_per_min", 10))
            sim["ev_budget"] = float(sim.get("ev_budget", 0.0)) + ev_per_min * max(0.0, float(dt)) / 60.0

            k_allow = min(3, int(sim["ev_budget"]))
            if k_allow <= 0:
                self.cpu = sim
                return

            # Modell + Gewichte
            use_ai = bool((self.cfg.OVERLAY or {}).get("cpu_sim_ai", True))
            model = self._cpu_ai_get_model() if use_ai else {"events": {}}

            now = time.time()
            self._cpu_state_init(sim, now, diff)  # state{'modes_in_progress','multiball_until','balls_next_ts'}
            st = sim.get("state", {})
            corp = self._cpu_cor_params_for(diff)
            caps = self._cpu_label_caps_for(diff)

            # Korrelierte Zusatz-Keys je Tick (zählen NICHT aufs Budget; simulieren “free” Ketten)
            try:
                extra = self._cpu_generate_correlated_events(sim, now, diff)
            except Exception:
                extra = {"event_keys": [], "inc_labels": {}}

            sd = sim.setdefault("session_deltas", {})
            ev = sim.setdefault("event_counts", {})
            cur_score = int(sim.get("score", 0) or 0)

            # 1) Zusätzliche Inkrement-Labels (z. B. Balls Played)
            for lab, inc in (extra.get("inc_labels") or {}).items():
                if inc and (caps.get(lab, 1_000_000) > int(sd.get(lab, 0))):
                    sd[lab] = int(sd.get(lab, 0)) + int(inc)

            # 2) Budgetierte Events auswählen und ausführen
            issued = 0
            keys_main = self._cpu_ai_pick_event_keys(k_allow, model) or []
            for key in keys_main:
                # Cap prüfen
                label = self._cpu_label_for_event(key)
                cap = caps.get(label)
                if cap is not None and int(sd.get(label, 0)) >= int(cap):
                    continue

                # zählen
                sd[label] = int(sd.get(label, 0)) + 1
                ev[key] = int(ev.get(key, 0)) + 1
                cur_score += int(max(0, self._cpu_score_for_event(key, params, model)))
                issued += 1

                # State-Updates + korrelierte Folge-Events
                # a) Multiball gestartet -> Fenster setzen
                if key == "multiball":
                    lo, hi = corp["mb_window_s"]
                    st["multiball_until"] = time.time() + random.uniform(lo, hi)

                # b) Ramp/Loop/Orbit -> ggf. Combo “kostenlos” hinterher
                if key in ("ramps", "loops", "orbit"):
                    if random.random() < float(corp["p_combo_after_ramp_loop"]):
                        lab_c = self._cpu_label_for_event("combo")
                        if caps.get(lab_c, 1_000_000) > int(sd.get(lab_c, 0)):
                            sd[lab_c] = int(sd.get(lab_c, 0)) + 1
                            ev["combo"] = int(ev.get("combo", 0)) + 1
                            cur_score += int(max(0, self._cpu_score_for_event("combo", params, model)))

                # c) Mode Start -> probabilistisch Completion nach Delay einplanen
                if key == "mode_starts":
                    plo, phi = corp["p_mode_complete"]
                    if random.random() < random.uniform(plo, phi):
                        dlo, dhi = corp["mode_delay_s"]
                        st.setdefault("modes_in_progress", []).append({"due": time.time() + random.uniform(dlo, dhi)})

                if issued >= k_allow:
                    break

            # 3) Korrelierte Events des Ticks (kostenlos) ausführen
            for kx in (extra.get("event_keys") or []):
                labx = self._cpu_label_for_event(kx)
                if caps.get(labx, 1_000_000) > int(sd.get(labx, 0)):
                    sd[labx] = int(sd.get(labx, 0)) + 1
                    ev[kx] = int(ev.get(kx, 0)) + 1
                    cur_score += int(max(0, self._cpu_score_for_event(kx, params, model)))

            # 4) Budget abbauen
            sim["ev_budget"] = max(0.0, float(sim.get("ev_budget", 0.0)) - issued)

            # 5) kleiner Jitter
            jitter = max(0, random.randint(0, int(0.04 * max(1, params.get("score_min", 50_000)))))
            cur_score += jitter

            sim["score"] = int(cur_score)
            self.cpu = sim
        except Exception as e:
            log(self.cfg, f"[CPU] ai tick failed: {e}", "WARN")
 

    def _cpu_sim_session_reset(self):
        """
        Setzt CPU-Sim Zähler für eine neue Session zurück (Score/Deltas/Playtime).
        """
        self.cpu = (self.cpu or {})
        self.cpu["active_play_seconds"] = 0.0
        self.cpu["session_deltas"] = {}
        self.cpu["event_counts"] = {}
        self.cpu["score"] = 0

    def _cpu_sim_init(self):
        """
        Initialize CPU simulation state from cfg.OVERLAY.
        - Sets active flag, difficulty (stored in German), params, and resets small runtime fields.
        - Does NOT start ticking by itself; live tick runs only when game_active and snapshot bootstrap done.
        """
        try:
            self.cpu = (self.cpu or {})
            ov = getattr(self.cfg, "OVERLAY", {}) or {}
            # Active flag
            self.cpu["active"] = bool(ov.get("cpu_sim_active", True))
            # Difficulty: accept EN synonyms but store as de ("leicht","mittel","schwer","pro")
            raw = str(ov.get("cpu_sim_difficulty", self.cpu.get("difficulty", "mittel"))).lower()
            map_en = {"easy": "leicht", "medium": "mittel", "difficult": "schwer", "hard": "schwer", "pro": "pro"}
            diff = map_en.get(raw, raw if raw in ("leicht", "mittel", "schwer", "pro") else "mittel")
            self.cpu["difficulty"] = diff
            # Params for live tick
            self.cpu["params"] = self._cpu_params_for(diff)
            # Budgets/state
            self.cpu.setdefault("ev_budget", 0.0)
            self.cpu.setdefault("score", 0)
            self.cpu.setdefault("active_play_seconds", 0.0)
            self.cpu.setdefault("session_deltas", {})
            self.cpu.setdefault("event_counts", {})
            self.cpu.setdefault("state", {})
        except Exception:
            # keep it silent; callers wrap in try/except already
            pass

 
    def set_cpu_sim_active(self, active: bool):
        """
        Ein-/Ausschalten der CPU-Simulation zur Laufzeit.
        Zustand wird dauerhaft in cfg.OVERLAY gespeichert.
        """
        self.cpu = (self.cpu or {})
        self.cpu["active"] = bool(active)

        # Direkt in der Config persistieren
        try:
            ov = getattr(self.cfg, "OVERLAY", {}) or {}
            ov["cpu_sim_active"] = bool(active)
            # Schwierigkeitsgrad sicherstellen, falls noch nicht vorhanden (deutsch)
            cur = str((self.cpu.get("difficulty") or ov.get("cpu_sim_difficulty") or "mittel")).lower()
            map_en = {"easy": "leicht", "medium": "mittel", "difficult": "schwer", "hard": "schwer", "pro": "pro"}
            cur = map_en.get(cur, cur if cur in ("leicht", "mittel", "schwer", "pro") else "mittel")
            ov["cpu_sim_difficulty"] = cur
            self.cfg.OVERLAY = ov
            self.cfg.save()
        except Exception:
            pass

        # Bei Aktivierung sauber initialisieren (und aktiv lassen)
        if active:
            try:
                diff = str(self.cfg.OVERLAY.get("cpu_sim_difficulty", self.cpu.get("difficulty", "mittel"))).lower()
                map_en = {"easy": "leicht", "medium": "mittel", "difficult": "schwer", "hard": "schwer", "pro": "pro"}
                self.cpu["difficulty"] = map_en.get(diff, diff if diff in ("leicht", "mittel", "schwer", "pro") else "mittel")
            except Exception:
                pass
            self._cpu_sim_init()


    def set_cpu_difficulty(self, difficulty: str):
        """
        Umschalten der CPU-Schwierigkeit zur Laufzeit.
        Persistiert in cfg.OVERLAY und setzt next_tick neu.
        Akzeptiert deutsche und englische Eingaben, speichert deutsch.
        """
        raw = str(difficulty or "mittel").lower()
        map_en = {"easy": "leicht", "medium": "mittel", "difficult": "schwer", "hard": "schwer", "pro": "pro"}
        d = map_en.get(raw, raw)
        if d not in ("leicht", "mittel", "schwer", "pro"):
            d = "mittel"

        self.cpu = (self.cpu or {})
        self.cpu["difficulty"] = d
        self.cpu["params"] = self._cpu_params_for(d)
        try:
            eps = float(self.cpu["params"].get("events_per_min", 10)) / 60.0
            ival = 1.0 / max(0.2, eps) if eps > 0 else 1.0
            self.cpu["next_tick"] = time.time() + ival
        except Exception:
            self.cpu["next_tick"] = time.time() + 1.0

        try:
            ov = getattr(self.cfg, "OVERLAY", {}) or {}
            ov["cpu_sim_difficulty"] = d
            if "cpu_sim_active" not in ov:
                ov["cpu_sim_active"] = bool(self.cpu.get("active", False))
            self.cfg.OVERLAY = ov
            self.cfg.save()
        except Exception:
            pass
        

    def _build_session_stats(self, start_audits: dict, end_audits: dict, duration_sec: int) -> dict:
        deltas = self._compute_session_deltas(start_audits, end_audits)
        events = self._build_events_from_deltas(deltas)
        score_final = self._find_score_from_audits(end_audits)
        events["devils_number"] = ("666" in str(score_final))
        initials = ""
        for k in end_audits.keys():
            if "initial" in str(k).lower():
                initials = str(end_audits.get(k) or "").strip()
                break
        events["initials"] = initials
        return {"score": score_final, "duration_sec": duration_sec, "events": events}

    def _collect_player_rules_for_rom(self, rom: str) -> list:
        rules = []
        rpath = os.path.join(p_rom_spec(self.cfg), f"{rom}.ach.json")
        if os.path.exists(rpath):
            data = load_json(rpath, {}) or {}
            if isinstance(data.get("rules"), list):
                rules.extend(data["rules"])

        # Custom: Platzhalter-Datei explizit überspringen
        cdir = p_custom(self.cfg)
        placeholder = "put_your_custom_achievements_here_click_me.json"
        if os.path.isdir(cdir):
            for fn in os.listdir(cdir):
                if not fn.lower().endswith(".json"):
                    continue
                if fn.lower() == placeholder:
                    continue  # nur diese eine Datei ignorieren
                data = load_json(os.path.join(cdir, fn), {}) or {}
                if isinstance(data.get("rules"), list):
                    rules.extend(data["rules"])
                for ex in data.get("examples", []) or []:
                    if isinstance(ex, dict) and ex.get("rom") == rom:
                        achs = ex.get("achievements", [])
                        if isinstance(achs, list):
                            rules.extend(achs)
        out, seen = [], set()
        for r in rules:
            t = r.get("title") or "Achievement"
            if t in seen:
                continue
            seen.add(t)
            out.append(r)
        return out

    def _evaluate_player_session_achievements(self, pid: int, rom: str) -> list:
        """
        Evaluates player-specific session achievements for the given ROM.
        Unterstützt NUR:
          - nvram_delta   (per-player Session-Deltas)
          - session_time  (Spielzeit pro Spieler)
        KEIN nvram_overall im Spieler-Kontext.
        """
        if pid not in self.players:
            return []
        player = self.players[pid]
        deltas = player.get("session_deltas", {}) or {}
        play_sec = int(player.get("active_play_seconds", 0.0))

        rules = self._collect_player_rules_for_rom(rom)
        awarded = []
        for rule in rules:
            cond = rule.get("condition", {}) or {}
            rtype = cond.get("type")
            field = cond.get("field")
            title = rule.get("title") or "Achievement"

            try:
                if rtype == "nvram_delta":
                    if not field or is_excluded_field(field):
                        continue
                    need = int(cond.get("min", 0))
                    if deltas.get(field, 0) >= need:
                        awarded.append(title)

                elif rtype == "session_time":
                    min_s = int(cond.get("min_seconds", cond.get("min", 0)))
                    if play_sec >= min_s:
                        awarded.append(title)

                # nvram_overall wird im Spieler-Kontext bewusst ignoriert
            except Exception:
                continue

        # Duplikate entfernen (pro Zielfeld nur einmal)
        out, seen_field = [], set()
        for title in awarded:
            parts = title.split("–")
            if len(parts) > 1:
                field_name = parts[-1].strip().split(" ")[0]
            else:
                field_name = title
            if field_name in seen_field:
                continue
            seen_field.add(field_name)
            out.append(title)
        return out

    def export_overlay_snapshot(self, end_audits: dict, duration_sec: int, on_demand: bool = False) -> str:
        """
        Export activePlayers JSON files for each player (atomic write).
        UPDATED: remove achievements from payloads entirely (GUI/Overlay should have no achievements of any kind).
        """
        self._latest_end_audits_cache = dict(end_audits)
        try:
            self._ball_finalize_current(end_audits, force=True)
        except Exception as e:
            log(self.cfg, f"[BALL] finalize current failed: {e}", "WARN")

        active_dir = os.path.join(p_highlights(self.cfg), "activePlayers")
        ensure_dir(active_dir)

        # Include provisional events of the current segment (overlay-only)
        provisional_events: dict = {}
        if self.include_current_segment_in_overlay and self.snapshot_mode and self.current_segment_provisional_diff:
            for label, delta in (self.current_segment_provisional_diff or {}).items():
                ll = label.lower()
                for ev_key, words in (self.EVENT_KEYWORDS or {}).items():
                    if any(w in ll for w in words):
                        provisional_events[ev_key] = provisional_events.get(ev_key, 0) + int(delta or 0)
                        break

        # Players 1..4 (no achievements in payload)
        for pid in range(1, 4 + 1):
            rec = self.players.get(pid)
            if not rec:
                payload = {
                    "player": pid,
                    "playtime_sec": 0,
                    "score": 0,
                    "highlights": {"Power": [], "Precision": [], "Fun": []},
                }
                save_json(os.path.join(active_dir, f"{self.current_rom}_P{pid}.json"), payload)
                continue

            play_sec = int(rec.get("active_play_seconds", 0.0))

            # Events from per-player deltas
            deltas_for_player = rec.get("session_deltas", {}) or {}
            merged_events = self._build_events_from_deltas(deltas_for_player)

            # Provisional events for active player
            if pid == self.snap_player and provisional_events:
                for k, v in provisional_events.items():
                    merged_events[k] = merged_events.get(k, 0) + int(v or 0)

            analysis_sec = play_sec if play_sec > 0 else int(duration_sec or 0)

            try:
                score_abs = int(self._find_score_from_audits(end_audits, pid=pid) or 0)
            except Exception:
                score_abs = 0

            pseudo_stats = {
                "score": score_abs,
                "duration_sec": analysis_sec,
                "events": merged_events,
            }

            try:
                highlights = self.analyze_session(pseudo_stats)
            except Exception as e:
                log(self.cfg, f"[HIGHLIGHTS] analyze_session failed for P{pid}: {e}", "WARN")
                highlights = {"Power": [], "Precision": [], "Fun": []}

            payload = {
                "player": pid,
                "playtime_sec": play_sec,
                "score": score_abs,
                "highlights": highlights,
            }
            save_json(os.path.join(active_dir, f"{self.current_rom}_P{pid}.json"), payload)

        # CPU (Player 5) – only if active; no achievements in payload
        try:
            sim = getattr(self, "cpu", {}) or {}
            if sim.get("active"):
                cpu_play = int(sim.get("active_play_seconds", 0.0))
                cpu_deltas = sim.get("session_deltas", {}) or {}
                cpu_events = self._build_events_from_deltas(cpu_deltas)
                pseudo_stats_cpu = {
                    "score": int(sim.get("score", 0) or 0),
                    "duration_sec": cpu_play,
                    "events": cpu_events,
                }
                try:
                    cpu_highlights = self.analyze_session(pseudo_stats_cpu)
                except Exception as e:
                    log(self.cfg, f"[HIGHLIGHTS] analyze_session failed for CPU: {e}", "WARN")
                    cpu_highlights = {"Power": [], "Precision": [], "Fun": []}

                cpu_payload = {
                    "player": 5,
                    "playtime_sec": cpu_play,
                    "score": int(sim.get("score", 0) or 0),
                    "highlights": cpu_highlights,
                }
                save_json(os.path.join(active_dir, f"{self.current_rom}_P5.json"), cpu_payload)
        except Exception as e:
            log(self.cfg, f"[CPU] overlay export failed: {e}", "WARN")

        if not on_demand:
            log(self.cfg, "[EXPORT] session-only activePlayers written")
        return active_dir

    def _evaluate_achievements(self, rom: str, start_audits: dict, end_audits: dict, duration_sec: int) -> tuple[list[str], list[str], list[dict]]:
        """
        Evaluate global achievements (scope='global') for the given ROM.
        Supported rule types: nvram_overall, nvram_delta, session_time.
        Returns:
          (awarded_titles, all_global_titles, awarded_meta)
          awarded_meta contains dict items: {'title': str, 'origin': str}
        """
        global_rules = self._collect_global_rules_for_rom(rom)

        # Prepare case-insensitive deltas Start -> End
        deltas_ci = {}
        for k, _ve in (end_audits or {}).items():
            try:
                ve_i = int(self._nv_get_int_ci(end_audits, str(k), 0))
                vs_i = int(self._nv_get_int_ci(start_audits, str(k), 0))
                d = ve_i - vs_i
            except Exception:
                d = 0
            if d < 0:
                d = 0
            deltas_ci[str(k)] = d

        awarded = []
        awarded_meta = []
        all_titles = []
        seen_all = set()
        seen_aw = set()

        for rule in global_rules:
            title = (rule.get("title") or "Achievement").strip()
            if title not in seen_all:
                seen_all.add(title)
                all_titles.append(title)

            cond = (rule.get("condition") or {}) if isinstance(rule, dict) else {}
            rtype = str(cond.get("type") or "").lower()
            origin = rule.get("_origin") or ""

            try:
                if rtype == "nvram_overall":
                    field = cond.get("field")
                    if not field or is_excluded_field(field):
                        continue
                    need = int(cond.get("min", 1))
                    sv = int(self._nv_get_int_ci(start_audits, field, 0))
                    ev = int(self._nv_get_int_ci(end_audits, field, 0))
                    if sv < need <= ev and title not in seen_aw:
                        awarded.append(title); seen_aw.add(title)
                        awarded_meta.append({"title": title, "origin": origin})

                elif rtype == "nvram_delta":
                    field = cond.get("field")
                    if not field or is_excluded_field(field):
                        continue
                    need = int(cond.get("min", 1))
                    de = int(self._nv_get_int_ci(end_audits, field, 0))
                    ds = int(self._nv_get_int_ci(start_audits, field, 0))
                    d = de - ds
                    if d < 0:
                        d = 0
                    if d >= need and title not in seen_aw:
                        awarded.append(title); seen_aw.add(title)
                        awarded_meta.append({"title": title, "origin": origin})

                elif rtype == "session_time":
                    min_s = int(cond.get("min_seconds", cond.get("min", 0)))
                    if int(duration_sec or 0) >= min_s and title not in seen_aw:
                        awarded.append(title); seen_aw.add(title)
                        awarded_meta.append({"title": title, "origin": origin})
            except Exception:
                continue

        return awarded, all_titles, awarded_meta
        
    def _collect_global_rules_for_rom(self, rom: str) -> list[dict]:
        """
        Collect global rules only (scope='global') for the given ROM.
        Sources:
          - global_achievements.json (all global rules)
          - rom_specific_achievements/<rom>.ach.json (should be empty for globals, but still tolerated)
          - custom_achievements/*.json (global rules and examples for this ROM)
        Each returned rule is annotated with '_origin' in {'global_achievements','rom_specific','custom'}.
        Titles are deduplicated.
        """
        rules_out = []
        seen_titles = set()

        # 1) global_achievements.json
        gp = f_global_ach(self.cfg)
        if os.path.exists(gp):
            data = load_json(gp, {}) or {}
            for r in (data.get("rules") or []):
                if not isinstance(r, dict):
                    continue
                if self._is_rule_global(r, origin="global_achievements"):
                    t = (r.get("title") or "Achievement").strip()
                    if t not in seen_titles:
                        seen_titles.add(t)
                        r2 = dict(r)
                        r2["_origin"] = "global_achievements"
                        rules_out.append(r2)

        # 2) ROM-specific (should no longer contain global rules, but we accept them if present)
        rpath = os.path.join(p_rom_spec(self.cfg), f"{rom}.ach.json")
        if os.path.exists(rpath):
            data = load_json(rpath, {}) or {}
            for r in (data.get("rules") or []):
                if not isinstance(r, dict):
                    continue
                if self._is_rule_global(r, origin="rom_specific"):
                    t = (r.get("title") or "Achievement").strip()
                    if t not in seen_titles:
                        seen_titles.add(t)
                        r2 = dict(r)
                        r2["_origin"] = "rom_specific"
                        rules_out.append(r2)

        # 3) custom_achievements (direct rules + examples for this ROM)
        cdir = p_custom(self.cfg)
        placeholder = "put_your_custom_achievements_here_click_me.json"
        if os.path.isdir(cdir):
            for fn in os.listdir(cdir):
                if not fn.lower().endswith(".json"):
                    continue
                if fn.lower() == placeholder:
                    continue
                fpath = os.path.join(cdir, fn)
                data = load_json(fpath, {}) or {}

                # a) direct rules
                for r in (data.get("rules") or []):
                    if not isinstance(r, dict):
                        continue
                    if self._is_rule_global(r, origin="custom"):
                        t = (r.get("title") or "Achievement").strip()
                        if t not in seen_titles:
                            seen_titles.add(t)
                            r2 = dict(r)
                            r2["_origin"] = "custom"
                            rules_out.append(r2)

                # b) examples for this ROM
                for ex in (data.get("examples") or []):
                    if not isinstance(ex, dict) or ex.get("rom") != rom:
                        continue
                    for r in (ex.get("achievements") or []):
                        if not isinstance(r, dict):
                            continue
                        if self._is_rule_global(r, origin="custom"):
                            t = (r.get("title") or "Achievement").strip()
                            if t not in seen_titles:
                                seen_titles.add(t)
                                r2 = dict(r)
                                r2["_origin"] = "custom"
                                rules_out.append(r2)

        return rules_out     
        
 
    def _is_rule_global(self, rule: dict, origin: str) -> bool:
        """
        Strikte Regel: Nur Regeln mit scope='global' werden als 'global' gewertet – unabhängig von Quelle.
        """
        scope = str(rule.get("scope") or "").strip().lower()
        return scope == "global"
 

    def _ensure_global_ach(self):
        """
        Legt global_achievements.json an, falls nicht vorhanden ODER wenn sie zu wenige Regeln (< 40) enthält.
        Befüllt mit ~50 abwechslungsreichen Global-Regeln (siehe _generate_default_global_rules).
        """
        path = f_global_ach(self.cfg)
        if os.path.exists(path):
            try:
                data = load_json(path, {}) or {}
                cur = data.get("rules") or []
                if isinstance(cur, list) and len(cur) >= 40:
                    # ausreichend bestückt – nichts tun
                    return
            except Exception:
                pass
        try:
            rules = self._generate_default_global_rules()
            save_json(path, {"rules": rules})
            log(self.cfg, f"global_achievements.json created/refreshed with {len(rules)} rules")
        except Exception as e:
            log(self.cfg, f"[GLOBAL_ACH] generation failed: {e}", "WARN")

    def _ensure_custom_placeholder(self):
        """
        Erzeugt die Beispiel-/Platzhalterdatei mit 'scope' pro Achievement.
        - Session-Achievements: scope='session'
        - Globales Beispiel: scope='global' (session_time 15 Minuten)
        """
        path = os.path.join(p_custom(self.cfg), "PUT_YOUR_CUSTOM_ACHIEVEMENTS_HERE_CLICK_ME.json")
        if not os.path.exists(path):
            payload = {
                "examples": [
                    {
                        "rom": "afm_113b",
                        "achievements": [
                            {
                                "title": "AFM – First Game (Session)",
                                "scope": "session",
                                "condition": {
                                    "type": "nvram_delta",
                                    "field": "Games Started",
                                    "min": 1
                                }
                            },
                            {
                                "title": "AFM – 10 Ramps (Session)",
                                "scope": "session",
                                "condition": {
                                    "type": "nvram_delta",
                                    "field": "Ramps Made",
                                    "min": 10
                                }
                            },
                            {
                                "title": "AFM – 5 Minutes (Session)",
                                "scope": "session",
                                "condition": {
                                    "type": "session_time",
                                    "min_seconds": 300
                                }
                            },
                            {
                                "title": "AFM – 15 Minutes (Global)",
                                "scope": "global",
                                "condition": {
                                    "type": "session_time",
                                    "min_seconds": 900
                                }
                            }
                        ]
                    }
                ]
            }
            save_json(path, payload)
            log(self.cfg, "Custom placeholder created")


    def _rolling_txt_limit(self, rom: Optional[str]):
        if not rom:
            return
        patterns = [
            os.path.join(p_session(self.cfg), f"{sanitize_filename(rom)}__*.txt"),
            os.path.join(p_session(self.cfg), f"*_{sanitize_filename(rom)}_*.txt")
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        while len(files) > ROLLING_HISTORY_PER_ROM:
            old = files.pop(0)
            try:
                os.remove(old)
                log(self.cfg, f"Session limit reached – removed oldest: {old}")
            except Exception as e:
                log(self.cfg, f"Could not remove old session: {e}", "WARN")

    def _highlights_history_limit(self, keep: int = 10):
        try:
            ensure_dir(p_highlights(self.cfg))
            combined = glob.glob(os.path.join(p_highlights(self.cfg), "*.session.json"))
            latest_path = os.path.join(p_highlights(self.cfg), "session_latest.json")
            cand = [p for p in combined if os.path.abspath(p) != os.path.abspath(latest_path)]
            cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for pth in cand[keep:]:
                try:
                    os.remove(pth)
                except Exception as e:
                    log(self.cfg, f"[HIGHLIGHTS] Could not remove {pth}: {e}", "WARN")
        except Exception as e:
            log(self.cfg, f"[HIGHLIGHTS] History enforcement failed: {e}", "WARN")

    # In class Watcher: PATCH _export_summary to also write a per-session *.session.json
    def _export_summary(self, end_audits: dict, duration_sec: int):
        """
        Write session summary JSON with a proper UTC timestamp.
        Enthält jetzt zusätzlich:
          - end_audits: kompletter globaler NVRAM-Snapshot am Ende
          - global_deltas: globale Deltas aus Start-/End-Audits
        NEU: legt zusätzlich eine datierte *.session.json im Highlights-Ordner ab,
             damit der System-Stats-Tab (Heatmaps) Daten laden kann.
        """
        from datetime import timezone
        summary_path = os.path.join(p_highlights(self.cfg), self.SUMMARY_FILENAME)
        try:
            best_ball = None
            try:
                balls = self.ball_track.get("balls", [])
                if balls:
                    best_ball = max(balls, key=lambda b: (int(b.get("score", 0)), int(b.get("duration", 0))))
            except Exception:
                best_ball = None

            # globale Deltas sicher ermitteln
            try:
                global_deltas = self._compute_session_deltas(self.start_audits, end_audits)
            except Exception:
                global_deltas = {}

            players_out = []
            for pid in range(1, 5):
                prec = self.players.get(pid)
                if not prec:
                    players_out.append({
                        "player": pid,
                        "playtime_sec": 0,
                        "deltas": {},
                        "events": {},
                    })
                    continue
                players_out.append({
                    "player": pid,
                    "playtime_sec": int(prec.get("active_play_seconds", 0.0)),
                    "deltas": {k: v for k, v in prec.get("session_deltas", {}).items() if "score" not in k.lower()},
                    "events": prec.get("event_counts", {}),
                })

            payload = {
                "rom": self.current_rom,
                "table": self.current_table,
                "duration_sec": duration_sec,
                "segments": self.snap_segment_index,
                "bootstrap_phase": self.bootstrap_phase,
                "whitelist_size": len(self.active_field_whitelist),
                "best_ball": best_ball,
                "players": players_out,
                "end_audits": end_audits,          # kompletter globaler End-Snapshot
                "global_deltas": global_deltas,     # globale Deltas Start->Ende
                "end_timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # 1) Aktuelle Summary (letzte Session)
            save_json(summary_path, payload)

            # 2) NEU: Persistente Session-Datei für System-Stats/Heatmaps
            try:
                ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                per_session_path = os.path.join(p_highlights(self.cfg), f"{ts_tag}.session.json")
                save_json(per_session_path, payload)
            except Exception as e:
                log(self.cfg, f"[SUMMARY] write per-session file failed: {e}", "WARN")

        except Exception as e:
            log(self.cfg, f"[SUMMARY] export failed: {e}", "WARN")

    # --- Global Self-learning (EMA, post-session, ROM-agnostisch) --------------------------
    def _ai_global_dir(self) -> str:
        """
        Globales AI-Verzeichnis: BASE\\AI
        """
        d = p_ai(self.cfg)
        ensure_dir(d)
        return d

    def _ai_global_model_path(self) -> str:
        """
        Pfad zum globalen Lernmodell (ROM-unabhängig).
        """
        return os.path.join(self._ai_global_dir(), "global.learning.json")

    def _ai_global_coach_path(self) -> str:
        """
        Pfad zu globalen Coach-Tipps (ROM-unabhängig).
        """
        return os.path.join(self._ai_global_dir(), "global.coach.json")

    def _ai_read_latest_summary(self) -> dict:
        """
        Liest die zuletzt exportierte Session-Summary (session_latest.summary.json).
        """
        try:
            path = os.path.join(p_highlights(self.cfg), self.SUMMARY_FILENAME)
            if os.path.isfile(path):
                return load_json(path, {}) or {}
        except Exception:
            pass
        return {}

    def _ai_aggregate_events_from_summary(self, s: dict) -> dict:
        """
        Summiert Events über alle Spieler aus der Summary.
        Rückgabe: {"jackpot":int, "multiball":int, ...}
        """
        totals = {}
        for p in (s.get("players") or []):
            ev = p.get("events") or {}
            for k, v in ev.items():
                try:
                    totals[k] = int(totals.get(k, 0)) + int(v or 0)
                except Exception:
                    pass
        return totals

    def _ai_load_global_model(self) -> dict:
        """
        Lädt das globale Lernmodell oder erzeugt Defaults.
        Struktur:
          {
            "updated": "...",
            "meta": {"sessions": 0, "ema_score": 0.0, "ema_duration": 0.0, "alpha": 0.25},
            "events": { "jackpot": {"w": 0.0, "n": 0}, ... }  # w = Wert/Event (EMA), n = Beobachtungen
          }
        """
        base = {
            "updated": "",
            "meta": {"sessions": 0, "ema_score": 0.0, "ema_duration": 0.0, "alpha": 0.25},
            "events": {}
        }
        path = self._ai_global_model_path()
        try:
            if os.path.isfile(path):
                cur = load_json(path, {}) or {}
                if isinstance(cur, dict):
                    m = cur.setdefault("meta", {})
                    if "alpha" not in m:
                        m["alpha"] = 0.25
                    return cur
        except Exception:
            pass
        return base

    def _ai_save_global_model(self, model: dict):
        try:
            model["updated"] = datetime.now(timezone.utc).isoformat()
            save_json(self._ai_global_model_path(), model)
        except Exception as e:
            log(self.cfg, f"[AI] save global model failed: {e}", "WARN")

# 1) In class Watcher: ersetze _ai_write_global_coach_tips durch diese Version (ohne 'tip'):

    def _ai_write_global_coach_tips(self, model: dict, top_k: int = 5):
        """
        Generate coach top events WITHOUT the template 'tip' text.
        Writes BASE\\AI\\global.coach.json. All text in English (data fields only).
        """
        events = model.get("events", {}) or {}
        meta = model.get("meta", {}) or {}

        items = []
        for k, slot in events.items():
            try:
                w = float(slot.get("w", 0.0))
                n = int(slot.get("n", 0))
            except Exception:
                w, n = 0.0, 0
            if n >= 2 and w > 0:
                items.append((k, w, n))
        items.sort(key=lambda t: t[1], reverse=True)
        top = items[:max(1, int(top_k))]

        advice = []
        for k, w, n in top:
            scenarios = []
            for delta in (1, 3, 5):
                scenarios.append({"delta": delta, "predicted_score_gain": int(round(delta * w))})
            advice.append({
                "event": k,
                "value_per_event": round(w, 2),
                "observations": n,
                "what_if": scenarios
                # 'tip' entfernt
            })

        payload = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "meta": meta,
            "top_events": advice
        }
        save_json(self._ai_global_coach_path(), payload)
        log(self.cfg, f"[AI] global coach tips written: {self._ai_global_coach_path()} without 'tip'")

    def _ai_update_global_learning(self, end_audits: dict, duration_sec: int):
        """
        Globales Post-Session Learning (ROM-unabhängig, EMA):
          - Zielvariable: finaler Score der Session (aus globalen End-Audits)
          - Features: aggregierte Event-Zähler über alle Spieler
          - Update-Regel: EMA auf 'Wert pro Event' (score / max(1, count))
          - Zusätzlich: ema_score/ema_duration, sessions++
          - Erzeugt globale Coach-Tipps (Top-Events + einfache What-if-Schätzung)
        Läuft ausschließlich nach der Session.
        """
        # 1) Daten laden
        summary = self._ai_read_latest_summary()
        if not summary:
            return
        events = self._ai_aggregate_events_from_summary(summary)

        # Ziel: finaler Score (global aus Audits)
        try:
            total_score = int(self._find_score_from_audits(end_audits) or 0)
        except Exception:
            total_score = 0

        # 2) Modell laden
        model = self._ai_load_global_model()
        meta = model.setdefault("meta", {})
        ev_store = model.setdefault("events", {})
        alpha = float(meta.get("alpha", 0.25))
        alpha = 0.25 if alpha <= 0 or alpha >= 1 else alpha

        # 3) EMA-Updates pro beobachtetem Event
        for ev_key, c in (events or {}).items():
            try:
                c = int(c or 0)
            except Exception:
                c = 0
            if c <= 0:
                continue
            sample = float(total_score) / float(max(1, c))
            slot = ev_store.setdefault(ev_key, {"w": 0.0, "n": 0})
            w_old = float(slot.get("w", 0.0))
            w_new = (1.0 - alpha) * w_old + alpha * float(sample)
            slot["w"] = float(w_new)
            slot["n"] = int(slot.get("n", 0)) + 1

        # 4) EMA für Score & Duration + Sessions-Inkrement
        try:
            ema_score = float(meta.get("ema_score", 0.0))
            ema_dur = float(meta.get("ema_duration", 0.0))
            meta["ema_score"] = (1.0 - alpha) * ema_score + alpha * float(total_score)
            meta["ema_duration"] = (1.0 - alpha) * ema_dur + alpha * float(max(1, int(duration_sec or 0)))
        except Exception:
            pass
        meta["sessions"] = int(meta.get("sessions", 0)) + 1

        # 5) Speichern + Coach-Tipps generieren
        self._ai_save_global_model(model)
        try:
            self._ai_write_global_coach_tips(model)
        except Exception as e:
            log(self.cfg, f"[AI] global coach export failed: {e}", "WARN")

    def _ai_global_bootstrap(self):
        """
        Legt beim Start den Ordner BASE\\AI an und erzeugt Default-Dateien,
        falls diese noch nicht existieren.
        """
        try:
            root = self._ai_global_dir()  # sorgt für ensure_dir
            # Modell anlegen, wenn nicht vorhanden
            model_path = self._ai_global_model_path()
            if not os.path.isfile(model_path):
                base_model = {
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "meta": {"sessions": 0, "ema_score": 0.0, "ema_duration": 0.0, "alpha": 0.25},
                    "events": {}
                }
                save_json(model_path, base_model)
                log(self.cfg, f"[AI] created model file: {model_path}")

            # Coach-Datei anlegen, wenn nicht vorhanden
            coach_path = self._ai_global_coach_path()
            if not os.path.isfile(coach_path):
                base_coach = {
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "meta": {"sessions": 0, "ema_score": 0.0, "ema_duration": 0.0, "alpha": 0.25},
                    "top_events": []
                }
                save_json(coach_path, base_coach)
                log(self.cfg, f"[AI] created coach file: {coach_path}")
        except Exception as e:
            log(self.cfg, f"[AI] bootstrap failed: {e}", "WARN")


    # --- AI Profiling (Skill Analysis & Trends, post-session; ENGLISH) ---------------------
    def _ai_history_dir(self) -> str:
        d = os.path.join(self._ai_global_dir(), "history")
        ensure_dir(d)
        return d

    def _ai_profile_path(self) -> str:
        return os.path.join(self._ai_global_dir(), "profile.json")

    def _ai_profile_bootstrap(self):
        try:
            ensure_dir(self._ai_history_dir())
            path = self._ai_profile_path()
            if not os.path.isfile(path):
                payload = {
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "sessions": 0,
                    "style": {"scores": {"precision": 0.0, "multiball": 0.0, "risk": 0.0}, "label": "unknown"},
                    "last_session": {},
                    "trends": {}
                }
                save_json(path, payload)
                log(self.cfg, f"[AI-PROFILE] created {path}")
        except Exception as e:
            log(self.cfg, f"[AI-PROFILE] bootstrap failed: {e}", "WARN")

    def _ach_state_load(self) -> dict:
        """
        Load the persistent achievements state from disk.
        Structure:
          { "global": { "<rom>": [ { "title": str, "origin": str, "ts": iso }, ... ] },
            "session": { "<rom>": [ { "title": str, "ts": iso }, ... ] } }
        """
        try:
            return load_json(f_achievements_state(self.cfg), {}) or {}
        except Exception:
            return {}

    def _ach_state_save(self, state: dict):
        """
        Save the persistent achievements state to disk.
        """
        try:
            save_json(f_achievements_state(self.cfg), state or {})
        except Exception:
            pass

    def _ach_record_unlocks(self, kind: str, rom: str, titles: list):
        """
        Record one-time unlocks (no duplicates per ROM and kind).
        kind: 'global' or 'session'
        titles: list of strings OR dicts like {'title': '...', 'origin': 'global_achievements'}
        """
        if not rom or not titles:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        state = self._ach_state_load()
        bucket = state.setdefault(kind, {})
        lst = bucket.setdefault(rom, [])
        seen = {str(e.get("title") if isinstance(e, dict) else e).strip() for e in lst}
        for t in titles:
            if isinstance(t, dict):
                title = str(t.get("title", "")).strip()
                if not title or title in seen:
                    continue
                entry = {"title": title, "ts": now_iso}
                if t.get("origin"):
                    entry["origin"] = str(t["origin"])
                lst.append(entry)
                seen.add(title)
            else:
                title = str(t).strip()
                if not title or title in seen:
                    continue
                lst.append({"title": title, "ts": now_iso})
                seen.add(title)
        self._ach_state_save(state)
    
    def _ai_build_features_from_summary(self, s: dict) -> dict:
        # Build session features from summary in ENGLISH
        try:
            duration_sec = int(s.get("duration_sec", 0) or 0)
        except Exception:
            duration_sec = 0
        minutes = max(1e-6, duration_sec / 60.0)

        # USE REAL SESSION END TIMESTAMP (fallback to now UTC)
        ts_iso = s.get("end_timestamp") or datetime.now(timezone.utc).isoformat()

        events = self._ai_aggregate_events_from_summary(s) if isinstance(s, dict) else {}
        def ev(k):
            try: return int(events.get(k, 0) or 0)
            except Exception: return 0

        precision_cnt = ev("ramps") + ev("loops") + ev("spinner") + ev("orbit") + ev("combo")
        precision_pm = float(precision_cnt) / minutes
        multiball_pm = (ev("multiball") + 0.8 * ev("jackpot")) / minutes
        jackpots_pm = ev("jackpot") / minutes
        risk_raw = (ev("tilt_warnings") + 2.0 * ev("tilt") - 0.5 * ev("ball_save"))
        risk_pm = max(0.0, risk_raw / minutes)
        ball_saves_pm = ev("ball_save") / minutes
        extra_balls_pm = ev("extra_ball") / minutes

        try:
            score_final = self._find_score_from_audits(s.get("end_audits", {}) or {})
        except Exception:
            score_final = 0
        spm = float(score_final) / minutes

        precision_score = float(max(0.0, min(100.0, precision_pm * 12.0)))
        power_score = float(max(0.0, min(100.0, multiball_pm * 20.0)))
        risk_score = float(max(0.0, min(100.0, risk_pm * 25.0)))

        scores = {"precision": round(precision_score, 2), "multiball": round(power_score, 2), "risk": round(risk_score, 2)}
        label_key = max(scores, key=lambda k: scores[k]) if max(scores.values()) >= 15.0 else "balanced"
        label_map = {"precision": "precise", "multiball": "multiball-oriented", "risk": "risky", "balanced": "balanced"}
        style_label = label_map.get(label_key, label_key)

        best_ball = None
        if isinstance(s.get("best_ball"), dict):
            bb = s.get("best_ball") or {}
            try:
                best_ball = {
                    "num": int(bb.get("num", 0) or 0),
                    "score": int(bb.get("score", 0) or 0),
                    "duration": int(bb.get("duration", 0) or 0),
                }
            except Exception:
                best_ball = None

        return {
            "ts": ts_iso,
            "rom": s.get("rom"),
            "table": s.get("table"),
            "duration_sec": int(duration_sec),
            "score": int(score_final),
            "score_per_min": round(spm, 3),
            "precision_per_min": round(precision_pm, 3),
            "multiball_per_min": round(multiball_pm, 3),
            "jackpots_per_min": round(jackpots_pm, 3),
            "tilt_warnings_per_min": round(ev("tilt_warnings") / minutes, 3) if minutes > 0 else 0.0,
            "tilts": ev("tilt"),
            "ball_saves_per_min": round(ball_saves_pm, 3),
            "extra_balls_per_min": round(extra_balls_pm, 3),
            "style_scores": scores,
            "style_label": style_label,
            "best_ball": best_ball,
        }

    def _ai_compute_trends(self, feats_list: list[dict]) -> dict:
        keys = [
            "score_per_min",
            "precision_per_min",
            "multiball_per_min",
            "jackpots_per_min",
            "ball_saves_per_min",
            "tilt_warnings_per_min",
        ]
        trends = {}
        if not feats_list:
            return trends

        last = feats_list[-20:]
        for k in keys:
            vals = [float(f.get(k, 0.0) or 0.0) for f in last if isinstance(f.get(k, None), (int, float))]
            if not vals:
                continue
            avg_all = sum(vals) / len(vals)
            tail = vals[-5:] if len(vals) >= 5 else vals[-len(vals):]
            prev = vals[-10:-5] if len(vals) >= 10 else vals[:-len(tail)]
            avg_tail = sum(tail) / max(1, len(tail))
            avg_prev = sum(prev) / max(1, len(prev)) if prev else avg_tail
            if avg_prev <= 1e-9:
                pct = 100.0 if avg_tail > 0 else 0.0
            else:
                pct = ((avg_tail - avg_prev) / abs(avg_prev)) * 100.0
            if pct > 5.0:
                t = "up"
            elif pct < -5.0:
                t = "down"
            else:
                t = "stable"
            trends[k] = {
                "avg": round(avg_all, 3),
                "avg_last5": round(avg_tail, 3),
                "delta_pct_vs_prev5": round(pct, 1),
                "trend": t
            }
        return trends

    def _ai_update_profile(self, end_audits: dict, duration_sec: int):
        """
        Post-session profiling:
          - read session_latest.summary.json
          - compute session features + style (EN)
          - write history\\TS.features.json
          - update BASE\\AI\\profile.json (style + trends over history)
        """
        s = self._ai_read_latest_summary()
        if not s:
            return

        feats = self._ai_build_features_from_summary(s)

        # history entry
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            hpath = os.path.join(self._ai_history_dir(), f"{ts}.features.json")
            save_json(hpath, feats)
        except Exception as e:
            log(self.cfg, f"[AI-PROFILE] write history failed: {e}", "WARN")

        # load history (max 50)
        hist_files = []
        try:
            hist_files = [os.path.join(self._ai_history_dir(), fn)
                          for fn in os.listdir(self._ai_history_dir()) if fn.lower().endswith(".features.json")]
            hist_files.sort(key=lambda p: os.path.getmtime(p))
        except Exception:
            hist_files = []
        feats_list = []
        for pth in hist_files[-50:]:
            try:
                d = load_json(pth, None)
                if isinstance(d, dict):
                    feats_list.append(d)
            except Exception:
                continue
        if not feats_list or feats_list[-1].get("ts") != feats.get("ts"):
            feats_list.append(feats)

        trends = self._ai_compute_trends(feats_list)

        style = {"scores": feats.get("style_scores", {}), "label": feats.get("style_label", "unknown")}
        profile_payload = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "sessions": len(feats_list),
            "style": style,
            "last_session": feats,
            "trends": trends
        }
        save_json(self._ai_profile_path(), profile_payload)
        log(self.cfg, f"[AI-PROFILE] profile updated: {self._ai_profile_path()}")



    def on_session_start(self, table_or_rom: str, is_rom: bool = False):
        """
        Session-Start:
        - Setzt current_rom/current_table, initialisiert Snapshot/Bootstrap-Basiswerte,
          lädt Whitelist, schreibt watcher_hook.ini nur bei Änderungen und bereitet Spielerstrukturen vor.
        """
        # activePlayers leeren
        try:
            active_dir = os.path.join(p_highlights(self.cfg), "activePlayers")
            if os.path.isdir(active_dir):
                for f in os.listdir(active_dir):
                    fp = os.path.join(active_dir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
            else:
                ensure_dir(active_dir)
        except Exception as e:
            log(self.cfg, f"[CLEANUP] activePlayers cleanup failed: {e}", "WARN")

        # ROM/Table setzen und INI einmalig sicherstellen
        if is_rom:
            self.current_rom = table_or_rom
            self.current_table = f"(ROM only: {self.current_rom})"
            self._table_load_ts = time.time()
            try:
                # INI nur schreiben, wenn sich etwas geändert hat (kein Spam)
                self._ensure_hook_ini_once()
            except Exception as e:
                log(self.cfg, f"[HOOK] ini-on-start failed: {e}", "WARN")
        else:
            self.current_table = table_or_rom

        # Grundzustand
        self.start_time = time.time()
        self.game_active = True
        self.players.clear()

        # NEU: CPU-Sim initialisieren/übernehmen (aktiv-Flag bleibt erhalten)
        try:
            self._cpu_sim_init()
        except Exception as e:
            log(self.cfg, f"[CPU] init failed: {e}", "WARN")

        # CPU-Sim pro Session zurücksetzen (nur wenn aktiv)
        try:
            if (self.cpu or {}).get("active"):
                self._cpu_sim_session_reset()
        except Exception as e:
            log(self.cfg, f"[CPU] session reset failed: {e}", "WARN")

        # Start-Audits laden und ROM-spezifische Achievements sicherstellen
        self.start_audits, _, _ = self.read_nvram_audits_with_autofix(self.current_rom)

        # Start lightweight sampler if no base map exists (auto-mapping)
        try:
            if not self._base_map_exists(self.current_rom):
                self._nvram_sampler_start(self.current_rom)
        except Exception:
            pass

        try:
            self._ensure_rom_specific(self.current_rom, self.start_audits)
        except Exception as e:
            log(self.cfg, f"[ROM_SPEC] generation failed: {e}", "WARN")

        # Spieler-Snapshots vorbereiten
        self._init_player_snaps(self.start_audits)
        self._last_audits_global = dict(self.start_audits)

        # Bootstrap-Basiswerte JEDE Session zurücksetzen
        try:
            self._snap_bootstrap_done = False
            self._snap_bootstrap_just_done = False
            self._snap_bootstrap_games = int(self._nv_get_int_ci(self.start_audits, "Games Started", 0))
            self._snap_bootstrap_balls = int(self._get_balls_played(self.start_audits) or 0)
            # Wichtig: CB-Basis auf 0 setzen, damit der DLL-Wert (cb >= 1) sofort triggern kann
            self._snap_bootstrap_cb = 0
        except Exception:
            pass

        # Snapshot-/Whitelist-Initialisierung
        if self.snapshot_mode:
            try:
                self._ball_reset(self.start_audits)
            except Exception as e:
                log(self.cfg, f"[BALL] reset failed: {e}", "WARN")
            try:
                self._fw_load_active_whitelist(self.current_rom)
            except Exception as e:
                log(self.cfg, f"[FW] load whitelist failed: {e}", "WARN")
            self.bootstrap_phase = (len(self.active_field_whitelist) == 0)
            if hasattr(self, "_snap_reset"):
                self._snap_reset(self.start_audits)
            else:
                log(self.cfg, "[SNAP] _snap_reset missing – disabling snapshot_mode", "WARN")
                self.snapshot_mode = False
        else:
            self.bootstrap_phase = False

        # Logging
        log(self.cfg, f"[SESSION] Start: table={self.current_table}, rom={self.current_rom}, bootstrap={self.bootstrap_phase}")
        if self.start_audits:
            log(self.cfg, f"[AUDITS] loaded: {len(self.start_audits)} keys")
        else:
            log(self.cfg, f"[AUDITS] none for {self.current_rom}")

    def _ensure_singleplayer_min_playtime(self, nplayers: int, duration_sec: int) -> None:
        """
        Ensures for single-player sessions that P1 active_play_seconds is at least the total session duration.
        This guards against edge cases where fallback allocation yields 0 for P1.
        """
        try:
            if int(nplayers) == 1:
                cur = int(self.players.get(1, {}).get("active_play_seconds") or 0)
                if cur < int(duration_sec):
                    self.players.setdefault(1, {})["active_play_seconds"] = int(duration_sec)
        except Exception:
            # Silent guard: never break session end flow
            pass


    # --- In class Watcher: unmittelbar VOR on_session_end() einfügen ---
    def _normalize_single_player_playtime(self, nplayers: int, end_audits: dict):
        """
        Singleplayer-Korrektur: Setzt P2..P4 Playtime auf 0, wenn keine Evidenz
        für Teilnahme vorhanden (Score==0 und keine Session-Deltas).
        """
        if nplayers > 1:
            return
        for pid in range(2, 5):
            try:
                rec = self.players.get(pid, {})
                if not rec:
                    continue
                cur_pt = int(rec.get("active_play_seconds", 0) or 0)
                if cur_pt <= 0:
                    continue
                try:
                    score = int(end_audits.get(f"P{pid} Score", 0) or 0)
                except Exception:
                    score = 0
                deltas = rec.get("session_deltas", {}) or {}
                has_delta = any(int(v or 0) > 0 for v in deltas.values())
                if score <= 0 and not has_delta:
                    self.players[pid]["active_play_seconds"] = 0
            except Exception:
                continue

            
    def on_session_end(self):
        if not self.game_active:
            return

        end_ts = time.time()
        duration_sec = int(end_ts - (self.start_time or end_ts))
        duration_str = str(timedelta(seconds=duration_sec))

        # Pre-kill snapshot override for challenges (prefer this to disk read)
        ch = getattr(self, "challenge", {}) or {}
        pre = ch.get("prekill_end") if isinstance(ch.get("prekill_end", None), dict) else None
        if pre:
            end_audits = dict(pre)
        else:
            # Read end audits (fallback to last known if necessary)
            try:
                end_audits, _, _ = self.read_nvram_audits_with_autofix(self.current_rom)
                if not end_audits:
                    raise RuntimeError("Empty end_audits")
            except Exception as e:
                log(self.cfg, f"[END] read end audits failed, using last known: {e}", "WARN")
                end_audits = dict(self._last_audits_global)

        # Determine players in game (for fallbacks and gating session achievements)
        players_detected = self._infer_players_in_game_from_audits(self.start_audits, end_audits)
        nplayers = max(1, min(4, int(players_detected or 1)))
        single_player_fast = (nplayers <= 1)

        # Compute per-player deltas (segment-based if available; otherwise fallbacks)
        seg_deltas = {}
        try:
            if getattr(self, "_snap_bootstrap_done", False) and int(self.snap_segment_index or 0) > 0:
                seg_deltas = self._compute_player_deltas(end_audits) or {}
                log(self.cfg, "[SNAP] segment-based attribution used (all players)")
            else:
                if single_player_fast:
                    seg_deltas = {1: self._compute_session_deltas(self.start_audits, end_audits)}
                    log(self.cfg, "[SNAP] single-player fallback (no segments)")
                else:
                    seg_deltas = self._compute_player_deltas_end_only(self.start_audits, end_audits) or {}
                    log(self.cfg, "[SNAP] end-only attribution used (score-weighted, no segments)")
        except Exception as e:
            seg_deltas = {}
            log(self.cfg, f"[SNAP] attribution failed: {e}", "WARN")

        try:
            log(self.cfg, f"[SNAP] summary: players={nplayers}, segments={int(self.snap_segment_index or 0)}, bootstrap={bool(getattr(self, '_snap_bootstrap_done', False))}")
        except Exception:
            pass

        # Merge computed deltas back into self.players
        if single_player_fast and (getattr(self, "_snap_bootstrap_done", False) is False or int(self.snap_segment_index or 0) <= 0):
            self.players.setdefault(1, {
                "start_audits": self._player_field_filter(self.start_audits, 1) or {"P1 Score": 0},
                "last_audits": self._player_field_filter(end_audits, 1) or {"P1 Score": 0},
                "active_play_seconds": float(self.players.get(1, {}).get("active_play_seconds", 0.0)),
                "start_time": self.players.get(1, {}).get("start_time", time.time()),
                "session_deltas": {},
                "event_counts": self.players.get(1, {}).get("event_counts", {}),
            })
            existing = self.players[1].get("session_deltas", {}) or {}
            computed = seg_deltas.get(1, {}) if isinstance(seg_deltas, dict) else (seg_deltas or {})
            merged = dict(existing)
            for k, v in (computed or {}).items():
                ex = int(merged.get(k, 0) or 0)
                cv = int(v or 0)
                if cv > ex:
                    merged[k] = cv
            self.players[1]["session_deltas"] = merged

            for pid in range(2, 5):
                if pid in self.players:
                    self.players[pid]["session_deltas"] = {}
            log(self.cfg, "[SNAP] single-player fast path applied")
        else:
            for pid in range(1, 5):
                self.players.setdefault(pid, {
                    "start_audits": self._player_field_filter(self.start_audits, pid) or {f"P{pid} Score": 0},
                    "last_audits": self._player_field_filter(end_audits, pid) or {f"P{pid} Score": 0},
                    "active_play_seconds": float(self.players.get(pid, {}).get("active_play_seconds", 0.0)),
                    "start_time": self.players.get(pid, {}).get("start_time", time.time()),
                    "session_deltas": self.players.get(pid, {}).get("session_deltas", {}),
                    "event_counts": self.players.get(pid, {}).get("event_counts", {}),
                })
                existing = self.players[pid].get("session_deltas", {}) or {}
                computed = seg_deltas.get(pid, {}) or {}
                merged = dict(existing)
                for k, v in computed.items():
                    ex = int(merged.get(k, 0) or 0)
                    cv = int(v or 0)
                    if cv > ex:
                        merged[k] = cv
                self.players[pid]["session_deltas"] = merged

        # Playtime fallback
        try:
            total_pt = 0.0
            for pid in range(1, nplayers + 1):
                total_pt += float(self.players.get(pid, {}).get("active_play_seconds", 0.0))
            if int(self.snap_segment_index or 0) == 0 or total_pt <= 1.0:
                score_deltas = self._score_deltas_from_audits(self.start_audits, end_audits, nplayers)
                tot_score = sum(score_deltas.values())
                weights = {}
                if tot_score > 0:
                    for pid in range(1, nplayers + 1):
                        weights[pid] = float(score_deltas.get(pid, 0))
                else:
                    nonzero = [pid for pid in range(1, nplayers + 1) if self._find_score_from_audits(end_audits, pid) > 0]
                    base_set = nonzero if nonzero else list(range(1, nplayers + 1))
                    for pid in range(1, nplayers + 1):
                        weights[pid] = 1.0 if pid in base_set else 0.0
                alloc = self._allocate_int_proportional(int(duration_sec), weights)
                for pid in range(1, nplayers + 1):
                    try:
                        self.players[pid]["active_play_seconds"] = int(alloc.get(pid, 0))
                    except Exception:
                        pass
                log(self.cfg, "[SNAP] playtime fallback assigned (end-only)")
                self._normalize_single_player_playtime(nplayers, end_audits)
        except Exception as e:
            log(self.cfg, f"[SNAP] playtime fallback failed: {e}", "WARN")

        # NEW: ensure P1 playtime >= session duration in single-player sessions
        self._ensure_singleplayer_min_playtime(nplayers, duration_sec)

        # Evaluate/persist achievements
        awarded_from_ga = []
        session_achs_p1 = []
        try:
            awarded, _all_global, awarded_meta = self._evaluate_achievements(self.current_rom, self.start_audits, end_audits, duration_sec)
        except Exception as e:
            log(self.cfg, f"[ACH] eval failed: {e}", "WARN")
            awarded, awarded_meta = [], []

        # Global: only origin='global_achievements'
        try:
            awarded_from_ga = [m for m in (awarded_meta or []) if (m.get("origin") == "global_achievements")]
            try:
                st = self._ach_state_load()
                already = {str(e.get("title", "")) for e in (st.get("global", {}).get(self.current_rom, []) or [])}
                if already:
                    awarded_from_ga = [m for m in awarded_from_ga if str(m.get("title", "")) not in already]
            except Exception:
                pass
            if awarded_from_ga:
                self._ach_record_unlocks("global", self.current_rom, awarded_from_ga)
        except Exception as e:
            log(self.cfg, f"[ACH] persist global failed: {e}", "WARN")

        # Session achievements: only 1-player (persist internally, but do not print in snapshot)
        try:
            if nplayers == 1:
                session_achs_p1 = self._evaluate_player_session_achievements(1, self.current_rom) or []
                try:
                    st = self._ach_state_load()
                    already_sess = {str(e.get("title", "")) for e in (st.get("session", {}).get(self.current_rom, []) or [])}
                    if already_sess:
                        session_achs_p1 = [t for t in session_achs_p1 if str(t) not in already_sess]
                except Exception:
                    pass
                if session_achs_p1:
                    self._ach_record_unlocks("session", self.current_rom, list(session_achs_p1))
                    # Optional toast display kept; remove if you also want no popups
                    try:
                        for t in session_achs_p1:
                            self.bridge.ach_toast_show.emit(str(t), self.current_rom or "", 5)
                    except Exception:
                        pass
            else:
                log(self.cfg, "[ACH] session achievements skipped (multi-player session)")
        except Exception as e:
            log(self.cfg, f"[ACH] persist session failed: {e}", "WARN")

        # TXT export (achievements intentionally not shown)
        txt_filename = (
            f"{sanitize_filename(self.current_rom)}__"
            f"{sanitize_filename(self.current_table)}__"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )
        path_txt = os.path.join(p_session(self.cfg), txt_filename)
        lines: List[str] = []
        lines.append(f"Table:    {self.current_table}")
        lines.append(f"ROM:      {self.current_rom}")
        lines.append(f"Duration: {duration_str}")

        best_ball = None
        try:
            balls = self.ball_track.get("balls", [])
            if balls:
                best_ball = max(balls, key=lambda b: (int(b.get("score", 0)), int(b.get("duration", 0))))
        except Exception:
            best_ball = None
        if best_ball:
            mm, ss = divmod(int(best_ball.get('duration', 0)), 60)
            lines.append(f"Best Ball: #{best_ball.get('num', 1)} – {mm}m {ss}s")

        lines.append("=== Global Snapshot ===")
        # Achievements intentionally omitted from Global snapshot

        NV_EXCLUDE_VALUE = 4_373_822
        SKIP_WEIRD_GE = 10_000_000
        CONTROL_FIELDS = {"current_player", "player_count", "current_ball", "balls played", "credits", "game over", "tilt warnings", "tilted"}

        lines.append("Audits (filtered):")
        try:
            for k in sorted(end_audits.keys(), key=lambda s: str(s).lower()):
                if not isinstance(k, str):
                    continue
                ll = k.lower()
                if ll in CONTROL_FIELDS or is_excluded_field(k) or self.NOISE_REGEX.search(k):
                    continue
                v = end_audits.get(k)
                if isinstance(v, (int, float)):
                    vi = int(v)
                    if vi == NV_EXCLUDE_VALUE or vi >= SKIP_WEIRD_GE:
                        continue
                lines.append(f"{k} {v}")
        except Exception as e:
            log(self.cfg, f"[AUDITS] print failed: {e}", "WARN")
        lines.append("")

        for pid in range(1, 5):
            prec = self.players.get(pid, {"active_play_seconds": 0.0, "session_deltas": {}, "event_counts": {}})
            lines.append(f"=== Player {pid} Snapshot ===")
            psec = int(prec.get("active_play_seconds", 0.0))
            lines.append(f"Playtime: {str(timedelta(seconds=psec))}")
            # Session achievements intentionally omitted in Player snapshots

            s_deltas = prec.get("session_deltas", {}) or {}
            lines.append("Session Deltas:")
            any_out = False
            if s_deltas:
                for k in sorted(s_deltas.keys()):
                    if not isinstance(k, str) or k.startswith("P"):
                        continue
                    kl = k.lower()
                    if ("score" in kl) or (kl in CONTROL_FIELDS) or is_excluded_field(k) or self.NOISE_REGEX.search(k):
                        continue
                    try:
                        val = int(s_deltas.get(k, 0) or 0)
                    except Exception:
                        val = 0
                    if val <= 0:
                        continue
                    if _is_weird_value(val):
                        continue
                    lines.append(f"  {k:<24} {val:>6}")
                    any_out = True
            if not any_out:
                lines.append("  (none)")
            lines.append("")

        write_text(path_txt, "\n".join(lines))
        log(self.cfg, f"[SESSION END] table={self.current_table}, rom={self.current_rom}, duration={duration_str}")
        self._rolling_txt_limit(self.current_rom)

        try:
            if str((self.cfg.OVERLAY or {}).get("cpu_sim_mode", "postgame")).lower() == "postgame":
                self._cpu_postgame_simulate(self.current_rom, duration_sec)
        except Exception as e:
            log(self.cfg, f"[CPU] postgame simulate failed: {e}", "WARN")

        try:
            self.export_overlay_snapshot(end_audits, duration_sec, on_demand=False)
        except Exception as e:
            log(self.cfg, f"[HIGHLIGHTS] export failed: {e}", "WARN")
        try:
            self._export_summary(end_audits, duration_sec)
        except Exception as e:
            log(self.cfg, f"[SUMMARY] failed: {e}", "WARN")

        try:
            self._ai_update_global_learning(end_audits, duration_sec)
        except Exception as e:
            log(self.cfg, f"[AI] global learning update failed: {e}", "WARN")

        try:
            self._ai_update_profile(end_audits, duration_sec)
        except Exception as e:
            log(self.cfg, f"[AI-PROFILE] update failed: {e}", "WARN")

        try:
            if self.current_rom and not self._base_map_exists(self.current_rom):
                self._nvram_autogen_map(self.current_rom)
        except Exception as e:
            log(self.cfg, f"[AUTOMAP] failed: {e}", "WARN")

        # Auto-show overlay unless a challenge requested to suppress it
        try:
            ch = getattr(self, "challenge", {}) or {}
            suppress = bool(ch.get("suppress_big_overlay_once", False))
            if suppress:
                ch["suppress_big_overlay_once"] = False
                self.challenge = ch
            elif (self.cfg.OVERLAY or {}).get("auto_show_on_end", True):
                self.bridge.overlay_show.emit()
        except Exception as e:
            log(self.cfg, f"[OVERLAY] auto-show emit failed: {e}", "WARN")

        # Challenge result banner + persist (only if challenge was active)
        try:
            ch = getattr(self, "challenge", {}) or {}

            # Timed: injiziere den besten P1‑Score in end_audits, damit das gespeicherte Ergebnis sicher ist
            if str(ch.get("kind", "")).lower() == "timed":
                try:
                    end_audits = self._inject_best_score_for_timed(end_audits)
                except Exception:
                    pass

            if ch.get("kind") in ("timed", "oneball"):
                self._challenge_record_result(str(ch.get("kind")), end_audits, duration_sec)
        except Exception as e:
            log(self.cfg, f"[CHALLENGE] result finalize failed: {e}", "WARN")

        # Reset session state
        self.current_table = None
        self.current_rom = None
        self.start_time = None
        self.game_active = False
        self.start_audits = {}
        self.players.clear()
        self.ball_track.update({"active": False, "index": 0, "start_time": None, "score_base": 0, "last_balls_played": None, "balls": []})
        self._last_audits_global = {}
        self.snap_initialized = False
        self.field_stats.clear()
        self.active_field_whitelist.clear()
        self.bootstrap_phase = False
        self.current_segment_provisional_diff = {}


   
    def force_flush(self) -> Dict[str, Any]:
        if not self.game_active or not self.current_rom:
            return {"ok": False, "reason": "no active game"}
        with self._flush_lock:
            audits, _, _ = self.read_nvram_audits_with_autofix(self.current_rom)
            if not audits:
                return {"ok": False, "reason": "no audits"}
            duration_sec = int(time.time() - (self.start_time or time.time()))
            latest_path = self.export_overlay_snapshot(audits, duration_sec, on_demand=True)
            try:
                st = os.stat(latest_path)
                return {"ok": True, "latest": latest_path, "size": int(st.st_size), "mtime": float(st.st_mtime)}
            except Exception:
                return {"ok": True, "latest": latest_path}

    def monitor_table(self) -> Optional[Dict[str, str]]:
        if not win32gui:
            return None
        def _cb(hwnd, acc):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title.startswith("Visual Pinball - ["):
                    acc.append(title)
        wins = []
        try:
            win32gui.EnumWindows(_cb, wins)
        except Exception:
            return None
        if not wins:
            return None
        title = wins[0]
        if not (title.startswith("Visual Pinball - [") and title.endswith("]")):
            return None
        table_fragment = title[len("Visual Pinball - ["):-1]
        vpx_filename = table_fragment if table_fragment.lower().endswith(".vpx") else table_fragment + ".vpx"
        vpx_path = os.path.join(self.cfg.TABLES_DIR, vpx_filename)
        if not os.path.isfile(vpx_path):
            alt = os.path.join(self.cfg.TABLES_DIR, table_fragment)
            vpx_path = alt if os.path.isfile(alt) else None
        rom = extract_cgamename_from_vpx(vpx_path) if vpx_path else None
        used_fallback = False
        if not rom:
            rom = fallback_rom_from_table_name(vpx_filename)
            used_fallback = True
        if not rom:
            return None
        if rom != self._last_logged_rom:
            log(self.cfg, f"[ROM] {'Fallback' if used_fallback else 'Script'}: {rom}")
            self._last_logged_rom = rom
        clean_table = table_fragment[:-4] if table_fragment.lower().endswith(".vpx") else table_fragment
        return {"table": clean_table, "rom": rom, "vpx_file": vpx_path or ""}

    def _thread_main(self):
        log(self.cfg, ">>> watcher thread running")
        active_rom = None
        if not hasattr(self, "_last_live_export_ts"):
            self._last_live_export_ts = 0.0

        while not self._stop.is_set():
            now_loop = time.time()
            dt = now_loop - getattr(self, "_last_tick_time", now_loop)
            if dt < 0 or dt > 5:
                dt = 0.5
            self._last_tick_time = now_loop

            try:
                upd = self.monitor_table()
            except Exception as e:
                log(self.cfg, f"[WATCHER] monitor error: {e}", "WARN")
                upd = None

            if upd:
                rom = (upd.get("rom") or "").strip()
                if active_rom is None and rom:
                    self.on_session_start(rom, is_rom=True)
                    active_rom = rom
                    # NEW: show mini info overlay if map is missing in BASE\NVRAM_Maps\maps
                    self._emit_mini_info_if_missing_map(rom, 5)

                elif active_rom and rom and rom != active_rom:
                    self.on_session_end()
                    active_rom = None
                    self.on_session_start(rom, is_rom=True)
                    active_rom = rom
                    # NEW: show mini info overlay if map is missing in BASE\NVRAM_Maps\maps
                    self._emit_mini_info_if_missing_map(rom, 5)

                # Injector früh starten (liefert live.session.json)
                try:
                    self._try_start_injectors_for_vpx()
                except Exception:
                    pass

                if active_rom:
                    # 1) Audits
                    audits, _, _ = self.read_nvram_audits_with_autofix(self.current_rom)

                    # Periodic sampler tick (collect nvram bytes for auto-mapping)
                    try:
                        self._nvram_sampler_tick()
                    except Exception:
                        pass

                    # 2) Control-Signale (DLL bevorzugt)
                    try:
                        controls = self._read_control_signals(self.current_rom) or {}
                    except Exception:
                        controls = {}
                    audits_ctl = dict(audits)
                    for k in ("current_player", "player_count", "current_ball", "Balls Played"):
                        if k in controls:
                            audits_ctl[k] = controls[k]

                    # Fallback: wenn die DLL nur 'current_ball' liefert, aber kein 'Balls Played',
                    # synthesize 'Balls Played' aus 'current_ball' (Ziel: One‑Ball sauber triggern)
                    if "Balls Played" not in audits_ctl and "current_ball" in audits_ctl:
                        try:
                            audits_ctl["Balls Played"] = int(audits_ctl["current_ball"])
                        except Exception:
                            pass

                    # NEU: One‑Ball unabhängig von snapshot_mode prüfen (Drain via bp oder cb)
                    try:
                        self._oneball_check_and_schedule(audits_ctl)
                    except Exception as e:
                        log(self.cfg, f"[CHALLENGE] one-ball check in loop failed: {e}", "WARN")

                    # optional: [CTRL]-Log …
                    try:
                        if bool(getattr(self.cfg, "LOG_CTRL", False)):
                            now_dbg = time.time()
                            if now_dbg - getattr(self, "_dbg_ctl_ts", 0.0) >= 1.0:
                                cp = audits_ctl.get("current_player")
                                pc = audits_ctl.get("player_count")
                                cb = audits_ctl.get("current_ball")
                                bp = audits_ctl.get("Balls Played")
                                tup = (cp, pc, cb, bp)
                                if tup != getattr(self, "_dbg_ctl_last", None):
                                    log(self.cfg, f"[CTRL] live cp={cp} pc={pc} cb={cb} bp={bp}")
                                    self._dbg_ctl_last = tup
                                self._dbg_ctl_ts = now_dbg
                    except Exception:
                        pass

                    # 3) PRE-GAME-Info …
                    try:
                        if not getattr(self, "_snap_bootstrap_done", False):
                            pc = int(audits_ctl.get("player_count", 0) or 0)
                            if pc and pc != int(getattr(self, "_pregame_last_pc", 0) or 0):
                                self._pregame_last_pc = pc
                                log(self.cfg, f"[SNAP] pregame player_count detected: {pc}")
                    except Exception:
                        pass

                    # 4) Live-Export alle 2s …
                    try:
                        now2 = time.time()
                        if self.current_rom and self.cfg.OVERLAY.get("live_updates", False) and (now2 - self._last_live_export_ts >= 2.0):
                            duration_sec = int(now2 - (self.start_time or now2))
                            self.export_overlay_snapshot(audits, duration_sec, on_demand=True)
                            self._last_live_export_ts = now2
                    except Exception as e:
                        log(self.cfg, f"[EXPORT] live export failed: {e}", "WARN")

                    # 5) SNAP-Bootstrap …
                    if self.snapshot_mode:
                        # (unverändert)
                        pass

                    # 6) Spielerzahl erkennen …
                    try:
                        if self.snapshot_mode and self.snap_initialized and hasattr(self, "_snap_detect_players"):
                            self._snap_detect_players(audits_ctl)
                    except Exception:
                        pass

                    # 7) Rotation …
                    if self.snapshot_mode and self.snap_initialized:
                        try:
                            seg_before = int(self.snap_segment_index or 0)
                            self._maybe_rotate_on_current_player(audits_ctl)
                            rotated = (int(self.snap_segment_index or 0) != seg_before)
                            if not rotated:
                                cur_bp = self._get_balls_played(audits_ctl)
                                if cur_bp is not None:
                                    if self.snap_last_balls_played is None:
                                        self.snap_last_balls_played = cur_bp
                                    elif cur_bp > int(self.snap_last_balls_played):
                                        if getattr(self, "_snap_bootstrap_just_done", False):
                                            self.snap_last_balls_played = cur_bp
                                            self._snap_bootstrap_just_done = False
                                        else:
                                            steps = int(cur_bp) - int(self.snap_last_balls_played)
                                            self.snap_last_balls_played = cur_bp
                                            self._snap_rotate(audits_ctl, steps=steps)
                            rotated = (int(self.snap_segment_index or 0) != seg_before)
                            if not rotated:
                                self._maybe_rotate_on_score_delta(audits)
                        except Exception as e:
                            log(self.cfg, f"[SNAP] rotate (cp/bp/score) failed: {e}", "WARN")

                        # 7.4) Externer Detector …
                        try:
                            if self._pending_detector_switch is not None:
                                target_pid = int(self._pending_detector_switch)
                                cur = int(self.snap_player or 1)
                                if target_pid != cur:
                                    try:
                                        self._snap_finalize_segment(audits_ctl, note="(detector)")
                                    except Exception as e:
                                        log(self.cfg, f"[SNAP] finalize(detector) failed: {e}", "WARN")
                                    self.players.setdefault(cur, {})
                                    self.players.setdefault(target_pid, {})
                                    self.players[cur]["end_audits"] = dict(audits_ctl)
                                    self.players[target_pid]["start_audits"] = dict(audits_ctl)
                                    self.snap_player = target_pid
                                    self.snap_start_audits = dict(audits_ctl)
                                    self.snap_segment_index += 1
                                    self.snap_segment_start_time = time.time()
                                    log(self.cfg, f"[SNAP] detector -> switched to P{self.snap_player}")
                                else:
                                    log(self.cfg, f"[SNAP] detector -> same player P{cur} (ignored)")
                                self._pending_detector_switch = None
                        except Exception as e:
                            log(self.cfg, f"[SNAP] detector-rotate failed: {e}", "WARN")

                    # 8) Live-Overlay-Diff …
                    if self.snapshot_mode and self.snap_initialized and self.include_current_segment_in_overlay:
                        try:
                            self.current_segment_provisional_diff = self._snap_diff(self.snap_start_audits, audits)
                        except Exception:
                            self.current_segment_provisional_diff = {}
                    else:
                        self.current_segment_provisional_diff = {}

                    # 9) current_player Quelle …
                    try:
                        cp_val = int(audits_ctl.get("current_player", audits.get("current_player", 1)) or 1)
                    except Exception:
                        cp_val = 1
                    if self.snapshot_mode and getattr(self, "_snap_bootstrap_done", False):
                        self.current_player = int(self.snap_player or cp_val or 1)
                    else:
                        self.current_player = cp_val

                    # 10) Spielzeit
                    try:
                        if self.current_player in self.players:
                            self.players[self.current_player]["active_play_seconds"] = \
                                float(self.players[self.current_player].get("active_play_seconds", 0.0)) + dt
                    except Exception:
                        pass

                    # 10.5) Live-Events attribuieren …
                    try:
                        changed = False
                        try:
                            changed = bool(self._attribute_events(audits_ctl))
                        except Exception:
                            changed = False
                        if changed and self.cfg.OVERLAY.get("live_updates", False):
                            try:
                                duration_now = int(time.time() - (self.start_time or time.time()))
                                self.export_overlay_snapshot(audits, duration_now, on_demand=True)
                                self._last_live_export_ts = time.time()
                            except Exception as e:
                                log(self.cfg, f"[EXPORT] live immediate export failed: {e}", "WARN")
                    except Exception as e:
                        log(self.cfg, f"[HIGHLIGHTS] live attribute failed: {e}", "WARN")

                    # 10.6) CPU-Sim TICK (NEU)
                    try:
                        self._cpu_sim_tick(dt)
                    except Exception as e:
                        log(self.cfg, f"[CPU] tick failed in loop: {e}", "WARN")

                    # 11) Player-Audits (nur Anzeige)
                    for pid in range(1, 5):
                        if pid not in self.players:
                            self.players[pid] = {
                                "start_audits": self._player_field_filter(self.start_audits, pid) or {f"P{pid} Score": 0},
                                "last_audits": self._player_field_filter(self.start_audits, pid) or {f"P{pid} Score": 0},
                                "active_play_seconds": 0.0,
                                "start_time": time.time(),
                                "session_deltas": {},
                                "event_counts": {},
                            }
                        player_audits = self._player_field_filter(audits, pid)
                        if player_audits:
                            self.players[pid]["last_audits"].update(player_audits)

                    # 10.7) Challenges tick (time/kill handling)
                    try:
                        self._challenge_tick(audits_ctl)
                    except Exception as e:
                        log(self.cfg, f"[CHALLENGE] tick failed in loop: {e}", "WARN")

                    # 12) Ball-Tracking (auch bei One‑Ball aktivieren)
                    if self.snapshot_mode or (getattr(self, "challenge", {}).get("active") and getattr(self, "challenge", {}).get("kind") == "oneball"):
                        try:
                            self._ball_update(audits_ctl)
                        except Exception as e:
                            log(self.cfg, f"[BALL] update failed: {e}", "WARN")

            else:
                if active_rom is not None:
                    self.on_session_end()
                    active_rom = None

            time.sleep(0.5)


    def start(self):
        """
        Bootstrap + Hintergrund-Thread starten. Idempotent.
        """
        if getattr(self, "thread", None) and self.thread.is_alive():
            return
        try:
            self.bootstrap()
        except Exception as e:
            log(self.cfg, f"[BOOTSTRAP] failed: {e}", "WARN")
        try:
            self._ensure_global_ach()
        except Exception as e:
            log(self.cfg, f"[GLOBAL_ACH] ensure failed: {e}", "WARN")
        try:
            self._ensure_custom_placeholder()
        except Exception as e:
            log(self.cfg, f"[CUSTOM_PLACEHOLDER] ensure failed: {e}", "WARN")
        try:
            self._start_detector_http(host="127.0.0.1", port=8765)
        except Exception as e:
            log(self.cfg, f"[HTTP] detector start failed: {e}", "WARN")
        try:
            self.start_prefetch_background()
        except Exception as e:
            log(self.cfg, f"[PREFETCH] auto-start failed: {e}", "WARN")

        self._stop.clear()
        self.thread = threading.Thread(target=self._thread_main, daemon=True, name="WatcherThread")
        self.thread.start()

# In Watcher.stop(): Injector-Kill einbauen (vor dem finalen Log)
    def stop(self):
        """
        Thread sauber stoppen und Session ggf. beenden.
        """
        try:
            self._stop.set()
            if getattr(self, "thread", None):
                self.thread.join(timeout=3)
        except Exception:
            pass

        # Session beenden, falls aktiv
        if self.game_active:
            try:
                self.on_session_end()
            except Exception as e:
                log(self.cfg, f"[WATCHER] on_session_end during stop failed: {e}", "WARN")

        # HTTP-Detector stoppen
        try:
            self._stop_detector_http()
        except Exception:
            pass

        # NEU: Injector-Prozesse beenden + verifizieren
        try:
            self._kill_injectors(force=True, verify=True, timeout_verify=3.0)
        except Exception as e:
            log(self.cfg, f"[HOOK] kill injectors failed: {e}", "WARN")

        # B2S-Prozess beenden, falls VR-Hide aktiv
        try:
            self._kill_b2s_process_if_enabled()
        except Exception:
            pass

        log(self.cfg, "[WATCHER] stopped")


            
class Bridge(QObject):
    overlay_trigger = pyqtSignal()
    overlay_show = pyqtSignal()  # explicit “show overlay” signal (no toggle)
    # Small info overlay with countdown (existing): args: rom, seconds
    mini_info_show = pyqtSignal(str, int)
    # Achievement toast (existing): title, rom, seconds
    ach_toast_show = pyqtSignal(str, str, int)

    # NEW: Challenges signals
    # Start/stop the bottom-left timer overlay (seconds total)
    challenge_timer_start = pyqtSignal(int)
    challenge_timer_stop = pyqtSignal()
    # Warm-up banner in the center (seconds, message)
    challenge_warmup_show = pyqtSignal(int, str)
    # Small centered result banner (message, seconds, color hex like "#FFFFFF")
    challenge_info_show = pyqtSignal(str, int, str)
    # Request the GUI to speak a short English phrase (volume is configured in Challenges tab)
    challenge_speak = pyqtSignal(str)

    def __init__(self):
        super().__init__()


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]

# Low-level keyboard hook id
WH_KEYBOARD_LL = 13

# --- replace GlobalKeyHook with multi-binding version ---
class GlobalKeyHook:
    """
    Installs a low-level keyboard hook (WH_KEYBOARD_LL) and invokes supplied callbacks
    when configured VKs are pressed. Works regardless of foreground window.

    Now supports multiple bindings:
      bindings: list of dicts with keys:
        - name: str
        - get_vk: callable() -> int
        - on_press: callable()
    """
    def __init__(self, bindings: list[dict]):
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        self._hook = None
        self._proc = None
        self._bindings = list(bindings or [])

    def update_bindings(self, bindings: list[dict]):
        self._bindings = list(bindings or [])

    def _callback(self, nCode, wParam, lParam):
        try:
            if nCode == 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                vk = int(kb.vkCode)
                # dispatch to all matching bindings
                for b in self._bindings:
                    try:
                        want = int(b.get("get_vk", lambda: -1)())
                    except Exception:
                        want = -1
                    if want and vk == want:
                        cb = b.get("on_press")
                        if cb:
                            # don’t steal focus, schedule on GUI thread
                            QTimer.singleShot(0, cb)
        except Exception:
            pass
        return self._user32.CallNextHookEx(self._hook, nCode, wParam, lParam)

    def install(self):
        if self._hook:
            return
        CMPFUNC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)
        self._proc = CMPFUNC(self._callback)
        hMod = self._kernel32.GetModuleHandleW(None)
        self._hook = self._user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, hMod, 0)

    def uninstall(self):
        if self._hook:
            try:
                self._user32.UnhookWindowsHookEx(self._hook)
            except Exception:
                pass
        self._hook = None
        self._proc = None
        
class SetupWizardDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Initial Setup – Achievement Paths")
        self.resize(640, 320)
        main = QVBoxLayout(self)
        info = QLabel(
            "Welcome!\n\n"
            "Select paths for:\n"
            "  1) Base achievements data\n"
            "  2) VPinMAME NVRAM directory\n"
            "  3) (Optional) Tables directory\n\n"
            "You can re-run this wizard later."
        )
        info.setWordWrap(True)
        main.addWidget(info)

        def row(label, val, title):
            lay = QHBoxLayout()
            edit = QLineEdit(val)
            btn = QPushButton("…")
            def pick():
                d = QFileDialog.getExistingDirectory(self, title, edit.text().strip() or os.path.expanduser("~"))
                if d:
                    edit.setText(d)
            btn.clicked.connect(pick)
            lay.addWidget(QLabel(label))
            lay.addWidget(edit, 1)
            lay.addWidget(btn)
            return lay, edit

        lay_base, self.ed_base = row("Base:", self.cfg.BASE, "Select Achievements Base Folder")
        lay_nv, self.ed_nvram = row("NVRAM:", self.cfg.NVRAM_DIR, "Select NVRAM Directory")
        lay_tab, self.ed_tables = row("Tables:", self.cfg.TABLES_DIR, "Select Tables Directory (optional)")
        main.addLayout(lay_base); main.addLayout(lay_nv); main.addLayout(lay_tab)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#c04020;font-weight:bold;")
        main.addWidget(self.lbl_status)

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Apply & Start")
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self._accept_if_valid)
        btns.addStretch(1); btns.addWidget(btn_cancel); btns.addWidget(btn_ok)
        main.addLayout(btns)

        self.btn_ok = btn_ok
        self.ed_base.textChanged.connect(self._validate)
        self.ed_nvram.textChanged.connect(self._validate)
        self._validate()

    def _validate(self):
        base = self.ed_base.text().strip()
        nvram = self.ed_nvram.text().strip()
        errors = []
        if not base:
            errors.append("Missing base path")
        if nvram and not os.path.isdir(nvram):
            errors.append("NVRAM dir does not exist")
        self.btn_ok.setEnabled(len(errors) == 0)
        self.lbl_status.setText(" / ".join(errors) if errors else "")

    def _accept_if_valid(self):
        self._validate()
        if not self.btn_ok.isEnabled():
            return
        self.cfg.BASE = os.path.abspath(self.ed_base.text().strip())
        if self.ed_nvram.text().strip():
            self.cfg.NVRAM_DIR = os.path.abspath(self.ed_nvram.text().strip())
        if self.ed_tables.text().strip():
            self.cfg.TABLES_DIR = os.path.abspath(self.ed_tables.text().strip())
        self.cfg.FIRST_RUN = False
        self._ensure_base_layout()
        self.cfg.save()
        log(self.cfg, f"[SETUP] BASE={self.cfg.BASE} NVRAM={self.cfg.NVRAM_DIR} TABLES={self.cfg.TABLES_DIR}")
        self.accept()

    def _ensure_base_layout(self):
        try:
            ensure_dir(self.cfg.BASE)
            for sub in [
                "NVRAM_Maps",
                "NVRAM_Maps/maps",
                "NVRAM_Maps/overrides",
                "session_stats",
                "session_stats/Highlights",
                "session_stats/whitelists",
                "rom_specific_achievements",
                "custom_achievements",
                "bin",   # Ablage für DLL/Injector
                "AI",    # NEU: globaler KI-Ordner
            ]:
                ensure_dir(os.path.join(self.cfg.BASE, sub))
        except Exception:
            pass


class OverlayWindow(QWidget):
    TITLE_OFFSET_X = 0
    TITLE_OFFSET_Y = 0
    CLAMP_TITLE = True
    ROTATION_DEBOUNCE_MS = 1

    def _resolve_background_url(self, bg: str) -> str | None:
        def is_img(p: str) -> bool:
            return p.lower().endswith((".png", ".jpg", ".jpeg"))
        if isinstance(bg, str) and bg and bg.lower() != "auto":
            if os.path.isfile(bg) and is_img(bg):
                return bg.replace("\\", "/")
        for fn in ("overlay_bg.png", "overlay_bg.jpg", "overlay_bg.jpeg"):
            p = os.path.join(APP_DIR, fn)
            if os.path.isfile(p):
                return p.replace("\\", "/")
        return None

    def _show_live_unrotated(self):
        try:
            self.rotated_label.hide()
        except Exception:
            pass
        try:
            self.container.show()
            self.text_container.show()
            self.title.show()
            self.body.show()
        except Exception:
            pass

    def _icon_local(self, key: str) -> str:
        """
        Small icon helper for the overlay (Best Ball / Extra Ball).
        By default use emojis (unless OVERLAY['prefer_ascii_icons'] is True).
        """
        use_emojis = not bool(self.parent_gui.cfg.OVERLAY.get("prefer_ascii_icons", False))
        if use_emojis:
            emoji_map = {
                "best_ball": "🔥",
                "extra_ball": "➕",
            }
            return emoji_map.get(key, "•")
        else:
            ascii_map = {
                "best_ball": "[BB]",
                "extra_ball": "[EB]",
            }
            return ascii_map.get(key, "[*]")


    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self._layout_positions)
        if self.portrait_mode:
            QTimer.singleShot(0, lambda: self.request_rotation(force=True))
        else:
            QTimer.singleShot(0, self._show_live_unrotated)
            
    def _alpha_bbox(self, img: QImage, min_alpha: int = 8) -> QRect:
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return QRect(0, 0, 0, 0)
        top = None
        left = None
        right = -1
        bottom = -1
        for y in range(h):
            for x in range(w):
                if img.pixelColor(x, y).alpha() >= min_alpha:
                    if top is None:
                        top = y
                    bottom = y
                    if left is None or x < left:
                        left = x
                    if x > right:
                        right = x
        if top is None:
            return QRect(0, 0, 0, 0)
        return QRect(left, top, right - left + 1, bottom - top + 1)

    def _ref_screen_geometry(self) -> QRect:
        try:
            win = self.windowHandle()
            if win and win.screen():
                return win.screen().geometry()
            scr = QApplication.primaryScreen()
            if scr:
                return scr.geometry()
        except Exception:
            pass
        screens = QApplication.screens() or []
        return screens[0].geometry() if screens else QRect(0, 0, 1280, 720)

    def _register_raw_input(self):
        try:
            hwnd = int(self.winId())
            register_raw_input_for_window(hwnd)
        except Exception:
            pass

    def __init__(self, parent: "MainWindow"):
        super().__init__(None)
        self.parent_gui = parent
        self.setWindowTitle("Watchtower Overlay")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        ov = self.parent_gui.cfg.OVERLAY
        self.scale_pct = int(ov.get("scale_pct", 100))
        self.portrait_mode = bool(ov.get("portrait_mode", True))

        self.rotate_ccw = bool(ov.get("portrait_rotate_ccw", True))
        self.position = "center"
        self.lines_per_category = int(ov.get("lines_per_category", 5))

        self.title_color = ov.get("title_color", "#FFFFFF")
        self.highlight_color = ov.get("highlight_color", "#FFFFFF")
        self.player_colors = [
            ov.get("player1_color", "#00B050"),
            ov.get("player2_color", "#00B050"),
            ov.get("player3_color", "#00B050"),
            ov.get("player4_color", "#00B050"),
        ]
        self.font_family = ov.get("font_family", "Segoe UI")
        self._base_title_size = int(ov.get("base_title_size", 36))
        self._base_body_size = int(ov.get("base_body_size", 20))
        self._base_hint_size = int(ov.get("base_hint_size", 16))
        self._body_pt = self._base_body_size

        self._current_combined = None
        self._current_title = None
        self._rotation_pending = False

        self._apply_geometry()
        self.bg_url = self._resolve_background_url(ov.get("background", "auto"))

        self.container = QWidget(self)
        self.container.setObjectName("overlay_bg")
        self.container.setGeometry(0, 0, self.width(), self.height())
        if self.bg_url:
            css = ("QWidget#overlay_bg {"
                   f"border-image: url('{self.bg_url}') 0 0 0 0 stretch stretch;"
                   "background:#000;border:2px solid rgba(255,255,255,80);border-radius:18px;}")
        else:
            css = ("QWidget#overlay_bg {background:#000;"
                   "border:2px solid rgba(255,255,255,80);border-radius:18px;}")
        self.container.setStyleSheet(css)

        self.text_container = QWidget(self)
        self.text_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.text_container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.text_container.setGeometry(0, 0, self.width(), self.height())

        self.title = QLabel("Highlights", self.text_container)
        self.body = QLabel(self.text_container)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setWordWrap(True)
        for lab in (self.title, self.body):
            lab.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            lab.setAutoFillBackground(False)

        self.title.setStyleSheet(f"color:{self.title_color};background:transparent;")
        self.body.setStyleSheet("color:#FFFFFF;background:transparent;")

        self._apply_scale(self.scale_pct)

        self.rotated_label = QLabel(self)
        self.rotated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rotated_label.setStyleSheet("background:transparent;")
        self.rotated_label.hide()

        self._rot_in_progress = False
        self._font_update_in_progress = False

        self._layout_positions()
        QTimer.singleShot(0, self._register_raw_input)

    def request_rotation(self, force: bool = False):
        if not self.portrait_mode:
            return
        if self._rotation_pending and not force:
            return
        self._rotation_pending = True
        def _do():
            try:
                self._apply_rotation_snapshot(force=True)
            finally:
                self._rotation_pending = False
        QTimer.singleShot(self.ROTATION_DEBOUNCE_MS if not force else 0, _do)

    def _apply_geometry(self):
        ref = self._ref_screen_geometry()
        if self.portrait_mode:
            base_h = int(ref.height() * 0.55)
            base_w = int(base_h * 9 / 16)
        else:
            base_w = int(ref.width() * 0.40)
            base_h = int(ref.height() * 0.30)
        w = max(120, int(base_w * self.scale_pct / 100))
        h = max(120, int(base_h * self.scale_pct / 100))
        screens = QApplication.screens() or []
        if screens:
            vgeo = screens[0].geometry()
            for s in screens[1:]:
                vgeo = vgeo.united(s.geometry())
        else:
            vgeo = QRect(0, 0, 1280, 720)
        ov = self.parent_gui.cfg.OVERLAY
        if ov.get("use_xy", False):
            x = int(ov.get("pos_x", 0))
            y = int(ov.get("pos_y", 0))
        else:
            pad = 20
            pos = (getattr(self, "position", "center") or "center").lower()
            mapping = {
                "top-left": (vgeo.left() + pad, vgeo.top() + pad),
                "top-right": (vgeo.right() - w - pad, vgeo.top() + pad),
                "bottom-left": (vgeo.left() + pad, vgeo.bottom() - h - pad),
                "bottom-right": (vgeo.right() - w - pad, vgeo.bottom() - h - pad),
                "center-top": (vgeo.left() + (vgeo.width() - w) // 2, vgeo.top() + pad),
                "center-bottom": (vgeo.left() + (vgeo.width() - w) // 2, vgeo.bottom() - h - pad),
                "center-left": (vgeo.left() + pad, vgeo.top() + (vgeo.height() - h) // 2),
                "center-right": (vgeo.right() - w - pad, vgeo.top() + (vgeo.height() - h) // 2),
                "center": (vgeo.left() + (vgeo.width() - w) // 2, vgeo.top() + (vgeo.height() - h) // 2)
            }
            x, y = mapping.get(pos, mapping["center"])
        self.setGeometry(x, y, w, h)
        if hasattr(self, "container"):
            self.container.setGeometry(0, 0, w, h)
        if hasattr(self, "text_container"):
            self.text_container.setGeometry(0, 0, w, h)

    def _layout_positions(self):
        self._layout_positions_for(self.width(), self.height())
        if self.portrait_mode:
            self.request_rotation()

    def _layout_positions_for(self, w: int, h: int, portrait_pre_render: bool = False):
        """
        Positioniert Title und Body.
        - Title bekommt die volle Breite und AlignHCenter, damit er immer mittig sitzt.
        - Entfernt jegliche Label-Margins/Indents (kein links-/rechts-Offset).
        - Funktioniert identisch für Landscape und Portrait (inkl. Pre-Render für Rotation).
        """
        if hasattr(self, "text_container"):
            self.text_container.setGeometry(0, 0, w, h)

        pad = 24
        # Title wirklich exakt horizontal zentrieren und ohne Einrückungen rendern
        try:
            self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.title.setIndent(0)
            self.title.setMargin(0)
            self.title.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass
        self.title.adjustSize()
        t_h = self.title.sizeHint().height()

        if not self.portrait_mode:
            # Landscape: Titel über volle Breite, Body darunter
            self.title.setGeometry(0, pad, w, t_h)
            body_top = self.title.y() + t_h + 10
            body_h = h - body_top - pad
            body_w = int(w * 0.9)
            body_x = (w - body_w) // 2
            try:
                self.body.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass
            self.body.setGeometry(body_x, body_top, body_w, max(80, body_h))
            return

        # Portrait: sowohl für Pre-Render als auch Live dasselbe Layout
        self.title.setGeometry(0, pad, w, t_h)
        body_w = int(w * 0.92)
        body_x = (w - body_w) // 2
        body_top = pad + t_h + 10
        body_h = h - body_top - pad
        try:
            self.body.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass
        self.body.setGeometry(body_x, body_top, body_w, max(80, body_h))

    def _apply_scale(self, scale_pct: int):
        r = max(0.5, min(3.2, scale_pct / 100.0))
        title_pt = max(12, int(round(self._base_title_size * r)))
        body_pt = max(10, int(round(self._base_body_size * r)))
        self._body_pt = body_pt
        self.title.setFont(QFont(self.font_family, title_pt, QFont.Weight.Bold))
        self.body.setFont(QFont(self.font_family, body_pt))
        self.body.setStyleSheet(f"color:#FFFFFF;background:transparent;font-size:{body_pt}pt;font-family:'{self.font_family}';")

    def _composition_mode_source_over(self):
        try:
            return QPainter.CompositionMode.CompositionMode_SourceOver  # PyQt6
        except Exception:
            try:
                return getattr(QPainter, "CompositionMode_SourceOver")  # Fallback
            except Exception:
                return None

    def _apply_rotation_snapshot(self, force: bool = False):
        """
        Portrait-Render: Pre-Layout im gedrehten Koordinatensystem, danach 90° rotieren.
        WICHTIG: Kein Alpha-Crop mehr – das verursachte bei Anti-Aliasing feine Offsets.
        Stattdessen das rotierte Bild als Ganzes zentriert (normalerweise exakt W×H) zeichnen.
        """
        if not self.portrait_mode:
            self.rotated_label.hide()
            self.container.show()
            self.text_container.show()
            self.title.show()
            self.body.show()
            return

        if getattr(self, "_rot_in_progress", False):
            return
        self._rot_in_progress = True
        try:
            W, H = self.width(), self.height()
            if W <= 0 or H <= 0:
                return

            angle = -90 if getattr(self, "rotate_ccw", True) else 90

            # Hintergrund (rotiert und skaliert auf Fenstergröße)
            if self.bg_url and os.path.isfile(self.bg_url):
                pm = QPixmap(self.bg_url)
                if not pm.isNull():
                    rot_pm = pm.transformed(QTransform().rotate(angle), Qt.TransformationMode.SmoothTransformation)
                    scaled = rot_pm.scaled(W, H, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                           Qt.TransformationMode.SmoothTransformation)
                    sw, sh = scaled.width(), scaled.height()
                    cx = max(0, (sw - W) // 2)
                    cy = max(0, (sh - H) // 2)
                    bg_img = scaled.copy(cx, cy, min(W, sw - cx), min(H, sh - cy)).toImage().convertToFormat(
                        QImage.Format.Format_ARGB32_Premultiplied)
                else:
                    bg_img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied); bg_img.fill(Qt.GlobalColor.black)
            else:
                bg_img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied); bg_img.fill(Qt.GlobalColor.black)

            # Pre-Layout: Wir layouten in (pre_w, pre_h) = (H, W)
            pre_w, pre_h = H, W
            old_geom = self.text_container.geometry()
            old_title_vis = self.title.isVisible()
            old_body_vis = self.body.isVisible()

            self.text_container.setGeometry(0, 0, pre_w, pre_h)
            self.title.setVisible(True)
            self.body.setVisible(True)

            # Normales Layout (Titel oben, Body darunter) – Titel ist auf volle Breite und AlignHCenter gesetzt
            self._layout_positions_for(pre_w, pre_h, portrait_pre_render=False)
            QApplication.processEvents()

            # Container (mit Title+Body) in Bild rendern
            content_pre = QImage(pre_w, pre_h, QImage.Format.Format_ARGB32_Premultiplied)
            content_pre.fill(Qt.GlobalColor.transparent)
            p_all = QPainter(content_pre)
            try:
                self.text_container.render(p_all)
            finally:
                p_all.end()

            # Rotieren (ergibt für 90° normalerweise Größe W×H)
            content_rot = content_pre.transformed(QTransform().rotate(angle), Qt.TransformationMode.SmoothTransformation)

            # Widgets verbergen – wir zeigen den flachen Snapshot
            self.container.hide()
            self.text_container.hide()

            # Finalbild zusammensetzen – KEIN Alpha-BBox-Crop mehr
            final_img = QImage(bg_img)
            p_final = QPainter(final_img)
            try:
                mode = self._composition_mode_source_over()
                if mode is not None:
                    p_final.setCompositionMode(mode)
                dx = (W - content_rot.width()) // 2
                dy = (H - content_rot.height()) // 2
                p_final.drawImage(dx, dy, content_rot)

                # Rahmen
                pen = QPen(QColor(255, 255, 255, 80))
                pen.setWidth(2)
                p_final.setPen(pen)
                p_final.setBrush(Qt.BrushStyle.NoBrush)
                p_final.drawRoundedRect(1, 1, W - 2, H - 2, 18, 18)
            finally:
                p_final.end()

            # Snapshot anzeigen
            self.text_container.setGeometry(old_geom)
            self.title.setVisible(old_title_vis)
            self.body.setVisible(old_body_vis)

            self.rotated_label.setGeometry(0, 0, W, H)
            self.rotated_label.setPixmap(QPixmap.fromImage(final_img))
            self.rotated_label.show()
            self.rotated_label.raise_()
        except Exception as e:
            print("[overlay] portrait render failed:", e)
            self.rotated_label.hide()
            self.container.show()
            self.text_container.show()
        finally:
            self._rot_in_progress = False

    def apply_colors_from_cfg(self, ov: dict):
        self.title_color = ov.get("title_color", self.title_color)
        self.highlight_color = ov.get("highlight_color", self.highlight_color)
        self.player_colors = [
            ov.get("player1_color", self.player_colors[0]),
            ov.get("player2_color", self.player_colors[1]),
            ov.get("player3_color", self.player_colors[2]),
            ov.get("player4_color", self.player_colors[3]),
        ]
        body_pt = self._body_pt
        self.title.setStyleSheet(f"color:{self.title_color};background:transparent;")
        self.body.setStyleSheet(f"color:#FFFFFF;background:transparent;font-size:{body_pt}pt;font-family:'{self.font_family}';")
        if self._current_combined:
            self._render_fixed_columns()
        else:
            self._layout_positions()
            self.request_rotation()

    def apply_font_from_cfg(self, ov: dict):
        if getattr(self, "_font_update_in_progress", False):
            return
        self._font_update_in_progress = True
        try:
            self.font_family = ov.get("font_family", self.font_family)
            self._base_body_size = int(ov.get("base_body_size", self._base_body_size))
            self._base_title_size = int(ov.get("base_title_size", self._base_title_size))
            self._base_hint_size = int(ov.get("base_hint_size", self._base_hint_size))
            self._apply_scale(self.scale_pct)
            def _finish():
                try:
                    if self._current_combined:
                        self._render_fixed_columns()
                    else:
                        self._layout_positions()
                        self.request_rotation(force=True)
                finally:
                    self._font_update_in_progress = False
            QTimer.singleShot(0, _finish)
        except Exception:
            self._font_update_in_progress = False

    def apply_portrait_from_cfg(self, ov: dict):
        self.portrait_mode = bool(ov.get("portrait_mode", self.portrait_mode))
        self.rotate_ccw = bool(ov.get("portrait_rotate_ccw", self.rotate_ccw))
        self._apply_geometry()
        self._layout_positions()
        if self.portrait_mode:
            self.request_rotation(force=True)
        else:
            self._show_live_unrotated()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.container.setGeometry(0, 0, self.width(), self.height())
        self._apply_geometry()
        self._layout_positions()
        if self.portrait_mode:
            self.request_rotation()
        else:
            self._show_live_unrotated()


    def set_placeholder(self, session_title: Optional[str] = None):
        self._current_combined = None
        self._current_title = session_title or "Highlights"
        self.title.setText(self._current_title)
        self.body.setText("<div>Loading highlights …</div>")
        self._layout_positions()
        self.request_rotation(force=True)



    def set_html(self, html: str, session_title: Optional[str] = None):
        """
        Render an arbitrary HTML page inside the overlay (portrait aware).
        If session_title == "", the title line is intentionally left empty.
        If session_title is None, fallback to 'Highlights'.
        """
        self._current_combined = None
        self._current_title = "Highlights" if session_title is None else session_title
        self.title.setText(self._current_title)
        body_pt = getattr(self, "_body_pt", 20)
        css = f"font-size:{body_pt}pt;font-family:'{self.font_family}';color:#FFFFFF;"
        self.body.setText(f"<div style='{css}'>{html}</div>")
        self._layout_positions()
        self.request_rotation(force=True)

    def set_combined(self, combined: dict, session_title: Optional[str] = None):
        """
        Combined multi-column highlights. If session_title == "", no title is shown.
        If session_title is None, fallback to 'Highlights'.
        """
        self._current_combined = combined or {}
        self._current_title = "Highlights" if session_title is None else session_title
        self._render_fixed_columns()

    def _render_fixed_columns(self):
        """
        Render columns for present entries (Global/Players/CPU) in the exact order of combined['players'].
        Each entry may provide an optional 'title' to override the default header.
        Zusätzlich: dedupliziere pro Spieler-ID zur Sicherheit.
        """
        self.title.setText(self._current_title or "Highlights")
        combined = self._current_combined or {}
        players_raw = combined.get("players", []) or []

        # Dedupe by id (first occurrence wins)
        seen_ids = set()
        players = []
        for e in players_raw:
            try:
                pid = int(e.get("id", e.get("player", 0)) or 0)
            except Exception:
                pid = 0
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            players.append(e)

        def esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def block(entry: dict):
            pid = 0
            try:
                pid = int(entry.get("id", entry.get("player", 0)) or 0)
            except Exception:
                pid = 0

            # Colors: P1..P4 from cfg, others neutral gray
            color = self.player_colors[pid - 1] if 1 <= pid <= 4 else "#7A7A7A"
            hld = (entry.get("highlights") or {})
            try:
                score_abs = int(entry.get("score", 0) or 0)
            except Exception:
                score_abs = 0

            # Header: optional override
            title = entry.get("title")
            if not title:
                if pid == 5:
                    title = "CPU"
                elif pid == 0:
                    title = "Global"
                else:
                    title = f"Player {pid}"

            # Header
            lines = []
            lines.append(f"<div style='font-weight:700;color:{color};margin-bottom:4px;'>{esc(title)}</div>")

            # Score total
            sc_txt = f"{score_abs:,d}".replace(",", ".")
            lines.append("<div style='font-weight:600;color:#FFFFFF;margin:6px 0 2px 0;'>Score</div>")
            lines.append(f"<div style='color:{self.highlight_color};'>{sc_txt}</div>")

            # Highlights
            for cat in ["Power", "Precision", "Fun"]:
                arr = hld.get(cat, [])
                if arr:
                    lines.append(f"<div style='font-weight:600;color:#FFFFFF;margin:6px 0 2px 0;'>{esc(cat)}</div>")
                    lines += [f"<div style='color:{self.highlight_color};'>{esc(x)}</div>" for x in arr[:5]]

            if len(lines) == 2:  # nur Header + Score
                lines.append("<div style='color:#888;'>–</div>")
            return "".join(lines)

        if not players:
            self.body.setText("<div>-</div>")
            self._layout_positions()
            self.request_rotation(force=True)
            return

        html = (
            "<table width='100%'><tr>" +
            "".join(f"<td valign='top' style='padding:0 14px;'>{block(p)}</td>" for p in players) +
            "</tr></table>"
        )
        body_pt = self._body_pt
        css = f"font-size:{body_pt}pt;font-family:'{self.font_family}';color:#FFFFFF;"
        self.body.setText(f"<div style='{css}'>{html}</div>")
        self._layout_positions()
        self.request_rotation(force=True)


class MiniInfoOverlay(QWidget):
    """
    Small, always-on-top, non-interactive overlay with dynamic size and
    a countdown that auto-closes after N seconds.
    Always centers itself on the primary monitor.
    Portrait-aware: when portrait_mode is enabled, the content is rotated
    by 90° (CCW or CW) to match the main overlay orientation.
    """
    def __init__(self, parent: "MainWindow"):
        super().__init__(None)
        self.parent_gui = parent
        self.setWindowTitle("Info")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Visual config
        ov = self.parent_gui.cfg.OVERLAY or {}
        base_pt = int(ov.get("base_body_size", 20))
        self._body_pt = max(12, base_pt + 3)           # etwas größer
        self._font_family = ov.get("font_family", "Segoe UI")
        self._red = "#FF3B30"                          # gut lesbares Rot
        self._hint = "#DDDDDD"                         # grauer Countdown
        self._bg_color = QColor(0, 0, 0, 190)
        self._radius = 16
        self._pad_w = 28
        self._pad_h = 22
        self._max_text_width = 520

        # Rotation flags (pro Render frisch aus cfg gelesen)
        self._portrait_mode = bool(ov.get("portrait_mode", True))
        self._rotate_ccw = bool(ov.get("portrait_rotate_ccw", True))

        # Laufzeit / Countdown
        self._remaining = 0
        self._base_msg = ""
        self._last_center = (960, 540)

        # Anzeige: Snapshot-Label (rotiert/unrotiert), kein interaktives Widget nötig
        self._snap_label = QLabel(self)
        self._snap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._snap_label.setStyleSheet("background:transparent;")

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

        self.hide()

    def _primary_center(self) -> tuple[int, int]:
        try:
            scr = QApplication.primaryScreen()
            geo = scr.geometry() if scr else QRect(0, 0, 1280, 720)
            return geo.left() + geo.width() // 2, geo.top() + geo.height() // 2
        except Exception:
            return 640, 360

    def _compose_html(self) -> str:
        return (
            f"<span style='color:{self._red};'>{self._base_msg}</span>"
            f"<br><span style='color:{self._hint};'>closing in {self._remaining}…</span>"
        )

    def _render_message_image(self, html: str) -> QImage:
        # Offscreen-Label erzeugen, um HTML maßzuschneidern
        tmp = QLabel()
        tmp.setTextFormat(Qt.TextFormat.RichText)
        tmp.setStyleSheet(f"color:{self._red};background:transparent;")
        tmp.setFont(QFont(self._font_family, self._body_pt))
        tmp.setWordWrap(True)
        tmp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tmp.setText(html)

        tmp.setFixedWidth(self._max_text_width)
        tmp.adjustSize()
        text_w = tmp.width()
        text_h = tmp.sizeHint().height()

        W = max(200, text_w + self._pad_w)
        H = max(60,  text_h + self._pad_h)

        img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        try:
            p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing, True)
            # Hintergrund
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._bg_color)
            p.drawRoundedRect(0, 0, W, H, self._radius, self._radius)
            # Text mittig
            margin_left = (W - text_w) // 2
            margin_top = (H - text_h) // 2
            tmp.render(p, QPoint(margin_left, margin_top))
        finally:
            p.end()
        return img

    def _refresh_view(self):
        # Portrait-/Drehrichtung frisch lesen
        ov = self.parent_gui.cfg.OVERLAY or {}
        self._portrait_mode = bool(ov.get("portrait_mode", True))
        self._rotate_ccw = bool(ov.get("portrait_rotate_ccw", True))

        html = self._compose_html()
        img = self._render_message_image(html)

        if self._portrait_mode:
            angle = -90 if self._rotate_ccw else 90
            img = img.transformed(QTransform().rotate(angle), Qt.TransformationMode.SmoothTransformation)

        W, H = img.width(), img.height()
        cx, cy = self._last_center
        x = int(cx - W // 2)
        y = int(cy - H // 2)

        self.setGeometry(x, y, W, H)
        self._snap_label.setGeometry(0, 0, W, H)
        self._snap_label.setPixmap(QPixmap.fromImage(img))
        self.show()
        self.raise_()

    def _on_tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            self._timer.stop()
            self.hide()
            return
        self._refresh_view()

    def show_info(self, message: str, seconds: int = 5, center: tuple[int, int] | None = None, color_hex: str | None = None):
        """
        Show message with a countdown (‘closing in X…’) and auto-hide.
        Always centers on the primary monitor. Optional text color override (hex).
        """
        self._base_msg = str(message or "").strip()
        self._remaining = max(1, int(seconds))
        # Update color if provided
        if color_hex:
            try:
                self._red = color_hex
            except Exception:
                pass
        # Always use primary center (ignores 'center' arg by design)
        self._last_center = self._primary_center()
        self._timer.stop()
        self._refresh_view()
        self._timer.start()



def read_active_players(base_dir: str):
    """
    Read all activePlayers JSONs, dedupe by player id (keep newest), sorted by id.
    UPDATED: strip achievements from returned payloads (GUI/Overlay must not see achievements).
    """
    ap_dir = os.path.join(base_dir, "session_stats", "Highlights", "activePlayers")
    if not os.path.isdir(ap_dir):
        return []

    candidates = []
    try:
        for fn in os.listdir(ap_dir):
            if not fn.lower().endswith(".json"):
                continue
            fpath = os.path.join(ap_dir, fn)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pid = int(data.get("player", data.get("id", 0)) or 0)
                if pid <= 0:
                    m = re.search(r"_P(\d+)\.json$", fn, re.IGNORECASE)
                    if m:
                        pid = int(m.group(1))
                st = os.stat(fpath)
                candidates.append((
                    pid,
                    float(getattr(st, "st_mtime", 0.0)),
                    {
                        "id": pid,
                        "highlights": data.get("highlights", {}),
                        "playtime_sec": data.get("playtime_sec", 0),
                        "score": int(data.get("score", 0) or 0),
                    }
                ))
            except Exception:
                continue
    except Exception:
        return []

    best_by_id: dict[int, tuple[float, dict]] = {}
    for pid, mtime, payload in candidates:
        if pid <= 0:
            continue
        old = best_by_id.get(pid)
        if (old is None) or (mtime > old[0]):
            best_by_id[pid] = (mtime, payload)

    result = [payload for _mt, payload in sorted(best_by_id.values(), key=lambda t: t[1].get("id", 0))]
    return result

class AchToastWindow(QWidget):
    """
    One-shot Steam-like achievement toast window:
      - Renders at bottom-right of the primary monitor
      - Auto-hides after N seconds
      - Portrait-aware: rotates content 90° CCW (or CW) based on config, while keeping bottom-right placement
    """
    finished = pyqtSignal()

    def __init__(self, parent: "MainWindow", title: str, rom: str, seconds: int = 5):
        super().__init__(None)
        self.parent_gui = parent
        self._title = str(title or "").strip()
        self._rom = str(rom or "").strip()
        self._seconds = max(1, int(seconds))

        self.setWindowTitle("Achievement")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._label = QLabel(self)
        self._label.show()  # <— ohne das wird nichts sichtbar

        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background:transparent;")

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._remaining = self._seconds

        self._render_and_place()
        self._timer.start()
        self.show()
        self.raise_()


    # In class AchToastWindow (einfügen)
    def _primary_geometry(self) -> QRect:
        try:
            scr = QApplication.primaryScreen()
            return scr.geometry() if scr else QRect(0, 0, 1280, 720)
        except Exception:
            return QRect(0, 0, 1280, 720)

    def _render_and_place(self):
        """
        Bild rendern und das Toast-Fenster unten rechts (Primary) platzieren.
        Rotation ist bereits in _compose_image() berücksichtigt.
        """
        try:
            img = self._compose_image()
            geo = self._primary_geometry()
            pad = 20
            W, H = img.width(), img.height()
            x = int(geo.right() - W - pad)
            y = int(geo.bottom() - H - pad)
            self.setGeometry(x, y, W, H)
            self._label.setGeometry(0, 0, W, H)
            self._label.setPixmap(QPixmap.fromImage(img))
        except Exception:
            pass

    def _icon_pixmap(self, size: int = 40) -> QPixmap:
        try:
            ic = self.parent_gui._get_icon()
            pm = ic.pixmap(size, size)
            if not pm.isNull():
                return pm
        except Exception:
            pass
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        try:
            p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing, True)
            p.setBrush(QColor(32, 32, 32, 240))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(0, 0, size, size, 6, 6)
            p.setPen(QColor("#FFFFFF"))
            f = QFont(self.parent_gui.cfg.OVERLAY.get("font_family", "Segoe UI"), max(8, size//3), QFont.Weight.Bold)
            p.setFont(f)
            p.drawText(pm.rect(), int(Qt.AlignmentFlag.AlignCenter), "AW")
        finally:
            p.end()
        return pm

    def _compose_image(self) -> QImage:
        ov = self.parent_gui.cfg.OVERLAY or {}
        family = ov.get("font_family", "Segoe UI")
        base_pt = int(ov.get("base_body_size", 20))
        f_small = QFont(family, max(10, base_pt - 4), QFont.Weight.Normal)
        f_title = QFont(family, max(12, base_pt + 2), QFont.Weight.Bold)
        f_rom = QFont(family, max(9, base_pt - 6), QFont.Weight.Normal)

        label1 = "Achievement unlocked"
        pad = 14
        gap = 10
        icon_sz = 44
        max_text_w = 360

        fm_small = QFontMetrics(f_small)
        fm_title = QFontMetrics(f_title)
        fm_rom = QFontMetrics(f_rom)

        flags = Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft
        rect_small = fm_small.boundingRect(0, 0, max_text_w, 1000, int(flags), label1)
        rect_title = fm_title.boundingRect(0, 0, max_text_w, 1000, int(flags), self._title)
        rom_text = self._rom if self._rom else ""
        rect_rom = fm_rom.boundingRect(0, 0, max_text_w, 1000, int(flags), rom_text) if rom_text else QRect(0, 0, 0, 0)

        text_w = max(rect_small.width(), rect_title.width(), rect_rom.width())
        text_h = rect_small.height() + 4 + rect_title.height() + (4 + rect_rom.height() if rom_text else 0)

        W = pad + icon_sz + gap + text_w + pad
        H = pad + max(icon_sz, text_h) + pad

        img = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)

        p = QPainter(img)
        try:
            p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing, True)

            # Background
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 210))
            p.drawRoundedRect(0, 0, W, H, 14, 14)

            # Icon
            pm = self._icon_pixmap(icon_sz)
            p.drawPixmap(pad, (H - icon_sz)//2, pm)

            # Texts
            x_text = pad + icon_sz + gap
            y_top = pad + max(0, (icon_sz - text_h)//2)

            p.setPen(QColor("#BBBBBB"))
            p.setFont(f_small)
            p.drawText(QRect(x_text, y_top, text_w, rect_small.height()), int(flags), label1)

            y = y_top + rect_small.height() + 4
            p.setPen(QColor("#FFFFFF"))
            p.setFont(f_title)
            p.drawText(QRect(x_text, y, text_w, rect_title.height()), int(flags), self._title)

            if rom_text:
                y += rect_title.height() + 4
                p.setPen(QColor("#CCCCCC"))
                p.setFont(f_rom)
                p.drawText(QRect(x_text, y, text_w, rect_rom.height()), int(flags), rom_text)
        finally:
            p.end()

        # Rotate if portrait
        portrait = bool(ov.get("portrait_mode", True))
        if portrait:
            ccw = bool(ov.get("portrait_rotate_ccw", True))
            angle = -90 if ccw else 90
            img = img.transformed(QTransform().rotate(angle), Qt.TransformationMode.SmoothTransformation)

        return img



    def _tick(self):
        self._remaining -= 1
        if self._remaining <= 0:
            try:
                self.finished.emit()
            except Exception:
                pass
            self.close()

    def closeEvent(self, _evt):
        try:
            self._timer.stop()
        except Exception:
            pass
        try:
            self.finished.emit()
        except Exception:
            pass
        super().closeEvent(_evt)



class AchToastManager(QObject):
    """
    Sequential toast manager:
      - enqueue(title, rom, seconds): queues a toast request
      - shows one AchToastWindow at a time
      - each toast uses its own window instance (as requested)
    """
    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.parent_gui = parent
        self._queue: list[tuple[str, str, int]] = []
        self._active = False

    def enqueue(self, title: str, rom: str, seconds: int = 5):
        self._queue.append((str(title or "").strip(), str(rom or "").strip(), max(1, int(seconds))))
        if not self._active:
            self._show_next()

    def _show_next(self):
        if not self._queue:
            self._active = False
            return
        self._active = True
        t, r, s = self._queue.pop(0)
        win = AchToastWindow(self.parent_gui, t, r, s)
        # When it finishes, chain the next
        win.finished.connect(self._on_finished)

    def _on_finished(self):
        # Small delay to avoid overlap flicker
        QTimer.singleShot(80, self._show_next)


class ChallengeCountdownOverlay(QWidget):
    def __init__(self, parent, total_seconds: int = 300):
        super().__init__(parent)
        self.parent_gui = parent
        self._left = max(1, int(total_seconds))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.resize(400, 120)
        self.show()

        # Echte Topmost-Priorität über DX/Borderless
        try:
            import win32gui, win32con
            hwnd = int(self.winId())
            win32gui.SetWindowPos(
                hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            )
        except Exception:
            pass

        self._render_and_place()

    def _tick(self):
        self._left -= 1
        if self._left <= 0:
            self._left = 0
            try:
                self._timer.stop()
                self._render_and_place()  # 00:00 sichtbar
            except Exception:
                pass
            QTimer.singleShot(3000, self._kill_vpx)  # +3s Puffer
            return
        self._render_and_place()


    # In class ChallengeCountdownOverlay: _kill_vpx ersetzen
    def _kill_vpx(self):
        """
        Beende NUR den Visual Pinball Player via ALT+F4; Fallback WM_CLOSE. Kein taskkill.
        """
        try:
            w = getattr(self.parent_gui, "watcher", None)
            if w and w._alt_f4_visual_pinball_player(wait_ms=3000):
                self.close()
                return
        except Exception:
            pass

        # Fallback: WM_CLOSE
        try:
            import win32gui, win32con
            def _cb(hwnd, _):
                try:
                    if not win32gui.IsWindowVisible(hwnd):
                        return True
                    title = (win32gui.GetWindowText(hwnd) or "").strip()
                    if title.startswith("Visual Pinball Player"):
                        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                except Exception:
                    pass
                return True
            win32gui.EnumWindows(_cb, None)
        except Exception:
            pass

        self.close()

    def _render_and_place(self):
        img = self._compose_image()
        if img is None:
            return
        W, H = img.width(), img.height()
        self.setFixedSize(W, H)

        # Immer unten-links (Portrait schon eingerechnet, weil img bereits rotiert ist)
        scr = QApplication.primaryScreen()
        geo = scr.geometry() if scr else QRect(0, 0, 1280, 720)
        pad = 40
        x = int(geo.left() + pad)
        y = int(geo.bottom() - H - pad)
        self.move(x, y)

        self._pix = QPixmap.fromImage(img)
        self.update()

    def _compose_image(self):
        w, h = 400, 120
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)

        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setPen(Qt.GlobalColor.white)

        # Hintergrund
        p.fillRect(0, 0, w, h, QColor(0, 0, 0, 180))

        # Zeit
        mins, secs = divmod(self._left, 60)
        txt = f"{mins:02d}:{secs:02d}"
        font = QFont("Segoe UI", 48, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, txt)
        p.end()

        # Portrait-Rotation per Config
        try:
            ov = self.parent_gui.cfg.OVERLAY or {}
            if ov.get("portrait_mode", False):
                angle = -90 if ov.get("portrait_rotate_ccw", True) else 90
                img = img.transformed(QTransform().rotate(angle), Qt.TransformationMode.SmoothTransformation)
        except Exception:
            pass

        return img

    def paintEvent(self, _evt):
        if hasattr(self, "_pix"):
            p = QPainter(self)
            p.drawPixmap(0, 0, self._pix)
            p.end()



# ---------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, watcher: Watcher, bridge: Bridge):
        super().__init__()
        self.cfg = cfg
        self.watcher = watcher
        self.bridge = bridge
        self.setWindowTitle("Achievement Watcher")
        self.resize(1100, 720)
        icon = self._get_icon()
        self.setWindowIcon(icon)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        # NEW: keep a handle to the main tab widget for Achievements tab builder
        self.main_tabs = tabs

        # Status tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        self.status_label = QLabel("Watcher: running")
        self.status_label.setStyleSheet("font: bold 14px 'Segoe UI'; color:#107c10;")
        row1 = QHBoxLayout()
        self.btn_restart = QPushButton("Restart Watcher")
        self.btn_restart.clicked.connect(self._restart_watcher)
        self.btn_quit = QPushButton("Quit GUI")
        self.btn_quit.clicked.connect(self.quit_all)
        row1.addWidget(self.btn_restart); row1.addWidget(self.btn_quit)
        status_layout.addWidget(self.status_label); status_layout.addLayout(row1)
        row2 = QHBoxLayout()
        self.btn_minimize = QPushButton("Minimize to tray")
        self.btn_minimize.clicked.connect(self.hide)
        row2.addWidget(self.btn_minimize); row2.addStretch(1)
        status_layout.addLayout(row2)
        tabs.addTab(status_tab, "Status")

        # Overlay tab (original UI restored)
        overlay_tab = QWidget(); ov_layout = QVBoxLayout(overlay_tab)
        self.bridge.overlay_trigger.connect(self._on_overlay_trigger)
        self.bridge.overlay_show.connect(self._show_overlay_latest)
        self.bridge.mini_info_show.connect(self._on_mini_info_show)

        self.bridge.ach_toast_show.connect(self._on_ach_toast_show)
        self._ach_toast_mgr = AchToastManager(self)
        
        
        
        # AI Evaluation tab (English)
        ai_tab = QWidget()
        ai_layout = QVBoxLayout(ai_tab)
        self.ai_view = QTextBrowser()
        self.ai_view.setToolTip("Shows AI outputs: Coach tips and Skill profile.")
        ai_layout.addWidget(self.ai_view)

        # Refresh-Button
        row_ai = QHBoxLayout()
        self.btn_ai_refresh = QPushButton("Refresh now")
        self.btn_ai_refresh.setToolTip("Reload AI Coach tips and Skill Profile from disk.")
        self.btn_ai_refresh.clicked.connect(self.update_ai_evaluation_tab)
        row_ai.addWidget(self.btn_ai_refresh)
        row_ai.addStretch(1)
        ai_layout.addLayout(row_ai)

        tabs.addTab(ai_tab, "AI Evaluation")

        # Initial fill
        try:
            self.update_ai_evaluation_tab()
        except Exception:
            pass

        # --- Portrait / Rotation ---
        row_portrait = QHBoxLayout()
        self.chk_portrait = QCheckBox("Portrait mode (rotate 90°)")
        self.chk_portrait.setChecked(bool(self.cfg.OVERLAY.get("portrait_mode", True)))
        self.chk_portrait.stateChanged.connect(self._on_portrait_toggle)
        # NEW: CCW/CW toggle
        self.chk_portrait_ccw = QCheckBox("Rotate CCW")
        self.chk_portrait_ccw.setChecked(bool(self.cfg.OVERLAY.get("portrait_rotate_ccw", True)))
        self.chk_portrait_ccw.stateChanged.connect(self._on_portrait_ccw_toggle)
        row_portrait.addWidget(self.chk_portrait)
        row_portrait.addWidget(self.chk_portrait_ccw)
        row_portrait.addStretch(1)

        # --- Scale ---
        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("Overlay size:"))
        self.sld_scale = QSlider(Qt.Orientation.Horizontal)
        self.sld_scale.setMinimum(30); self.sld_scale.setMaximum(300)
        self.sld_scale.setValue(int(self.cfg.OVERLAY.get("scale_pct", 100)))
        self.sld_scale.valueChanged.connect(self._on_overlay_scale)
        row_scale.addWidget(self.sld_scale)
        self.lbl_scale = QLabel(f"{self.sld_scale.value()}%")
        row_scale.addWidget(self.lbl_scale)

        # --- XY ---
        row_xy = QHBoxLayout()
        self.chk_use_xy = QCheckBox("Use X/Y")
        self.chk_use_xy.setChecked(bool(self.cfg.OVERLAY.get("use_xy", False)))
        self.chk_use_xy.stateChanged.connect(self._on_use_xy_changed)
        row_xy.addWidget(self.chk_use_xy); row_xy.addSpacing(12)
        self.spn_x = QSpinBox(); self.spn_x.setRange(-100000, 100000)
        self.spn_x.setValue(int(self.cfg.OVERLAY.get("pos_x", 100))); self.spn_x.valueChanged.connect(self._on_xy_changed)
        row_xy.addWidget(QLabel("X:")); row_xy.addWidget(self.spn_x); row_xy.addSpacing(12)
        self.spn_y = QSpinBox(); self.spn_y.setRange(-100000, 100000)
        self.spn_y.setValue(int(self.cfg.OVERLAY.get("pos_y", 100))); self.spn_y.valueChanged.connect(self._on_xy_changed)
        row_xy.addWidget(QLabel("Y:")); row_xy.addWidget(self.spn_y); row_xy.addStretch(1)

        # --- Auto-show overlay after VPX closes ---
        row_auto_show = QHBoxLayout()
        self.chk_auto_show = QCheckBox("Auto-show overlay after VPX closes")
        self.chk_auto_show.setChecked(bool(self.cfg.OVERLAY.get("auto_show_on_end", True)))
        self.chk_auto_show.stateChanged.connect(self._on_auto_show_toggle)
        row_auto_show.addWidget(self.chk_auto_show)
        row_auto_show.addStretch(1)

        # Add rows to overlay layout (VR row removed)
        for lay in (row_portrait, row_scale, row_xy, row_auto_show):
            ov_layout.addLayout(lay)

        grp_toggle = QGroupBox("Toggle overlay (keyboard or joystick)")
        gl_toggle = QVBoxLayout(grp_toggle)
        row_t1 = QHBoxLayout()
        row_t1.addWidget(QLabel("Source:"))
        self.cmb_toggle_src = QComboBox()
        self.cmb_toggle_src.addItems(["keyboard", "joystick"])
        self.cmb_toggle_src.setCurrentText(self.cfg.OVERLAY.get("toggle_input_source", "keyboard"))
        self.cmb_toggle_src.currentTextChanged.connect(self._on_toggle_source_changed)
        row_t1.addWidget(self.cmb_toggle_src); row_t1.addStretch(1)
        row_t2 = QHBoxLayout()
        self.btn_bind_toggle = QPushButton("Bind toggle…")
        self.btn_bind_toggle.clicked.connect(self._on_bind_toggle_clicked)
        self.lbl_toggle_binding = QLabel(self._toggle_binding_label_text())
        row_t2.addWidget(self.btn_bind_toggle); row_t2.addWidget(self.lbl_toggle_binding); row_t2.addStretch(1)
        gl_toggle.addLayout(row_t1); gl_toggle.addLayout(row_t2)

        grp_colors = QGroupBox("Overlay colors"); gl_colors = QVBoxLayout(grp_colors)
        row_col1 = QHBoxLayout()
        row_col1.addWidget(QLabel("Title:"))
        self.btn_col_title = QPushButton("Pick")
        self.lbl_col_title = QLabel(self.cfg.OVERLAY.get("title_color", "#FFFFFF"))
        self.btn_col_title.clicked.connect(lambda: self._pick_color("title_color", self.lbl_col_title))
        row_col1.addWidget(self.btn_col_title); row_col1.addWidget(self.lbl_col_title)
        row_col1.addSpacing(12); row_col1.addWidget(QLabel("Highlights:"))
        self.btn_col_high = QPushButton("Pick")
        self.lbl_col_high = QLabel(self.cfg.OVERLAY.get("highlight_color", "#FFFFFF"))
        self.btn_col_high.clicked.connect(lambda: self._pick_color("highlight_color", self.lbl_col_high))
        row_col1.addWidget(self.btn_col_high); row_col1.addWidget(self.lbl_col_high); row_col1.addStretch(1)

        row_col2 = QHBoxLayout()
        for pid, key in enumerate(["player1_color", "player2_color", "player3_color", "player4_color"], start=1):
            row_col2.addWidget(QLabel(f"Player {pid}:"))
            btn = QPushButton("Pick")
            lbl = QLabel(self.cfg.OVERLAY.get(key, "#00B050"))
            btn.clicked.connect(lambda _, k=key, l=lbl: self._pick_color(k, l))
            row_col2.addWidget(btn); row_col2.addWidget(lbl); row_col2.addSpacing(10)
            setattr(self, f"lbl_{key}", lbl)
        row_col2.addStretch(1)
        gl_colors.addLayout(row_col1); gl_colors.addLayout(row_col2)

        grp_font = QGroupBox("Overlay font"); gl_font = QVBoxLayout(grp_font)
        row_font1 = QHBoxLayout()
        row_font1.addWidget(QLabel("Font family:"))
        self.cmb_font_family = QFontComboBox()
        self.cmb_font_family.setCurrentFont(QFont(self.cfg.OVERLAY.get("font_family", "Segoe UI")))
        self.cmb_font_family.currentFontChanged.connect(self._on_font_family_changed)
        row_font1.addWidget(self.cmb_font_family); row_font1.addStretch(1)
        row_font2 = QHBoxLayout()
        row_font2.addWidget(QLabel("Body font size:"))
        self.spn_font_size = QSpinBox()
        self.spn_font_size.setRange(8, 64)
        self.spn_font_size.setValue(int(self.cfg.OVERLAY.get("base_body_size", 20)))
        self.spn_font_size.valueChanged.connect(self._on_font_size_changed)
        row_font2.addWidget(self.spn_font_size); row_font2.addStretch(1)
        gl_font.addLayout(row_font1); gl_font.addLayout(row_font2)

        row_test = QHBoxLayout()
        self.btn_toggle_now = QPushButton("Toggle overlay now (test)")
        self.btn_toggle_now.clicked.connect(self._toggle_overlay)
        self.btn_hide = QPushButton("Hide overlay")
        self.btn_hide.clicked.connect(self._hide_overlay)
        row_test.addWidget(self.btn_toggle_now); row_test.addWidget(self.btn_hide); row_test.addStretch(1)

        info = QLabel("Optional background: place overlay_bg.png/jpg next to the executable.")
        info.setStyleSheet("color:#555;")

        ov_layout.addWidget(grp_toggle)
        ov_layout.addWidget(grp_colors)
        ov_layout.addWidget(grp_font)
        ov_layout.addLayout(row_test)
        ov_layout.addWidget(info)
        overlay_tab.setLayout(ov_layout)
        tabs.addTab(overlay_tab, "Overlay")

        # Logs tab
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        tabs.addTab(log_tab, "Logs")

        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        self._build_challenges_tab()

        # Paths (promoted to attributes so we can tool-tip them)
        self.base_label = QLabel(f"BASE: {self.cfg.BASE}")
        self.btn_base = QPushButton("Change BASE")
        self.btn_base.clicked.connect(self.change_base)
        self.btn_base.setToolTip("Change the base directory for achievements data.")

        self.nvram_label = QLabel(f"NVRAM: {self.cfg.NVRAM_DIR}")
        self.btn_nvram = QPushButton("Change NVRAM")
        self.btn_nvram.clicked.connect(self.change_nvram)
        self.btn_nvram.setToolTip("Change the VPinMAME NVRAM directory.")

        self.tables_label = QLabel(f"TABLES (optional): {self.cfg.TABLES_DIR}")
        self.btn_tables = QPushButton("Change TABLES (optional)")
        self.btn_tables.clicked.connect(self.change_tables)
        self.btn_tables.setToolTip("Change the Tables directory (optional).")

        # Repair / Prefetch
        self.btn_repair = QPushButton("Repair data folders (recreate + fetch index)")
        self.btn_repair.clicked.connect(self._repair_data_folders)
        self.btn_repair.setToolTip("Recreate the base folder structure and fetch index/rom-names if missing.")
        settings_layout.addWidget(self.btn_repair)

        self.btn_prefetch = QPushButton("Cache maps now (prefetch)")
        self.btn_prefetch.clicked.connect(self._prefetch_maps_now)
        self.btn_prefetch.setToolTip("Cache missing NVRAM maps in the background. See watcher.log for progress.")
        settings_layout.addWidget(self.btn_prefetch)

        # Add remaining path widgets
        for w in (self.base_label, self.btn_base, self.nvram_label, self.btn_nvram, self.tables_label, self.btn_tables):
            settings_layout.addWidget(w)

        tabs.addTab(settings_tab, "Settings")

        # Stats tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_tabs = QTabWidget()
        self.stats_tabs.setToolTip("Shows statistics for Global, Players 1–4, and CPU simulation.")
        self.stats_views: Dict[str | int, QTextBrowser] = {}

        # Global
        self.stats_views["global"] = QTextBrowser()
        self.stats_tabs.addTab(self.stats_views["global"], "Global")

        # Players 1..4
        for i in range(1, 5):
            self.stats_views[i] = QTextBrowser()
            self.stats_tabs.addTab(self.stats_views[i], f"Player {i}")

        # CPU-Sim (once)
        self.stats_views["cpu"] = QTextBrowser()
        self.stats_tabs.addTab(self.stats_views["cpu"], "CPU-Sim")

        # CPU-Sim Controls (checkbox + difficulty button) above tabs
        row_cpu = QHBoxLayout()
        self.chk_cpu_active = QCheckBox("CPU-Sim active")
        try:
            self.chk_cpu_active.setChecked(bool((self.watcher.cpu or {}).get("active", False)))
        except Exception:
            self.chk_cpu_active.setChecked(False)
        self.chk_cpu_active.setToolTip("Toggle CPU simulation on/off.")
        self.chk_cpu_active.stateChanged.connect(self._on_cpu_active_changed)

        self.btn_cpu_diff = QPushButton(self._cpu_diff_label())
        self.btn_cpu_diff.setToolTip("Cycle difficulty: Easy → Medium → Hard → Pro.")
        self.btn_cpu_diff.clicked.connect(self._on_cpu_cycle_difficulty)

        row_cpu.addWidget(self.chk_cpu_active)
        row_cpu.addSpacing(12)
        row_cpu.addWidget(self.btn_cpu_diff)
        row_cpu.addStretch(1)

        stats_layout.addLayout(row_cpu)
        stats_layout.addWidget(self.stats_tabs)
        tabs.addTab(stats_tab, "Stats")

        # Timers
        self.timer_logs = QTimer(self)
        self.timer_logs.timeout.connect(self._update_logs)
        self.timer_logs.start(1200)

        self.timer_stats = QTimer(self)
        self.timer_stats.timeout.connect(self.update_stats)
        self.timer_stats.start(4000)

        self.overlay_refresh_timer = QTimer(self)
        self.overlay_refresh_timer.setInterval(2000)
        self.overlay_refresh_timer.timeout.connect(self._refresh_overlay_live)
        if bool(self.cfg.OVERLAY.get("live_updates", False)): self.overlay_refresh_timer.start()

        # Joystick poll
        self._joy_toggle_last_mask = 0
        self._joy_toggle_timer = QTimer(self)
        self._joy_toggle_timer.setInterval(50)
        self._joy_toggle_timer.timeout.connect(self._on_joy_toggle_poll)
        self._apply_toggle_source()

        # Keyboard raw input
        self._last_toggle_ts = 0.0

        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = QSystemTrayIcon(icon, self)
            menu = QMenu()
            menu.addAction("Open", self._show_from_tray)
            menu.addAction("Quit GUI", self.quit_all)
            self.tray.setContextMenu(menu)
            self.tray.show()
        else:
            self.tray = None
         
        self._overlay_cycle = {"sections": [], "idx": -1}
        self._overlay_busy = False
        self._overlay_last_action = 0.0 
        self.overlay: Optional[OverlayWindow] = None

        self.watcher.start()
        self._apply_theme()

        self._init_tooltips_main()
        self._init_overlay_tooltips()

        # NEW: build Achievements main tab (two subtabs) and start a refresh timer
        try:
            self._build_achievements_tab()
        except Exception:
            pass
        try:
            self._init_achievements_timer()
        except Exception:
            pass

        # --- System Stats tab ---
        try:
            self._build_system_stats_tab()
            self._stats_refresh_now()  # initial fill
        except Exception:
            pass

    def _first_screen_geometry(self) -> QRect:
        """
        Geometry des ersten Windows-Monitors (QApplication.screens()[0]).
        Fallback: Primary-Screen, sonst 1280x720.
        """
        try:
            screens = QApplication.screens() or []
            if screens:
                return screens[0].geometry()
            scr = QApplication.primaryScreen()
            if scr:
                return scr.geometry()
        except Exception:
            pass
        return QRect(0, 0, 1280, 720)


    # Helper in MainWindow:
    def _msgbox_topmost(self, kind: str, title: str, text: str):
        box = QMessageBox(self)
        box.setWindowTitle(str(title))
        box.setText(str(text))
        box.setIcon(QMessageBox.Icon.Information if kind == "info" else QMessageBox.Icon.Warning)
        # Always on top without stealing focus
        box.setWindowFlags(box.windowFlags() | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        box.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        box.setModal(True)
        box.show()
        box.raise_()
        return box.exec()

    def _on_challenge_timer_start(self, total_seconds: int):
        """
        Startet den kleinen Countdown unten links NICHT sofort, sondern erst nach 30s Warm‑Up.
        - Nach 30s wird der Timer mit (total_seconds - 30) gestartet.
        - WICHTIG: Keine Guard-Abfrage mehr auf watcher.game_active – Countdown startet robust auch bei kurzen Scanner-Glitches.
        """
        try:
            # Vorherige Delay-Timer abbrechen
            try:
                if hasattr(self, "_challenge_timer_delay") and self._challenge_timer_delay:
                    self._challenge_timer_delay.stop()
                    self._challenge_timer_delay.deleteLater()
            except Exception:
                pass
            self._challenge_timer_delay = None

            # Vorheriges Countdown-Fenster schließen/löschen
            try:
                if hasattr(self, "_challenge_timer") and self._challenge_timer:
                    self._challenge_timer.close()
                    self._challenge_timer.deleteLater()
            except Exception:
                pass
            self._challenge_timer = None

            warmup_sec = 30
            play_sec = max(1, int(total_seconds or 0) - warmup_sec)

            # Verzögert nach Warm‑Up starten
            self._challenge_timer_delay = QTimer(self)
            self._challenge_timer_delay.setSingleShot(True)

            def _spawn():
                # Safety: vorhandenes Fenster weg
                try:
                    if hasattr(self, "_challenge_timer") and self._challenge_timer:
                        self._challenge_timer.close()
                        self._challenge_timer.deleteLater()
                except Exception:
                    pass
                self._challenge_timer = None

                # Countdown-Fenster erzeugen (defensiv)
                try:
                    # Log-Hilfe (optional, falls du in watcher.log schauen willst)
                    log(self.cfg, f"[CHALLENGE] countdown spawn – seconds={play_sec}")
                except Exception:
                    pass

                try:
                    self._challenge_timer = ChallengeCountdownOverlay(self, play_sec)
                except Exception:
                    self._challenge_timer = None

            # Start nach Warm‑up
            self._challenge_timer_delay.timeout.connect(lambda: QTimer.singleShot(0, _spawn))
            self._challenge_timer_delay.start(warmup_sec * 1000)
        except Exception:
            pass

    # In class MainWindow: BEIDE Vorkommen von _start_timed_challenge_ui anpassen – KEINE Sprachausgabe hier
    def _start_timed_challenge_ui(self):
        if not self.watcher or not self.watcher.game_active:
            QMessageBox.information(self, "Challenge",
                                    "Start a table first. The challenge key works only while a game is running.")
            return
        try:
            # keine Sprachausgabe hier – sie kommt im Warm‑up‑Handler, wenn das Overlay sichtbar ist
            self.watcher.start_timed_challenge(total_seconds=330)
        except Exception:
            pass

    def _start_one_ball_challenge_ui(self):
        if not self.watcher or not self.watcher.game_active:
            QMessageBox.information(
                self, "Challenge",
                "Start a table first. The challenge key works only while a game is running."
            )
            return
        try:
            self.watcher.start_one_ball_challenge()
        except Exception:
            pass


    def _repair_data_folders(self):
        try:
            ensure_dir(self.cfg.BASE)
            for sub in [
                "NVRAM_Maps","NVRAM_Maps/maps","NVRAM_Maps/overrides","session_stats",
                "session_stats/Highlights","session_stats/whitelists","rom_specific_achievements","custom_achievements",
            ]:
                ensure_dir(os.path.join(self.cfg.BASE, sub))
            try:
                self.watcher.bootstrap()
            except Exception as e:
                log(self.cfg, f"[REPAIR] bootstrap failed: {e}", "WARN")
            self._msgbox_topmost("info", "Repair",
                                 "Base folders repaired.\n\nIf maps are still missing, please click 'Cache maps now (prefetch)'\nor simply start a ROM (maps will then be loaded on demand).")
            log(self.cfg, "[REPAIR] base folders ensured and index/romnames fetched (if missing)")
        except Exception as e:
            log(self.cfg, f"[REPAIR] failed: {e}", "ERROR")
            self._msgbox_topmost("warn", "Repair", f"Repair failed:\n{e}")

    def _mods_for_vk(self, vk: int) -> int:
        """
        Force no modifiers for WM_HOTKEY registrations, so plain letters work without Alt.
        NOTE: Plain letters can konflikieren mit anderen Programmen. F-Keys bleiben empfohlen.
        """
        return 0
     
    # ===== Challenges tab (GUI) =====
    def _build_challenges_tab(self):
        """
        Build the 'Challenges' main tab:
        - Two modes: Timed (5:30) and One-Ball
        - Per-mode input source (keyboard/joystick), bind button, current binding label
        - Volume slider for voice announcements
        - Latest results for the current ROM
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Timed challenge row
        grp_t = QGroupBox("Timed Challenge (5:00)")
        lay_t = QHBoxLayout(grp_t)

        self.btn_ch_timed_start = QPushButton("Start Timed Challenge")
        self.btn_ch_timed_start.setToolTip("Start the 5:00 Timed Challenge. Works only while a game is running.")
        self.btn_ch_timed_start.clicked.connect(self._start_timed_challenge_ui)

        self.cmb_ch_timed_src = QComboBox()
        self.cmb_ch_timed_src.addItems(["keyboard", "joystick"])
        self.cmb_ch_timed_src.setCurrentText(self.cfg.OVERLAY.get("challenge_time_input_source", "keyboard"))
        self.cmb_ch_timed_src.setToolTip("Input source for the Timed Challenge hotkey/button.")
        self.cmb_ch_timed_src.currentTextChanged.connect(self._on_ch_timed_src_changed)

        self.btn_ch_timed_bind = QPushButton("Bind…")
        self.btn_ch_timed_bind.setToolTip("Bind a key or joystick button for starting the Timed Challenge.")
        self.btn_ch_timed_bind.clicked.connect(self._on_bind_ch_timed_clicked)

        self.lbl_ch_timed_binding = QLabel(self._ch_binding_label_text(kind="time"))
        lay_t.addWidget(self.btn_ch_timed_start)
        lay_t.addSpacing(14)
        lay_t.addWidget(QLabel("Source:"))
        lay_t.addWidget(self.cmb_ch_timed_src)
        lay_t.addWidget(self.btn_ch_timed_bind)
        lay_t.addWidget(self.lbl_ch_timed_binding)
        lay_t.addStretch(1)

        # One-Ball challenge row
        grp_o = QGroupBox("One-Ball Challenge")
        lay_o = QHBoxLayout(grp_o)

        self.btn_ch_one_start = QPushButton("Start One-Ball Challenge")
        self.btn_ch_one_start.setToolTip("Arm the one-ball challenge. Ends after your first ball drains.")
        self.btn_ch_one_start.clicked.connect(self._start_one_ball_challenge_ui)

        self.cmb_ch_one_src = QComboBox()
        self.cmb_ch_one_src.addItems(["keyboard", "joystick"])
        self.cmb_ch_one_src.setCurrentText(self.cfg.OVERLAY.get("challenge_one_input_source", "keyboard"))
        self.cmb_ch_one_src.setToolTip("Input source for the One-Ball Challenge hotkey/button.")
        self.cmb_ch_one_src.currentTextChanged.connect(self._on_ch_one_src_changed)

        self.btn_ch_one_bind = QPushButton("Bind…")
        self.btn_ch_one_bind.setToolTip("Bind a key or joystick button for starting the One-Ball Challenge.")
        self.btn_ch_one_bind.clicked.connect(self._on_bind_ch_one_clicked)

        self.lbl_ch_one_binding = QLabel(self._ch_binding_label_text(kind="one"))
        lay_o.addWidget(self.btn_ch_one_start)
        lay_o.addSpacing(14)
        lay_o.addWidget(QLabel("Source:"))
        lay_o.addWidget(self.cmb_ch_one_src)
        lay_o.addWidget(self.btn_ch_one_bind)
        lay_o.addWidget(self.lbl_ch_one_binding)
        lay_o.addStretch(1)

        # Voice volume row
        grp_v = QGroupBox("Voice")
        lay_v = QHBoxLayout(grp_v)
        lay_v.addWidget(QLabel("Volume:"))
        self.sld_ch_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_ch_volume.setRange(0, 100)
        self.sld_ch_volume.setValue(int(self.cfg.OVERLAY.get("challenges_voice_volume", 80)))
        self.sld_ch_volume.valueChanged.connect(self._on_ch_volume_changed)
        lay_v.addWidget(self.sld_ch_volume)
        self.lbl_ch_volume = QLabel(f"{self.sld_ch_volume.value()}%")
        lay_v.addWidget(self.lbl_ch_volume)
        lay_v.addStretch(1)
        grp_v.setToolTip("Adjust the voice volume for challenge announcements.")

        # Results view
        self.ch_results_view = QTextBrowser()
        self.ch_results_view.setToolTip("Latest challenge results for the current ROM.")
        self._update_challenges_results_view()

        layout.addWidget(grp_t)
        layout.addWidget(grp_o)
        layout.addWidget(grp_v)
        layout.addWidget(QLabel("Latest results – Timed vs One-Ball"))
        layout.addWidget(self.ch_results_view)
        self.main_tabs.addTab(tab, "Challenges")

        # Connect bridge signals
        self.bridge.challenge_warmup_show.connect(self._on_challenge_warmup_show)
        self.bridge.challenge_timer_start.connect(self._on_challenge_timer_start)
        self.bridge.challenge_timer_stop.connect(self._on_challenge_timer_stop)
        self.bridge.challenge_info_show.connect(self._on_challenge_info_show)
        self.bridge.challenge_speak.connect(self._on_challenge_speak)

    def _on_ch_volume_changed(self, val: int):
        self.lbl_ch_volume.setText(f"{val}%")
        self.cfg.OVERLAY["challenges_voice_volume"] = int(val)
        self.cfg.save()

    def _ch_results_path(self, rom: str) -> str:
        return os.path.join(self.cfg.BASE, "challenges", "history", f"{sanitize_filename(rom)}.json")

    # In class MainWindow: REPLACE the whole _update_challenges_results_view with the two-column scoreboard
    def _update_challenges_results_view(self):
        """
        Show latest challenge scores in two side-by-side columns:
          - Left: Timed (score, rom)
          - Right: One-Ball (score, rom)
        Aggregated across all ROMs, up to 20 entries per column, newest first.
        """
        import glob
        from datetime import datetime

        hist_dir = os.path.join(self.cfg.BASE, "challenges", "history")

        def _parse_ts(iso: str, fallback_mtime: float = 0.0) -> float:
            try:
                if not iso:
                    return float(fallback_mtime or 0.0)
                # Accept 'Z' and timezone offsets
                s = str(iso).strip().replace("Z", "+00:00")
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                return float(fallback_mtime or 0.0)

        if not os.path.isdir(hist_dir):
            self.ch_results_view.setHtml("<div>(no results)</div>")
            return

        timed = []
        oneball = []

        # Collect all results from all ROM history files
        for fp in glob.glob(os.path.join(hist_dir, "*.json")):
            try:
                st = os.stat(fp)
                base_mtime = float(getattr(st, "st_mtime", 0.0))
            except Exception:
                base_mtime = 0.0
            data = load_json(fp, {"results": []}) or {"results": []}
            for it in (data.get("results") or []):
                if not isinstance(it, dict):
                    continue
                try:
                    rom = str(it.get("rom", "")).strip()
                    score = int(it.get("score", 0) or 0)
                    kind = str(it.get("kind", "")).strip().lower()
                    ts = _parse_ts(it.get("ts", ""), base_mtime)
                except Exception:
                    continue
                entry = {"rom": rom, "score": score, "ts": ts}
                if kind == "timed":
                    timed.append(entry)
                elif kind == "oneball" or kind == "one-ball":
                    oneball.append(entry)

        # Sort newest first and keep up to 20 each
        timed.sort(key=lambda e: e["ts"], reverse=True)
        oneball.sort(key=lambda e: e["ts"], reverse=True)
        timed = timed[:20]
        oneball = oneball[:20]

        # Build HTML (two columns)
        css = (
            "<style>"
            "table{border-collapse:collapse}"
            "th,td{padding:4px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}"
            ".col{vertical-align:top;padding:0 10px}"
            ".hdr{font-weight:700;margin-bottom:6px}"
            "</style>"
        )

        def _col(title: str, items: list[dict]) -> str:
            head = "<tr><th align='right'>Score</th><th align='left'>ROM</th></tr>"
            rows = []
            for e in items:
                rows.append(
                    f"<tr>"
                    f"<td align='right'>{self._fmt_int(int(e['score']))}</td>"
                    f"<td>{(e['rom'] or '-')}</td>"
                    f"</tr>"
                )
            body = "".join(rows) if rows else "<tr><td colspan='2'>(keine)</td></tr>"
            return f"<div class='hdr'>{title}</div><table>{head}{body}</table>"

        html = (
            css +
            "<table width='100%'><tr>"
            f"<td class='col' width='50%'>{_col('Timed', timed)}</td>"
            f"<td class='col' width='50%'>{_col('One-Ball', oneball)}</td>"
            "</tr></table>"
        )

        self.ch_results_view.setHtml(html)


    def _on_ch_timed_src_changed(self, src: str):
        self.cfg.OVERLAY["challenge_time_input_source"] = src
        self.cfg.save()
        self.lbl_ch_timed_binding.setText(self._ch_binding_label_text(kind="time"))
        # Reinstall + Poll-Entscheidung aktualisieren
        self._refresh_input_bindings()
        self._apply_toggle_source()

    def _on_ch_one_src_changed(self, src: str):
        self.cfg.OVERLAY["challenge_one_input_source"] = src
        self.cfg.save()
        self.lbl_ch_one_binding.setText(self._ch_binding_label_text(kind="one"))
        # Reinstall + Poll-Entscheidung aktualisieren
        self._refresh_input_bindings()
        self._apply_toggle_source()

    def _ch_binding_label_text(self, kind: str) -> str:
        if kind == "time":
            src = self.cfg.OVERLAY.get("challenge_time_input_source", "keyboard")
            if src == "joystick":
                btn = int(self.cfg.OVERLAY.get("challenge_time_joy_button", 3))
                return f"Current: joystick button {btn}"
            else:
                vk = int(self.cfg.OVERLAY.get("challenge_time_vk", 121))  # F10 default
                return f"Current: {vk_to_name(vk)}"
        else:
            src = self.cfg.OVERLAY.get("challenge_one_input_source", "keyboard")
            if src == "joystick":
                btn = int(self.cfg.OVERLAY.get("challenge_one_joy_button", 4))
                return f"Current: joystick button {btn}"
            else:
                vk = int(self.cfg.OVERLAY.get("challenge_one_vk", 122))  # F11 default
                return f"Current: {vk_to_name(vk)}"

    def _on_bind_ch_timed_clicked(self):
        self._bind_challenge_key(kind="time")

    def _on_bind_ch_one_clicked(self):
        self._bind_challenge_key(kind="one")

    def _bind_challenge_key(self, kind: str):
        """
        Bind keyboard or joystick for a specific challenge ('time' | 'one').
        UI/flow matches the overlay binding UX.
        """
        src = self.cfg.OVERLAY.get("challenge_time_input_source" if kind=="time" else "challenge_one_input_source", "keyboard")

        if src == "joystick":
            # Simple capture of next pressed button (timeout 10s)
            dlg = QDialog(self)
            dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
            dlg.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            dlg.setWindowTitle("Joystick binding")
            dlg.resize(420, 160)
            lay = QVBoxLayout(dlg)
            lbl = QLabel("Press any joystick button to bind…\n(Timeout in 10 seconds; ESC to cancel)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)

            cancelled = {"flag": False}
            def keyPressEvent(evt):
                if evt.key() == Qt.Key.Key_Escape:
                    cancelled["flag"] = True
                    dlg.reject()
            dlg.keyPressEvent = keyPressEvent  # type: ignore

            def _read_buttons_mask() -> int:
                jix = JOYINFOEX()
                jix.dwSize = ctypes.sizeof(JOYINFOEX)
                jix.dwFlags = JOY_RETURNALL
                mask_all = 0
                for jid in range(16):
                    try:
                        if _joyGetPosEx(jid, ctypes.byref(jix)) == JOYERR_NOERROR:
                            mask_all |= int(jix.dwButtons)
                    except Exception:
                        continue
                return mask_all

            baseline = _read_buttons_mask()
            start_ts = time.time()
            timer = QTimer(dlg)

            def _poll():
                nonlocal baseline
                if cancelled["flag"]:
                    timer.stop()
                    return
                try:
                    mask = _read_buttons_mask()
                    newly = mask & ~baseline
                    baseline = mask
                    if newly:
                        lsb = newly & -newly
                        idx = lsb.bit_length() - 1
                        button_num = idx + 1
                        key_btn = "challenge_time_joy_button" if kind=="time" else "challenge_one_joy_button"
                        self.cfg.OVERLAY[key_btn] = int(button_num)
                        self.cfg.save()
                        if kind == "time":
                            self.lbl_ch_timed_binding.setText(self._ch_binding_label_text(kind="time"))
                        else:
                            self.lbl_ch_one_binding.setText(self._ch_binding_label_text(kind="one"))
                        timer.stop()
                        dlg.accept()
                        # Reinstall inputs after binding
                        self._refresh_input_bindings()
                        return
                    if time.time() - start_ts > 10.0:
                        timer.stop()
                        dlg.reject()
                        return
                except Exception:
                    pass

            timer.setInterval(35)
            timer.timeout.connect(_poll)
            timer.start()
            dlg.exec()
            return

        # Keyboard bind
        class _TmpVKFilter(QAbstractNativeEventFilter):
            def __init__(self, cb):
                super().__init__()
                self.cb = cb
                self._done = False
            def nativeEventFilter(self, eventType, message):
                if self._done:
                    return False, 0
                try:
                    if eventType == b"windows_generic_MSG":
                        msg = ctypes.wintypes.MSG.from_address(int(message))
                        if msg.message in (WM_KEYDOWN, WM_SYSKEYDOWN):
                            vk = int(msg.wParam)
                            self._done = True
                            self.cb(vk)
                except Exception:
                    pass
                return False, 0

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard binding")
        dlg.resize(360, 140)
        # Always on top + non-activating
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        dlg.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        lay = QVBoxLayout(dlg)
        lbl = QLabel("Press any key... (ESC to cancel)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)

        cancelled = {"flag": False}
        def on_vk(vk: int):
            if cancelled["flag"]:
                return
            key = "challenge_time_vk" if kind=="time" else "challenge_one_vk"
            self.cfg.OVERLAY[key] = int(vk)
            self.cfg.save()
            if kind == "time":
                self.lbl_ch_timed_binding.setText(self._ch_binding_label_text(kind="time"))
            else:
                self.lbl_ch_one_binding.setText(self._ch_binding_label_text(kind="one"))
            try:
                QCoreApplication.instance().removeNativeEventFilter(fil)  # type: ignore
            except Exception:
                pass
            dlg.accept()
            # Reinstall inputs after binding
            self._refresh_input_bindings()

        def keyPressEvent(evt):
            if evt.key() == Qt.Key.Key_Escape:
                cancelled["flag"] = True
                try:
                    QCoreApplication.instance().removeNativeEventFilter(fil)  # type: ignore
                except Exception:
                    pass
                dlg.reject()
        dlg.keyPressEvent = keyPressEvent  # type: ignore

        fil = _TmpVKFilter(on_vk)
        QCoreApplication.instance().installNativeEventFilter(fil)
        dlg.exec()



 
# ... in class MainWindow, replace _build_system_stats_tab with this version ...
    def _build_system_stats_tab(self):
        """
        Build the 'System Stats' main tab with category sub‑tabs.
        Tooltips are set in English on both the view and the tab button.
        NOTE: Reduced tab set (Overview, Top & Least, Trends, Heatmaps, Records).
        Adds a small explanation at the top of the tab.
        """
        stats_tab = QWidget()
        layout = QVBoxLayout(stats_tab)

        # Small explanation
        info = QLabel(
            "System Stats shows Overview, Top & Least Played, Trends, Heatmaps, and Records.\n"
            "Heatmaps display the current month; if there is no activity this month, the latest month with activity is shown automatically."
        )
        info.setStyleSheet("color:#555;margin-bottom:6px;")
        layout.addWidget(info)

        # Sub-tabs (reduced set)
        self.sys_tabs = QTabWidget()
        self.sys_views = {}

        def add_subtab(key: str, title: str, tooltip: str = ""):
            w = QWidget()
            vlayout = QVBoxLayout(w)
            view = QTextBrowser()
            if tooltip:
                view.setToolTip(tooltip)
            vlayout.addWidget(view)
            idx = self.sys_tabs.addTab(w, title)
            if tooltip:
                self.sys_tabs.setTabToolTip(idx, tooltip)
            self.sys_views[key] = view

        add_subtab("overview", "Overview", "Totals and recent activity")
        add_subtab("topleast", "Top & Least Played", "Tables by playtime/plays")
        add_subtab("trends", "Trends", "Weekly plays and duration trends")
        add_subtab("heat", "Heatmaps", "Monthly calendars: playtime/day and sessions/day")
        add_subtab("records", "Records", "Longest sessions, highest scores")

        # Refresh row
        row = QHBoxLayout()
        self.btn_sys_refresh = QPushButton("Refresh System Stats")
        self.btn_sys_refresh.setToolTip("Reload all statistics from disk (history and sessions).")
        self.btn_sys_refresh.clicked.connect(self._stats_refresh_now)
        row.addWidget(self.btn_sys_refresh)
        row.addStretch(1)

        layout.addLayout(row)
        layout.addWidget(self.sys_tabs)
        self.main_tabs.addTab(stats_tab, "System Stats")

        # Small periodic refresher (lightweight; relies on caching)
        self.timer_sys_stats = QTimer(self)
        self.timer_sys_stats.setInterval(15000)  # 15s
        self.timer_sys_stats.timeout.connect(self._stats_refresh_now)
        self.timer_sys_stats.start()

    def _tz_for_heatmaps(self) -> "ZoneInfo | None":
        """
        Return local system timezone only (no external services, no user input).
        """
        try:
            return datetime.now().astimezone().tzinfo  # type: ignore
        except Exception:
            return None


    def _stats_render_heatmap(self, sessions: list) -> str:
        """
        Monthly calendars only:
          - Playtime/day (target month)
          - Sessions/day (target month)
        Uses local system time.

        Target month selection:
          - Default: current month
          - If there is no activity in the current month, automatically switch to the
            month of the latest session so the calendar is never “empty”.
        """
        # Local timezone (system)
        tz = self._tz_for_heatmaps()

        # Helper: normalize dt from a session item to naive local datetime
        def to_local_naive(dt: datetime) -> datetime:
            if dt is None:
                return None  # type: ignore
            try:
                if dt.tzinfo is not None:
                    return dt.astimezone().replace(tzinfo=None)
            except Exception:
                pass
            return dt

        # Collect all session datetimes (local-naive)
        all_dts: list[datetime] = []
        for s in sessions or []:
            dt = s.get("_dt")
            if isinstance(dt, datetime):
                all_dts.append(to_local_naive(dt))

        # Choose target month/year
        from datetime import date as _date
        import calendar

        now_local = datetime.now().astimezone().replace(tzinfo=None)
        target_year = now_local.year
        target_month = now_local.month

        # Check if current month has any activity; if not, use latest session month
        has_current_month = any((d.year == target_year and d.month == target_month) for d in all_dts)
        if not has_current_month and all_dts:
            latest = max(all_dts)
            target_year, target_month = latest.year, latest.month

        month_name = datetime(target_year, target_month, 1).strftime("%B %Y")

        # Aggregates per day for the target month
        from collections import defaultdict
        day_counts: dict[_date, int] = defaultdict(int)
        day_secs: dict[_date, int] = defaultdict(int)

        for s in sessions or []:
            dt = s.get("_dt")
            if not isinstance(dt, datetime):
                continue
            # normalize to local-naive
            dt_local = to_local_naive(dt)
            if dt_local.year == target_year and dt_local.month == target_month:
                d = dt_local.date()
                day_counts[d] += 1
                try:
                    day_secs[d] += int(s.get("duration_sec", 0) or 0)
                except Exception:
                    pass

        # Build calendar weeks (Mon..Sun)
        calendar.setfirstweekday(calendar.MONDAY)
        weeks = calendar.monthcalendar(target_year, target_month)  # 0 => out-of-month day

        css = (
            "<style>"
            "table{border-collapse:collapse}"
            "th,td{padding:6px 8px;border:1px solid #e5e5e5;white-space:nowrap;text-align:center;vertical-align:middle}"
            ".muted{color:#777}"
            ".dim{color:#bbb}"
            "</style>"
        )

        def fmt_hms(seconds: int) -> str:
            try:
                seconds = int(seconds or 0)
            except Exception:
                seconds = 0
            d = seconds // 86400
            h = (seconds % 86400) // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            if d > 0:
                return f"{d}d {h:02d}:{m:02d}:{s:02d}"
            return f"{h:02d}:{m:02d}:{s:02d}"

        def cal_table(title: str, value_for_day):
            head = "<tr>" + "".join(f"<th>{w}</th>" for w in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]) + "</tr>"
            rows_html = []
            for w in weeks:
                tds = []
                for day_num in w:
                    if day_num == 0:
                        tds.append("<td class='dim'> </td>")
                    else:
                        d = datetime(target_year, target_month, day_num).date()
                        val_txt = value_for_day(d)
                        tds.append(
                            f"<td><div style='font-weight:600'>{day_num}</div>"
                            f"<div class='muted'>{val_txt}</div></td>"
                        )
                rows_html.append("<tr>" + "".join(tds) + "</tr>")
            return f"<h4>{title} – {month_name}</h4><table>{head}{''.join(rows_html)}</table>"

        cal_playtime = cal_table(
            "Monthly calendar – Playtime/day",
            lambda d: fmt_hms(day_secs.get(d, 0))
        )
        cal_sessions = cal_table(
            "Monthly calendar – Sessions/day",
            lambda d: f"{int(day_counts.get(d, 0)):,d}".replace(",", ".")
        )

        return css + cal_playtime + cal_sessions

    def _stats_cache_reset(self):
        self._stats_cache = {
            "features": {"mtimes": {}, "data": []},
            "sessions": {"mtimes": {}, "data": []},
            "maps": {"mtimes": {}, "index": {}},
        }

# In class MainWindow: make _stats_load_features timezone-safe (normalize to naive local datetime)
    def _stats_load_features(self) -> list:
        """
        Load BASE/AI/history/*.features.json with simple mtime cache.
        Each item augmented with parsed 'dt' (datetime) and 'ymd', 'wk' keys.
        NOTE: _dt is normalized to naive local datetime to avoid aware/naive comparison issues.
        """
        try:
            cache = getattr(self, "_stats_cache", None)
            if cache is None:
                self._stats_cache_reset()
                cache = self._stats_cache
            bucket = cache["features"]
            root = self._stats_paths()["history"]
            if not os.path.isdir(root):
                return []
            out = list(bucket.get("data", []))
            mtimes = dict(bucket.get("mtimes", {}))

            updated = False
            for fn in os.listdir(root):
                if not fn.lower().endswith(".features.json"):
                    continue
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                m = float(getattr(st, "st_mtime", 0.0))
                if mtimes.get(p) == m:
                    continue
                # (Re)load
                try:
                    data = load_json(p, None)
                    if not isinstance(data, dict):
                        mtimes[p] = m
                        continue
                    # Normalize timestamp
                    ts = data.get("ts") or data.get("updated") or ""
                    try:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    except Exception:
                        dt = datetime.fromtimestamp(m)
                    # Normalize to naive local datetime to avoid aware/naive comparisons
                    try:
                        if dt.tzinfo is not None:
                            dt = dt.astimezone().replace(tzinfo=None)
                    except Exception:
                        pass

                    data["_dt"] = dt
                    data["_ymd"] = dt.date().isoformat()
                    # Robust ISO calendar week (supports both tuple and namedtuple)
                    iso = dt.isocalendar()
                    try:
                        y = iso.year; w = iso.week
                    except Exception:
                        y, w, _ = iso  # tuple fallback
                    data["_week"] = f"{int(y)}-W{int(w):02d}"
                    out.append(data)
                finally:
                    mtimes[p] = m
                    updated = True

            if updated:
                out.sort(key=lambda d: d.get("_dt") or datetime(1970, 1, 1))
                cache["features"]["data"] = out
                cache["features"]["mtimes"] = mtimes
            return cache["features"]["data"]
        except Exception:
            return []


    def _stats_paths(self) -> dict:
        """
        Centralizes folder paths used by the System Stats tab.
        """
        base = self.cfg.BASE
        return {
            "history": os.path.join(base, "AI", "history"),
            "sessions": os.path.join(base, "session_stats", "Highlights"),
            "maps": os.path.join(base, "NVRAM_Maps", "maps"),
            "overrides": os.path.join(base, "NVRAM_Maps", "overrides"),
        }

# ... in class MainWindow, replace _stats_refresh_now with this version ...
    def _stats_refresh_now(self):
        """
        Recompute all System Stats sub-tabs from disk (cached loads).
        Robust: each sub-tab is rendered independently so one failure
        does not blank all other sub-tabs.
        """
        # Load sources (defensive)
        try:
            feats = self._stats_load_features()
        except Exception as e:
            log(self.cfg, f"[SYSSTATS] load features failed: {e}", "WARN")
            feats = []
        try:
            sess = self._stats_load_sessions()
        except Exception as e:
            log(self.cfg, f"[SYSSTATS] load sessions failed: {e}", "WARN")
            sess = []

        def safe_set(key: str, builder):
            html = "<div>(no data)</div>"
            try:
                html = builder()
            except Exception as e:
                log(self.cfg, f"[SYSSTATS] render {key} failed: {e}", "WARN")
                html = "<div style='color:#c33'>Error rendering section. Check watcher.log.</div>"
            try:
                view = self.sys_views.get(key)
                if view:
                    view.setHtml(html)
                else:
                    log(self.cfg, f"[SYSSTATS] view not found for key '{key}'", "WARN")
            except Exception as e:
                log(self.cfg, f"[SYSSTATS] setHtml {key} failed: {e}", "WARN")

        # Render only the kept tabs
        safe_set("overview",  lambda: self._stats_render_overview(feats))
        safe_set("topleast",  lambda: self._stats_render_top_least(feats))
        safe_set("trends",    lambda: self._stats_render_trends(feats))
        safe_set("heat",      lambda: self._stats_render_heatmap(sess))
        safe_set("records",   lambda: self._stats_render_records(feats, sess))
            
    # In class MainWindow: also normalize _dt in sessions loader
    def _stats_load_sessions(self) -> list:
        """
        Load BASE/session_stats/Highlights/*.session.json with mtime cache.
        Augment items with parsed 'dt' from 'end_timestamp' or file mtime.
        NOTE: _dt is normalized to naive local datetime to avoid aware/naive comparison issues.
        """
        try:
            cache = getattr(self, "_stats_cache", None)
            if cache is None:
                self._stats_cache_reset()
                cache = self._stats_cache
            bucket = cache["sessions"]
            root = self._stats_paths()["sessions"]
            if not os.path.isdir(root):
                return []
            out = list(bucket.get("data", []))
            mtimes = dict(bucket.get("mtimes", {}))
            updated = False

            for fn in os.listdir(root):
                if not fn.lower().endswith(".session.json"):
                    continue
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p); m = float(getattr(st, "st_mtime", 0.0))
                except Exception:
                    continue
                if mtimes.get(p) == m:
                    continue
                try:
                    data = load_json(p, None)
                    if not isinstance(data, dict):
                        mtimes[p] = m
                        continue
                    iso = data.get("end_timestamp", "")
                    try:
                        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
                    except Exception:
                        dt = datetime.fromtimestamp(m)
                    # Normalize to naive local datetime
                    try:
                        if dt.tzinfo is not None:
                            dt = dt.astimezone().replace(tzinfo=None)
                    except Exception:
                        pass

                    data["_dt"] = dt
                    out.append(data)
                finally:
                    mtimes[p] = m
                    updated = True

            if updated:
                out.sort(key=lambda d: d.get("_dt") or datetime(1970,1,1))
                bucket["data"] = out
                bucket["mtimes"] = mtimes
            return bucket["data"]
        except Exception:
            return []

    def _stats_index_maps(self) -> dict:
        """
        Build a small index: rom -> {'type':'base'|'override'|'auto'|'unknown'}
        We inspect files in maps/ and overrides/; for 'auto' we check payload.generated==true.
        """
        try:
            cache = getattr(self, "_stats_cache", None)
            if cache is None:
                self._stats_cache_reset()
                cache = self._stats_cache
            bucket = cache["maps"]
            root_maps = self._stats_paths()["maps"]
            root_ovr = self._stats_paths()["overrides"]
            idx = {}
            mtimes = dict(bucket.get("mtimes", {}))
            updated = False

            def mark(rom, kind):
                prev = idx.get(rom, {}).get("type")
                if prev in ("override",) and kind != "override":
                    return
                idx[rom] = {"type": kind}

            # Maps
            if os.path.isdir(root_maps):
                for fn in os.listdir(root_maps):
                    if not (fn.lower().endswith(".json") or fn.lower().endswith(".map.json")):
                        continue
                    rom = os.path.splitext(fn)[0].replace(".map", "")
                    p = os.path.join(root_maps, fn)
                    try:
                        st = os.stat(p); m = float(st.st_mtime)
                    except Exception:
                        continue
                    if mtimes.get(p) != m:
                        updated = True
                        mtimes[p] = m
                    payload = load_json(p, {}) or {}
                    if payload.get("generated") is True:
                        mark(rom, "auto")
                    else:
                        mark(rom, "base")

            # Overrides
            if os.path.isdir(root_ovr):
                for fn in os.listdir(root_ovr):
                    if not fn.lower().endswith(".json"):
                        continue
                    rom = os.path.splitext(fn)[0]
                    p = os.path.join(root_ovr, fn)
                    try:
                        st = os.stat(p); m = float(st.st_mtime)
                    except Exception:
                        continue
                    if mtimes.get(p) != m:
                        updated = True
                        mtimes[p] = m
                    mark(rom, "override")

            if updated:
                bucket["index"] = idx
                bucket["mtimes"] = mtimes
            return bucket.get("index", {})
        except Exception:
            return {}


    # ---------- Helpers (formatting, math, parsing) ----------

    def _fmt_hms(self, seconds: int) -> str:
        try:
            seconds = int(seconds or 0)
        except Exception:
            seconds = 0
        d = seconds // 86400
        h = (seconds % 86400) // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if d > 0:
            return f"{d}d {h:02d}:{m:02d}:{s:02d}"
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _fmt_int(self, n: int) -> str:
        try:
            return f"{int(n):,d}".replace(",", ".")
        except Exception:
            return str(n)

    def _median(self, arr: list[float]) -> float:
        a = sorted([float(x) for x in arr if isinstance(x, (int, float))])
        if not a:
            return 0.0
        n = len(a)
        mid = n // 2
        if n % 2 == 1:
            return a[mid]
        return 0.5 * (a[mid - 1] + a[mid])

    def _percentile(self, arr: list[float], p: float) -> float:
        a = sorted([float(x) for x in arr if isinstance(x, (int, float))])
        if not a:
            return 0.0
        k = max(0, min(len(a) - 1, int(round((p / 100.0) * (len(a) - 1)))))
        return a[k]

    def _extract_maker_year(self, table: str) -> tuple[str, int | None]:
        """
        Try to extract manufacturer and year from strings like:
          "ATTACK FROM MARS (BALLY 1995) (VISUAL PINBALL X)"
        Returns (maker, year_or_None).
        """
        try:
            s = str(table or "")
            m = re.search(r"\(([A-Za-z0-9&.\- ]+)\s+(\d{4})\)", s)
            if m:
                maker = m.group(1).strip()
                year = int(m.group(2))
                return maker, year
            m2 = re.search(r"\((\d{4})\)", s)
            if m2:
                return "", int(m2.group(1))
        except Exception:
            pass
        return "", None

    # ---------- Renderers per category ----------

    def _stats_render_overview(self, feats: list) -> str:
        total_time = sum(int(f.get("duration_sec", 0) or 0) for f in feats)
        plays = len(feats)
        tables = {str(f.get("table","")).strip() for f in feats if f.get("table")}
        uniq_tables = len(tables)
        durations = [int(f.get("duration_sec", 0) or 0) for f in feats]
        avg_dur = int(sum(durations) / len(durations)) if durations else 0
        med_dur = int(self._median(durations)) if durations else 0

        # Recent 10 by timestamp desc
        recent = sorted(feats, key=lambda f: f.get("_dt") or datetime(1970,1,1), reverse=True)[:10]

        css = ("<style>"
               "table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}"
               ".cards{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 10px 0}"
               ".card{background:#141414;color:#fff;padding:12px 16px;border-radius:10px;min-width:210px}"
               ".muted{color:#777}"
               "</style>")
        cards = (
            f"<div class='cards'>"
            f"<div class='card'><div class='muted'>Total System Time</div><div style='font-weight:700;font-size:18px'>{self._fmt_hms(total_time)}</div></div>"
            f"<div class='card'><div class='muted'>Total Plays</div><div style='font-weight:700;font-size:18px'>{self._fmt_int(plays)}</div></div>"
            f"<div class='card'><div class='muted'>Unique Tables</div><div style='font-weight:700;font-size:18px'>{self._fmt_int(uniq_tables)}</div></div>"
            f"<div class='card'><div class='muted'>Average Session</div><div style='font-weight:700;font-size:18px'>{self._fmt_hms(avg_dur)}</div></div>"
            f"<div class='card'><div class='muted'>Median Session</div><div style='font-weight:700;font-size:18px'>{self._fmt_hms(med_dur)}</div></div>"
            f"</div>"
        )

        rows = []
        rows.append("<tr><th align='left'>When</th><th align='left'>Table</th><th align='right'>Duration</th></tr>")
        for f in recent:
            dt = f.get("_dt")
            when = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else "-"
            table = str(f.get("table","") or f.get("rom","")).strip()
            dur = self._fmt_hms(int(f.get("duration_sec", 0) or 0))
            rows.append(f"<tr><td>{when}</td><td>{table}</td><td align='right'>{dur}</td></tr>")
        recent_tbl = "<table>" + "".join(rows) + "</table>"

        return css + "<h3>Overview</h3>" + cards + "<h4>Recent activity</h4>" + recent_tbl

    def _stats_render_top_least(self, feats: list) -> str:
        # Aggregate by table
        agg = defaultdict(lambda: {"plays": 0, "time": 0, "durations": []})
        for f in feats:
            key = str(f.get("table") or f.get("rom") or "").strip()
            if not key:
                continue
            d = int(f.get("duration_sec", 0) or 0)
            agg[key]["plays"] += 1
            agg[key]["time"] += d
            agg[key]["durations"].append(d)

        def row_for(k, meta):
            avg = int(sum(meta["durations"])/len(meta["durations"])) if meta["durations"] else 0
            med = int(self._median(meta["durations"])) if meta["durations"] else 0
            return f"<tr><td>{k}</td><td align='right'>{self._fmt_hms(meta['time'])}</td><td align='right'>{self._fmt_int(meta['plays'])}</td><td align='right'>{self._fmt_hms(avg)}</td><td align='right'>{self._fmt_hms(med)}</td></tr>"

        by_time = sorted(agg.items(), key=lambda kv: (-(kv[1]["time"]), kv[0].lower()))[:10]
        by_plays = sorted(agg.items(), key=lambda kv: (-(kv[1]["plays"]), kv[0].lower()))[:10]
        least_time = [kv for kv in sorted(agg.items(), key=lambda kv: (kv[1]["time"], kv[0].lower())) if kv[1]["time"] > 0][:10]

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")

        def tbl(title, items):
            head = "<tr><th align='left'>Table</th><th align='right'>Playtime</th><th align='right'>Plays</th><th align='right'>Avg</th><th align='right'>Median</th></tr>"
            body = "".join([row_for(k, v) for k, v in items])
            return f"<h4>{title}</h4><table>{head}{body}</table>"

        return css + \
            tbl("Top 10 — Playtime", by_time) + \
            tbl("Top 10 — Plays", by_plays) + \
            tbl("Least Played by Time (> 0)", least_time)

    def _stats_render_trends(self, feats: list) -> str:
        # Weekly buckets
        per_week = defaultdict(lambda: {"plays": 0, "time": 0, "dur": []})
        for f in feats:
            wk = str(f.get("_week") or "")
            d = int(f.get("duration_sec", 0) or 0)
            per_week[wk]["plays"] += 1
            per_week[wk]["time"] += d
            per_week[wk]["dur"].append(d)

        weeks = sorted(per_week.keys())
        # Keep latest 12 weeks for the table
        if len(weeks) > 12:
            weeks = weeks[-12:]

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")
        rows = ["<tr><th align='left'>Week</th><th align='right'>Plays</th><th align='right'>Playtime</th><th align='right'>Avg Duration</th></tr>"]
        for wk in weeks:
            meta = per_week[wk]
            avg = int(sum(meta["dur"])/len(meta["dur"])) if meta["dur"] else 0
            rows.append(f"<tr><td>{wk}</td><td align='right'>{self._fmt_int(meta['plays'])}</td><td align='right'>{self._fmt_hms(meta['time'])}</td><td align='right'>{self._fmt_hms(avg)}</td></tr>")

        # Trending ROMs last 30d vs previous 30d
        now = datetime.now()
        d30 = now - timedelta(days=30)
        d60 = now - timedelta(days=60)
        play_30 = defaultdict(int)
        play_prev = defaultdict(int)
        for f in feats:
            dt = f.get("_dt")
            if not isinstance(dt, datetime):
                continue
            key = str(f.get("rom") or f.get("table") or "").strip()
            if not key:
                continue
            dur = int(f.get("duration_sec", 0) or 0)
            if d30 <= dt <= now:
                play_30[key] += dur
            elif d60 <= dt < d30:
                play_prev[key] += dur
        trending = []
        for k in set(list(play_30.keys()) + list(play_prev.keys())):
            cur = play_30.get(k, 0)
            prev = play_prev.get(k, 0)
            delta = cur - prev
            trending.append((k, cur, delta))
        trending.sort(key=lambda t: (-t[2], -t[1], t[0].lower()))
        trending = trending[:10]

        head = "<tr><th align='left'>ROM/Table</th><th align='right'>Playtime (30d)</th><th align='right'>Δ vs prev 30d</th></tr>"
        body = "".join([f"<tr><td>{k}</td><td align='right'>{self._fmt_hms(cur)}</td><td align='right'>{('+' if d>=0 else '')+self._fmt_hms(abs(d))}</td></tr>" for k, cur, d in trending])

        return css + "<h4>Plays per week (last 12)</h4><table>" + "".join(rows) + "</table>" + \
               "<h4>Trending ROMs (last 30 days vs previous 30 days)</h4><table>" + head + body + "</table>"


    def _stats_render_kpis(self, feats: list) -> str:
        # Collect per-minute KPIs
        def val(f, key):
            try: return float(f.get(key, 0) or 0)
            except Exception: return 0.0

        scorepm = [val(f, "score_per_min") for f in feats]
        precpm  = [val(f, "precision_per_min") for f in feats]
        mbpm    = [val(f, "multiball_per_min") for f in feats]
        jppm    = [val(f, "jackpots_per_min") for f in feats]
        twpm    = [val(f, "tilt_warnings_per_min") for f in feats]

        def stat_row(title, arr):
            if not arr:
                return f"<tr><td>{title}</td><td align='right'>-</td><td align='right'>-</td></tr>"
            p50 = self._percentile(arr, 50)
            p90 = self._percentile(arr, 90)
            return f"<tr><td>{title}</td><td align='right'>{p50:.2f}</td><td align='right'>{p90:.2f}</td></tr>"

        # Top 10 sessions by score/min
        top_spm = sorted(feats, key=lambda f: f.get("score_per_min", 0) or 0, reverse=True)[:10]

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")

        head1 = "<tr><th align='left'>Metric</th><th align='right'>P50</th><th align='right'>P90</th></tr>"
        body1 = "".join([
            stat_row("Score/min", scorepm),
            stat_row("Precision/min", precpm),
            stat_row("Multiball/min", mbpm),
            stat_row("Jackpots/min", jppm),
            stat_row("Tilt warnings/min", twpm),
        ])

        head2 = "<tr><th align='left'>When</th><th align='left'>Table</th><th align='right'>Score/min</th><th align='right'>Duration</th></tr>"
        body2 = []
        for f in top_spm:
            dt = f.get("_dt"); when = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else "-"
            table = str(f.get("table","") or f.get("rom","")).strip()
            spm = float(f.get("score_per_min", 0) or 0)
            dur = self._fmt_hms(int(f.get("duration_sec", 0) or 0))
            body2.append(f"<tr><td>{when}</td><td>{table}</td><td align='right'>{spm:.2f}</td><td align='right'>{dur}</td></tr>")

        return css + "<h4>Per-minute KPIs (P50/P90)</h4><table>" + head1 + body1 + "</table>" + \
               "<h4>Top 10 sessions by Score/min</h4><table>" + head2 + "".join(body2) + "</table>"
               
    def _stats_render_profiles(self, feats: list, sess: list) -> str:
        # Aggregate per ROM
        rom_agg = defaultdict(lambda: {"time": 0, "plays": 0, "dur": [], "scores": [], "spm": []})
        for f in feats:
            rom = str(f.get("rom") or f.get("table") or "").strip()
            if not rom:
                continue
            rom_agg[rom]["time"] += int(f.get("duration_sec", 0) or 0)
            rom_agg[rom]["plays"] += 1
            rom_agg[rom]["dur"].append(int(f.get("duration_sec", 0) or 0))
            if isinstance(f.get("score", None), (int, float)):
                rom_agg[rom]["scores"].append(int(f.get("score", 0) or 0))
            if isinstance(f.get("score_per_min", None), (int, float)):
                rom_agg[rom]["spm"].append(float(f.get("score_per_min", 0) or 0))

        # Event efficiency from sessions (global_deltas)
        eff = defaultdict(lambda: {"jackpots": 0, "multiball": 0, "modes_started": 0, "modes_completed": 0})
        for s in sess:
            rom = str(s.get("rom") or "").strip()
            if not rom:
                continue
            gd = s.get("global_deltas", {}) or {}
            def gi(label): 
                try: return int(gd.get(label, 0) or 0)
                except Exception: return 0
            eff[rom]["jackpots"] += gi("Jackpots")
            eff[rom]["multiball"] += gi("Total Multiballs")
            eff[rom]["modes_started"] += gi("Modes Started")
            eff[rom]["modes_completed"] += gi("Modes Completed")

        # Build rows sorted by total playtime desc
        items = []
        for rom, a in rom_agg.items():
            avg = int(sum(a["dur"])/len(a["dur"])) if a["dur"] else 0
            med = int(self._median(a["dur"])) if a["dur"] else 0
            best5 = sorted(a["scores"], reverse=True)[:5]
            best5_spm = sorted(a["spm"], reverse=True)[:5]
            e = eff.get(rom, {})
            jp = int(e.get("jackpots", 0)); mb = int(e.get("multiball", 0))
            ms = int(e.get("modes_started", 0)); mc = int(e.get("modes_completed", 0))
            ratio_jp_mb = (jp / mb) if mb > 0 else 0.0
            rate_modes = (mc / ms) if ms > 0 else 0.0
            items.append((rom, a["time"], a["plays"], avg, med, best5, best5_spm, ratio_jp_mb, rate_modes))
        items.sort(key=lambda t: (-t[1], t[0].lower()))

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}"
               ".muted{color:#777}</style>")
        head = ("<tr><th align='left'>ROM</th><th align='right'>Playtime</th><th align='right'>Plays</th>"
                "<th align='right'>Avg</th><th align='right'>Median</th>"
                "<th align='right'>Best Scores</th><th align='right'>Best Score/min</th>"
                "<th align='right'>Jackpots/MB</th><th align='right'>Mode Completion</th></tr>")
        body = []
        for rom, tot, plays, avg, med, best, bestspm, rj, rm in items[:50]:
            def s_list(nums, fmt="{:,d}"):
                if not nums: return "-"
                return ", ".join([fmt.format(int(x)).replace(",", ".") if isinstance(x, (int, float)) and fmt=="{:,d}" else f"{float(x):.2f}" for x in nums])
            body.append(
                f"<tr>"
                f"<td>{rom}</td>"
                f"<td align='right'>{self._fmt_hms(tot)}</td>"
                f"<td align='right'>{self._fmt_int(plays)}</td>"
                f"<td align='right'>{self._fmt_hms(avg)}</td>"
                f"<td align='right'>{self._fmt_hms(med)}</td>"
                f"<td align='right'>{s_list(best)}</td>"
                f"<td align='right'>{s_list(bestspm, fmt='float')}</td>"
                f"<td align='right'>{rj:.2f}</td>"
                f"<td align='right'>{rm:.2f}</td>"
                f"</tr>"
            )

        return css + "<h4>Per-ROM profiles (Top 50 by playtime)</h4><table>" + head + "".join(body) + "</table>"               


    def _stats_render_multiplayer(self, sess: list) -> str:
        solo = 0; multi = 0
        wins = 0; total_multi = 0
        diffs = []
        for s in sess:
            players = s.get("players", []) or []
            n_active = sum(1 for p in players if int(p.get("playtime_sec", 0) or 0) > 0)
            if n_active <= 1:
                solo += 1
                continue
            multi += 1
            # P1 win by end_audits scores
            end = s.get("end_audits", {}) or {}
            def gi(k):
                try: return int(end.get(k, 0) or 0)
                except Exception: return 0
            p1 = gi("P1 Score")
            others = max(gi("P2 Score"), gi("P3 Score"), gi("P4 Score"))
            if (p1 > 0 or others > 0):
                total_multi += 1
                if p1 >= others:
                    wins += 1
                diffs.append(p1 - others)

        win_rate = (wins / total_multi * 100.0) if total_multi > 0 else 0.0
        avg_diff = (sum(diffs)/len(diffs)) if diffs else 0.0

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")
        rows = [
            f"<tr><td>Solo sessions</td><td align='right'>{self._fmt_int(solo)}</td></tr>",
            f"<tr><td>Multiplayer sessions</td><td align='right'>{self._fmt_int(multi)}</td></tr>",
            f"<tr><td>P1 win rate (multiplayer)</td><td align='right'>{win_rate:.1f}%</td></tr>",
            f"<tr><td>Avg P1 score diff vs best opponent</td><td align='right'>{self._fmt_int(int(avg_diff))}</td></tr>",
        ]
        return css + "<h4>Multiplayer summary</h4><table>" + "".join(rows) + "</table>"

    def _stats_render_map_whitelist(self, sess: list, maps_index: dict) -> str:
        # Avg whitelist size per ROM
        wl = defaultdict(lambda: {"sum": 0, "n": 0})
        for s in sess:
            rom = str(s.get("rom") or "").strip()
            if not rom:
                continue
            w = int(s.get("whitelist_size", 0) or 0)
            wl[rom]["sum"] += w
            wl[rom]["n"] += 1

        items = []
        for rom, meta in wl.items():
            avg = (meta["sum"]/meta["n"]) if meta["n"]>0 else 0.0
            mtype = (maps_index.get(rom, {}) or {}).get("type", "unknown")
            items.append((rom, meta["n"], avg, mtype))
        # Sort: most sessions, then avg wl size desc
        items.sort(key=lambda t: (-t[1], -t[2], t[0].lower()))

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")
        head = "<tr><th align='left'>ROM</th><th align='right'>Sessions</th><th align='right'>Avg Whitelist Size</th><th align='left'>Map Type</th></tr>"
        body = "".join([
            f"<tr><td>{rom}</td><td align='right'>{self._fmt_int(n)}</td><td align='right'>{avg:.1f}</td><td>{mtype}</td></tr>"
            for rom, n, avg, mtype in items[:100]
        ])
        return css + "<h4>Map provenance & whitelist quality</h4><table>" + head + body + "</table>"

    def _stats_render_records(self, feats: list, sess: list) -> str:
        # Longest sessions
        longest = sorted(feats, key=lambda f: int(f.get("duration_sec", 0) or 0), reverse=True)[:10]
        # Highest scores (globally)
        hi_scores = [f for f in feats if isinstance(f.get("score", None), (int, float))]
        hi_scores.sort(key=lambda f: int(f.get("score", 0) or 0), reverse=True)
        hi_scores = hi_scores[:10]

        # Best ball (from features if present, fall back to sessions)
        best_balls = []
        for f in feats:
            bb = f.get("best_ball")
            if isinstance(bb, dict):
                best_balls.append((f, bb))
        if not best_balls:
            for s in sess:
                bb = s.get("best_ball")
                if isinstance(bb, dict):
                    # make a feature-like shell
                    f = {"_dt": s.get("_dt"), "table": s.get("table"), "rom": s.get("rom")}
                    best_balls.append((f, bb))
        # sort by score then duration
        best_balls.sort(key=lambda t: (int(t[1].get("score", 0) or 0), int(t[1].get("duration", 0) or 0)), reverse=True)
        best_balls = best_balls[:10]

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")

        def tbl(title, head, rows):
            return f"<h4>{title}</h4><table>{head}{''.join(rows)}</table>"

        # Longest
        head1 = "<tr><th align='left'>When</th><th align='left'>Table</th><th align='right'>Duration</th></tr>"
        rows1 = []
        for f in longest:
            dt = f.get("_dt"); when = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else "-"
            table = str(f.get("table","") or f.get("rom","")).strip()
            rows1.append(f"<tr><td>{when}</td><td>{table}</td><td align='right'>{self._fmt_hms(int(f.get('duration_sec',0) or 0))}</td></tr>")

        # Highest scores
        head2 = "<tr><th align='left'>When</th><th align='left'>Table</th><th align='right'>Score</th></tr>"
        rows2 = []
        for f in hi_scores:
            dt = f.get("_dt"); when = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else "-"
            table = str(f.get("table","") or f.get("rom","")).strip()
            rows2.append(f"<tr><td>{when}</td><td>{table}</td><td align='right'>{self._fmt_int(int(f.get('score',0) or 0))}</td></tr>")

        # Best Ball
        head3 = "<tr><th align='left'>When</th><th align='left'>Table</th><th align='right'>Ball #</th><th align='right'>Score</th><th align='right'>Duration</th></tr>"
        rows3 = []
        for f, bb in best_balls:
            dt = f.get("_dt"); when = dt.strftime("%Y-%m-%d %H:%M") if isinstance(dt, datetime) else "-"
            table = str(f.get("table","") or f.get("rom","")).strip()
            num = int(bb.get("num", 0) or 0)
            sc = self._fmt_int(int(bb.get("score", 0) or 0))
            dur = self._fmt_hms(int(bb.get("duration", 0) or 0))
            rows3.append(f"<tr><td>{when}</td><td>{table}</td><td align='right'>{num}</td><td align='right'>{sc}</td><td align='right'>{dur}</td></tr>")

        return css + tbl("Longest sessions", head1, rows1) + tbl("Highest scores", head2, rows2) + tbl("Best ball (score & duration)", head3, rows3)


    def _stats_render_manu_era(self, feats: list) -> str:
        per_maker = defaultdict(lambda: {"plays": 0, "time": 0})
        per_decade = defaultdict(lambda: {"plays": 0, "time": 0})
        for f in feats:
            maker, year = self._extract_maker_year(str(f.get("table","")))
            d = int(f.get("duration_sec", 0) or 0)
            per_maker[maker]["plays"] += 1
            per_maker[maker]["time"] += d
            if isinstance(year, int):
                decade = f"{year//10*10}s"
                per_decade[decade]["plays"] += 1
                per_decade[decade]["time"] += d

        makers = sorted(per_maker.items(), key=lambda kv: (-kv[1]["time"], kv[0].lower()))[:20]
        decades = sorted(per_decade.items(), key=lambda kv: (-kv[1]["time"], kv[0]))

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")
        def tbl(title, items):
            head = "<tr><th align='left'>Group</th><th align='right'>Playtime</th><th align='right'>Plays</th></tr>"
            body = "".join([f"<tr><td>{k or '(unknown)'}</td><td align='right'>{self._fmt_hms(v['time'])}</td><td align='right'>{self._fmt_int(v['plays'])}</td></tr>" for k, v in items])
            return f"<h4>{title}</h4><table>{head}{body}</table>"

        return css + tbl("Manufacturer (Top 20 by playtime)", makers) + tbl("Decade", decades)

    def _stats_render_misc(self, feats: list) -> str:
        # Sessions/day last 90 days
        now = datetime.now()
        d90 = now - timedelta(days=90)
        per_day = defaultdict(int)
        for f in feats:
            dt = f.get("_dt")
            if not isinstance(dt, datetime): 
                continue
            if dt >= d90:
                per_day[dt.date().isoformat()] += 1
        days = sorted(per_day.items(), key=lambda kv: kv[0])  # asc by date

        # Avg sessions per table
        by_table = defaultdict(int)
        for f in feats:
            key = str(f.get("table") or f.get("rom") or "").strip()
            if key:
                by_table[key] += 1
        avg_per_table = (sum(by_table.values())/len(by_table)) if by_table else 0.0

        # Highest median duration tables (top 10, min 3 plays)
        dur_by_table = defaultdict(list)
        for f in feats:
            key = str(f.get("table") or f.get("rom") or "").strip()
            if key:
                dur_by_table[key].append(int(f.get("duration_sec", 0) or 0))
        med_items = []
        for k, arr in dur_by_table.items():
            if len(arr) >= 3:
                med_items.append((k, self._median(arr), len(arr)))
        med_items.sort(key=lambda t: (-t[1], -t[2], t[0].lower()))
        med_items = med_items[:10]

        css = ("<style>table{border-collapse:collapse;width:100%}"
               "th,td{padding:6px 8px;border-bottom:1px solid #e5e5e5;white-space:nowrap}</style>")

        # Sessions/day (compact table)
        head1 = "<tr><th align='left'>Date</th><th align='right'>Sessions</th></tr>"
        rows1 = "".join([f"<tr><td>{d}</td><td align='right'>{self._fmt_int(n)}</td></tr>" for d, n in days[-30:]])

        # Highest median duration
        head2 = "<tr><th align='left'>Table</th><th align='right'>Median</th><th align='right'>Plays</th></tr>"
        rows2 = "".join([f"<tr><td>{k}</td><td align='right'>{self._fmt_hms(int(m))}</td><td align='right'>{self._fmt_int(n)}</td></tr>" for k, m, n in med_items])

        cards = (
            "<div class='cards' style='display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 10px 0'>"
            f"<div class='card' style='background:#141414;color:#fff;padding:12px 16px;border-radius:10px;min-width:210px'>"
            f"<div class='muted' style='color:#bbb'>Avg Sessions per Table</div>"
            f"<div style='font-weight:700;font-size:18px'>{avg_per_table:.2f}</div></div></div>"
        )

        return css + "<h4>Sessions per day (last 30)</h4><table>" + head1 + rows1 + "</table>" + \
               cards + "<h4>Highest median duration (min 3 plays)</h4><table>" + head2 + rows2 + "</table>"



    def quit_all(self):
        self.cfg.save()
        # Tray zuerst verstecken, damit closeEvent NICHT in "hide to tray" fällt
        try:
            if self.tray:
                self.tray.hide()
        except Exception:
            pass
        # Watcher explicit stoppen (inkl. Injector-Kill)
        try:
            if getattr(self, "watcher", None):
                self.watcher.stop()
        except Exception:
            pass
        # GUI schließen und App beenden
        try:
            self.close()
        except Exception:
            pass
        try:
            QApplication.instance().quit()
        except Exception:
            pass

            


    def _prefetch_maps_now(self):
        """
        Start prefetch in the background and show the dynamic target path (BASE\\NVRAM_Maps\\maps).
        """
        try:
            self.watcher.start_prefetch_background()
            maps_dir = os.path.join(self.cfg.BASE, "NVRAM_Maps", "maps")
            QMessageBox.information(
                self, "Prefetch",
                f"Prefetch started. Missing maps are being cached in the background at:\n"
                f"{maps_dir}\n"
                "See watcher.log for progress."
            )
            log(self.cfg, "[PREFETCH] started by user")
        except Exception as e:
            log(self.cfg, f"[PREFETCH] failed: {e}", "ERROR")
            QMessageBox.warning(self, "Prefetch", f"Prefetch failed:\n{e}")
  
          
    # --- Theme / Tooltips ---
    def _style(self, widget, css: str):
        try:
            if widget:
                widget.setStyleSheet(css)
        except Exception:
            pass

    def _apply_theme(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        p = app.palette()
        p.setColor(p.ColorRole.Window, QColor("#f3f3f3"))
        p.setColor(p.ColorRole.WindowText, QColor("#202020"))
        p.setColor(p.ColorRole.Base, QColor("#ffffff"))
        p.setColor(p.ColorRole.Text, QColor("#202020"))
        p.setColor(p.ColorRole.Button, QColor("#e5e5e5"))
        p.setColor(p.ColorRole.ButtonText, QColor("#202020"))
        p.setColor(p.ColorRole.Highlight, QColor("#0078d7"))
        p.setColor(p.ColorRole.HighlightedText, QColor("#ffffff"))
        app.setPalette(p)
        app.setFont(QFont("Segoe UI", 10))

        # Buttons
        self._style(getattr(self, "btn_minimize", None),
                    "background:#0078d7;color:white;border-radius:6px;padding:6px 12px;")
        self._style(getattr(self, "btn_quit", None),
                    "background:#7a7a7a;color:white;border-radius:6px;padding:6px 12px;")
        self._style(getattr(self, "btn_restart", None),
                    "background:#00cc6a;color:white;border-radius:6px;padding:6px 12px;")

        # SICHER: Icon überall setzen (Fenster + Tray) – bevorzugt watcher.ico
        try:
            icon = self._get_icon()
            self.setWindowIcon(icon)
            if getattr(self, "tray", None):
                # explizit setzen, auch wenn bereits im Konstruktor übergeben
                self.tray.setIcon(icon)
        except Exception:
            pass

    def _init_tooltips_main(self):
        tips = {
            "btn_restart": "Restart the internal watcher thread.",
            "btn_quit": "Exit the GUI (will stop watcher and capture).",
            "btn_minimize": "Minimize window to tray (if available).",
            "status_label": "Watcher status label."
        }
        apply_tooltips(self, tips)

    def _init_overlay_tooltips(self):
        """
        Sets tooltips for controls in the Overlay tab.
        All text in English.
        """
        tips = {
            "chk_portrait": "Rotate and render the overlay in portrait (90° CCW).",
            "sld_scale": "Overall scaling factor of the overlay window (percent).",
            "lbl_scale": "Current overlay scale in percent.",
            "chk_use_xy": "If enabled, use fixed X/Y coordinates instead of automatic placement.",
            "spn_x": "Overlay X position (used only if 'Use X/Y' is enabled).",
            "spn_y": "Overlay Y position (used only if 'Use X/Y' is enabled).",
            "cmb_toggle_src": "Input source for the overlay toggle hotkey (keyboard or joystick).",
            "btn_bind_toggle": "Bind a key or joystick button to toggle the overlay visibility.",
            "lbl_toggle_binding": "Displays the currently bound toggle key/button.",
            "btn_col_title": "Pick the title (header) color.",
            "lbl_col_title": "Current title color.",
            "btn_col_high": "Pick the highlight text color.",
            "lbl_col_high": "Current highlight text color.",
            "lbl_player1_color": "Current color for Player 1 highlight section.",
            "lbl_player2_color": "Current color for Player 2 highlight section.",
            "lbl_player3_color": "Current color for Player 3 highlight section.",
            "lbl_player4_color": "Current color for Player 4 highlight section.",
            "cmb_font_family": "Font family used for title and body text.",
            "spn_font_size": "Base body font size (title & hint sizes derive from this).",
            "btn_toggle_now": "Show or hide the overlay immediately (test).",
            "btn_hide": "Hide the overlay if it is visible.",
        }
        apply_tooltips(self, tips)
 
    def _build_achievements_tab(self):
        """
        Create the main 'Achievements' tab with two subtabs:
          - 'Global-NVRAM': per ROM, only achievements from global_achievements.json
          - 'PL-Achievements': per ROM, all persisted session achievements (ROM-specific/custom)
        Tooltips are in English.
        """
        # Ensure we have the main tab widget
        if not hasattr(self, "main_tabs") or self.main_tabs is None:
            return

        ach_tab = QWidget()
        ach_layout = QVBoxLayout(ach_tab)

        self.ach_tabs = QTabWidget()
        self.ach_view_global = QTextBrowser()
        self.ach_view_pl = QTextBrowser()

        # Tooltips in English
        self.ach_tabs.setToolTip("Shows unlocked achievements across all ROMs.")
        self.ach_view_global.setToolTip("Global achievements unlocked from global_achievements.json (per ROM, one-time).")
        self.ach_view_pl.setToolTip("Player-level session achievements (ROM-specific/custom), recorded only from 1-player sessions.")

        self.ach_tabs.addTab(self.ach_view_global, "Global-NVRAM")
        self.ach_tabs.addTab(self.ach_view_pl, "PL-Achievements")
        ach_layout.addWidget(self.ach_tabs)
        ach_tab.setLayout(ach_layout)

        # Add as a top-level tab
        self.main_tabs.addTab(ach_tab, "Achievements")

        # Initial fill
        try:
            self.update_achievements_tab()
        except Exception:
            pass

    def update_achievements_tab(self):
        """
        Build HTML for:
          - Global-NVRAM: per ROM only achievements with origin='global_achievements'
          - PL-Achievements: per ROM all session achievements (1-player only, persisted)
        Renders one column per ROM from left to right.
        """
        state = load_json(f_achievements_state(self.cfg), {}) or {}
        global_map = state.get("global", {}) or {}
        session_map = state.get("session", {}) or {}

        def build_columns_html(data_map: dict, filter_origin_ga_only: bool = False) -> str:
            roms = sorted(data_map.keys(), key=lambda s: str(s).lower())
            if not roms:
                return "<div>(no data)</div>"
            cols = []
            for rom in roms:
                entries = data_map.get(rom, []) or []
                items = []
                for e in entries:
                    if isinstance(e, dict):
                        title = str(e.get("title", "")).strip()
                        origin = str(e.get("origin", "")).strip()
                    else:
                        title = str(e).strip()
                        origin = ""
                    if not title:
                        continue
                    if filter_origin_ga_only and origin and origin != "global_achievements":
                        continue
                    items.append(title)
                if filter_origin_ga_only and not items:
                    continue  # skip empty ROM column when filtering
                lines = [f"<div style='font-weight:700;margin-bottom:4px;'>{rom}</div>"]
                if items:
                    for it in items:
                        lines.append(f"<div style='margin:2px 0;'>{it}</div>")
                else:
                    lines.append("<div style='color:#888;'>–</div>")
                cols.append("".join(lines))
            if not cols:
                return "<div>(no data)</div>"
            html = "<table width='100%'><tr>" + "".join(
                f"<td valign='top' style='padding:0 14px;'>{c}</td>" for c in cols
            ) + "</tr></table>"
            return html

        # Global-NVRAM (only origin=global_achievements)
        try:
            html_g = build_columns_html(global_map, filter_origin_ga_only=True)
            self.ach_view_global.setHtml(html_g)
        except Exception:
            pass

        # PL-Achievements (all session unlocks)
        try:
            html_pl = build_columns_html(session_map, filter_origin_ga_only=False)
            self.ach_view_pl.setHtml(html_pl)
        except Exception:
            pass

    def _init_achievements_timer(self):
        """
        Set up a small timer to refresh the Achievements tab periodically.
        """
        try:
            self.timer_achievements = QTimer(self)
            self.timer_achievements.setInterval(5000)  # 5 seconds
            self.timer_achievements.timeout.connect(self.update_achievements_tab)
            self.timer_achievements.start()
        except Exception:
            pass 
 
 
    def _get_icon(self) -> QIcon:
        """
        Use ONLY 'watcher.ico' shipped with the app (bundle or next to the EXE).
        """
        # 1) Bundle (PyInstaller) – resource_path
        try:
            p = resource_path("watcher.ico")
            if os.path.isfile(p):
                ic = QIcon(p)
                if not ic.isNull():
                    return ic
        except Exception:
            pass

        # 2) Neben der EXE (Dev/Portable)
        try:
            p2 = os.path.join(APP_DIR, "watcher.ico")
            if os.path.isfile(p2):
                ic = QIcon(p2)
                if not ic.isNull():
                    return ic
        except Exception:
            pass

        # 3) Minimaler Fallback (sollte nicht eintreten, wenn watcher.ico korrekt mitgeliefert ist)
        pm = QPixmap(32, 32)
        pm.fill(Qt.GlobalColor.transparent)
        try:
            painter = QPainter(pm)
            painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing, True)
            painter.setBrush(QColor("#202020"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(1, 1, 30, 30, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.drawText(pm.rect(), int(Qt.AlignmentFlag.AlignCenter), "AW")
            painter.end()
        except Exception:
            pm.fill(QColor("#202020"))
        return QIcon(pm)

    def _on_portrait_ccw_toggle(self, state: int):
        is_ccw = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.cfg.OVERLAY["portrait_rotate_ccw"] = is_ccw
        self.cfg.save()
        if self.overlay:
            self.overlay.apply_portrait_from_cfg(self.cfg.OVERLAY)
            self.overlay.request_rotation(force=True)
            
    def _show_from_tray(self):
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def closeEvent(self, event):
        """
        Clean shutdown: save config, uninstall keyboard hook, unregister WM_HOTKEY, stop watcher thread.
        """
        self.cfg.save()
        try:
            if getattr(self, "tray", None) and self.tray and self.tray.isVisible():
                self.hide()
                event.ignore()
                return
        except Exception:
            pass
        try:
            self._unregister_global_hotkeys()
        except Exception:
            pass
        try:
            self._uninstall_global_keyboard_hook()
        except Exception:
            pass
        try:
            if getattr(self, "watcher", None):
                self.watcher.stop()
        except Exception:
            pass
        event.accept()

    def change_base(self):
        d = QFileDialog.getExistingDirectory(self, "Select BASE directory", self.cfg.BASE)
        if d:
            self.cfg.BASE = d
            self.base_label.setText(f"BASE: {d}")
            self.cfg.save()

    def change_nvram(self):
        d = QFileDialog.getExistingDirectory(self, "Select NVRAM directory", self.cfg.NVRAM_DIR)
        if d:
            self.cfg.NVRAM_DIR = d
            self.nvram_label.setText(f"NVRAM: {d}")
            self.cfg.save()

    def change_tables(self):
        d = QFileDialog.getExistingDirectory(self, "Select TABLES directory", self.cfg.TABLES_DIR)
        if d:
            self.cfg.TABLES_DIR = d
            self.tables_label.setText(f"TABLES (optional): {d}")
            self.cfg.save()

    # --- Logs / Stats ---
    def _latest_log_path(self) -> Optional[str]:
        watcher_path = f_log(self.cfg)
        cands = []
        if os.path.isfile(watcher_path):
            cands.append(watcher_path)
        cands.extend(glob.glob(os.path.join(self.cfg.BASE, "*.log")))
        if not cands:
            return None
        return max(cands, key=lambda x: os.path.getmtime(x))

    def _update_logs(self):
        latest = self._latest_log_path()
        if not latest:
            return
        try:
            # Scroll-Position merken (proportional)
            v = self.log_view.verticalScrollBar()
            old_val = v.value()
            old_max = max(1, v.maximum())
            at_bottom_before = (old_val >= old_max - 2)
            ratio = old_val / old_max if old_max > 0 else 0.0

            with open(latest, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            self.log_view.setPlainText(content)

            # Scroll-Position wiederherstellen
            new_max = max(1, v.maximum())
            if at_bottom_before:
                v.setValue(v.maximum())
            else:
                new_val = int(round(ratio * new_max))
                v.setValue(max(0, min(new_val, new_max)))
        except Exception:
            pass

    def _extract_block(self, text: str, header: str) -> str:
        """
        Parse a '=== <header> ===' block into HTML.
        UPDATED: Skips achievement sections entirely:
          - "Achievements (unlocked)"
          - "Session Achievements"
        Renders remaining sections (e.g., 'Audits (filtered)', 'Session Deltas', etc.) into columns.
        """
        lines = text.splitlines()
        block = []
        capture = False
        for line in lines:
            s = line.strip()
            if s.startswith(f"=== {header} ==="):
                capture = True
                block = []
                continue
            if capture and s.startswith("===") and not s.startswith(f"=== {header} ==="):
                break
            if capture:
                block.append(line)

        if not block:
            return f"<p>No data found for {header}</p>"

        style = """
        <style>
        table { border-collapse: collapse; }
        .inner td { padding: 3px 6px; white-space:nowrap; }
        .inner td:first-child { text-align: left; }
        .inner td:last-child { text-align: right; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        h3 { margin-top: 12px; }
        h4 { margin-top: 10px; margin-bottom: 4px; border-bottom: 1px solid #ccc; }
        </style>
        """
        html = style + f"<h3>{header}</h3>"

        current_section = None
        rows: List[Tuple[str, str]] = []
        skip_section = False

        def flush():
            nonlocal html, rows, current_section
            if rows:
                sec_title = current_section or ""
                if sec_title:
                    html += f"<h4>{sec_title}</h4>"
                html += self._render_multi_columns(rows, 4)
                rows = []

        for raw in block:
            stripped = raw.rstrip()
            if not stripped:
                continue

            st = stripped.strip()

            # Section header?
            if st.endswith(":"):
                tag = st[:-1].strip()
                flush()
                low = tag.lower()
                if low in ("achievements (unlocked)", "session achievements"):
                    current_section = None
                    skip_section = True
                else:
                    current_section = tag
                    skip_section = False
                continue

            if skip_section:
                continue

            # Parse rows
            parts = st.split()
            if len(parts) >= 2:
                key = " ".join(parts[:-1])
                val = parts[-1]
                rows.append((key, val))
            else:
                rows.append((st, ""))

        flush()
        return html


    def _read_latest_session_txt(self) -> str:
        """
        Reads the latest *.txt from BASE\\session_stats or returns "".
        """
        stats_dir = os.path.join(self.cfg.BASE, "session_stats")
        if not os.path.isdir(stats_dir):
            return ""
        try:
            txt_files = [os.path.join(stats_dir, fn) for fn in os.listdir(stats_dir)
                         if fn.lower().endswith(".txt")]
            if not txt_files:
                return ""
            latest = max(txt_files, key=lambda p: os.path.getmtime(p))
            with open(latest, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


    @staticmethod
    def _render_multi_columns(rows: List[Tuple[str, str]], columns: int) -> str:
        if columns <= 0:
            columns = 1
        per_col = (len(rows) + columns - 1) // columns
        html = "<table width='100%'><tr>"
        for c in range(columns):
            start = c * per_col
            end = start + per_col
            col_rows = rows[start:end]
            html += "<td valign='top'><table class='inner'>"
            for k, v in col_rows:
                html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            html += "</table></td>"
        html += "</tr></table>"
        return html




    def _build_global_html_nonzero(self, content: str) -> str:
        """
        Build HTML for Global Snapshot (page body) without achievements.
        UPDATED: Only render 'Audits (filtered)' non-zero rows.
        """
        if not content:
            return "<div>(no data)</div>"

        lines = content.splitlines()
        in_global = False
        in_audits = False
        rows: List[Tuple[str, str]] = []

        for raw in lines:
            s = raw.rstrip()
            st = s.strip()

            if st.startswith("=== Global Snapshot ==="):
                in_global = True
                in_audits = False
                continue
            if in_global and st.startswith("===") and not st.startswith("=== Global Snapshot ==="):
                break
            if not in_global:
                continue

            if st.endswith(":"):
                tag = st[:-1].strip().lower()
                if tag == "achievements (unlocked)":
                    in_audits = False
                    continue
                in_audits = (tag == "audits (filtered)")
                continue

            if in_audits:
                parts = st.split()
                if len(parts) >= 2:
                    key = " ".join(parts[:-1])
                    val = parts[-1]
                    try:
                        ival = int(val)
                    except Exception:
                        continue
                    if ival != 0:
                        rows.append((key, val))
                continue

        style = """
        <style>
          table { border-collapse: collapse; }
          .inner td { padding: 3px 6px; white-space:nowrap; }
          .inner td:first-child { text-align: left; }
          .inner td:last-child { text-align: right; }
          tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
        """

        parts: List[str] = [style]
        if rows:
            parts.append("<div style='font-weight:600;color:#FFFFFF;margin:4px 0 4px 0;'>Audits (filtered)</div>")
            parts.append(self._render_multi_columns(rows, 4))
        else:
            parts.append("<div>(no data)</div>")

        return "".join(parts)

    def _parse_player_snapshot(self, content: str, pid: int) -> dict:
        """
        Parse '=== Player X Snapshot ===' block into:
          { 'playtime': str, 'achievements': [str], 'deltas': List[Tuple[label, val]] }
        """
        out = {"playtime": "", "achievements": [], "deltas": []}
        if not content:
            return out

        lines = content.splitlines()
        in_block = False
        in_achs = False
        in_deltas = False

        for raw in lines:
            s = raw.rstrip()   # rechts trimmen, links beibehalten (wichtig für Einrückungen)
            st = s.strip()
            if st.startswith(f"=== Player {pid} Snapshot ==="):
                in_block = True
                in_achs = False
                in_deltas = False
                continue
            if in_block and st.startswith("===") and not st.startswith(f"=== Player {pid} Snapshot ==="):
                break
            if not in_block:
                continue

            if st.lower().startswith("playtime:"):
                out["playtime"] = st.partition(":")[2].strip()
                continue

            if st.endswith(":"):
                t = st[:-1].strip().lower()
                in_achs = (t == "session achievements")
                in_deltas = (t == "session deltas")
                continue

            # ACHTUNG: auf s (mit führenden Spaces) prüfen, nicht auf st
            if in_achs and st:
                if (s.startswith("  ") or s.startswith("\t")):
                    out["achievements"].append(st)
                continue

            if in_deltas and st:
                if (s.startswith("  ") or s.startswith("\t")):
                    # lines like "  Ramps Made              12"
                    parts = st.split()
                    if len(parts) >= 2:
                        key = " ".join(parts[:-1])
                        val = parts[-1]
                        try:
                            ival = int(val)
                            if ival > 0:
                                out["deltas"].append((key, val))
                        except Exception:
                            pass
                continue

        return out

    def _build_player_snapshots_html(self, content: str) -> str:
        """
        Structured columns for P1–P4 and optional CPU.
        UPDATED: No 'Session Achievements' rendering – only Playtime and Session Deltas.
        """
        payloads = read_active_players(self.cfg.BASE) or []

        part_ids = set()
        for p in payloads:
            try:
                pid = int(p.get("id", 0) or 0)
            except Exception:
                pid = 0
            if pid < 1 or pid > 4:
                continue
            played = False
            try:
                if int(p.get("playtime_sec", 0) or 0) > 0:
                    played = True
            except Exception:
                pass
            try:
                if int(p.get("score", 0) or 0) > 0:
                    played = True
            except Exception:
                pass
            h = p.get("highlights", {}) or {}
            if any(h.get(cat) for cat in ("Power", "Precision", "Fun")):
                played = True
            if played:
                part_ids.add(pid)

        if not part_ids:
            part_ids = {1}

        def parse_player(pid: int) -> dict:
            out = {"playtime": "", "deltas": []}
            if not content:
                return out
            lines = content.splitlines()
            in_block = False; in_deltas = False; in_achs = False
            for raw in lines:
                s = raw.rstrip()
                st = s.strip()
                if st.startswith(f"=== Player {pid} Snapshot ==="):
                    in_block = True; in_deltas = False; in_achs = False
                    continue
                if in_block and st.startswith("===") and not st.startswith(f"=== Player {pid} Snapshot ==="):
                    break
                if not in_block:
                    continue
                if st.lower().startswith("playtime:"):
                    out["playtime"] = st.partition(":")[2].strip()
                    continue
                if st.endswith(":"):
                    tag = st[:-1].strip().lower()
                    in_achs = (tag == "session achievements")
                    in_deltas = (tag == "session deltas")
                    continue
                if in_achs:
                    continue
                if in_deltas and st and (s.startswith("  ") or s.startswith("\t")):
                    parts = st.split()
                    if len(parts) >= 2:
                        key = " ".join(parts[:-1]); val = parts[-1]
                        try:
                            ival = int(val)
                            if ival > 0:
                                out["deltas"].append((key, val))
                        except Exception:
                            pass
            return out

        def pcolor(pid: int) -> str:
            if 1 <= pid <= 4:
                return self.cfg.OVERLAY.get(f"player{pid}_color", "#00B050")
            return "#7A7A7A"

        def esc(x: Any) -> str:
            return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def col_html(title: str, color: str, playtime: str, deltas: list) -> str:
            parts = []
            if title:
                parts.append(f"<div style='font-weight:700;color:{color};margin-bottom:4px;'>{esc(title)}</div>")
            if playtime:
                parts.append("<div style='font-weight:600;color:#FFFFFF;margin:4px 0 2px 0;'>Playtime</div>")
                parts.append(f"<div style='color:#FFFFFF'>{esc(playtime)}</div>")
            if deltas:
                parts.append("<div style='font-weight:600;color:#FFFFFF;margin:6px 0 2px 0;'>Session deltas</div>")
                parts.append(self._render_multi_columns(deltas, 2))
            if not deltas and not playtime:
                parts.append("<div style='color:#888;'>–</div>")
            return "".join(parts)

        cols: list[str] = []
        for pid in sorted(part_ids):
            d = parse_player(pid)
            cols.append(col_html(f"P{pid}", pcolor(pid), d.get("playtime", ""), d.get("deltas", [])))

        # Optional CPU column (no achievements)
        try:
            sim = getattr(self.watcher, "cpu", {}) or {}
            if bool(sim.get("active", False)):
                from datetime import timedelta
                play = int(sim.get("active_play_seconds", 0) or 0)
                deltas = sim.get("session_deltas", {}) or {}
                rows = []
                try:
                    sorted_items = sorted(deltas.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]).lower()))
                except Exception:
                    sorted_items = list(deltas.items())
                for k, v in sorted_items:
                    try:
                        ival = int(v or 0)
                    except Exception:
                        ival = 0
                    if ival > 0:
                        rows.append((str(k), str(ival)))
                cols.append(col_html("CPU", pcolor(5), str(timedelta(seconds=play)), rows))
        except Exception:
            pass

        style = """
        <style>
          table { border-collapse: collapse; }
          .inner td { padding: 3px 6px; white-space:nowrap; }
          .inner td:first-child { text-align: left; }
          .inner td:last-child { text-align: right; }
          tr:nth-child(even) { background-color: #f9f9f9; }
        </style>
        """
        table = "<table width='100%'><tr>" + "".join(f"<td valign='top' style='padding:0 14px;'>{c}</td>" for c in cols) + "</tr></table>"
        return style + table



    def update_stats(self):
        """
        Aktualisiert die Inhalte des 'Stats'-Reiters:
          - Global/Spieler 1..4 aus der neuesten Session-Textdatei (Scrollposition behalten)
          - CPU-Sim live aus watcher.cpu (Scrollposition behalten)
        """
        stats_dir = os.path.join(self.cfg.BASE, "session_stats")

        # CPU-Tab immer live aktualisieren (auch wenn keine Session-TXTs existieren)
        if not os.path.isdir(stats_dir):
            try:
                self._update_cpu_stats_tab()
            except Exception:
                pass
            return

        # Letzte Session-Textdatei laden
        try:
            txt_files = [os.path.join(stats_dir, fn) for fn in os.listdir(stats_dir)
                         if fn.lower().endswith(".txt")]
            if txt_files:
                latest = max(txt_files, key=lambda p: os.path.getmtime(p))
                with open(latest, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                content = ""
        except Exception:
            content = ""

        # Hilfsfunktion: HTML setzen und Scrollposition bewahren
        def _set_html_preserve_scroll(browser: QTextBrowser, html: str):
            try:
                sb = browser.verticalScrollBar()
                old_val = sb.value()
                old_max = max(1, sb.maximum())
                at_bottom_before = (old_val >= old_max - 2)
                ratio = old_val / old_max if old_max > 0 else 0.0

                browser.setHtml(html)

                new_max = max(1, sb.maximum())
                if at_bottom_before:
                    sb.setValue(sb.maximum())
                else:
                    new_val = int(round(ratio * new_max))
                    sb.setValue(max(0, min(new_val, new_max)))
            except Exception:
                try:
                    browser.setHtml(html)
                except Exception:
                    pass

        # Global-Block
        try:
            if "global" in self.stats_views:
                html = self._extract_block(content, "Global Snapshot")
                _set_html_preserve_scroll(self.stats_views["global"], html)
        except Exception:
            pass

        # Spieler 1..4
        for i in range(1, 5):
            try:
                if i in self.stats_views:
                    html = self._extract_block(content, f"Player {i} Snapshot")
                    _set_html_preserve_scroll(self.stats_views[i], html)
            except Exception:
                pass

        # CPU-Sim live anzeigen
        try:
            self._update_cpu_stats_tab()
        except Exception:
            pass

        # AI Evaluation Tab ebenfalls refreshen
        try:
            self.update_ai_evaluation_tab()
        except Exception:
            pass




    def _refresh_overlay_live(self):
        """
        Refresh overlay during live mode.
        UPDATED: Remove achievements from combined players payload.
        """
        if not bool(self.cfg.OVERLAY.get("live_updates", False)):
            return
        if not self.overlay or not self.overlay.isVisible():
            return
        if not self.watcher or not self.watcher.game_active or not self.watcher.current_rom:
            return
        try:
            if (time.monotonic() - getattr(self, "_overlay_last_action", 0.0)) < 0.35:
                return
        except Exception:
            pass
        if getattr(self, "_overlay_busy", False):
            return

        try:
            self._overlay_busy = True
            try:
                self.watcher.force_flush()
            except Exception:
                pass

            try:
                idx = int(getattr(self, "_overlay_cycle", {}).get("idx", -1))
            except Exception:
                idx = -1

            if idx >= 0:
                self._prepare_overlay_sections()
                secs = self._overlay_cycle.get("sections", [])
                if not secs:
                    self._hide_overlay()
                    self._overlay_cycle = {"sections": [], "idx": -1}
                    return
                if idx >= len(secs):
                    idx = len(secs) - 1
                    self._overlay_cycle["idx"] = idx
                self._show_overlay_section(secs[idx])
                return

            try:
                players = read_active_players(self.cfg.BASE)
                combined = {"players": []}
                for p in players:
                    combined["players"].append({
                        "id": p.get("id"),
                        "highlights": p.get("highlights", {}),
                        "playtime_sec": p.get("playtime_sec", 0),
                        "score": p.get("score", 0),
                    })
                if combined["players"]:
                    self.overlay.set_combined(combined, session_title="Active Player Highlights")
            except Exception:
                pass
        finally:
            self._overlay_busy = False


    def _has_highlights(self, entry: dict) -> bool:
        h = entry.get("highlights", {}) or {}
        for cat in ("Power", "Precision", "Fun"):
            if h.get(cat):
                return True
        return False

    def _build_global_section(self) -> dict | None:
        """
        Aggregiert globale Events aus der Summary und baut einen 'Global'-Block (wie ein Reiter).
        """
        try:
            s = self.watcher._ai_read_latest_summary()  # nutzt Watcher-Helper
            if not s:
                return None
            events_totals = self.watcher._ai_aggregate_events_from_summary(s) or {}
            try:
                score_final = int(self.watcher._find_score_from_audits(s.get("end_audits", {}) or {}))
            except Exception:
                score_final = 0
            duration_sec = int(s.get("duration_sec", 0) or 0)
            pseudo_stats = {"score": score_final, "duration_sec": duration_sec, "events": events_totals}
            highlights = self.watcher.analyze_session(pseudo_stats) or {}
            # nur aufnehmen, wenn irgendwas drin ist
            nonempty = any(highlights.get(k) for k in ("Power", "Precision", "Fun")) or score_final > 0
            if not nonempty:
                return None
            return {
                "id": 0,  # neutral
                "title": "Global",
                "highlights": highlights,
                "score": score_final,
                "playtime_sec": duration_sec
            }
        except Exception:
            return None

  


    def _prepare_overlay_sections(self):
        """
        Build cycle pages (Active Player Highlights, Global Snapshot, Player Snapshots).
        UPDATED: strip achievements from all overlay sections.
        """
        def _played_entry(p: dict) -> bool:
            try:
                if int(p.get("playtime_sec", 0) or 0) > 0:
                    return True
            except Exception:
                pass
            try:
                if int(p.get("score", 0) or 0) > 0:
                    return True
            except Exception:
                pass
            h = p.get("highlights", {}) or {}
            return any(h.get(cat) for cat in ("Power", "Precision", "Fun"))

        sections = []

        # 1) Combined multi-column highlights (players + optional CPU)
        players_raw = read_active_players(self.cfg.BASE)
        combined_players = []
        if players_raw:
            for p in players_raw:
                if not _played_entry(p):
                    continue
                combined_players.append({
                    "id": int(p.get("id", 0)),
                    "highlights": p.get("highlights", {}),
                    "playtime_sec": p.get("playtime_sec", 0),
                    "score": int(p.get("score", 0) or 0),
                })

        # CPU optional
        try:
            sim = getattr(self.watcher, "cpu", {}) or {}
            if bool(sim.get("active", False)):
                deltas = sim.get("session_deltas", {}) or {}
                events = self.watcher._build_events_from_deltas(deltas)
                pseudo_stats = {
                    "score": int(sim.get("score", 0) or 0),
                    "duration_sec": int(sim.get("active_play_seconds", 0) or 0),
                    "events": events,
                }
                cpu_highlights = self.watcher.analyze_session(pseudo_stats) or {"Power": [], "Precision": [], "Fun": []}
                combined_players.append({
                    "id": 5,
                    "title": "CPU",
                    "highlights": cpu_highlights,
                    "playtime_sec": int(sim.get("active_play_seconds", 0) or 0),
                    "score": int(sim.get("score", 0) or 0),
                })
        except Exception:
            pass

        if combined_players:
            title = "Active Player Highlights"
            if len(combined_players) == 1:
                only = combined_players[0].get("id", 0)
                title = "CPU Highlights" if only == 5 else f"Player {only} Highlights"

            sections.append({
                "kind": "combined_players",
                "players": combined_players,
                "title": title
            })

        content = self._read_latest_session_txt()

        # 2) Global Snapshot (no achievements)
        if content:
            html_global = self._build_global_html_nonzero(content)
            sections.append({
                "kind": "html",
                "html": html_global,
                "title": "Global Snapshot"
            })

        # 3) Player Snapshots (no achievements)
        if content:
            html_players = self._build_player_snapshots_html(content)
            sections.append({
                "kind": "html",
                "html": html_players,
                "title": "Player Snapshots"
            })

        self._overlay_cycle = {"sections": sections, "idx": -1}



    def _show_overlay_section(self, payload: dict):
        """
        Show exactly one cycle page:
          - kind=combined_players → multi-column highlights
          - kind=html            → render provided HTML page
          - default              → single player highlight block (fallback; hier nicht genutzt)
        """
        self._ensure_overlay()
        kind = str(payload.get("kind", "")).lower()
        title = str(payload.get("title", "") or "").strip()

        if kind == "combined_players":
            combined = {"players": payload.get("players", [])}
            self.overlay.set_combined(combined, session_title=title or "Active Player Highlights")
            self.overlay.show(); self.overlay.raise_()
            return

        if kind == "html":
            html = payload.get("html", "") or "<div>-</div>"
            # WICHTIG: keinen Fallback 'Highlights' erzwingen, damit der gewünschte Titel angezeigt wird
            self.overlay.set_html(html, session_title=title)
            self.overlay.show(); self.overlay.raise_()
            return

        # Fallback (sollte im 3-Seiten-Flow nicht auftreten)
        combined = {"players": [payload]}
        title2 = f"Highlights – {payload.get('title','')}".strip()
        self.overlay.set_combined(combined, session_title=title2)
        self.overlay.show(); self.overlay.raise_()


    def _cycle_overlay_button(self):
        """
        Toggle/cycle pages:
          - If closed: prepare sections and show first page.
          - If open: advance to next page; close at end.

        CHANGE: Allow toggling even while a game is active.
        """
        if getattr(self, "_overlay_busy", False):
            return
        self._overlay_busy = True
        try:
            ov = getattr(self, "overlay", None)
            if not ov or not ov.isVisible():
                # Prepare and start at page 0
                self._prepare_overlay_sections()
                secs = self._overlay_cycle.get("sections", [])
                if not secs:
                    self._msgbox_topmost("info", "Overlay", "No contents available (Global/Player).")
                    return
                self._overlay_cycle["idx"] = 0
                self._show_overlay_section(secs[0])
            else:
                secs = self._overlay_cycle.get("sections", [])
                if not secs:
                    self._prepare_overlay_sections()
                    secs = self._overlay_cycle.get("sections", [])
                    if not secs:
                        self._hide_overlay()
                        self._overlay_cycle = {"sections": [], "idx": -1}
                        return
                    self._overlay_cycle["idx"] = 0
                    self._show_overlay_section(secs[0])
                    return

                idx = int(self._overlay_cycle.get("idx", -1))
                idx = 0 if idx < 0 else idx + 1
                if idx >= len(secs):
                    self._hide_overlay()
                    self._overlay_cycle = {"sections": [], "idx": -1}
                else:
                    self._overlay_cycle["idx"] = idx
                    self._show_overlay_section(secs[idx])
        finally:
            self._overlay_last_action = time.monotonic()
            self._overlay_busy = False



    def update_ai_evaluation_tab(self):
        """
        Build English HTML from BASE/AI/global.coach.json and BASE/AI/profile.json.
        Coach table WITHOUT the 'Tip' column.
        """
        if not hasattr(self, "ai_view"):
            return

        try:
            coach = {}
            profile = {}
            try:
                coach = load_json(os.path.join(self.cfg.BASE, "AI", "global.coach.json"), {}) or {}
            except Exception:
                coach = {}
            try:
                profile = load_json(os.path.join(self.cfg.BASE, "AI", "profile.json"), {}) or {}
            except Exception:
                profile = {}

            # Preserve scroll
            sb = self.ai_view.verticalScrollBar()
            old_val = sb.value(); old_max = max(1, sb.maximum())
            at_bottom_before = (old_val >= old_max - 2)
            ratio = old_val / old_max if old_max > 0 else 0.0

            def esc(x):
                return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

            html = []
            html.append("<style>"
                        "table{border-collapse:collapse}"
                        "th,td{padding:4px 8px;border-bottom:1px solid #ddd;white-space:nowrap}"
                        "h3{margin:8px 0 4px 0}"
                        "h4{margin:8px 0 4px 0;border-bottom:1px solid #ccc}"
                        ".muted{color:#777}"
                        "</style>")

            # Coach tips (NO tip column)
            html.append("<h3>AI Coach – Top Events</h3>")
            te = coach.get("top_events", []) or []
            if te:
                html.append("<table>")
                html.append("<tr><th>Event</th><th>Value/Event</th><th>Obs.</th><th>What-if (+1/+3/+5)</th></tr>")
                for item in te[:10]:
                    event = esc(item.get("event", ""))
                    vpe = item.get("value_per_event", 0)
                    obs = item.get("observations", 0)
                    wis = item.get("what_if", []) or []
                    wi_text = ", ".join([f"+{w.get('delta',0)} → {int(w.get('predicted_score_gain',0)):,d}".replace(",", ".") for w in wis])
                    html.append(f"<tr><td>{event}</td><td>{vpe}</td><td>{obs}</td><td>{wi_text}</td></tr>")
                html.append("</table>")
                upd = coach.get("updated", "")
                if upd:
                    html.append(f"<div class='muted'>Updated: {esc(upd)}</div>")
            else:
                html.append("<div>(No coach data yet)</div>")

            # Profile (unverändert, nur Anzeige)
            html.append("<h3>Skill Profile</h3>")
            if profile:
                style = profile.get("style", {}) or {}
                scores = style.get("scores", {}) or {}
                label = esc(style.get("label", "unknown"))
                sessions = int(profile.get("sessions", 0) or 0)
                html.append(f"<div>Sessions tracked: {sessions}</div>")
                html.append(f"<div>Style: <b>{label}</b> &nbsp; "
                            f"(precision={scores.get('precision',0)}, multiball={scores.get('multiball',0)}, risk={scores.get('risk',0)})</div>")

                last = profile.get("last_session", {}) or {}
                if last:
                    html.append("<h4>Last session</h4>")
                    rows = [
                        ("Score", f"{int(last.get('score',0)):,d}".replace(",", ".")),
                        ("Score/min", last.get("score_per_min", 0)),
                        ("Precision/min", last.get("precision_per_min", 0)),
                        ("Multiball/min", last.get("multiball_per_min", 0)),
                        ("Jackpots/min", last.get("jackpots_per_min", 0)),
                        ("Ball saves/min", last.get("ball_saves_per_min", 0)),
                        ("Extra balls/min", last.get("extra_balls_per_min", 0)),
                        ("Tilt warnings/min", last.get("tilt_warnings_per_min", 0)),
                        ("Tilts", last.get("tilts", 0)),
                        ("Duration (s)", last.get("duration_sec", 0)),
                        ("ROM", esc(last.get("rom","") or "")),
                        ("Table", esc(last.get("table","") or "")),
                    ]
                    html.append("<table>")
                    for k,v in rows:
                        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
                    html.append("</table>")

                trends = profile.get("trends", {}) or {}
                if trends:
                    html.append("<h4>Trends (avg vs. last 5 sessions)</h4>")
                    html.append("<table>")
                    html.append("<tr><th>Metric</th><th>Avg</th><th>Avg last 5</th><th>Δ% vs prev5</th><th>Trend</th></tr>")
                    for k, t in trends.items():
                        avg = t.get("avg", 0)
                        avg5 = t.get("avg_last5", 0)
                        pct = t.get("delta_pct_vs_prev5", 0)
                        tr = t.get("trend", "stable")
                        arrow = "▲" if tr == "up" else ("▼" if tr == "down" else "→")
                        html.append(f"<tr><td>{esc(k)}</td><td>{avg}</td><td>{avg5}</td><td>{pct}%</td><td>{arrow} {tr}</td></tr>")
                    html.append("</table>")

                upd = profile.get("updated", "")
                if upd:
                    html.append(f"<div class='muted'>Updated: {esc(upd)}</div>")
            else:
                html.append("<div>(No profile data yet)</div>")

            self.ai_view.setHtml("".join(html))

            # Scroll back
            new_max = max(1, sb.maximum())
            if at_bottom_before:
                sb.setValue(sb.maximum())
            else:
                new_val = int(round(ratio * new_max))
                sb.setValue(max(0, min(new_val, new_max)))
        except Exception:
            # no crash
            pass

    # ===== Challenges: TTS (best-effort via SAPI) and bridge handlers =====
    def _speak_en(self, text: str):
        """
        Speak a short English phrase using SAPI if available (best-effort).
        Volume from cfg.OVERLAY['challenges_voice_volume'] (0..100).
        Non-blocking; silently ignores errors.
        """
        try:
            vol = int(self.cfg.OVERLAY.get("challenges_voice_volume", 80))
            vol = max(0, min(100, vol))
            # SAPI via win32com (if available)
            try:
                import win32com.client  # type: ignore
                sp = win32com.client.Dispatch("SAPI.SpVoice")
                sp.Volume = vol
                sp.Speak(str(text))
                return
            except Exception:
                pass
            # Fallback: no-op if SAPI unavailable
        except Exception:
            pass

    # In class MainWindow (ersetzen)
    def _on_challenge_warmup_show(self, seconds: int, message: str):
        try:
            if not hasattr(self, "_mini_overlay") or self._mini_overlay is None:
                self._mini_overlay = MiniInfoOverlay(self)
            self._mini_overlay.show_info(str(message), max(1, int(seconds)), color_hex="#FF3B30")

            # Sprachausgabe nur einmal, nachdem das Overlay angezeigt wurde
            if not hasattr(self, "_ch_last_spoken"):
                self._ch_last_spoken = {}
            now = time.time()
            last = float(self._ch_last_spoken.get("timed", 0.0) or 0.0)
            if now - last > 2.0:
                QTimer.singleShot(0, lambda: self._speak_en("Timed challenge started"))
                self._ch_last_spoken["timed"] = now
        except Exception:
            pass

    def _on_challenge_timer_stop(self):
        """
        Stop the countdown overlay and any pending delayed start.
        """
        try:
            if hasattr(self, "_challenge_timer_delay") and self._challenge_timer_delay:
                self._challenge_timer_delay.stop()
                self._challenge_timer_delay.deleteLater()
        except Exception:
            pass
        self._challenge_timer_delay = None

        try:
            if hasattr(self, "_challenge_timer") and self._challenge_timer:
                self._challenge_timer.close()
                self._challenge_timer.deleteLater()
        except Exception:
            pass
        self._challenge_timer = None

    def _on_challenge_info_show(self, message: str, seconds: int, color_hex: str = "#FFFFFF"):
        try:
            if not hasattr(self, "_mini_overlay") or self._mini_overlay is None:
                self._mini_overlay = MiniInfoOverlay(self)
            self._mini_overlay.show_info(str(message), max(1, int(seconds)), color_hex=str(color_hex or "#FFFFFF"))
        except Exception:
            pass

    def _on_challenge_speak(self, phrase: str):
        self._speak_en(str(phrase or ""))


    def _update_cpu_stats_tab(self):
        """
        Build HTML for the 'CPU Sim' sub-tab from watcher.cpu (session deltas, play time, difficulty).
        English UI strings. Shows whether a game is active.
        """
        if "cpu" not in self.stats_views:
            return
        view = self.stats_views["cpu"]

        # Scroll-Position vorher merken
        try:
            sb = view.verticalScrollBar()
            old_val = sb.value()
            old_max = max(1, sb.maximum())
            at_bottom_before = (old_val >= old_max - 2)
            ratio = old_val / old_max if old_max > 0 else 0.0
        except Exception:
            sb = None
            at_bottom_before = False
            ratio = 0.0

        sim = getattr(self.watcher, "cpu", {}) or {}
        active = bool(sim.get("active"))
        # Schwierigkeit aus Watcher, sonst Config
        diff = str(sim.get("difficulty", self.cfg.OVERLAY.get("cpu_sim_difficulty", "mittel"))).lower()
        play = int(sim.get("active_play_seconds", 0.0) or 0)
        score = int(sim.get("score", 0) or 0)
        rows = []
        deltas = sim.get("session_deltas", {}) or {}

        # Sort: value desc, then label
        try:
            sorted_items = sorted(deltas.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]).lower()))
        except Exception:
            sorted_items = list(deltas.items())

        for k, v in sorted_items:
            try:
                ival = int(v or 0)
            except Exception:
                ival = 0
            if ival <= 0:
                continue
            rows.append((str(k), f"{ival}"))

        from datetime import timedelta
        diff_map = {"leicht": "Easy", "mittel": "Medium", "schwer": "Hard", "pro": "Pro"}
        head = []
        head.append(f"<h3>CPU Sim</h3>")
        head.append(f"<div>Active: {'Yes' if active else 'No'}</div>")
        head.append(f"<div>Game active: {'Yes' if self.watcher.game_active else 'No'}</div>")
        head.append(f"<div>Difficulty: {diff_map.get(diff, diff.capitalize())}</div>")
        head.append(f"<div>Play time: {str(timedelta(seconds=play))}</div>")
        head.append(f"<div>Score (internal): {score:,d}</div>".replace(",", "."))

        if rows:
            tbl = self._render_multi_columns(rows, 3)
            html = (
                "<style>table { border-collapse: collapse; }"
                ".inner td { padding: 3px 6px; white-space:nowrap; }"
                ".inner td:first-child { text-align: left; }"
                ".inner td:last-child { text-align: right; }"
                "tr:nth-child(even) { background-color: #f9f9f9; }</style>"
                + "\n".join(head) + "<h4>Session deltas</h4>" + tbl
            )
        else:
            hint = "(No deltas – start a ROM and enable CPU Sim)" if not self.watcher.game_active else "(No deltas yet)"
            html = "\n".join(head) + f"<p>{hint}</p>"

        # HTML setzen und Scroll-Position wiederherstellen
        try:
            view.setHtml(html)
            if sb is not None:
                new_max = max(1, sb.maximum())
                if at_bottom_before:
                    sb.setValue(sb.maximum())
                else:
                    new_val = int(round(ratio * new_max))
                    sb.setValue(max(0, min(new_val, new_max)))
        except Exception:
            pass

    def _cpu_diff_label(self) -> str:
        """
        Button label for the difficulty switch (English).
        Bevorzugt Config-Wert, falls Watcher noch nicht initialisiert ist.
        """
        try:
            cur = str((self.watcher.cpu or {}).get(
                "difficulty",
                self.cfg.OVERLAY.get("cpu_sim_difficulty", "mittel")
            )).lower()
        except Exception:
            cur = str(self.cfg.OVERLAY.get("cpu_sim_difficulty", "mittel")).lower()
        # auch englische Synonyme robust abfangen
        syn = {"easy": "leicht", "medium": "mittel", "difficult": "schwer", "hard": "schwer", "pro": "pro"}
        cur = syn.get(cur, cur)
        mapping = {"leicht": "Easy", "mittel": "Medium", "schwer": "Hard", "pro": "Pro"}
        return f"CPU difficulty: {mapping.get(cur, 'Medium')} (click to cycle)"

    def _on_cpu_cycle_difficulty(self):
        """
        Button-Handler: Schwierigkeit zyklisch umschalten:
          leicht -> mittel -> schwer -> pro -> leicht ...
        """
        order = ["leicht", "mittel", "schwer", "pro"]
        try:
            cur = str((self.watcher.cpu or {}).get("difficulty", "mittel")).lower()
        except Exception:
            cur = "mittel"
        try:
            idx = order.index(cur)
        except ValueError:
            idx = 1
        nxt = order[(idx + 1) % len(order)]
        try:
            self.watcher.set_cpu_difficulty(nxt)
        except Exception:
            pass
        try:
            self.btn_cpu_diff.setText(self._cpu_diff_label())
        except Exception:
            pass

    def _on_cpu_active_changed(self, state: int):
        """
        Checkbox-Handler: CPU-Simulation aktivieren/deaktivieren.
        """
        active = (Qt.CheckState(state) == Qt.CheckState.Checked)
        try:
            self.watcher.set_cpu_sim_active(active)
        except Exception:
            pass


    # --- Overlay control ---
    def _ensure_overlay(self):
        if self.overlay is None:
            self.overlay = OverlayWindow(self)
        self.overlay.portrait_mode = bool(self.cfg.OVERLAY.get("portrait_mode", True))
        self.overlay._apply_geometry()
        self.overlay._layout_positions()
        self.overlay.request_rotation(force=True)

    def _on_auto_show_toggle(self, state: int):
        """
        Checkbox handler: enable/disable automatic overlay show after VPX closes.
        """
        self.cfg.OVERLAY["auto_show_on_end"] = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.cfg.save()

    def _show_overlay_latest(self):
        """
        After session end: open the normal overlay with the 3-page cycle:
          1) Active Player Highlights
          2) Global Snapshot
          3) Player Snapshots
        """
        try:
            # Prepare pages from the latest data and show page 0
            self._prepare_overlay_sections()
            secs = self._overlay_cycle.get("sections", [])
            if not secs:
                return
            self._ensure_overlay()
            self._overlay_cycle["idx"] = 0
            self._show_overlay_section(secs[0])
        except Exception:
            pass

    def _on_mini_info_show(self, rom: str, seconds: int = 7):
        """
        Show the small info overlay centered on the primary monitor,
        with a countdown. Auto-closes after countdown finishes.
        """
        try:
            if not hasattr(self, "_mini_overlay") or self._mini_overlay is None:
                self._mini_overlay = MiniInfoOverlay(self)
            msg = f"No data yet – NVRAM map not found for {rom}"
            self._mini_overlay.show_info(msg, seconds=max(1, int(seconds)))
        except Exception:
            pass

    def _on_ach_toast_show(self, title: str, rom: str, seconds: int = 5):
        """
        Receive achievement toast requests (from Watcher via Bridge)
        and enqueue them for sequential display (5s each).
        """
        try:
            self._ach_toast_mgr.enqueue(title, rom, max(1, int(seconds)))
        except Exception:
            pass


    def _compute_overlay_anchor(self) -> tuple[int, int]:
        """
        Compute the anchor point (center) derived from how the main overlay is placed:
        - If use_xy: use (pos_x, pos_y) directly as center.
        - Else: reproduce OverlayWindow's auto placement and take the geometric center.
        """
        try:
            ov = self.cfg.OVERLAY or {}
            # Reference virtual geometry (union of screens)
            screens = QApplication.screens() or []
            if screens:
                vgeo = screens[0].geometry()
                for s in screens[1:]:
                    vgeo = vgeo.united(s.geometry())
            else:
                vgeo = QRect(0, 0, 1280, 720)

            if ov.get("use_xy", False):
                return int(ov.get("pos_x", 100)), int(ov.get("pos_y", 100))

            # Approximate big overlay size like OverlayWindow._apply_geometry
            portrait_mode = bool(ov.get("portrait_mode", True))
            scale_pct = int(ov.get("scale_pct", 100))
            if portrait_mode:
                base_h = int(vgeo.height() * 0.55)
                base_w = int(base_h * 9 / 16)
            else:
                base_w = int(vgeo.width() * 0.40)
                base_h = int(vgeo.height() * 0.30)
            w = max(120, int(base_w * scale_pct / 100))
            h = max(120, int(base_h * scale_pct / 100))

            pad = 20
            pos = "center"
            mapping = {
                "top-left": (vgeo.left() + pad, vgeo.top() + pad),
                "top-right": (vgeo.right() - w - pad, vgeo.top() + pad),
                "bottom-left": (vgeo.left() + pad, vgeo.bottom() - h - pad),
                "bottom-right": (vgeo.right() - w - pad, vgeo.bottom() - h - pad),
                "center-top": (vgeo.left() + (vgeo.width() - w) // 2, vgeo.top() + pad),
                "center-bottom": (vgeo.left() + (vgeo.width() - w) // 2, vgeo.bottom() - h - pad),
                "center-left": (vgeo.left() + pad, vgeo.top() + (vgeo.height() - h) // 2),
                "center-right": (vgeo.right() - w - pad, vgeo.top() + (vgeo.height() - h) // 2),
                "center": (vgeo.left() + (vgeo.width() - w) // 2, vgeo.top() + (vgeo.height() - h) // 2)
            }
            x, y = mapping.get(pos, mapping["center"])
            cx = x + w // 2
            cy = y + h // 2
            return int(cx), int(cy)
        except Exception:
            # Fallback roughly center of primary screen
            try:
                scr = QApplication.primaryScreen()
                geo = scr.geometry() if scr else QRect(0, 0, 1280, 720)
                return geo.left() + geo.width() // 2, geo.top() + geo.height() // 2
            except Exception:
                return 640, 360

    def _hide_overlay(self):
        if self.overlay and self.overlay.isVisible():
            self.overlay.hide()


    def _toggle_overlay(self):
        """
        Toggle overlay. Uses Cycle directly (first page = all player highlights).
        If a session is active and live_updates is enabled, force-flush first.
        """
        if self.watcher and self.watcher.game_active and self.watcher.current_rom:
            if bool(self.cfg.OVERLAY.get("live_updates", False)):
                try:
                    self.watcher.force_flush()
                except Exception:
                    pass
        # Nur Cycle benutzen (kein Vorab-Rendern)
        self._cycle_overlay_button()
            


    def _on_toggle_keyboard_event(self):
        """
        Toggle overlay via keyboard (global). Debounced and guarded against concurrent renders.
        """
        now = time.monotonic()
        # etwas großzügigeres Debounce gegen Doppelfeuer
        if now - getattr(self, "_last_toggle_ts", 0.0) < 0.40:
            return
        self._last_toggle_ts = now
        # Verhindere Reentrancy, wenn gerade gerendert wird
        if getattr(self, "_overlay_busy", False):
            return
        self._cycle_overlay_button()

    def _on_joy_toggle_poll(self):
        """
        Poll joystick buttons for:
          - Overlay toggle (if source=joystick)
          - Timed Challenge (if source=joystick)
          - One-Ball Challenge (if source=joystick)
        Debounced against last mask.

        IMPORTANT: Do NOT early-return when no game is active.
        We call the UI functions directly; they show the top-most
        info box themselves if a game is not running.
        """
        # Build interest map
        want = {}
        if self.cfg.OVERLAY.get("toggle_input_source", "keyboard") == "joystick":
            want["overlay"] = int(self.cfg.OVERLAY.get("toggle_joy_button", 2))
        if self.cfg.OVERLAY.get("challenge_time_input_source", "keyboard") == "joystick":
            want["time"] = int(self.cfg.OVERLAY.get("challenge_time_joy_button", 3))
        if self.cfg.OVERLAY.get("challenge_one_input_source", "keyboard") == "joystick":
            want["one"] = int(self.cfg.OVERLAY.get("challenge_one_joy_button", 4))
        if not want:
            self._joy_toggle_last_mask = 0
            return

        jix = JOYINFOEX()
        jix.dwSize = ctypes.sizeof(JOYINFOEX)
        jix.dwFlags = JOY_RETURNALL
        mask_all = 0
        for jid in range(16):
            if _joyGetPosEx(jid, ctypes.byref(jix)) != JOYERR_NOERROR:
                continue
            mask_all |= int(jix.dwButtons)
        newly = (mask_all & ~getattr(self, "_joy_toggle_last_mask", 0))
        self._joy_toggle_last_mask = mask_all

        def _bit_for(btn: int) -> int:
            return 1 << max(0, int(btn) - 1)

        # Overlay
        if "overlay" in want and (newly & _bit_for(want["overlay"])) != 0:
            self._cycle_overlay_button()
            return

        # Challenges: let UI check game_active and show info popup if needed
        if "time" in want and (newly & _bit_for(want["time"])) != 0:
            self._start_timed_challenge_ui()
            return
        if "one" in want and (newly & _bit_for(want["one"])) != 0:
            self._start_one_ball_challenge_ui()
            return
        
    def _on_portrait_toggle(self, state: int):
        is_checked = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.cfg.OVERLAY["portrait_mode"] = is_checked
        self.cfg.save()
        if self.overlay:
            self.overlay.apply_portrait_from_cfg(self.cfg.OVERLAY)

    def _on_overlay_scale(self, val: int):
        self.lbl_scale.setText(f"{val}%")
        self.cfg.OVERLAY["scale_pct"] = int(val)
        self.cfg.save()
        if self.overlay:
            self.overlay.scale_pct = int(val)
            self.overlay._apply_scale(int(val))
            self.overlay._apply_geometry()
            self.overlay._layout_positions()
            self.overlay.request_rotation(force=True)

    def _on_use_xy_changed(self, state: int):
        is_checked = (Qt.CheckState(state) == Qt.CheckState.Checked)
        self.cfg.OVERLAY["use_xy"] = is_checked
        self.cfg.save()
        if self.overlay:
            self.overlay._apply_geometry()
            self.overlay._layout_positions()
            self.overlay.request_rotation(force=True)

    def _on_xy_changed(self, _val: int):
        self.cfg.OVERLAY["pos_x"] = int(self.spn_x.value())
        self.cfg.OVERLAY["pos_y"] = int(self.spn_y.value())
        self.cfg.save()
        if self.overlay and self.cfg.OVERLAY.get("use_xy", False):
            self.overlay._apply_geometry()
            self.overlay._layout_positions()
            self.overlay.request_rotation(force=True)

    def _on_toggle_source_changed(self, src: str):
        self.cfg.OVERLAY["toggle_input_source"] = src
        self.cfg.save()
        self.lbl_toggle_binding.setText(self._toggle_binding_label_text())
        self._apply_toggle_source()
        # Reinstall inputs on source change
        self._refresh_input_bindings()

    def _apply_toggle_source(self):
        """
        Startet Joystick-Poll, wenn mind. eine Aktion Source=joystick hat.
        Stoppt ihn sonst. (Keyboard-Hotkeys werden separat registriert.)
        """
        try:
            src_overlay = str(self.cfg.OVERLAY.get("toggle_input_source", "keyboard")).lower()
            src_time    = str(self.cfg.OVERLAY.get("challenge_time_input_source", "keyboard")).lower()
            src_one     = str(self.cfg.OVERLAY.get("challenge_one_input_source", "keyboard")).lower()

            need_poll = (src_overlay == "joystick") or (src_time == "joystick") or (src_one == "joystick")
            if need_poll:
                self._joy_toggle_timer.start()
            else:
                self._joy_toggle_timer.stop()
                self._joy_toggle_last_mask = 0
        except Exception:
            try:
                self._joy_toggle_timer.stop()
            except Exception:
                pass
            self._joy_toggle_last_mask = 0
            
     # --- helper: refresh input bindings (reinstall hook + hotkeys) ---
    def _refresh_input_bindings(self):
        """
        Reinstall the low-level keyboard hook and WM_HOTKEY registrations
        after any binding or input-source change.
        """
        try:
            self._install_global_keyboard_hook()
        except Exception:
            pass
        try:
            self._register_global_hotkeys()
        except Exception:
            pass           

    def _on_bind_toggle_clicked(self):
        """
        Bindet entweder eine Joystick-Taste oder eine Tastaturtaste als Overlay-Toggle.
        Auswahl basiert auf cfg.OVERLAY["toggle_input_source"].
        """
        src = self.cfg.OVERLAY.get("toggle_input_source", "keyboard")

        # --- Joystick ---
        if src == "joystick":
            dlg = QDialog(self)
            dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
            dlg.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            dlg.setWindowTitle("Joystick binding")
            dlg.resize(420, 160)
            lay = QVBoxLayout(dlg)
            lbl = QLabel("Press any joystick button to bind…\n(Timeout in 10 seconds; ESC to cancel)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)

            cancelled = {"flag": False}
            def keyPressEvent(evt):
                if evt.key() == Qt.Key.Key_Escape:
                    cancelled["flag"] = True
                    dlg.reject()
            dlg.keyPressEvent = keyPressEvent  # type: ignore

            def _read_buttons_mask() -> int:
                jix = JOYINFOEX()
                jix.dwSize = ctypes.sizeof(JOYINFOEX)
                jix.dwFlags = JOY_RETURNALL
                mask_all = 0
                for jid in range(16):
                    try:
                        if _joyGetPosEx(jid, ctypes.byref(jix)) == JOYERR_NOERROR:
                            mask_all |= int(jix.dwButtons)
                    except Exception:
                        continue
                return mask_all

            baseline = _read_buttons_mask()
            start_ts = time.time()
            timer = QTimer(dlg)
            def _poll():
                nonlocal baseline
                if cancelled["flag"]:
                    timer.stop()
                    return
                try:
                    mask = _read_buttons_mask()
                    newly = mask & ~baseline
                    baseline = mask
                    if newly:
                        lsb = newly & -newly
                        idx = lsb.bit_length() - 1
                        button_num = idx + 1
                        self.cfg.OVERLAY["toggle_joy_button"] = int(button_num)
                        self.cfg.save()
                        self.lbl_toggle_binding.setText(self._toggle_binding_label_text())
                        timer.stop()
                        dlg.accept()
                        # sofort aktivieren
                        self._refresh_input_bindings()
                        return
                    if time.time() - start_ts > 10.0:
                        timer.stop()
                        dlg.reject()
                except Exception:
                    pass
            timer.setInterval(35)
            timer.timeout.connect(_poll)
            timer.start()
            dlg.exec()
            return

        # --- Keyboard ---
        class _TmpVKFilter(QAbstractNativeEventFilter):
            def __init__(self, cb):
                super().__init__()
                self.cb = cb
                self._done = False
            def nativeEventFilter(self, eventType, message):
                if self._done:
                    return False, 0
                try:
                    if eventType == b"windows_generic_MSG":
                        msg = ctypes.wintypes.MSG.from_address(int(message))
                        if msg.message in (WM_KEYDOWN, WM_SYSKEYDOWN):
                            vk = int(msg.wParam)
                            self._done = True
                            self.cb(vk)
                except Exception:
                    pass
                return False, 0

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard binding")
        dlg.resize(360, 140)
        lay = QVBoxLayout(dlg)
        lbl = QLabel("Press any key... (ESC to cancel)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)

        cancelled = {"flag": False}
        def keyPressEvent(evt):
            if evt.key() == Qt.Key.Key_Escape:
                cancelled["flag"] = True
                try:
                    QCoreApplication.instance().removeNativeEventFilter(fil)  # type: ignore
                except Exception:
                    pass
                dlg.reject()
        dlg.keyPressEvent = keyPressEvent  # type: ignore

        def on_vk(vk: int):
            if cancelled["flag"]:
                return
            try:
                QCoreApplication.instance().removeNativeEventFilter(fil)  # type: ignore
            except Exception:
                pass
            self.cfg.OVERLAY["toggle_vk"] = int(vk)
            self.cfg.save()
            self.lbl_toggle_binding.setText(self._toggle_binding_label_text())
            dlg.accept()
            # sofort aktivieren
            self._refresh_input_bindings()

        fil = _TmpVKFilter(on_vk)
        QCoreApplication.instance().installNativeEventFilter(fil)
        dlg.exec()


    def _toggle_binding_label_text(self) -> str:
        src = self.cfg.OVERLAY.get("toggle_input_source", "keyboard")
        if src == "joystick":
            btn = int(self.cfg.OVERLAY.get("toggle_joy_button", 2))
            return f"Current: joystick button {btn}"
        else:
            vk = int(self.cfg.OVERLAY.get("toggle_vk", 120))
            return f"Current: {vk_to_name(vk)}"

    def _on_overlay_trigger(self):
        self._toggle_overlay()

    def _on_font_family_changed(self, qfont: QFont):
        family = qfont.family()
        self.cfg.OVERLAY["font_family"] = family
        self.cfg.save()
        if self.overlay:
            self.overlay.apply_font_from_cfg(self.cfg.OVERLAY)

    def _on_font_size_changed(self, val: int):
        body = int(val)
        self.cfg.OVERLAY["base_body_size"] = body
        self.cfg.OVERLAY["base_title_size"] = int(round(body * 1.8))
        self.cfg.OVERLAY["base_hint_size"] = int(round(body * 0.8))
        self.cfg.save()
        if self.overlay:
            self.overlay.apply_font_from_cfg(self.cfg.OVERLAY)
            self.overlay._apply_geometry()
            self.overlay._layout_positions()
            self.overlay.request_rotation(force=True)

    def _pick_color(self, key: str, label_widget: QLabel):
        initial = QColor(self.cfg.OVERLAY.get(key, "#FFFFFF"))
        color = QColorDialog.getColor(initial, self, f"Choose {key.replace('_',' ')}")
        if not color.isValid():
            return
        hexc = color.name(QColor.NameFormat.HexRgb)
        self.cfg.OVERLAY[key] = hexc
        self.cfg.save()
        try:
            label_widget.setText(hexc)
        except Exception:
            pass
        if self.overlay:
            self.overlay.apply_colors_from_cfg(self.cfg.OVERLAY)

    def _restart_watcher(self):
        try:
            if self.watcher:
                self.watcher.stop()
        except Exception:
            pass
        self.watcher = Watcher(self.cfg, self.bridge)
        self.watcher.start()
        self.status_label.setText("Watcher: running")
        self.status_label.setStyleSheet("font: bold 14px 'Segoe UI'; color:#107c10;")
        
    def _install_global_keyboard_hook(self):
        """
        Keyboard-Hook deaktiviert. Wir nutzen WM_HOTKEY für alle Keyboard-Bindings.
        Leise bleiben (kein Info-Log).
        """
        try:
            if getattr(self, "_global_keyhook", None):
                try:
                    self._global_keyhook.uninstall()
                except Exception:
                    pass
            self._global_keyhook = None
            # Keine INFO-Logs hier mehr
        except Exception as e:
            log(self.cfg, f"[HOTKEY] disable hook failed: {e}", "WARN")

    def _register_global_hotkeys(self):
        """
        Registriert WM_HOTKEY nur für Aktionen mit Source=keyboard.
        Joystick wird separat gepollt, wenn Source=joystick.
        """
        try:
            # Clean previous registrations + filter
            try:
                self._unregister_global_hotkeys()
            except Exception:
                pass

            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            hwnd = int(self.winId())
            MOD_NOREPEAT = 0x4000

            ids = {
                "overlay_toggle": 0xA11,
                "challenge_time": 0xA12,
                "challenge_one":  0xA13,
            }

            # Quellen
            src_overlay = str(self.cfg.OVERLAY.get("toggle_input_source", "keyboard")).lower()
            src_time    = str(self.cfg.OVERLAY.get("challenge_time_input_source", "keyboard")).lower()
            src_one     = str(self.cfg.OVERLAY.get("challenge_one_input_source", "keyboard")).lower()

            # VKs
            vk_overlay = int(self.cfg.OVERLAY.get("toggle_vk", 120))          # F9
            vk_time    = int(self.cfg.OVERLAY.get("challenge_time_vk", 121))  # F10
            vk_one     = int(self.cfg.OVERLAY.get("challenge_one_vk", 122))   # F11

            def _reg(name: str, _id: int, vk: int):
                mods = (self._mods_for_vk(vk) | MOD_NOREPEAT)
                if not user32.RegisterHotKey(wintypes.HWND(hwnd), _id, mods, vk):
                    log(self.cfg, f"[HOTKEY] RegisterHotKey failed for {name} vk={vk} mod={mods}", "WARN")

            # Nur registrieren, wenn Source=keyboard
            if src_overlay == "keyboard":
                _reg("overlay", ids["overlay_toggle"], vk_overlay)
            if src_time == "keyboard":
                _reg("timed",   ids["challenge_time"], vk_time)
            if src_one == "keyboard":
                _reg("oneball", ids["challenge_one"],  vk_one)

            # Ein Filter für alle (feuert nur, wenn registriert)
            class _HotkeyFilter(QAbstractNativeEventFilter):
                def __init__(self, parent_ref, ids_map):
                    super().__init__()
                    self.p = parent_ref
                    self.ids = ids_map
                def nativeEventFilter(self, eventType, message):
                    try:
                        if eventType == b"windows_generic_MSG":
                            msg = ctypes.wintypes.MSG.from_address(int(message))
                            if msg.message == WM_HOTKEY:
                                hid = int(msg.wParam)
                                if hid == self.ids["overlay_toggle"]:
                                    QTimer.singleShot(0, self.p._on_toggle_keyboard_event)
                                elif hid == self.ids["challenge_time"]:
                                    QTimer.singleShot(0, self.p._start_timed_challenge_ui)
                                elif hid == self.ids["challenge_one"]:
                                    QTimer.singleShot(0, self.p._start_one_ball_challenge_ui)
                    except Exception:
                        pass
                    return False, 0

            self._hotkey_ids = ids
            self._hotkey_filter = _HotkeyFilter(self, ids)
            QCoreApplication.instance().installNativeEventFilter(self._hotkey_filter)

        except Exception as e:
            log(self.cfg, f"[HOTKEY] register failed: {e}", "WARN")
       
    def _uninstall_global_keyboard_hook(self):
        """
        Uninstall the global low-level keyboard hook.
        """
        try:
            if getattr(self, "_global_keyhook", None):
                self._global_keyhook.uninstall()
                self._global_keyhook = None
                log(self.cfg, "[HOOK] Global keyboard hook uninstalled")
        except Exception as e:
            log(self.cfg, f"[HOOK] uninstall failed: {e}", "WARN")


    def _unregister_global_hotkeys(self):
        """
        Unregister global WM_HOTKEYs and remove filter.
        """
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            hwnd = int(self.winId())
            if getattr(self, "_hotkey_ids", None):
                for _name, _id in list(self._hotkey_ids.items()):
                    try:
                        user32.UnregisterHotKey(wintypes.HWND(hwnd), _id)
                    except Exception:
                        pass
            self._hotkey_ids = {}
        except Exception:
            pass
        try:
            if getattr(self, "_hotkey_filter", None):
                QCoreApplication.instance().removeNativeEventFilter(self._hotkey_filter)  # type: ignore
        except Exception:
            pass
        self._hotkey_filter = None

    
    def _init_settings_tooltips(self):
        """
        English tooltips for the Settings tab (paths, repair, prefetch).
        Safely checks widget existence to avoid AttributeError if UI elements are not present.
        """
        def _set_tip(attr: str, tip: str):
            try:
                w = getattr(self, attr, None)
                if w:
                    w.setToolTip(tip)
            except Exception:
                pass

        # Repair / Prefetch
        _set_tip("btn_repair", "Recreate the base folder structure and fetch index/ROM-names if missing.")
        _set_tip("btn_prefetch", "Cache missing NVRAM maps in the background. See watcher.log for progress.")

        # Paths
        _set_tip("base_label",   "Current base directory for achievements data.")
        _set_tip("btn_base",     "Change the base directory for achievements data.")
        _set_tip("nvram_label",  "Current VPinMAME NVRAM directory.")
        _set_tip("btn_nvram",    "Change the VPinMAME NVRAM directory.")
        _set_tip("tables_label", "Current Tables directory (optional).")
        _set_tip("btn_tables",   "Change the Tables directory (optional).")
            
            
            
            
# ---------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------
def main():
    cfg = AppConfig.load()
    app = QApplication(sys.argv)
    need_wizard = cfg.FIRST_RUN or not os.path.isdir(cfg.BASE)
    if need_wizard:
        if not os.path.isdir(cfg.BASE):
            home_alt = os.path.join(os.path.expanduser("~"), "Achievements")
            if not os.path.exists(cfg.BASE) and not os.path.exists(home_alt):
                cfg.BASE = home_alt
        wiz = SetupWizardDialog(cfg)
        if wiz.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
    for sub in [
        "NVRAM_Maps/maps",
        "NVRAM_Maps/overrides",
        "session_stats/Highlights",
        "session_stats/whitelists",
        "rom_specific_achievements",
        "custom_achievements",
        "AI",
    ]:
        ensure_dir(os.path.join(cfg.BASE, sub))

    bridge = Bridge()
    watcher = Watcher(cfg, bridge)

    # Wichtig: CPU-Sim-Zustand und Schwierigkeit aus der Config laden,
    # bevor die GUI die Checkbox initialisiert.
    try:
        watcher._cpu_sim_init()
    except Exception:
        pass

    win = MainWindow(cfg, watcher, bridge)

    # Low-level hook + WM_HOTKEY-Fallback
    try:
        win._install_global_keyboard_hook()
    except Exception:
        pass
    try:
        win._register_global_hotkeys()
    except Exception:
        pass

    if cfg.FIRST_RUN:
        cfg.FIRST_RUN = False
        cfg.save()
    win.hide()
    code = app.exec()
    cfg.save()
    sys.exit(code)

if __name__ == "__main__":
    main()
