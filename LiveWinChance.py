"""
LoL Live Game Tracker + Win Probability
Pollt die lokale Live Client Data API alle 30s und postet den aktuellen Gamestate
inkl. KI-basierter Win-Probability.

Requirements:
    pip install requests colorama torch numpy
"""

import requests
import time
import os
import json
import urllib3
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta

# SSL-Warnings unterdrücken (Riot nutzt self-signed cert lokal)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False
    print("Tipp: 'pip install colorama' für bunte Ausgabe\n")

BASE_URL      = "https://127.0.0.1:2999/liveclientdata"
POLL_INTERVAL = 30  # Sekunden
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))

# ── Farb-Helfer ────────────────────────────────────────────────────────────────

def c(text, color="", bold=False):
    if not COLOR:
        return str(text)
    b = Style.BRIGHT if bold else ""
    return f"{b}{color}{text}{Style.RESET_ALL}"

def red(t, bold=False):    return c(t, Fore.RED, bold)
def blue(t, bold=False):   return c(t, Fore.BLUE, bold)
def yellow(t, bold=False): return c(t, Fore.YELLOW, bold)
def green(t, bold=False):  return c(t, Fore.GREEN, bold)
def cyan(t, bold=False):   return c(t, Fore.CYAN, bold)
def white(t, bold=False):  return c(t, Fore.WHITE, bold)
def dim(t):                return c(t, Style.DIM) if COLOR else str(t)

# ── Data Dragon Item Cache ─────────────────────────────────────────────────────

ITEM_COSTS = {}  # item_id (int) → total gold cost (int)

def load_item_costs():
    """Lädt Item-Goldwerte einmalig von Data Dragon beim Start."""
    global ITEM_COSTS
    try:
        ver_r   = requests.get(
            "https://ddragon.leagueoflegends.com/api/versions.json", timeout=5
        )
        version = ver_r.json()[0]
        item_r  = requests.get(
            f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json",
            timeout=10,
        )
        items = item_r.json().get("data", {})
        ITEM_COSTS = {
            int(iid): d.get("gold", {}).get("total", 0)
            for iid, d in items.items()
        }
        print(cyan(f"  ✓ Item-Daten geladen ({len(ITEM_COSTS)} Items, Patch {version})"))
    except Exception as e:
        print(yellow(f"  ⚠ Item-Daten nicht geladen ({e}) — Gold wird auf 0 gesetzt"))

def player_item_gold(player: dict) -> int:
    """Summe der Total-Goldwerte aller Items eines Spielers."""
    total = 0
    for slot in player.get("items", []):
        iid = slot.get("itemID", 0)
        total += ITEM_COSTS.get(iid, 0)
    return total

# ── Modell laden ───────────────────────────────────────────────────────────────

class WinProbModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def load_model():
    model_path   = os.path.join(BASE_DIR, "lol_model.pt")
    scaler_path  = os.path.join(BASE_DIR, "lol_scaler.json")
    feature_path = os.path.join(BASE_DIR, "lol_features.json")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu")
    model = WinProbModel(
        checkpoint["input_dim"],
        checkpoint["hidden"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(scaler_path) as f:
        scaler = json.load(f)
    with open(feature_path) as f:
        features = json.load(f)

    return model, scaler, features

MODEL, SCALER, FEATURES = load_model()

if MODEL is None:
    print(dim("  [Win Probability nicht verfügbar — lol_model.pt nicht gefunden]"))

# ── API Calls ──────────────────────────────────────────────────────────────────

def get(endpoint):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", verify=False, timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        print(red(f"  API Fehler ({endpoint}): {e}"))
        return None

# ── Daten-Extraktion ───────────────────────────────────────────────────────────

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))[2:]  # MM:SS

def get_team(player):
    return player.get("team", "?")

def get_kda(scores):
    k = scores.get("kills", 0)
    d = scores.get("deaths", 0)
    a = scores.get("assists", 0)
    return k, d, a

def build_scoreline(players):
    order = [p for p in players if get_team(p) == "ORDER"]
    chaos = [p for p in players if get_team(p) == "CHAOS"]
    return order, chaos

def build_name_to_team(players):
    mapping = {}
    for p in players:
        team = get_team(p)
        for key in ("riotIdGameName", "summonerName", "riotId"):
            name = p.get(key, "")
            if name:
                mapping[name] = team
                mapping[name.lower()] = team
    return mapping

def resolve_team(name, name_to_team):
    return name_to_team.get(name) or name_to_team.get(name.lower())

def count_objectives(events, players):
    dragons    = {"ORDER": 0, "CHAOS": 0}
    turrets    = {"ORDER": 0, "CHAOS": 0}
    barons     = {"ORDER": 0, "CHAOS": 0}
    inhibitors = {"ORDER": 0, "CHAOS": 0}

    n2t = build_name_to_team(players)

    for e in events:
        t      = e.get("EventName", "")
        killer = e.get("KillerName", "")

        raw_team = e.get("KillingTeam") or e.get("KillerTeam") or ""
        team = raw_team if raw_team in ("ORDER", "CHAOS") else resolve_team(killer, n2t)

        if team not in ("ORDER", "CHAOS"):
            continue

        if t == "DragonKill":
            dragons[team] += 1
        elif t == "BaronKill":
            barons[team] += 1
        elif t == "TurretKilled":
            turrets[team] += 1
        elif t == "InhibKilled":
            inhibitors[team] += 1

    return dragons, barons, turrets, inhibitors

# ── Win Probability ────────────────────────────────────────────────────────────

def build_snapshot(players, events, game_time):
    snap = {"minute": int(game_time // 60)}

    order = [p for p in players if get_team(p) == "ORDER"]
    chaos = [p for p in players if get_team(p) == "CHAOS"]

    # Gold = nur ausgegebenes Gold via Items (entspricht totalGold aus Timeline)
    snap["gold_blue"] = sum(player_item_gold(p) for p in order)
    snap["gold_red"]  = sum(player_item_gold(p) for p in chaos)
    snap["gold_diff"] = snap["gold_blue"] - snap["gold_red"]
    snap["cs_blue"]   = sum(p.get("scores", {}).get("creepScore", 0) for p in order)
    snap["cs_red"]    = sum(p.get("scores", {}).get("creepScore", 0) for p in chaos)

    dragons, barons, turrets, inhibitors = count_objectives(events, players)
    snap["kills_blue"]   = sum(p.get("scores", {}).get("kills", 0) for p in order)
    snap["kills_red"]    = sum(p.get("scores", {}).get("kills", 0) for p in chaos)
    snap["dragons_blue"] = dragons["ORDER"]
    snap["dragons_red"]  = dragons["CHAOS"]
    snap["barons_blue"]  = barons["ORDER"]
    snap["barons_red"]   = barons["CHAOS"]
    snap["turrets_blue"] = turrets["ORDER"]
    snap["turrets_red"]  = turrets["CHAOS"]
    snap["inhibs_blue"]  = inhibitors["ORDER"]
    snap["inhibs_red"]   = inhibitors["CHAOS"]

    for i, p in enumerate(players, 1):
        scores = p.get("scores", {})
        prefix = f"p{i}"
        snap[f"{prefix}_team"]    = "blue" if get_team(p) == "ORDER" else "red"
        snap[f"{prefix}_champ"]   = p.get("championName", "?")
        snap[f"{prefix}_kills"]   = scores.get("kills", 0)
        snap[f"{prefix}_deaths"]  = scores.get("deaths", 0)
        snap[f"{prefix}_assists"] = scores.get("assists", 0)
        snap[f"{prefix}_cs"]      = scores.get("creepScore", 0)
        snap[f"{prefix}_gold"]    = player_item_gold(p)
        snap[f"{prefix}_level"]   = p.get("level", 1)
        # snap[f"{prefix}_xp"]      = 0

    return snap

def predict_winprob(snap):
    if MODEL is None:
        
        return None

    mean = np.array(SCALER["mean"], dtype=np.float32)
    std  = np.array(SCALER["std"],  dtype=np.float32)

    row = []
    for feat in FEATURES:
        val = snap.get(feat, 0)
        if isinstance(val, str):
            val = 1.0 if val == "blue" else 0.0
        row.append(float(val))

    X = np.array([row], dtype=np.float32)
    X = (X - mean) / std

    with torch.no_grad():
        prob = MODEL(torch.tensor(X)).item()
    # # temporär zum debuggen
    # raw_row = []
    # for feat in FEATURES:
    #     val = snap.get(feat, 0)
    #     if isinstance(val, str):
    #         val = 1.0 if val == "blue" else 0.0
    #     raw_row.append((feat, float(val)))

    # print("\n  DEBUG RAW (erste 10 Features):")
    # for fname, fval in raw_row[:10]:
    #     print(f"    {fname:<20} {fval}")

    # X_scaled = (np.array([[v for _, v in raw_row]], dtype=np.float32) - mean) / std
    # print("\n  DEBUG SCALED (erste 10):")
    # for i, (fname, _) in enumerate(raw_row[:10]):
    #     print(f"    {fname:<20} {X_scaled[0][i]:.3f}")
    return prob

# ── Ausgabe ────────────────────────────────────────────────────────────────────

def print_header(game_time):
    os.system("cls" if os.name == "nt" else "clear")
    bar = "═" * 62
    print(cyan(f"╔{bar}╗", bold=True))
    print(cyan(f"║{'LoL LIVE TRACKER':^62}║", bold=True))
    print(cyan(f"║{('⏱  ' + format_time(game_time)):^62}║", bold=True))
    print(cyan(f"╚{bar}╝", bold=True))

def print_scoreline(order, chaos):
    ok = sum(p.get("scores", {}).get("kills", 0) for p in order)
    ck = sum(p.get("scores", {}).get("kills", 0) for p in chaos)
    score_str = f"  {blue('BLUE', bold=True)}  {ok}  :  {ck}  {red('RED', bold=True)}"
    print(f"\n{'SCORE':─<10}{score_str}")

def print_team(players, label, color_fn):
    print(f"\n  {color_fn('▌ ' + label, bold=True)}")
    print(f"  {'Champion':<18} {'KDA':<12} {'CS':<6} {'Gold (Items)':<14} {'Lvl'}")
    print(f"  {'─'*18} {'─'*12} {'─'*6} {'─'*14} {'─'*3}")
    for p in players:
        champ   = p.get("championName", "?")[:17]
        scores  = p.get("scores", {})
        k, d, a = get_kda(scores)
        cs      = scores.get("creepScore", 0)
        gold    = player_item_gold(p)
        lvl     = p.get("level", 1)
        kda_str = f"{k}/{d}/{a}"
        print(f"  {champ:<18} {kda_str:<12} {cs:<6} {gold:<14} {lvl}")

def print_objectives(events, players):
    dragons, barons, turrets, inhibitors = count_objectives(events, players)
    print(f"\n  {'OBJECTIVES':─<55}")
    row = lambda label, o, c: (
        f"  {label:<14} {blue(str(o)):>5}  vs  {red(str(c)):<5}"
    )
    print(row("🐉 Dragons",    dragons["ORDER"],    dragons["CHAOS"]))
    print(row("👑 Barons",     barons["ORDER"],     barons["CHAOS"]))
    print(row("🏰 Turrets",    turrets["ORDER"],    turrets["CHAOS"]))
    print(row("💀 Inhibitors", inhibitors["ORDER"], inhibitors["CHAOS"]))

def print_winprob(players, events, game_time):
    snap = build_snapshot(players, events, game_time)
    prob = predict_winprob(snap)

    if prob is None:
        return

    prob_blue = prob * 100
    prob_red  = (1 - prob) * 100
    bar_len   = int(prob * 40)
    bar_blue  = "█" * bar_len
    bar_red   = "█" * (40 - bar_len)

    print(f"\n  {'WIN PROBABILITY':─<55}")
    print(f"  {blue('BLUE', bold=True)} {blue(bar_blue)}{red(bar_red)} {red('RED', bold=True)}")
    print(f"  {blue(f'{prob_blue:.1f}%', bold=True):<30}{red(f'{prob_red:.1f}%', bold=True):>10}")

def print_last_events(events, n=5):
    if not events:
        return
    print(f"\n  {'LETZTE EVENTS':─<55}")
    for e in events[-n:]:
        t    = e.get("EventName", "?")
        et   = format_time(e.get("EventTime", 0))
        name = e.get("KillerName", e.get("VictimName", ""))
        print(f"  {dim(et)}  {yellow(t):<30} {dim(name)}")

def print_footer(next_update):
    print(f"\n{dim('─'*64)}")
    print(dim(f"  Nächstes Update in {next_update}s  |  Ctrl+C zum Beenden"))

# ── Haupt-Loop ─────────────────────────────────────────────────────────────────

def poll():
    print(cyan("\n  LoL Live Tracker startet...", bold=True))
    load_item_costs()  # Data Dragon einmalig beim Start laden
    print(cyan("\n  Warte auf laufende Partie...", bold=True))

    while True:
        data = get("/allgamedata")

        if data is None:
            print(red("  Keine Verbindung – läuft das Spiel? (retry in 10s)"))
            time.sleep(10)
            continue

        game      = data.get("gameData", {})
        players   = data.get("allPlayers", [])
        events    = data.get("events", {}).get("Events", [])
        game_time = game.get("gameTime", 0)

        order, chaos = build_scoreline(players)

        print_header(game_time)
        print_scoreline(order, chaos)
        print_team(order, "BLUE SIDE", blue)
        print_team(chaos, "RED SIDE",  red)
        print_objectives(events, players)
        print_winprob(players, events, game_time)
        print_last_events(events, n=6)
        print_footer(POLL_INTERVAL)

        time.sleep(POLL_INTERVAL)

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        poll()
    except KeyboardInterrupt:
        print(f"\n{yellow('  Tracker beendet. GG!')}\n")