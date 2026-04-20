"""
LoL Dataset Collector
Sammelt die letzten N ranked Games (Solo/Duo + Flex) mit Minuten-Snapshots ab Minute 5.
Ausgabe als CSV für Win-Probability Training.

Requirements:
    pip install requests

Usage:
    python lol_dataset_collector.py
    → Fragt nach API Key und Riot ID beim Start
"""

import requests
import time
import csv
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

REGION_PLATFORM = "euw1"          # Plattform (für Summoner/Match-IDs)
REGION_CLUSTER  = "europe"        # Cluster (für Match v5 API)
QUEUE_IDS       = {420, 440}      # 420 = Solo/Duo, 440 = Flex
SNAPSHOT_START  = 5               # Snapshots ab Minute X
GAMES_TO_FETCH  = 100
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lol_dataset.csv")
RATE_LIMIT_PAUSE = 1.25           # Sekunden zwischen Requests (Dev Key: 100/2min)

# ── HTTP ───────────────────────────────────────────────────────────────────────

def get(url, api_key, params=None):
    headers = {"X-Riot-Token": api_key}
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=10)
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 10))
            print(f"  Rate limit hit — warte {retry}s...")
            time.sleep(retry)
            return get(url, api_key, params)
        r.raise_for_status()
        time.sleep(RATE_LIMIT_PAUSE)
        return r.json()
    except Exception as e:
        print(f"  Fehler: {e} | URL: {url}")
        return None

# ── Summoner ───────────────────────────────────────────────────────────────────

def get_puuid(api_key, game_name, tag_line):
    url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    data = get(url, api_key)
    return data.get("puuid") if data else None

# ── Match IDs ──────────────────────────────────────────────────────────────────

def get_match_ids(api_key, puuid, count=100):
    url = f"https://{REGION_CLUSTER}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"type": "ranked", "count": count, "start": 0}
    return get(url, api_key, params) or []

# ── Match Info ─────────────────────────────────────────────────────────────────

def get_match_info(api_key, match_id):
    url = f"https://{REGION_CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return get(url, api_key)

def get_match_timeline(api_key, match_id):
    url = f"https://{REGION_CLUSTER}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return get(url, api_key)

# ── Verarbeitung ───────────────────────────────────────────────────────────────

def get_participant_index(match_info, puuid):
    """Gibt 0-basierten Index des eigenen Spielers zurück."""
    for i, p in enumerate(match_info["metadata"]["participants"]):
        if p == puuid:
            return i
    return None

def extract_snapshots(match_info, timeline, puuid):
    """
    Gibt eine Liste von Dicts zurück — einen pro Minute ab SNAPSHOT_START.
    Jedes Dict = eine Zeile im CSV.
    """
    participants = match_info["info"]["participants"]
    own_idx      = get_participant_index(match_info, puuid)
    own_champ = participants[own_idx]["championName"] if own_idx is not None else "?"
    match_id     = match_info["metadata"]["matchId"]
    queue_id     = match_info["info"]["queueId"]

    # Wer hat gewonnen? Team 100 = Blue, Team 200 = Red
    winner_team = None
    for p in participants:
        if p["win"]:
            winner_team = p["teamId"]
            break

    # Eigener Spieler → Team bestimmen
    own_team_id = participants[own_idx]["teamId"] if own_idx is not None else None

    frames = timeline.get("info", {}).get("frames", [])
    snapshots = []

    for frame in frames:
        minute = frame.get("timestamp", 0) // 60000
        if minute < SNAPSHOT_START:
            continue

        pf = frame.get("participantFrames", {})

        # ── Team-Aggregates ────────────────────────────────────────────────
        team_gold   = {100: 0, 200: 0}
        team_kills  = {100: 0, 200: 0}  # kumuliert via Events bis zu dieser Minute
        team_cs     = {100: 0, 200: 0}

        for i, p in enumerate(participants):
            tid  = p["teamId"]
            pkey = str(i + 1)
            pdata = pf.get(pkey, {})
            team_gold[tid]  += pdata.get("totalGold", 0)
            team_cs[tid]    += pdata.get("jungleMinionsKilled", 0) + pdata.get("minionsKilled", 0)

        # ── Objectives bis zu dieser Minute zählen ─────────────────────────
        obj = {
            "dragons_blue": 0, "dragons_red": 0,
            "barons_blue":  0, "barons_red":  0,
            "turrets_blue": 0, "turrets_red":  0,
            "inhibs_blue":  0, "inhibs_red":   0,
            "kills_blue":   0, "kills_red":    0,
        }

        for f2 in frames:
            if f2.get("timestamp", 0) > frame["timestamp"]:
                break
            for e in f2.get("events", []):
                etype = e.get("type", "")
                if etype == "CHAMPION_KILL":
                    killer_id = e.get("killerId", 0)
                    if 1 <= killer_id <= 10:
                        killer_team = participants[killer_id - 1]["teamId"]
                        if killer_team == 100: obj["kills_blue"] += 1
                        else:                  obj["kills_red"]  += 1
                elif etype == "ELITE_MONSTER_KILL":
                    mtype = e.get("monsterType", "")
                    killer_team = e.get("killerTeamId", 0)
                    if mtype == "DRAGON":
                        if killer_team == 100: obj["dragons_blue"] += 1
                        else:                  obj["dragons_red"]  += 1
                    elif mtype == "BARON_NASHOR":
                        if killer_team == 100: obj["barons_blue"] += 1
                        else:                  obj["barons_red"]  += 1
                elif etype == "BUILDING_KILL":
                    btype = e.get("buildingType", "")
                    killer_team = e.get("teamId", 0)  # teamId = Team das die Struktur VERLOREN hat → invertieren
                    attacking_team = 200 if killer_team == 100 else 100
                    if btype == "TOWER_BUILDING":
                        if attacking_team == 100: obj["turrets_blue"] += 1
                        else:                     obj["turrets_red"]  += 1
                    elif btype == "INHIBITOR_BUILDING":
                        if attacking_team == 100: obj["inhibs_blue"] += 1
                        else:                     obj["inhibs_red"]  += 1

        # ── Alle 10 Spieler einzeln ────────────────────────────────────────
        player_data = {}
        for i, p in enumerate(participants):
            pkey  = str(i + 1)
            pdata = pf.get(pkey, {})
            prefix = f"p{i+1}"
            cs = pdata.get("minionsKilled", 0) + pdata.get("jungleMinionsKilled", 0)
            player_data[f"{prefix}_champ"]  = p["championName"]
            player_data[f"{prefix}_team"]   = "blue" if p["teamId"] == 100 else "red"
            player_data[f"{prefix}_kills"]  = pdata.get("kills", 0) if "kills" in pdata else ""
            player_data[f"{prefix}_deaths"] = pdata.get("deaths", 0) if "deaths" in pdata else ""
            player_data[f"{prefix}_assists"]= pdata.get("assists", 0) if "assists" in pdata else ""
            player_data[f"{prefix}_cs"]     = cs
            player_data[f"{prefix}_gold"]   = pdata.get("totalGold", 0)
            player_data[f"{prefix}_level"]  = pdata.get("level", 1)
            player_data[f"{prefix}_xp"]     = pdata.get("xp", 0)

        # KDA aus participantFrames fehlt leider — Kills/Deaths/Assists kumuliert
        # aus Events hochzählen (zuverlässiger)
        kda = {str(i+1): {"k": 0, "d": 0, "a": 0} for i in range(10)}
        for f2 in frames:
            if f2.get("timestamp", 0) > frame["timestamp"]:
                break
            for e in f2.get("events", []):
                if e.get("type") == "CHAMPION_KILL":
                    kid = str(e.get("killerId", 0))
                    vid = str(e.get("victimId", 0))
                    if kid in kda: kda[kid]["k"] += 1
                    if vid in kda: kda[vid]["d"] += 1
                    for a in e.get("assistingParticipantIds", []):
                        aid = str(a)
                        if aid in kda: kda[aid]["a"] += 1

        for i in range(10):
            prefix = f"p{i+1}"
            pkey   = str(i + 1)
            player_data[f"{prefix}_kills"]  = kda[pkey]["k"]
            player_data[f"{prefix}_deaths"] = kda[pkey]["d"]
            player_data[f"{prefix}_assists"]= kda[pkey]["a"]

        # ── Label & Meta ───────────────────────────────────────────────────
        own_team_label = "blue" if own_team_id == 100 else "red"
        won = 1 if winner_team == own_team_id else 0

        row = {
            "match_id":       match_id,
            "queue_id":       queue_id,
            "minute":         minute,
            "own_team":       own_team_label,
            "own_champ":      own_champ,
            "result":         won,
            # Team-Aggregate
            "gold_blue":      team_gold[100],
            "gold_red":       team_gold[200],
            "gold_diff":      team_gold[100] - team_gold[200],
            "cs_blue":        team_cs[100],
            "cs_red":         team_cs[200],
            **obj,
            # Alle Spieler
            **player_data,
        }
        snapshots.append(row)

    return snapshots

# ── CSV Writer ─────────────────────────────────────────────────────────────────

def write_rows(rows, filepath, write_header):
    if not rows:
        return
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  LoL Dataset Collector")
    print("═" * 60)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("RiotAPI", "").strip()
    riot_id   = input("  Riot ID (Name#TAG):       ").strip()

    if "#" not in riot_id:
        print("  ❌ Riot ID muss im Format 'Name#TAG' sein.")
        return

    game_name, tag_line = riot_id.rsplit("#", 1)

    print(f"\n  Suche PUUID für {game_name}#{tag_line}...")
    puuid = get_puuid(api_key, game_name, tag_line)
    if not puuid:
        print("  ❌ Spieler nicht gefunden. API Key oder Riot ID prüfen.")
        return
    print(f"  ✓ PUUID: {puuid[:16]}...")

    # Match IDs sammeln
    print(f"\n  Lade letzte {GAMES_TO_FETCH} ranked Matches...")
    all_ids = get_match_ids(api_key, puuid, count=GAMES_TO_FETCH)
    print(f"  ✓ {len(all_ids)} gefunden")

    # Deduplizieren (theoretisch nicht nötig, aber sicher ist sicher)
    all_ids = list(dict.fromkeys(all_ids))[:GAMES_TO_FETCH]
    print(f"\n  Gesamt: {len(all_ids)} Matches\n")

    # Output vorbereiten
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    total_rows = 0
    write_header = True

    for i, match_id in enumerate(all_ids):
        print(f"  [{i+1:>3}/{len(all_ids)}] {match_id}", end=" ... ")

        info = get_match_info(api_key, match_id)
        if not info:
            print("skip (kein Info)")
            continue

        # Queue nochmal prüfen (Sicherheit)
        if info["info"].get("queueId") not in QUEUE_IDS:
            print("skip (falscher Queue)")
            continue

        timeline = get_match_timeline(api_key, match_id)
        if not timeline:
            print("skip (keine Timeline)")
            continue

        rows = extract_snapshots(info, timeline, puuid)
        write_rows(rows, OUTPUT_FILE, write_header)
        write_header = False
        total_rows  += len(rows)

        print(f"{len(rows)} snapshots")

    print(f"\n{'═'*60}")
    print(f"  ✓ Fertig! {total_rows} Zeilen in '{OUTPUT_FILE}'")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()