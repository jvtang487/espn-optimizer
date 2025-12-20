import streamlit as st
import pandas as pd
import numpy as np
import pulp
import requests
from espn_api.football import League

# --- 1. CONFIGURATION & MAPPING ---
TEAM_MAP = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WSH': 'Washington Commanders'
}

POSITIONAL_SCALES = {
    'QB': [0.85, 1.10], 'RB': [0.75, 1.25], 'WR': [0.80, 1.20],
    'TE': [0.70, 1.15], 'DEF': [0.60, 1.40], 'K': [0.95, 1.05]
}

# --- 2. HELPER FUNCTIONS ---
@st.cache_data
def get_vegas_totals(api_key):
    # If no API key, return empty dict to avoid crashing
    if not api_key or api_key == "your_key_here": return {}
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {'apiKey': api_key, 'regions': 'us', 'markets': 'totals', 'oddsFormat': 'american'}
    try:
        response = requests.get(url, params=params).json()
        return {game['home_team']: game['bookmakers'][0]['markets'][0]['outcomes'][0]['point'] for game in response}
    except: return {}

def solve_lineup_pulp(df, points_col):
    prob = pulp.LpProblem("Optimizer", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("p", df.index, cat='Binary')
    prob += pulp.lpSum([df.loc[i, points_col] * player_vars[i] for i in df.index])
    
    # Roster Constraints
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 9
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'QB']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'RB']) >= 2
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'WR']) >= 2
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'TE']) >= 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] in ['RB', 'WR', 'TE']]) == 6
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'K']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] in ['D/ST', 'DEF']]) == 1
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return [i for i in df.index if player_vars[i].varValue == 1]



# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Fantasy Strategist", layout="wide")
st.title("Fantasy Football Lineup Optimizer")

with st.sidebar:
    st.header("Settings")
    league_id = st.number_input("League ID", value=1821237400)
    year = st.number_input("Year", value=2025)
    swid = st.text_input("SWID", value="{70B0D541-7666-4B31-8869-4361D56693B2}", type="password")
    s2 = st.text_input("ESPN_S2", value="AEBCl...", type="password")
    team_ID = st.number_input("Team ID", value=2)
    vegas_key = st.text_input("Odds API Key", value="your_key_here")
    week = st.slider("Week", 1, 18, 4)

if st.button("Run Analysis"):
    # 1. Connect to ESPN
    league = League(league_id=league_id, year=year, espn_s2=s2, swid=swid)
    team = league.teams[team_ID - 1] # Adjust logic to find specific team
    
    # 2. Get Data
    matchups = league.box_scores(week)
    my_matchup = next(m for m in matchups if m.home_team.team_id == team.team_id or m.away_team.team_id == team.team_id)
    lineup = my_matchup.home_lineup if my_matchup.home_team.team_id == team.team_id else my_matchup.away_lineup
    
    # 3. Build DataFrame
    data = []
    totals_dict = get_vegas_totals(vegas_key)
    for p in lineup:
        full_name = TEAM_MAP.get(p.proTeam, "Unknown")
        game_total = totals_dict.get(full_name, 44.0)
        
        # Adjustment Math
        low, high = POSITIONAL_SCALES.get(p.position, [0.9, 1.1])
        matchup_scale = np.interp(p.pro_pos_rank, [1, 32], [low, high])
        vegas_scale = 1.1 if game_total > 50 else (0.9 if game_total < 40 else 1.0)
        
        adj_proj = (p.projected_points * 0.4) * matchup_scale * vegas_scale
        
        data.append({"Name": p.name, "Position": p.position, "Projected": p.projected_points, "Adj_Proj": adj_proj})
    
    df = pd.DataFrame(data)
    
    # 4. Solve
    starter_indices = solve_lineup_pulp(df, "Adj_Proj")
    df['Status'] = ["🚀 Starter" if i in starter_indices else "📋 Bench" for i in df.index]
    
    # 5. Display Result
    st.dataframe(df.sort_values("Status", ascending=False), use_container_width=True)