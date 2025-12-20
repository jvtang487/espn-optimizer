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

# --- 2. ROBUST HELPER FUNCTIONS ---
@st.cache_data(ttl=3600) # Cache this for 1 hour to save API calls
def get_vegas_totals(api_key):
    """
    Fetches odds from The Odds API.
    Returns empty dict if key is missing or invalid (Safe Fallback).
    """
    if not api_key or "your_key" in api_key: 
        return {}
    
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {
        'apiKey': api_key, 
        'regions': 'us', 
        'markets': 'totals', 
        'oddsFormat': 'american'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return {}
            
        data = response.json()
        totals = {}
        for game in data:
            # Get the Over/Under from the first available bookmaker
            try:
                line = game['bookmakers'][0]['markets'][0]['outcomes'][0]['point']
                totals[game['home_team']] = line
                totals[game['away_team']] = line
            except (IndexError, KeyError):
                continue
        return totals
    except Exception:
        return {}

def solve_lineup_pulp(df, points_col):
    """
    Uses Linear Programming to find the optimal roster.
    """
    # Create the LP problem
    prob = pulp.LpProblem("Fantasy_Optimizer", pulp.LpMaximize)
    
    # Decision Variables: 1 if selected, 0 if not
    player_vars = pulp.LpVariable.dicts("player", df.index, cat='Binary')
    
    # Objective: Maximize Points
    prob += pulp.lpSum([df.loc[i, points_col] * player_vars[i] for i in df.index])
    
    # --- CONSTRAINTS ---
    # 1. Total Starters = 9
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 9
    
    # 2. Positional Requirements
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'QB']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'RB']) >= 2
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'WR']) >= 2
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'TE']) >= 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] == 'K']) == 1
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] in ['D/ST', 'DEF']]) == 1
    
    # 3. Flex Control (RB + WR + TE must equal 6 total slots)
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'Position'] in ['RB', 'WR', 'TE']]) == 6

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Return list of indices that were selected
    return [i for i in df.index if player_vars[i].varValue == 1]

# --- 3. STREAMLIT APP LOGIC ---
st.set_page_config(page_title="Fantasy Strategist", layout="wide", page_icon="🏈")
st.title("🏈 Fantasy Football Lineup Optimizer")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ League Settings")
    league_id = st.number_input("League ID", value=1821237400)
    year = st.number_input("Year", value=2025)
    swid = st.text_input("SWID (Cookies)", value="{70B0D541...}", type="password")
    espn_s2 = st.text_input("ESPN_S2 (Cookies)", value="AEBCl...", type="password")
    team_id = st.number_input("Team ID", value=2)
    
    st.divider()
    
    st.header("🎲 Vegas Settings")
    vegas_key = st.text_input("Odds API Key (Optional)", type="password")
    st.info("If no key is provided, all games default to neutral (44.0 pts).")
    
    week = st.slider("Select Week", min_value=1, max_value=18, value=4)
    
    run_btn = st.button("🚀 Run Analysis", type="primary")

if run_btn:
    try:
        with st.spinner('Connecting to ESPN...'):
            # Initialize connection
            league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
            
            # Find YOUR team (logic: usually team_id matches index+1, but safer to search)
            # For this demo, we assume you are Team 3. Ideally, add a selector.
            my_team = league.teams[team_id - 1] 

            # Get the Box Score for the specific week
            box_scores = league.box_scores(week)
            
            # Find the match involving your team
            my_matchup = None
            for match in box_scores:
                if match.home_team == my_team or match.away_team == my_team:
                    my_matchup = match
                    break
            
            if not my_matchup:
                st.error("Could not find a matchup for this team in this week.")
                st.stop()

            # Determine which lineup list to use (Home vs Away)
            if my_matchup.home_team == my_team:
                roster = my_matchup.home_lineup
            else:
                roster = my_matchup.away_lineup

        # --- DATA PROCESSING ---
        st.success(f"Fetched Roster for: **{my_team.team_name}**")
        
        totals_dict = get_vegas_totals(vegas_key)
        roster_data = []

        for p in roster:
            if p.position == 'DT' or p.position == 'IR': continue # Skip injured/bench slots if needed

            # 1. Map Team Name
            full_team_name = TEAM_MAP.get(p.proTeam, "Unknown")
            
            # 2. Get Vegas Total (Default to 44.0 if missing)
            game_total = totals_dict.get(full_team_name, 44.0)
            
            # 3. Calculate Vegas Multiplier
            if game_total > 50: vegas_scale = 1.10    # Shootout Boost
            elif game_total < 40: vegas_scale = 0.90  # Slugfest Penalty
            else: vegas_scale = 1.0
            
            # 4. Calculate Matchup Scale
            low, high = POSITIONAL_SCALES.get(p.position, [0.9, 1.1])
            # Handle cases where rank might be 0 (bye week)
            opp_rank = p.pro_pos_rank if p.pro_pos_rank > 0 else 16 
            matchup_scale = np.interp(opp_rank, [1, 32], [low, high])

            # --- CRITICAL FIX: HANDLE ZERO AVERAGE ---
            # If avg_points is 0 (hasn't played enough), use projected as a proxy
            # This prevents the 65% weight from crushing their score.
            try:
                avg_pts = league.player_info(p.name).avg_points
            except:
                avg_pts = p.projected_points

            # 5. Weighted Formula
            baseline = (p.projected_points * 0.4) + (avg_pts * 0.6)
            adjusted_proj = baseline * matchup_scale * vegas_scale
            
            roster_data.append({
                "Name": p.name,
                "Position": p.position,
                "Team": p.proTeam,
                "Opp_Rank": p.pro_pos_rank,
                "Projected": p.projected_points,
                "Avg_Points": avg_pts,
                "Adj_Proj": adjusted_proj,
                "Vegas_Total": game_total,
                "Actual": p.points
            })

        df = pd.DataFrame(roster_data)

        # --- OPTIMIZATION ---
        starter_indices = solve_lineup_pulp(df, "Adj_Proj")
        
        # Label rows
        df['Status'] = df.index.map(lambda x: "🚀 Starter" if x in starter_indices else "📋 Bench")
        
        # Sort for display (Starters first, then by score)
        df['Sort_Status'] = pd.Categorical(df['Status'], ["🚀 Starter", "📋 Bench"])
        df = df.sort_values(["Sort_Status", "Adj_Proj"], ascending=[True, False]).drop(columns=["Sort_Status"])

        # --- DISPLAY ---
        
        # 1. Summary Metrics
        st.subheader("📊 Optimization Results")
        col1, col2 = st.columns(2)
        starters = df[df['Status'] == "🚀 Starter"]
        
        with col1:
            st.metric("Projected Lineup Score", f"{starters['Adj_Proj'].sum():.1f}")
        with col2:
            st.metric("ESPN Standard Projection", f"{starters['Projected'].sum():.1f}")

        # 2. Main Table with Highlighting
        st.markdown("### Your Optimized Lineup")
        
        def highlight_starters(s):
            return ['background-color: #d1e7dd; color: black' if s.Status == "🚀 Starter" else '' for _ in s]

        st.dataframe(
            df.style.apply(highlight_starters, axis=1)
              .format({"Projected": "{:.1f}", "Avg_Points": "{:.1f}", "Adj_Proj": "{:.1f}", "Vegas_Total": "{:.1f}"}),
            use_container_width=True,
            height=600
        )

        # 3. Debug Section (To see why scores are low)
        with st.expander("🕵️ Debug: See Calculation Details"):
            st.write("If 'Avg_Points' is 0, we used 'Projected' to prevent the score from dropping too low.")
            st.dataframe(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Double check your League ID, Year, and Cookies (SWID/S2).")