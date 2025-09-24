from espn_api.football import League
import numpy as np
import pandas as pd
import pulp

league = League(league_id=1821237400, 
                year=2025, 
                espn_s2="AEBCl45LNO4%2FYiR8%2BVvz3OZsi0Quj3hBiHdLNLuQ8rBll4JymvtmMMmWsatNxXJb2LF%2BVNdPHzrsOpm9kHrqyCU%2B2Cui%2BwagofEMzlcOCK32xm1hoMAXyQc%2FmCLFaqIOY5C%2BwNFxFvtheAd0scKPCfZpEYG102%2F6AgMJxei7v6JTUf5crqcDD9GV889uUoIPPKo7t4nc%2F2W7Ke7k3804q%2B8k3yq4eQq1xYFYp6lr01KWEh7dO3DDXlSWpKJJ8JQpFx6ia0UlIXT39O2lRlxokODiz8liFax%2FNcnvm3uTmaGGkg%3D%3D", 
                swid="{70B0D541-7666-4B31-8869-4361D56693B2}")
team_id = 3
my_team = league.teams[team_id-1]  # pick your team


print("My team:", my_team.team_name)

week = 3 
matchups = league.box_scores(week)
my_matchup = next(m for m in matchups if m.home_team.team_id == team_id or m.away_team.team_id == team_id)


# Figure out if you're home or away
if my_matchup.home_team.team_id == team_id:
    my_lineup = my_matchup.home_lineup
    opponent_lineup = my_matchup.away_lineup
else:
    my_lineup = my_matchup.away_lineup
    opponent_lineup = my_matchup.home_lineup

def adjust_projection(proj, rank, fppg, weight_proj=0.4, weight_fppg=0.6):
    # Rank 1 = toughest, Rank 32 = easiest
    # Scale: 0.8x for rank 1 → 1.2x for rank 32
    scale = np.interp(rank, [1, 32], [0.45, 1.2])

    baseline = (proj * weight_proj) + (fppg * weight_fppg)
    return baseline * scale

projected = 0
adj_projected = 0
actual = 0

lineup_data = []

for player in my_lineup:
    adj_proj = adjust_projection(player.projected_points, player.pro_pos_rank, league.player_info(player.name).avg_points)
    
    lineup_data.append({
        "Name": player.name,
        "Position": player.position,
        "Projected": player.projected_points,
        "Actual": player.points,
        "Adjusted_Projected": adj_proj,
        "Opp": player.pro_opponent,
        "Opp_Rank": player.pro_pos_rank,
    })

    projected += player.projected_points
    adj_projected += adj_proj
    actual += player.points

df = pd.DataFrame(lineup_data)
df["is_QB"] = (df["Position"] == "QB").astype(int)
df["is_RB"] = (df["Position"] == "RB").astype(int)
df["is_WR"] = (df["Position"] == "WR").astype(int)
df["is_TE"] = (df["Position"] == "TE").astype(int)
df["is_K"]  = (df["Position"] == "K").astype(int)
df["is_DEF"] = (df["Position"] == "D/ST").astype(int)

# Create decision variables for each player (1 if selected, 0 otherwise)
player_vars = {i: pulp.LpVariable(f"player_{i}", cat="Binary") for i in df.index}

# Define problem
prob = pulp.LpProblem("Fantasy_Lineup", pulp.LpMaximize)

# Objective: maximize total adjusted projected points
prob += pulp.lpSum([df.loc[i, "Adjusted_Projected"] * player_vars[i] for i in df.index])

# Constraints
prob += pulp.lpSum([df.loc[i, "is_QB"] * player_vars[i] for i in df.index]) == 1
prob += pulp.lpSum([df.loc[i, "is_RB"] * player_vars[i] for i in df.index]) == 2
prob += pulp.lpSum([df.loc[i, "is_WR"] * player_vars[i] for i in df.index]) == 2
prob += pulp.lpSum([df.loc[i, "is_TE"] * player_vars[i] for i in df.index]) == 1
prob += pulp.lpSum([df.loc[i, "is_K"]  * player_vars[i] for i in df.index]) == 1
prob += pulp.lpSum([df.loc[i, "is_DEF"] * player_vars[i] for i in df.index]) == 1

# FLEX: one more from RB/WR/TE
prob += pulp.lpSum([
    (df.loc[i, "is_RB"] + df.loc[i, "is_WR"] + df.loc[i, "is_TE"]) * player_vars[i] 
    for i in df.index
]) == 6

# Solve
prob.solve()

optimal_lineup = df[[player_vars[i].value() == 1 for i in df.index]]
print(optimal_lineup[["Name", "Position", "Adjusted_Projected", "Projected", "Actual"]])
print("Total Projected Points:", optimal_lineup["Adjusted_Projected"].sum(), "Total Actual:", optimal_lineup["Actual"].sum())
#print("Actual: ", actual, " Projected:", projected, " Diff: ", actual - projected, " Adj Proj :", adj_projected, " Diff: ", actual - adj_projected)
