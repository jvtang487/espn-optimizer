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

week = 4 
matchups = league.box_scores(week)
my_matchup = next(m for m in matchups if m.home_team.team_id == team_id or m.away_team.team_id == team_id)


# Figure out if you're home or away
if my_matchup.home_team.team_id == team_id:
    my_lineup = my_matchup.home_lineup
    opponent_lineup = my_matchup.away_lineup
else:
    my_lineup = my_matchup.away_lineup
    opponent_lineup = my_matchup.home_lineup

def adjust_projection(proj, rank, fppg, weight_proj=0.35, weight_fppg=0.65):
    # Rank 1 = toughest, Rank 32 = easiest
    # Scale: 0.8x for rank 1 → 1.2x for rank 32
    scale = np.interp(rank, [1, 32], [0.47, 1.2])

    baseline = (proj * weight_proj) + (fppg * weight_fppg)
    return baseline * scale


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


df = pd.DataFrame(lineup_data)
def greedy_lineup(df, points_col="Adjusted_Projected"):
    lineup = []

    # QB
    qb = df[df["Position"] == "QB"].sort_values(points_col, ascending=False).head(1)
    lineup.append((qb.iloc[0]["Name"], "QB", qb.iloc[0]["Adjusted_Projected"],qb.iloc[0]["Projected"],qb.iloc[0]["Actual"]))

    # RBs
    rbs = df[df["Position"] == "RB"].sort_values(points_col, ascending=False).head(2)
    for _, row in rbs.iterrows():
        lineup.append((row["Name"], "RB", row["Adjusted_Projected"],row["Projected"],row["Actual"]))

    # WRs
    wrs = df[df["Position"] == "WR"].sort_values(points_col, ascending=False).head(2)
    for _, row in wrs.iterrows():
        lineup.append((row["Name"], "WR", row["Adjusted_Projected"],row["Projected"],row["Actual"]))

    # TE
    te = df[df["Position"] == "TE"].sort_values(points_col, ascending=False).head(1)
    lineup.append((te.iloc[0]["Name"], "TE", te.iloc[0]["Adjusted_Projected"],te.iloc[0]["Projected"],te.iloc[0]["Actual"]))

    # Kicker
    k = df[df["Position"] == "K"].sort_values(points_col, ascending=False).head(1)
    lineup.append((k.iloc[0]["Name"], "K", k.iloc[0]["Adjusted_Projected"],k.iloc[0]["Projected"],k.iloc[0]["Actual"]))

    # Defense
    d = df[df["Position"].isin(["DEF", "D/ST"])].sort_values(points_col, ascending=False).head(1)
    lineup.append((d.iloc[0]["Name"], "DEF", d.iloc[0]["Adjusted_Projected"],d.iloc[0]["Projected"],d.iloc[0]["Actual"]))

    # FLEX (best remaining RB/WR/TE not already chosen)
    taken_names = [x[0] for x in lineup]
    flex_pool = df[df["Position"].isin(["RB", "WR", "TE"]) & ~df["Name"].isin(taken_names)]
    flex = flex_pool.sort_values(points_col, ascending=False).head(1)
    lineup.append((flex.iloc[0]["Name"], "FLEX", flex.iloc[0]["Adjusted_Projected"],flex.iloc[0]["Projected"],flex.iloc[0]["Actual"]))

    # Final lineup DataFrame
    return pd.DataFrame(lineup, columns=["Name", "Role", "Adjusted_Projected", "Projected", "Actual"])
myOptimizedLineup = greedy_lineup(df, "Adjusted_Projected")
espnOptimized = greedy_lineup(df, "Projected")
print(myOptimizedLineup)
print("Total Adjusted Projected Points:", myOptimizedLineup["Adjusted_Projected"].sum())
print("Total Projected Points:", myOptimizedLineup["Projected"].sum())
print("Total Actual Points:", myOptimizedLineup["Actual"].sum())
print(espnOptimized)
print("Total Adjusted Projected Points:", espnOptimized["Adjusted_Projected"].sum())
print("Total Projected Points:", espnOptimized["Projected"].sum())
print("Total Actual Points:", espnOptimized["Actual"].sum())