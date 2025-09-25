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
        "Slot": player.slot_position,
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
    qb = df[(df["Position"] == "QB") & (df["Slot"] != "BE")].sort_values(points_col, ascending=False).head(1)
    lineup.append((qb.iloc[0]["Name"], "QB", qb.iloc[0]["Adjusted_Projected"], qb.iloc[0]["Projected"], qb.iloc[0]["Actual"]))

    # RBs
    rbs = df[(df["Position"] == "RB") & (df["Slot"] != "BE")].sort_values(points_col, ascending=False).head(2)
    for _, row in rbs.iterrows():
        lineup.append((row["Name"], "RB", row["Adjusted_Projected"], row["Projected"], row["Actual"]))

    # WRs
    wrs = df[(df["Position"] == "WR") & (df["Slot"] != "BE")].sort_values(points_col, ascending=False).head(2)
    for _, row in wrs.iterrows():
        lineup.append((row["Name"], "WR", row["Adjusted_Projected"], row["Projected"], row["Actual"]))

    # TE
    te = df[(df["Position"] == "TE") & (df["Slot"] != "BE")].sort_values(points_col, ascending=False).head(1)
    lineup.append((te.iloc[0]["Name"], "TE", te.iloc[0]["Adjusted_Projected"], te.iloc[0]["Projected"], te.iloc[0]["Actual"]))

    # Kicker
    k = df[(df["Position"] == "K") & (df["Slot"] != "BE")].sort_values(points_col, ascending=False).head(1)
    lineup.append((k.iloc[0]["Name"], "K", k.iloc[0]["Adjusted_Projected"], k.iloc[0]["Projected"], k.iloc[0]["Actual"]))

    # Defense
    d = df[(df["Position"].isin(["DEF", "D/ST"])) & (df["Slot"] != "BE")].sort_values(points_col, ascending=False).head(1)
    lineup.append((d.iloc[0]["Name"], "DEF", d.iloc[0]["Adjusted_Projected"], d.iloc[0]["Projected"], d.iloc[0]["Actual"]))

    # FLEX (best remaining RB/WR/TE not already chosen, excluding bench)
    taken_names = [x[0] for x in lineup]
    flex_pool = df[df["Position"].isin(["RB", "WR", "TE"]) & (df["Slot"] != "BE") & ~df["Name"].isin(taken_names)]
    flex = flex_pool.sort_values(points_col, ascending=False).head(1)
    if not flex.empty:
        lineup.append((flex.iloc[0]["Name"], "FLEX", flex.iloc[0]["Adjusted_Projected"], flex.iloc[0]["Projected"], flex.iloc[0]["Actual"]))

    # Final lineup DataFrame
    optimized_lineup = pd.DataFrame(lineup, columns=["Name", "Role", "Adjusted_Projected", "Projected", "Actual"])

    # Bench sorted by adjusted projections
    bench = df[df["Slot"] == "BE"].sort_values(points_col, ascending=False)
    bench_display = bench[["Name", "Position", "Adjusted_Projected", "Projected", "Actual"]].reset_index(drop=True)

    lineup_totals = {
        "Total_Adjusted": optimized_lineup["Adjusted_Projected"].sum(),
        "Total_Projected": optimized_lineup["Projected"].sum(),
        "Total_Actual": optimized_lineup["Actual"].sum(),
    }

    bench_totals = {
        "Total_Adjusted": bench_display["Adjusted_Projected"].sum(),
        "Total_Projected": bench_display["Projected"].sum(),
        "Total_Actual": bench_display["Actual"].sum(),
    }

    return optimized_lineup, bench_display, lineup_totals, bench_totals

myOptimizedLineup, myBench, lineup_totals, bench_totals = greedy_lineup(df, "Adjusted_Projected")
espnLineup, espnBench, espnlineup_totals, espnbench_totals = greedy_lineup(df, "Projected")

print("=== Optimized Lineup by Adj Proj ===")
print(myOptimizedLineup)
print("Lineup Totals:", lineup_totals)

print("\n=== Bench (sorted by Adjusted Projection) ===")
print(myBench)
print("Bench Totals:", bench_totals)

print("=== Optimized Lineup by Projection ===")
print(espnLineup)
print("Lineup Totals:", espnlineup_totals)

print("\n=== Bench (sorted by Projection) ===")
print(espnBench)
print("Bench Totals:", espnbench_totals)