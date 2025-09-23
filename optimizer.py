from espn_api.football import League
import numpy as np

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

def adjust_projection(proj, rank, fppg, weight_proj=0.63, weight_fppg=0.37):
    # Rank 1 = toughest, Rank 32 = easiest
    # Scale: 0.8x for rank 1 → 1.2x for rank 32
    scale = np.interp(rank, [1, 32], [0.525, 1.35])

    baseline = (proj * weight_proj) + (fppg * weight_fppg)
    return baseline * scale

projected = 0
adj_projected = 0
actual = 0
    
for player in my_lineup:
    adj_proj = adjust_projection(player.projected_points, player.pro_pos_rank, league.player_info(player.name).avg_points)
    print(player.name, 
          "Projected:", player.projected_points, 
          "Actual:", player.points,
          "Opp Rank:", player.pro_opponent, player.pro_pos_rank,
          "Adj Proj", adj_proj)

    projected += player.projected_points
    adj_projected += adj_proj
    actual += player.points
    
print("Actual: ", actual, " Projected:", projected, " Diff: ", actual - projected, " Adj Proj :", adj_projected, " Diff: ", actual - adj_projected)
