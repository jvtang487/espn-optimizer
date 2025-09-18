from espn_api.football import League
import requests
import pandas as pd

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
    
# Convert lineup to a DataFrame
data = []
for p in my_lineup:
    data.append({
        "Slot": p.slot_position,
        "Name": p.name,
        "Pos": p.position,
        "Team": p.proTeam,
        "Opp": p.pro_opponent,
        "Proj_Points": p.projected_points,
        "Actual_Points": p.points
    })


teamdf = pd.DataFrame(data)



positions = ["qb", "rb", "wr", "te", "k", "dst"]
dvp_data = {}

for pos in positions:
    url = f"https://www.fantasypros.com/nfl/points-allowed/{pos}.php"
    tables = pd.read_html(requests.get(url).text)

    # First table is the one we need
    rb_dvp = tables[0]

    # Clean up team abbreviations
    rb_dvp = rb_dvp.rename(columns={"Team": "Opp"})
    rb_dvp["Opp"] = rb_dvp["Opp"].str.extract(r"([A-Z]{2,3})")  # e.g., "DAL", "PHI"
    rb_dvp = rb_dvp[["Opp", "FPTS/G"]]  # Fantasy points allowed per game
    rb_dvp.head()

# Example: RB defense vs position
#print(dvp_data["RB"].head())
rb_df = teamdf[teamdf["Pos"] == "RB"].copy()

# Merge on Opponent
rb_df = rb_df.merge(rb_dvp, on="Opp", how="left")

print(rb_df[["Name", "Pos", "Proj_Points", "Opp", "FPTS/G"]])