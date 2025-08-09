#!/usr/bin/env python3
"""
Team selector using CSV data for playing 11s
"""

import pandas as pd
import numpy as np

class CSVTeamSelector:
    def __init__(self, csv_file='latest_teams_data.csv'):
        """Initialize with CSV data"""
        self.df = pd.read_csv(csv_file)
        self.england_players = self.df[self.df['Team'] == 'England'].to_dict('records')
        self.australia_players = self.df[self.df['Team'] == 'Australia'].to_dict('records')
    
    def select_playing_11_for_venue(self, team, venue, pitch_conditions):
        """
        Select playing 11 for a specific team and venue based on pitch conditions
        """
        if team.lower() == 'england':
            players = self.england_players
        else:
            players = self.australia_players
        
        # Categorize players
        batters = [p for p in players if 'Batter' in p['Role'] or p['Role'] == 'Opener' or p['Role'] == 'Top_Order_Batter']
        allrounders = [p for p in players if 'Allrounder' in p['Role'] or p['Role'] == 'Allrounder']
        bowlers = [p for p in players if 'Bowler' in p['Role'] or p['Role'] == 'Fast_Bowler' or p['Role'] == 'Spinner']
        
        # Sort players based on pitch conditions
        self._sort_players_by_conditions(batters, allrounders, bowlers, pitch_conditions, venue)
        
        # Select playing 11
        playing_11 = []
        selected_names = set()
        
        # 1. Openers (2)
        openers_selected = 0
        for player in batters:
            if player['Role'] == 'Opener' and player['Player'] not in selected_names and openers_selected < 2:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Opening batsman with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                openers_selected += 1
        
        # 2. Number 3
        for player in batters:
            if player['Role'] == 'Top_Order_Batter' and player['Player'] not in selected_names:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Number 3 batsman with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                break
        
        # 3. Middle Order (2)
        middle_order_selected = 0
        for player in batters:
            if player['Role'] == 'Middle_Order_Batter' and player['Player'] not in selected_names and middle_order_selected < 2:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Middle order batsman with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                middle_order_selected += 1
        
        # 4. All-rounder
        for player in allrounders:
            if player['Player'] not in selected_names:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"All-rounder with {player['Batting_Avg']} batting and {player['Bowling_Avg']} bowling average"
                })
                selected_names.add(player['Player'])
                break
        
        # 5. Wicket-keeper
        for player in batters:
            if player['Role'] == 'Wicketkeeper_Batter' and player['Player'] not in selected_names:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Wicket-keeper batsman with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                break
        
        # 6. Bowlers (4)
        # Fast bowlers first - ensure Starc is always selected for Australia
        fast_bowlers_selected = 0
        
        # First, always select Starc if available for Australia
        if team.lower() == 'australia':
            for player in bowlers:
                if 'Starc' in player['Player'] and player['Player'] not in selected_names and 'Fast_Bowler' in player['Role']:
                    playing_11.append({
                        "name": player['Player'],
                        "role": player['Role'],
                        "position": str(len(playing_11) + 1),
                        "batting_avg": player['Batting_Avg'],
                        "bowling_avg": player['Bowling_Avg'],
                        "reason": f"Fast bowler with {player['Bowling_Avg']} bowling average"
                    })
                    selected_names.add(player['Player'])
                    fast_bowlers_selected += 1
                    break
        
        # Then select other fast bowlers
        for player in bowlers:
            if 'Fast_Bowler' in player['Role'] and player['Player'] not in selected_names and fast_bowlers_selected < 3:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Fast bowler with {player['Bowling_Avg']} bowling average"
                })
                selected_names.add(player['Player'])
                fast_bowlers_selected += 1
        
        # Add captain and vice captain information
        captain = None
        vice_captain = None
        
        if team.lower() == 'australia':
            # Find Cummins as captain
            for player in playing_11:
                if 'Cummins' in player['name']:
                    captain = player['name']
                    break
        elif team.lower() == 'england':
            # Find Stokes as captain
            for player in playing_11:
                if 'Stokes' in player['name']:
                    captain = player['name']
                    break
        
        # Add captain and vice captain to each player
        for player in playing_11:
            player['captain'] = captain
            player['vice_captain'] = vice_captain
        
        # Spinner
        for player in bowlers:
            if 'Spinner' in player['Role'] and player['Player'] not in selected_names:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Spinner with {player['Bowling_Avg']} bowling average"
                })
                selected_names.add(player['Player'])
                break
        
        # Fill remaining positions if needed
        while len(playing_11) < 11:
            for player in players:
                if player['Player'] not in selected_names:
                    playing_11.append({
                        "name": player['Player'],
                        "role": player['Role'],
                        "position": str(len(playing_11) + 1),
                        "batting_avg": player['Batting_Avg'],
                        "bowling_avg": player['Bowling_Avg'],
                        "reason": f"Additional player with {player['Batting_Avg']} batting and {player['Bowling_Avg']} bowling average"
                    })
                    selected_names.add(player['Player'])
                    break
        
        return playing_11
    
    def get_series_combined_11(self):
        """
        Generate combined 11 for the entire series using CSV data only
        """
        all_players = self.england_players + self.australia_players
        
        # Categorize all players
        batters = [p for p in all_players if 'Batter' in p['Role'] or p['Role'] == 'Opener' or p['Role'] == 'Top_Order_Batter']
        allrounders = [p for p in all_players if 'Allrounder' in p['Role'] or p['Role'] == 'Allrounder']
        bowlers = [p for p in all_players if 'Bowler' in p['Role'] or p['Role'] == 'Fast_Bowler' or p['Role'] == 'Spinner']
        
        # Sort by performance
        batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
        allrounders.sort(key=lambda x: (float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0) + 
                                    (float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50), reverse=True)
        bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
        
        # Always prefer Starc over Boland and Cummins
        if any('Starc' in p['Player'] for p in bowlers):
            starc_players = [p for p in bowlers if 'Starc' in p['Player']]
            cummins_players = [p for p in bowlers if 'Cummins' in p['Player']]
            boland_players = [p for p in bowlers if 'Boland' in p['Player']]
            other_bowlers = [p for p in bowlers if 'Starc' not in p['Player'] and 'Cummins' not in p['Player'] and 'Boland' not in p['Player']]
            bowlers = starc_players + cummins_players + boland_players + other_bowlers
        
        combined_11 = []
        selected_names = set()
        
        # Select best players for each position
        # 1. Openers (2)
        openers_selected = 0
        for player in batters:
            if player['Role'] == 'Opener' and player['Player'] not in selected_names and openers_selected < 2:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best opener with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                openers_selected += 1
        
        # 2. Number 3
        for player in batters:
            if player['Role'] == 'Top_Order_Batter' and player['Player'] not in selected_names:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best number 3 with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                break
        
        # 3. Middle Order (2)
        middle_order_selected = 0
        for player in batters:
            if player['Role'] == 'Middle_Order_Batter' and player['Player'] not in selected_names and middle_order_selected < 2:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best middle order with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                middle_order_selected += 1
        
        # 4. All-rounder
        for player in allrounders:
            if player['Player'] not in selected_names:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best all-rounder with {player['Batting_Avg']} batting and {player['Bowling_Avg']} bowling"
                })
                selected_names.add(player['Player'])
                break
        
        # 5. Wicket-keeper
        for player in batters:
            if player['Role'] == 'Wicketkeeper_Batter' and player['Player'] not in selected_names:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best wicket-keeper with {player['Batting_Avg']} average"
                })
                selected_names.add(player['Player'])
                break
        
        # 6. Bowlers (4)
        # Fast bowlers first - ensure Starc is always selected
        fast_bowlers_selected = 0
        
        # First, always select Starc if available
        for player in bowlers:
            if 'Starc' in player['Player'] and player['Player'] not in selected_names and 'Fast_Bowler' in player['Role']:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best fast bowler with {player['Bowling_Avg']} bowling average"
                })
                selected_names.add(player['Player'])
                fast_bowlers_selected += 1
                break
        
        # Then select other fast bowlers
        for player in bowlers:
            if 'Fast_Bowler' in player['Role'] and player['Player'] not in selected_names and fast_bowlers_selected < 3:
                combined_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(combined_11) + 1),
                    "team": player['Team'],
                    "batting_avg": player['Batting_Avg'],
                    "bowling_avg": player['Bowling_Avg'],
                    "reason": f"Best fast bowler with {player['Bowling_Avg']} bowling average"
                })
                selected_names.add(player['Player'])
                fast_bowlers_selected += 1
        
        # Spinner - ensure Sohaib is selected if available
        spinner_selected = False
        for player in bowlers:
            if 'Spinner' in player['Role'] and player['Player'] not in selected_names:
                # Prioritize Sohaib if available
                if 'Sohaib' in player['Player'] or not spinner_selected:
                    combined_11.append({
                        "name": player['Player'],
                        "role": player['Role'],
                        "position": str(len(combined_11) + 1),
                        "team": player['Team'],
                        "batting_avg": player['Batting_Avg'],
                        "bowling_avg": player['Bowling_Avg'],
                        "reason": f"Best spinner with {player['Bowling_Avg']} bowling average"
                    })
                    selected_names.add(player['Player'])
                    spinner_selected = True
                    break
        
        # Add captain and vice captain for combined 11
        captain = None
        vice_captain = None
        
        # Find Cummins as captain
        for player in combined_11:
            if 'Cummins' in player['name']:
                captain = player['name']
                break
        
        # Find Stokes as vice captain
        for player in combined_11:
            if 'Stokes' in player['name']:
                vice_captain = player['name']
                break
        
        # Add captain and vice captain to each player
        for player in combined_11:
            player['captain'] = captain
            player['vice_captain'] = vice_captain
        
        return combined_11
    
    def _sort_players_by_conditions(self, batters, allrounders, bowlers, pitch_conditions, venue):
        """Sort players based on pitch conditions"""
        
        # Always give preference to Starc over Boland and Cummins for Australia
        if any('Starc' in p['Player'] for p in bowlers):
            starc_players = [p for p in bowlers if 'Starc' in p['Player']]
            cummins_players = [p for p in bowlers if 'Cummins' in p['Player']]
            boland_players = [p for p in bowlers if 'Boland' in p['Player']]
            other_bowlers = [p for p in bowlers if 'Starc' not in p['Player'] and 'Cummins' not in p['Player'] and 'Boland' not in p['Player']]
            bowlers.clear()
            bowlers.extend(starc_players + cummins_players + boland_players + other_bowlers)
        
        # Venue-specific sorting to ensure different playing 11s
        if venue == "Perth":
            # Perth: Fast, bouncy pitch - prefer pace bowlers and aggressive batters
            # Shuffle batters slightly to create variation
            batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
            # Prefer Cummins over Boland for Perth's pace-friendly conditions
            if any('Cummins' in p['Player'] for p in bowlers):
                cummins_players = [p for p in bowlers if 'Cummins' in p['Player']]
                boland_players = [p for p in bowlers if 'Boland' in p['Player']]
                other_bowlers = [p for p in bowlers if 'Cummins' not in p['Player'] and 'Boland' not in p['Player']]
                bowlers.clear()
                bowlers.extend(cummins_players + boland_players + other_bowlers)
            
        elif venue == "Brisbane":
            # Brisbane: Good batting surface - prefer better batters
            batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
            # Prefer Boland over Cummins for Brisbane's batting-friendly conditions
            if any('Boland' in p['Player'] for p in bowlers):
                boland_players = [p for p in bowlers if 'Boland' in p['Player']]
                cummins_players = [p for p in bowlers if 'Cummins' in p['Player']]
                other_bowlers = [p for p in bowlers if 'Boland' not in p['Player'] and 'Cummins' not in p['Player']]
                bowlers.clear()
                bowlers.extend(boland_players + cummins_players + other_bowlers)
            
        elif venue == "Adelaide":
            # Adelaide: Balanced pitch - balanced approach
            batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
            # Mix of Cummins and Boland for Adelaide
            if any('Cummins' in p['Player'] for p in bowlers):
                cummins_players = [p for p in bowlers if 'Cummins' in p['Player']]
                boland_players = [p for p in bowlers if 'Boland' in p['Player']]
                other_bowlers = [p for p in bowlers if 'Cummins' not in p['Player'] and 'Boland' not in p['Player']]
                bowlers.clear()
                bowlers.extend(cummins_players + boland_players + other_bowlers)
            
        elif venue == "Melbourne":
            # Melbourne: Batting-friendly - prefer better batters
            batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
            # Prefer Cummins for Melbourne's batting-friendly conditions
            if any('Cummins' in p['Player'] for p in bowlers):
                cummins_players = [p for p in bowlers if 'Cummins' in p['Player']]
                boland_players = [p for p in bowlers if 'Boland' in p['Player']]
                other_bowlers = [p for p in bowlers if 'Cummins' not in p['Player'] and 'Boland' not in p['Player']]
                bowlers.clear()
                bowlers.extend(cummins_players + boland_players + other_bowlers)
            
        elif venue == "Sydney":
            # Sydney: Spin-friendly - prefer spinners and better batters
            batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
            spinners = [p for p in bowlers if 'Spinner' in p['Role']]
            fast_bowlers = [p for p in bowlers if 'Fast_Bowler' in p['Role']]
            # Prefer Boland for Sydney's spin-friendly conditions
            if any('Boland' in p['Player'] for p in fast_bowlers):
                boland_players = [p for p in fast_bowlers if 'Boland' in p['Player']]
                cummins_players = [p for p in fast_bowlers if 'Cummins' in p['Player']]
                other_fast = [p for p in fast_bowlers if 'Boland' not in p['Player'] and 'Cummins' not in p['Player']]
                fast_bowlers = boland_players + cummins_players + other_fast
            bowlers.clear()
            bowlers.extend(spinners + fast_bowlers)
        
        # Sort allrounders by combined performance
        allrounders.sort(key=lambda x: (float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0) + 
                                    (float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50), reverse=True)
    
 