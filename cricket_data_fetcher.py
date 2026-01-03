import requests
import os
import json
import pandas as pd
from typing import List, Dict, Optional

class CricketDataFetcher:
    def __init__(self):
        self.cricapi_key = os.getenv('CRICAPI_KEY', '')
        self.espn_api_key = os.getenv('ESPN_CRICINFO_API_KEY', '')
        self.cache = {}
    
    def get_ashes_squads(self, team: str) -> List[Dict]:
        team_lower = team.lower()
        
        if self.cricapi_key:
            try:
                squad = self._get_squad_from_cricapi(team)
                if squad:
                    return squad
            except Exception as e:
                print(f"CricAPI error: {e}")
        
        if self.espn_api_key:
            try:
                squad = self._get_squad_from_espn(team)
                if squad:
                    return squad
            except Exception as e:
                print(f"ESPN API error: {e}")
        
        return self._get_squad_from_csv(team)
    
    def get_players_by_country(self, country: str) -> List[Dict]:
        
        country_lower = country.lower()
        
        if self.cricapi_key:
            try:
                squad = self._get_squad_from_cricapi(country)
                if squad:
                    return squad
            except Exception as e:
                print(f"CricAPI error for {country}: {e}")
        
        return self._get_international_players(country)
    
    def _get_international_players(self, country: str) -> List[Dict]:
        
        international_squads = {
            'India': [
                {'Player': 'Rohit Sharma', 'Team': 'India', 'Role': 'Opener', 'Batting_Avg': 46.5, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 14, 'Match_Wins': 5, 'Player_of_Match_Count': 3},
                {'Player': 'Virat Kohli', 'Team': 'India', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 49.3, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 15, 'Match_Wins': 6, 'Player_of_Match_Count': 4},
                {'Player': 'Jasprit Bumrah', 'Team': 'India', 'Role': 'Fast_Bowler', 'Batting_Avg': 8.5, 'Bowling_Avg': 22.1, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 7, 'Match_Wins': 5, 'Player_of_Match_Count': 2},
            ],
            'Pakistan': [
                {'Player': 'Babar Azam', 'Team': 'Pakistan', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 47.8, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 9, 'Match_Wins': 4, 'Player_of_Match_Count': 3},
                {'Player': 'Shaheen Afridi', 'Team': 'Pakistan', 'Role': 'Fast_Bowler', 'Batting_Avg': 9.2, 'Bowling_Avg': 24.5, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Good', 'Experience_Years': 6, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
            ],
            'New Zealand': [
                {'Player': 'Kane Williamson', 'Team': 'New Zealand', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 54.9, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 13, 'Match_Wins': 5, 'Player_of_Match_Count': 3},
                {'Player': 'Trent Boult', 'Team': 'New Zealand', 'Role': 'Fast_Bowler', 'Batting_Avg': 10.8, 'Bowling_Avg': 27.2, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 11, 'Match_Wins': 4, 'Player_of_Match_Count': 1},
            ],
            'South Africa': [
                {'Player': 'Kagiso Rabada', 'Team': 'South Africa', 'Role': 'Fast_Bowler', 'Batting_Avg': 11.2, 'Bowling_Avg': 22.8, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 9, 'Match_Wins': 4, 'Player_of_Match_Count': 2},
            ],
        }
        return international_squads.get(country.title(), [])
    
    def get_all_players(self) -> List[Dict]:
        
        all_players = []
        
        england = self.get_ashes_squads('England')
        australia = self.get_ashes_squads('Australia')
        all_players.extend(england)
        all_players.extend(australia)
        
        major_teams = ['India', 'Pakistan', 'New Zealand', 'South Africa', 'West Indies', 'Sri Lanka', 'Bangladesh']
        for team in major_teams:
            try:
                players = self.get_players_by_country(team)
                if players:
                    all_players.extend(players)
            except Exception as e:
                print(f"Error loading {team} players: {e}")
        
        return all_players
    
    def find_player(self, player_name: str, country: str = None) -> Optional[Dict]:
        
        if country:
            players = self.get_players_by_country(country)
            for player in players:
                if player['Player'].lower() == player_name.lower():
                    return player
        
        england = self.get_ashes_squads('England')
        australia = self.get_ashes_squads('Australia')
        
        for player in england + australia:
            if player['Player'].lower() == player_name.lower():
                return player
        
        all_players = self.get_all_players()
        for player in all_players:
            if player['Player'].lower() == player_name.lower():
                return player
        
        return None
    
    def _get_squad_from_cricapi(self, team: str) -> Optional[List[Dict]]:
        """
        Fetch squad data from CricAPI (cricapi.com)
        Requires API key from https://www.cricapi.com/
        """
        if not self.cricapi_key:
            return None
        
        team_search_terms = {
            'england': ['england', 'eng'],
            'australia': ['australia', 'aus']
        }
        
        search_terms = team_search_terms.get(team.lower(), [team.lower()])
        squad = []
        
        try:
            for search_term in search_terms:
                url = "https://cricapi.com/api/playerFinder"
                params = {
                    'apikey': self.cricapi_key,
                    'name': search_term
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    players = data.get('data', [])
                    
                    if players:
                        for player_info in players[:30]:
                            player_id = player_info.get('pid')
                            player_name = player_info.get('name', '')
                            
                            if not player_id or not player_name:
                                continue
                            
                            player_stats = self._get_player_stats_from_cricapi(player_id)
                            
                            squad.append({
                                'Player': player_name,
                                'Team': team.title(),
                                'Role': self._infer_role_from_cricapi(player_info, player_stats),
                                'Batting_Avg': self._get_batting_avg_from_stats(player_stats),
                                'Bowling_Avg': self._get_bowling_avg_from_stats(player_stats),
                                'Recent_Form': 'Good',
                                'Venue_Performance': 'Good',
                                'Experience_Years': self._get_experience_from_stats(player_stats),
                                'Match_Wins': 0,
                                'Player_of_Match_Count': 0
                            })
                        
                        if squad:
                            print(f"✓ Fetched {len(squad)} players from CricAPI for {team}")
                            return squad
                
                matches_url = "https://cricapi.com/api/matches"
                matches_params = {
                    'apikey': self.cricapi_key,
                    'date': '2025-11-01'
                }
                
                matches_response = requests.get(matches_url, params=matches_params, timeout=15)
                if matches_response.status_code == 200:
                    matches_data = matches_response.json()
                    pass
                    
        except requests.exceptions.RequestException as e:
            print(f"CricAPI network error: {e}")
            return None
        except Exception as e:
            print(f"Error fetching from CricAPI: {e}")
            return None
        
        return squad if squad else None
    
    def _get_player_stats_from_cricapi(self, player_id: str) -> Optional[Dict]:
        
        if not self.cricapi_key or not player_id:
            return None
        
        try:
            url = "https://cricapi.com/api/playerStats"
            params = {
                'apikey': self.cricapi_key,
                'pid': player_id
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
           
            pass
        
        return None
    
    def _get_batting_avg_from_stats(self, stats: Optional[Dict]) -> float:
        
        if not stats:
            return 0.0
        
        avg = (stats.get('data', {}).get('batting', {}).get('average') or
               stats.get('batting_average') or
               stats.get('battingAvg') or
               stats.get('avg', 0))
        
        try:
            return float(avg) if avg else 0.0
        except:
            return 0.0
    
    def _get_bowling_avg_from_stats(self, stats: Optional[Dict]) -> float:
        
        if not stats:
            return 0.0
        
        avg = (stats.get('data', {}).get('bowling', {}).get('average') or
               stats.get('bowling_average') or
               stats.get('bowlingAvg') or
               stats.get('bowl_avg', 0))
        
        try:
            return float(avg) if avg else 0.0
        except:
            return 0.0
    
    def _get_experience_from_stats(self, stats: Optional[Dict]) -> int:
        
        if not stats:
            return 0
        
        
        debut = stats.get('data', {}).get('debut') or stats.get('debut_year')
        if debut:
            try:
                debut_year = int(str(debut)[:4]) if len(str(debut)) >= 4 else 0
                if debut_year > 2000:
                    return 2025 - debut_year
            except:
                pass
        return 0
    
    def _infer_role_from_cricapi(self, player_info: Dict, stats: Optional[Dict] = None) -> str:
        
        role = player_info.get('role', '').lower()
        name = player_info.get('name', '').lower()
        
        if stats:
            batting = stats.get('data', {}).get('batting', {})
            bowling = stats.get('data', {}).get('bowling', {})
            
            has_batting = batting and len(batting) > 0
            has_bowling = bowling and len(bowling) > 0
            
            if has_batting and has_bowling:
                return 'Allrounder'
            elif has_bowling:
                if 'spin' in role or 'spinner' in name:
                    return 'Spinner'
                return 'Fast_Bowler'
            elif has_batting:
                if 'keeper' in role or 'wicket' in role:
                    return 'Wicketkeeper_Batter'
                return 'Middle_Order_Batter'
        
        if 'batsman' in role or 'batter' in role:
            if 'opener' in role:
                return 'Opener'
            return 'Middle_Order_Batter'
        elif 'bowler' in role:
            if 'spin' in role:
                return 'Spinner'
            return 'Fast_Bowler'
        elif 'all' in role or 'rounder' in role:
            return 'Allrounder'
        elif 'keeper' in role or 'wicket' in role:
            return 'Wicketkeeper_Batter'
        
        return 'Middle_Order_Batter' 
    
    def _get_squad_from_espn(self, team: str) -> Optional[List[Dict]]:
       
        try:
            return None
        except Exception as e:
            print(f"ESPN API not available: {e}")
            return None
    
    def _get_squad_from_csv(self, team: str) -> List[Dict]:
        """Fallback: Get default squad data (no CSV dependency) - Updated Ashes 2025 Squads"""
        default_squads = {
            'England': [
                {'Player': 'Zak Crawley', 'Team': 'England', 'Role': 'Opener', 'Batting_Avg': 45.2, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Good', 'Experience_Years': 6, 'Match_Wins': 4, 'Player_of_Match_Count': 2},
                {'Player': 'Ben Duckett', 'Team': 'England', 'Role': 'Opener', 'Batting_Avg': 38.7, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 5, 'Match_Wins': 3, 'Player_of_Match_Count': 0},
                {'Player': 'Ollie Pope', 'Team': 'England', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 35.8, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 7, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Joe Root', 'Team': 'England', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 52.3, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 13, 'Match_Wins': 4, 'Player_of_Match_Count': 3},
                {'Player': 'Harry Brook', 'Team': 'England', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 36.5, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Good', 'Experience_Years': 4, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Dan Lawrence', 'Team': 'England', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 29.3, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Average', 'Experience_Years': 4, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
                {'Player': 'Jamie Smith', 'Team': 'England', 'Role': 'Wicketkeeper_Batter', 'Batting_Avg': 31.2, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 3, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
                {'Player': 'Ben Stokes', 'Team': 'England', 'Role': 'Allrounder', 'Batting_Avg': 36.2, 'Bowling_Avg': 32.1, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Good', 'Experience_Years': 11, 'Match_Wins': 4, 'Player_of_Match_Count': 2},
                {'Player': 'Liam Livingstone', 'Team': 'England', 'Role': 'Batting_Allrounder', 'Batting_Avg': 28.5, 'Bowling_Avg': 34.2, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 3, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
                {'Player': 'Brydon Carse', 'Team': 'England', 'Role': 'Bowling_Allrounder', 'Batting_Avg': 18.7, 'Bowling_Avg': 28.9, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 4, 'Match_Wins': 3, 'Player_of_Match_Count': 0},
                {'Player': 'Mark Wood', 'Team': 'England', 'Role': 'Fast_Bowler', 'Batting_Avg': 18.5, 'Bowling_Avg': 28.5, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 8, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Jofra Archer', 'Team': 'England', 'Role': 'Fast_Bowler', 'Batting_Avg': 10.5, 'Bowling_Avg': 23.8, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 5, 'Match_Wins': 4, 'Player_of_Match_Count': 2},
                {'Player': 'Gus Atkinson', 'Team': 'England', 'Role': 'Fast_Bowler', 'Batting_Avg': 11.2, 'Bowling_Avg': 24.5, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Good', 'Experience_Years': 3, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Josh Tongue', 'Team': 'England', 'Role': 'Fast_Bowler', 'Batting_Avg': 18.9, 'Bowling_Avg': 29.4, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 4, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
            ],
            'Australia': [
                {'Player': 'Usman Khawaja', 'Team': 'Australia', 'Role': 'Opener', 'Batting_Avg': 48.9, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 13, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Cameron Bancroft', 'Team': 'Australia', 'Role': 'Opener', 'Batting_Avg': 39.2, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 6, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
                {'Player': 'Marnus Labuschagne', 'Team': 'Australia', 'Role': 'Top_Order_Batter', 'Batting_Avg': 52.3, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 7, 'Match_Wins': 3, 'Player_of_Match_Count': 2},
                {'Player': 'Steven Smith', 'Team': 'Australia', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 58.7, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 15, 'Match_Wins': 3, 'Player_of_Match_Count': 2},
                {'Player': 'Travis Head', 'Team': 'Australia', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 42.1, 'Bowling_Avg': 0, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 9, 'Match_Wins': 3, 'Player_of_Match_Count': 2},
                {'Player': 'Matt Renshaw', 'Team': 'Australia', 'Role': 'Middle_Order_Batter', 'Batting_Avg': 29.5, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 5, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
                {'Player': 'Alex Carey', 'Team': 'Australia', 'Role': 'Wicketkeeper_Batter', 'Batting_Avg': 32.8, 'Bowling_Avg': 0, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 6, 'Match_Wins': 3, 'Player_of_Match_Count': 0},
                {'Player': 'Cameron Green', 'Team': 'Australia', 'Role': 'Batting_Allrounder', 'Batting_Avg': 35.6, 'Bowling_Avg': 29.8, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 5, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Mitchell Marsh', 'Team': 'Australia', 'Role': 'Allrounder', 'Batting_Avg': 28.5, 'Bowling_Avg': 33.2, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 10, 'Match_Wins': 3, 'Player_of_Match_Count': 0},
                {'Player': 'Pat Cummins', 'Team': 'Australia', 'Role': 'Fast_Bowler', 'Batting_Avg': 16.4, 'Bowling_Avg': 21.6, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 11, 'Match_Wins': 3, 'Player_of_Match_Count': 2},
                {'Player': 'Mitchell Starc', 'Team': 'Australia', 'Role': 'Fast_Bowler', 'Batting_Avg': 22.1, 'Bowling_Avg': 27.3, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 13, 'Match_Wins': 3, 'Player_of_Match_Count': 1},
                {'Player': 'Josh Hazlewood', 'Team': 'Australia', 'Role': 'Fast_Bowler', 'Batting_Avg': 12.9, 'Bowling_Avg': 25.7, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 11, 'Match_Wins': 3, 'Player_of_Match_Count': 0},
                {'Player': 'Scott Boland', 'Team': 'Australia', 'Role': 'Fast_Bowler', 'Batting_Avg': 8.9, 'Bowling_Avg': 18.2, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 4, 'Match_Wins': 3, 'Player_of_Match_Count': 2},
                {'Player': 'Lance Morris', 'Team': 'Australia', 'Role': 'Fast_Bowler', 'Batting_Avg': 9.5, 'Bowling_Avg': 26.8, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 2, 'Match_Wins': 1, 'Player_of_Match_Count': 0},
                {'Player': 'Nathan Lyon', 'Team': 'Australia', 'Role': 'Spinner', 'Batting_Avg': 12.5, 'Bowling_Avg': 31.5, 'Recent_Form': 'Excellent', 'Venue_Performance': 'Excellent', 'Experience_Years': 13, 'Match_Wins': 4, 'Player_of_Match_Count': 1},
                {'Player': 'Todd Murphy', 'Team': 'Australia', 'Role': 'Spinner', 'Batting_Avg': 11.2, 'Bowling_Avg': 28.5, 'Recent_Form': 'Good', 'Venue_Performance': 'Good', 'Experience_Years': 2, 'Match_Wins': 2, 'Player_of_Match_Count': 0},
            ]
        }
        return default_squads.get(team.title(), [])
    
    def _infer_role_from_cricapi(self, player: Dict) -> str:
        
        role = player.get('role', '').lower()
        
        if 'batsman' in role or 'batter' in role:
            if 'opener' in role:
                return 'Opener'
            return 'Middle_Order_Batter'
        elif 'bowler' in role:
            if 'spin' in role:
                return 'Spinner'
            return 'Fast_Bowler'
        elif 'all' in role or 'rounder' in role:
            return 'Allrounder'
        elif 'keeper' in role or 'wicket' in role:
            return 'Wicketkeeper_Batter'
        
        return 'Middle_Order_Batter'
    
    def _get_batting_avg(self, player: Dict) -> float:
       
        avg = player.get('batting_average', 0) or \
              player.get('battingAvg', 0) or \
              player.get('avg', 0)
        try:
            return float(avg) if avg else 0.0
        except:
            return 0.0
    
    def _get_bowling_avg(self, player: Dict) -> float:
        
        avg = player.get('bowling_average', 0) or \
              player.get('bowlingAvg', 0) or \
              player.get('bowl_avg', 0)
        try:
            return float(avg) if avg else 0.0
        except:
            return 0.0
    
    def _get_experience(self, player: Dict) -> int:
       
        exp = player.get('experience', 0) or \
              player.get('years', 0) or \
              player.get('career_years', 0)
        try:
            return int(exp) if exp else 0
        except:
            return 0
    
    def update_team_data(self, team: str) -> bool:
        
        try:
            squad = self.get_ashes_squads(team)
            if squad and len(squad) > 0:
                # Cache the data in memory
                self.cache[team.lower()] = squad
            return True
        except Exception as e:
            print(f"Error updating team data: {e}")
            return False
    
    def get_player_stats(self, player_name: str, team: str = None) -> Optional[Dict]:
        
        if self.cricapi_key:
            try:
                url = f"https://cricapi.com/api/playerStats"
                params = {
                    'apikey': self.cricapi_key,
                    'pid': self._get_player_id(player_name)
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                print(f"Error fetching player stats: {e}")
        
        return None
    
    def _get_player_id(self, player_name: str) -> Optional[str]:
        
        return None

    def get_live_playing_11(self, match_id: str) -> Optional[Dict]:
        
        if not self.cricapi_key or not match_id:
            return None

        try:
            url = "https://api.cricapi.com/v1/match_scorecard"
            params = {"apikey": self.cricapi_key, "id": match_id}
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                parsed = self._parse_playing_11_response(resp.json())
                if parsed:
                    parsed["source"] = "cricapi.com v1 scorecard"
                    return parsed
        except Exception as e:
            print(f"CricAPI scorecard error: {e}")

        try:
            legacy_url = "https://cricapi.com/api/fantasySummary"
            params = {"apikey": self.cricapi_key, "unique_id": match_id}
            resp = requests.get(legacy_url, params=params, timeout=15)
            if resp.status_code == 200:
                parsed = self._parse_playing_11_response(resp.json())
                if parsed:
                    parsed["source"] = "cricapi.com fantasySummary"
                    return parsed
        except Exception as e:
            print(f"CricAPI legacy error: {e}")

        return None

    def _parse_playing_11_response(self, data: Dict) -> Optional[Dict]:
        
        if not data:
            return None

        candidates = []
        if isinstance(data, dict):
            candidates.append(data.get("data", {}))
            candidates.append(data.get("info", {}))
            candidates.append(data)

        team_entries = None
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            team_entries = (
                candidate.get("teams")
                or candidate.get("teamInfo")
                or candidate.get("teaminfo")
                or candidate.get("squads")
            )
            if team_entries:
                break

        if not team_entries or not isinstance(team_entries, list):
            return None

        normalized_teams = []
        for team in team_entries[:2]:
            if not isinstance(team, dict):
                continue
            name = (
                team.get("name")
                or team.get("teamName")
                or team.get("shortname")
                or team.get("title")
            )
            players = (
                team.get("players")
                or team.get("player")
                or team.get("playing11")
                or team.get("playing_11")
                or team.get("xi")
            )
            normalized_players = self._normalize_players(players)
            if name and normalized_players:
                normalized_teams.append(
                    {"team": name, "playing_11": normalized_players}
                )

        if len(normalized_teams) < 2:
            return None

        return {
            "home_team": normalized_teams[0]["team"],
            "away_team": normalized_teams[1]["team"],
            "home_playing_11": normalized_teams[0]["playing_11"],
            "away_playing_11": normalized_teams[1]["playing_11"],
        }

    def _normalize_players(self, players) -> List[Dict]:
       
        if not players or not isinstance(players, list):
            return []

        normalized = []
        for idx, player in enumerate(players[:11], start=1):
            if isinstance(player, dict):
                name = (
                    player.get("name")
                    or player.get("player")
                    or player.get("fullName")
                    or player.get("title")
                )
                role = (
                    player.get("role")
                    or player.get("playingRole")
                    or player.get("playerRole")
                    or ""
                )
            else:
                name = str(player)
                role = ""

            if name:
                normalized.append(
                    {"position": idx, "name": name, "role": role or "Unknown"}
                )

        return normalized

def get_real_time_squads():
    
    fetcher = CricketDataFetcher()
    
    england_squad = fetcher.get_ashes_squads('England')
    australia_squad = fetcher.get_ashes_squads('Australia')
    
    return {
        'england': england_squad,
        'australia': australia_squad
    }

if __name__ == "__main__":
    fetcher = CricketDataFetcher()
    
    print("Fetching England squad...")
    eng_squad = fetcher.get_ashes_squads('England')
    print(f"Found {len(eng_squad)} England players")
    
    print("\nFetching Australia squad...")
    aus_squad = fetcher.get_ashes_squads('Australia')
    print(f"Found {len(aus_squad)} Australia players")

