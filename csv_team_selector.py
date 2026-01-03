#!/usr/bin/env python3
"""
Team selector using API data for playing 11s
No CSV dependency - all data fetched from APIs
"""

import pandas as pd
import numpy as np

class APITeamSelector:
    """
    Team selector that uses API data exclusively
    No CSV dependency - fetches data from CricAPI, ESPN Cricinfo, or uses default data
    """
    
    def __init__(self, include_international=False):
        """
        Initialize with API data only
        If include_international is True, loads players from all countries
        """
        self.fetcher = None
        self.england_players = []
        self.australia_players = []
        self.all_players = []
        self.include_international = include_international
        self._load_from_api()
    
    def _load_from_api(self):
        """Load data from APIs only"""
        try:
            from cricket_data_fetcher import CricketDataFetcher
            self.fetcher = CricketDataFetcher()
            
            england_squad = self.fetcher.get_ashes_squads('England')
            australia_squad = self.fetcher.get_ashes_squads('Australia')
            
            if england_squad and len(england_squad) > 0:
                self.england_players = england_squad
                print(f"✓ Loaded {len(self.england_players)} England players from API")
            else:
                raise ValueError("No England squad data available")
            
            if australia_squad and len(australia_squad) > 0:
                self.australia_players = australia_squad
                print(f"✓ Loaded {len(self.australia_players)} Australia players from API")
            else:
                raise ValueError("No Australia squad data available")
            
            if self.include_international:
                try:
                    self.all_players = self.fetcher.get_all_players()
                    print(f"✓ Loaded {len(self.all_players)} total players (including international)")
                except Exception as e:
                    print(f"Warning: Could not load international players: {e}")
                    self.all_players = self.england_players + self.australia_players
            else:
                self.all_players = self.england_players + self.australia_players
                
        except Exception as e:
            print(f"Error loading from API: {e}")
            raise Exception(f"Failed to load team data: {e}. Please check API keys or network connection.")
    
    def refresh_data(self):
        """Refresh team data from API"""
        try:
            if not self.fetcher:
                from cricket_data_fetcher import CricketDataFetcher
                self.fetcher = CricketDataFetcher()
            
            england_squad = self.fetcher.get_ashes_squads('England')
            australia_squad = self.fetcher.get_ashes_squads('Australia')
            
            if england_squad and len(england_squad) > 0:
                self.england_players = england_squad
                print(f"✓ Refreshed {len(self.england_players)} England players")
            
            if australia_squad and len(australia_squad) > 0:
                self.australia_players = australia_squad
                print(f"✓ Refreshed {len(self.australia_players)} Australia players")
            
            return True
        except Exception as e:
            print(f"Error refreshing data: {e}")
            return False

    def _ashes_pool(self, country=None):
        """Return Ashes-only player pool, optionally filtered by country"""
        if not self.england_players or not self.australia_players:
            self._load_from_api()

        pool = self.england_players + self.australia_players
        if country and country.lower() in ['england', 'australia']:
            pool = [p for p in pool if p.get('Team', '').lower() == country.lower()]                                                                           
        return pool
    
    def safe_float(self, value, default=0.0):
        """Helper function to safely convert values to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_player_recommendations(self, preference, count=5, country=None):
        """
        Content-Based Recommendation System: Recommends players based on user preference.
        Uses player attributes (batting avg, bowling avg) to find similar players.
        Algorithm: Feature-based similarity matching
        Supports players from any country if country parameter is provided
        """
        if not self.include_international:
            all_players = self._ashes_pool(country)
        else:
            if country:
                if not self.fetcher:
                    from cricket_data_fetcher import CricketDataFetcher
                    self.fetcher = CricketDataFetcher()
                all_players = self.fetcher.get_players_by_country(country)
                if not all_players:
                    all_players = self.all_players
            else:
                all_players = self.all_players

        if preference.lower() == 'batting':
            sorted_players = sorted(
                all_players, 
                key=lambda x: self.safe_float(x.get('Batting_Avg'), 0.0), 
                reverse=True
            )
            metric_name = 'Batting_Avg'
            reason = 'Content-Based: Best Batting Average'
            
        elif preference.lower() == 'bowling':
            sorted_players = sorted(
                all_players, 
                key=lambda x: self.safe_float(x.get('Bowling_Avg'), 50.0), 
                reverse=False
            )
            sorted_players = [p for p in sorted_players if self.safe_float(p.get('Bowling_Avg')) > 0 or 'Bowler' in p['Role'] or 'Allrounder' in p['Role']]
            metric_name = 'Bowling_Avg'
            reason = 'Content-Based: Best Bowling Average'
            
        elif preference.lower() == 'allrounder':
            sorted_players = sorted(
                all_players,
                key=lambda x: (self.safe_float(x.get('Batting_Avg'), 0.0) * 0.5) + 
                             (50.0 - self.safe_float(x.get('Bowling_Avg'), 50.0)) * 0.5,
                reverse=True
            )
            metric_name = 'Allrounder_Score'
            reason = 'Content-Based: Best All-rounder (Combined Score)'
        else:
            return []

        recommendations = []
        selected_names = set()
        for player in sorted_players:
            if player['Player'] not in selected_names and len(recommendations) < count:
                recommendations.append({
                    "name": player['Player'],
                    "team": player['Team'],
                    "role": player['Role'],
                    "metric": metric_name,
                    "value": player.get(metric_name, 'N/A'),
                    "reason": reason,
                    "algorithm": "Content-Based Filtering"
                })
                selected_names.add(player['Player'])
                
        return recommendations
    
    def collaborative_filtering_recommendations(self, reference_player_name, count=5, country=None):
        """
        Collaborative Filtering: Recommends players similar to a reference player.
        Algorithm: User-based collaborative filtering using cosine similarity.
        Finds players with similar performance profiles.
        Works with players from any country.
        """
        if not self.include_international:
            comparison_pool = self._ashes_pool()
            if country and country.lower() in ['england', 'australia']:
                comparison_pool = self._ashes_pool(country)

            reference_player = next(
                (p for p in comparison_pool if p['Player'].lower() == reference_player_name.lower()),
                None
            )
            if not reference_player:
                return []
            all_players = comparison_pool
        else:
            if not self.fetcher:
                from cricket_data_fetcher import CricketDataFetcher
                self.fetcher = CricketDataFetcher()

            reference_player = self.fetcher.find_player(reference_player_name, country)                                                                         
            if not reference_player:
                return []

            all_players = self.all_players
            if country:
                country_players = self.fetcher.get_players_by_country(country)
                if country_players:
                    all_players = country_players + [p for p in self.all_players if p['Team'].lower() != country.lower()]                                       
        
        def calculate_similarity(p1, p2):
            """Calculate cosine similarity between two players"""
            features1 = [
                self.safe_float(p1.get('Batting_Avg', 0)),
                self.safe_float(p1.get('Bowling_Avg', 50)),
                self.safe_float(p1.get('Experience_Years', 0)),
                self.safe_float(p1.get('Match_Wins', 0))
            ]
            features2 = [
                self.safe_float(p2.get('Batting_Avg', 0)),
                self.safe_float(p2.get('Bowling_Avg', 50)),
                self.safe_float(p2.get('Experience_Years', 0)),
                self.safe_float(p2.get('Match_Wins', 0))
            ]
            
            norm1 = np.linalg.norm(features1) if np.linalg.norm(features1) > 0 else 1
            norm2 = np.linalg.norm(features2) if np.linalg.norm(features2) > 0 else 1
            
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return similarity
        
        similarities = []
        for player in all_players:
            if player['Player'] != reference_player['Player']:
                sim = calculate_similarity(reference_player, player)
                similarities.append({
                    "name": player['Player'],
                    "team": player['Team'],
                    "role": player['Role'],
                    "similarity_score": round(sim, 3),
                    "batting_avg": self.safe_float(player.get('Batting_Avg', 0)),
                    "bowling_avg": self.safe_float(player.get('Bowling_Avg', 50)),
                    "reason": f"Similar to {reference_player_name}",
                    "algorithm": "Collaborative Filtering (User-Based)"
                })
        
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:count]
    
    def hybrid_recommendations(self, preference='batting', venue=None, count=5, country=None):
        """
        Hybrid Recommendation System: Combines content-based and collaborative filtering.
        Algorithm: Weighted combination of multiple recommendation approaches.
        Supports players from any country.
        """
        if not self.include_international:
            all_players = self._ashes_pool(country)
        else:
            if country:
                if not self.fetcher:
                    from cricket_data_fetcher import CricketDataFetcher
                    self.fetcher = CricketDataFetcher()
                all_players = self.fetcher.get_players_by_country(country)
                if not all_players:
                    all_players = self.all_players
            else:
                all_players = self.all_players
        
        content_scores = {}
        for player in all_players:
            score = 0.0
            if preference.lower() == 'batting':
                score = self.safe_float(player.get('Batting_Avg', 0)) / 100.0
            elif preference.lower() == 'bowling':
                bowling_avg = self.safe_float(player.get('Bowling_Avg', 50))
                score = (50.0 - bowling_avg) / 50.0 if bowling_avg > 0 else 0
            elif preference.lower() == 'allrounder':
                bat_score = self.safe_float(player.get('Batting_Avg', 0)) / 100.0
                bowl_avg = self.safe_float(player.get('Bowling_Avg', 50))
                bowl_score = (50.0 - bowl_avg) / 50.0 if bowl_avg > 0 else 0
                score = (bat_score * 0.5) + (bowl_score * 0.5)
            
            venue_boost = 0.0
            if venue and player.get('Venue_Performance') == 'Excellent':
                venue_boost = 0.2
            elif venue and player.get('Venue_Performance') == 'Good':
                venue_boost = 0.1
            
            # Recent form boost
            form_boost = 0.0
            if player.get('Recent_Form') == 'Excellent':
                form_boost = 0.15
            elif player.get('Recent_Form') == 'Good':
                form_boost = 0.05
            
            content_scores[player['Player']] = score + venue_boost + form_boost
        
        collab_scores = {}
        max_experience = max([self.safe_float(p.get('Experience_Years', 0)) for p in all_players], default=1)
        max_wins = max([self.safe_float(p.get('Match_Wins', 0)) for p in all_players], default=1)
        
        for player in all_players:
            exp_score = self.safe_float(player.get('Experience_Years', 0)) / max_experience
            wins_score = self.safe_float(player.get('Match_Wins', 0)) / max_wins
            collab_scores[player['Player']] = (exp_score * 0.4) + (wins_score * 0.6)
        
        hybrid_scores = []
        for player in all_players:
            content = content_scores.get(player['Player'], 0)
            collab = collab_scores.get(player['Player'], 0)
            hybrid_score = (content * 0.6) + (collab * 0.4)
            
            hybrid_scores.append({
                "name": player['Player'],
                "team": player['Team'],
                "role": player['Role'],
                "hybrid_score": round(hybrid_score, 3),
                "content_score": round(content, 3),
                "collaborative_score": round(collab, 3),
                "batting_avg": self.safe_float(player.get('Batting_Avg', 0)),
                "bowling_avg": self.safe_float(player.get('Bowling_Avg', 50)),
                "reason": f"Hybrid: Best {preference} with venue and form consideration",
                "algorithm": "Hybrid Recommendation System"
            })
        
        hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_scores[:count]
    
    def venue_based_recommendations(self, venue, count=5, country=None):
        """
        Venue-Based Recommendation: Recommends players based on venue characteristics.
        Algorithm: Content-based filtering with venue-specific features.
        Supports players from any country.
        """
        if not self.include_international:
            all_players = self._ashes_pool(country)
        else:
            if country:
                if not self.fetcher:
                    from cricket_data_fetcher import CricketDataFetcher
                    self.fetcher = CricketDataFetcher()
                all_players = self.fetcher.get_players_by_country(country)
                if not all_players:
                    all_players = self.all_players
            else:
                all_players = self.all_players
        
        venue_requirements = {
            'Perth': {'pace_friendly': True, 'spin_friendly': False, 'batting_friendly': 'Moderate'},
            'Brisbane': {'pace_friendly': True, 'spin_friendly': False, 'batting_friendly': 'High'},
            'Adelaide': {'pace_friendly': True, 'spin_friendly': True, 'batting_friendly': 'High'},
            'Melbourne': {'pace_friendly': True, 'spin_friendly': False, 'batting_friendly': 'Very High'},
            'Sydney': {'pace_friendly': False, 'spin_friendly': True, 'batting_friendly': 'Moderate'}
        }
        
        venue_req = venue_requirements.get(venue, {})
        
        venue_scores = []
        for player in all_players:
            score = 0.0
            role = player.get('Role', '')
            
            if venue_req.get('pace_friendly') and 'Fast_Bowler' in role:
                score += 0.3
            
            if venue_req.get('spin_friendly') and 'Spinner' in role:
                score += 0.3
            
            if venue_req.get('batting_friendly') in ['High', 'Very High']:
                if 'Batter' in role or 'Opener' in role:
                    score += 0.2
            
            if player.get('Venue_Performance') == 'Excellent':
                score += 0.25
            elif player.get('Venue_Performance') == 'Good':
                score += 0.15
            
            if player.get('Recent_Form') == 'Excellent':
                score += 0.15
            
            venue_scores.append({
                "name": player['Player'],
                "team": player['Team'],
                "role": player['Role'],
                "venue_score": round(score, 3),
                "batting_avg": self.safe_float(player.get('Batting_Avg', 0)),
                "bowling_avg": self.safe_float(player.get('Bowling_Avg', 50)),
                "venue_performance": player.get('Venue_Performance', 'N/A'),
                "reason": f"Optimal for {venue} venue conditions",
                "algorithm": "Venue-Based Recommendation"
            })
        
        venue_scores.sort(key=lambda x: x['venue_score'], reverse=True)
        return venue_scores[:count]
    
    def similarity_based_recommendations(self, player_name, count=5, country=None):
        """
        Item-Based Collaborative Filtering: Finds players similar to the given player.
        Algorithm: Item-item similarity using Euclidean distance.
        Works with players from any country.
        """
        if not self.include_international:
            search_pool = self._ashes_pool(country)
            target_player = next(
                (p for p in search_pool if p['Player'].lower() == player_name.lower()),
                None
            )
            if not target_player:
                return []
            all_players = search_pool
        else:
            if not self.fetcher:
                from cricket_data_fetcher import CricketDataFetcher
                self.fetcher = CricketDataFetcher()

            target_player = self.fetcher.find_player(player_name, country)
            if not target_player:
                return []

            all_players = self.all_players
            if country:
                country_players = self.fetcher.get_players_by_country(country)  
                if country_players:
                    all_players = country_players + [p for p in self.all_players if p['Team'].lower() != country.lower()]                                       
                else:
                    all_players = self.all_players
        
        def euclidean_distance(p1, p2):
            features1 = np.array([
                self.safe_float(p1.get('Batting_Avg', 0)),
                self.safe_float(p1.get('Bowling_Avg', 50)),
                self.safe_float(p1.get('Experience_Years', 0)),
                self.safe_float(p1.get('Match_Wins', 0)),
                self.safe_float(p1.get('Player_of_Match_Count', 0))
            ])
            features2 = np.array([
                self.safe_float(p2.get('Batting_Avg', 0)),
                self.safe_float(p2.get('Bowling_Avg', 50)),
                self.safe_float(p2.get('Experience_Years', 0)),
                self.safe_float(p2.get('Match_Wins', 0)),
                self.safe_float(p2.get('Player_of_Match_Count', 0))
            ])
            distance = np.linalg.norm(features1 - features2)
            similarity = 1 / (1 + distance)
            return similarity
        
        similarities = []
        for player in all_players:
            if player['Player'] != target_player['Player']:
                sim = euclidean_distance(target_player, player)
                similarities.append({
                    "name": player['Player'],
                    "team": player['Team'],
                    "role": player['Role'],
                    "similarity": round(sim, 3),
                    "batting_avg": self.safe_float(player.get('Batting_Avg', 0)),
                    "bowling_avg": self.safe_float(player.get('Bowling_Avg', 50)),
                    "reason": f"Similar profile to {player_name}",
                    "algorithm": "Item-Based Collaborative Filtering"
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:count]
    
    def popularity_based_recommendations(self, count=5, country=None):
        """
        Popularity-Based Recommendation: Recommends most popular/successful players.
        Algorithm: Ranking based on multiple popularity metrics.
        Supports players from any country.
        """
        if not self.include_international:
            all_players = self._ashes_pool(country)
        else:
            if country:
                if not self.fetcher:
                    from cricket_data_fetcher import CricketDataFetcher
                    self.fetcher = CricketDataFetcher()
                all_players = self.fetcher.get_players_by_country(country)
                if not all_players:
                    all_players = self.all_players
            else:
                all_players = self.all_players
        
        popularity_scores = []
        for player in all_players:
            score = 0.0
            
            wins = self.safe_float(player.get('Match_Wins', 0))
            score += wins * 0.3
            
            pom_count = self.safe_float(player.get('Player_of_Match_Count', 0))
            score += pom_count * 0.4
            
            experience = self.safe_float(player.get('Experience_Years', 0))
            score += experience * 0.2
            
            batting_avg = self.safe_float(player.get('Batting_Avg', 0))
            bowling_avg = self.safe_float(player.get('Bowling_Avg', 50))
            
            if batting_avg > 40:
                score += 0.05
            if bowling_avg > 0 and bowling_avg < 30:
                score += 0.05
            
            popularity_scores.append({
                "name": player['Player'],
                "team": player['Team'],
                "role": player['Role'],
                "popularity_score": round(score, 3),
                "match_wins": wins,
                "player_of_match": pom_count,
                "experience": experience,
                "batting_avg": batting_avg,
                "bowling_avg": bowling_avg if bowling_avg < 50 else 0,
                "reason": "Most popular/successful players",
                "algorithm": "Popularity-Based Recommendation"
            })
        
        popularity_scores.sort(key=lambda x: x['popularity_score'], reverse=True)
        return popularity_scores[:count]
    
    def select_playing_11_for_venue(self, team, venue, pitch_conditions):
        """
        Select playing 11 for a specific team and venue based on pitch conditions
        Returns a dictionary with playing_11 list and strategy string
        Ensures exactly 11 players are always selected
        """
        if team.lower() == 'england':
            players = self.england_players
            captain = 'Ben Stokes'
        else:
            players = self.australia_players
            captain = 'Pat Cummins'
        
        pace_friendly = pitch_conditions.get('pace_friendly', True)
        spin_friendly = pitch_conditions.get('spin_friendly', False)
        batting_friendly = pitch_conditions.get('batting_friendly', 'Moderate')
        
        selected = []
        positions = set()
        
        def add_player(player, role_override=None):
            if player['Player'] not in positions and len(selected) < 11:
                selected.append({
                    'position': len(selected) + 1,
                    'name': player['Player'],
                    'role': role_override or player.get('Role', ''),
                    'captain': captain
                })
                positions.add(player['Player'])
                return True
            return False
        
        openers = [p for p in players if 'Opener' in p.get('Role', '')]
        openers.sort(key=lambda x: self.safe_float(x.get('Batting_Avg', 0)), reverse=True)
        for opener in openers[:2]:
            add_player(opener)
        
        middle_order = [p for p in players if ('Middle_Order' in p.get('Role', '') or 'Top_Order' in p.get('Role', '')) and p['Player'] not in positions]
        middle_order.sort(key=lambda x: self.safe_float(x.get('Batting_Avg', 0)), reverse=True)
        for player in middle_order[:3]:
            if len(selected) < 11:
                add_player(player)
        
        keepers = [p for p in players if 'Wicketkeeper' in p.get('Role', '') and p['Player'] not in positions]
        if keepers:
            add_player(keepers[0])
        else:
            backup_keepers = [p for p in players if 'Batter' in p.get('Role', '') and p['Player'] not in positions]
            if backup_keepers:
                add_player(backup_keepers[0], 'Wicketkeeper_Batter')
        
        if team.lower() == 'australia':
            allrounders = [p for p in players if 'Allrounder' in p.get('Role', '') and p['Player'] not in positions and p['Player'] not in ['Mitchell Marsh', 'Pat Cummins']]
            allrounders.sort(key=lambda x: (self.safe_float(x.get('Batting_Avg', 0)) + (50 - self.safe_float(x.get('Bowling_Avg', 50)))), reverse=True)
            for player in allrounders[:2]:
                if len(selected) < 11:
                    add_player(player)
        elif team.lower() == 'england':
            allrounders = [p for p in players if 'Allrounder' in p.get('Role', '') and p['Player'] not in positions and p['Player'] != 'Chris Woakes']
            # Prioritize Brydon Carse for England
            carse = next((p for p in allrounders if p['Player'] == 'Brydon Carse'), None)
            if carse:
                allrounders = [p for p in allrounders if p['Player'] != 'Brydon Carse']
                allrounders.insert(0, carse)
            allrounders.sort(key=lambda x: (
                0 if x['Player'] == 'Brydon Carse' 
                else (self.safe_float(x.get('Batting_Avg', 0)) + (50 - self.safe_float(x.get('Bowling_Avg', 50))))
            ), reverse=True)
            for player in allrounders[:2]:
                if len(selected) < 11:
                    add_player(player)
        else:
            allrounders = [p for p in players if 'Allrounder' in p.get('Role', '') and p['Player'] not in positions]
            allrounders.sort(key=lambda x: (self.safe_float(x.get('Batting_Avg', 0)) + (50 - self.safe_float(x.get('Bowling_Avg', 50)))), reverse=True)
            for player in allrounders[:2]:
                if len(selected) < 11:
                    add_player(player)
        
        remaining_slots = 11 - len(selected)
        
        if spin_friendly and remaining_slots > 0:
            if team.lower() == 'england':
                spinners = []
            else:
                spinners = [p for p in players if 'Spinner' in p.get('Role', '') and p['Player'] not in positions]
            if spinners:
                spinners.sort(key=lambda x: self.safe_float(x.get('Bowling_Avg', 50)))
                spinner_count = min(2, remaining_slots)
                for spinner in spinners[:spinner_count]:
                    if len(selected) < 11:
                        add_player(spinner)
        
        remaining_slots = 11 - len(selected)
        if remaining_slots > 0:
            if team.lower() == 'england':
                fast_bowlers = [p for p in players if 'Fast_Bowler' in p.get('Role', '') and p['Player'] not in positions and p['Player'] != 'Ollie Robinson']
            else:
                fast_bowlers = [p for p in players if 'Fast_Bowler' in p.get('Role', '') and p['Player'] not in positions]
            
            if team.lower() == 'australia':
                cummins = next((p for p in fast_bowlers if p['Player'] == 'Pat Cummins'), None)
                starc = next((p for p in fast_bowlers if p['Player'] == 'Mitchell Starc'), None)
                if cummins:
                    fast_bowlers = [p for p in fast_bowlers if p['Player'] != 'Pat Cummins']
                    fast_bowlers.insert(0, cummins)
                if starc:
                    fast_bowlers = [p for p in fast_bowlers if p['Player'] != 'Mitchell Starc']
                    fast_bowlers.insert(1 if cummins else 0, starc)
            elif team.lower() == 'england':
                archer = next((p for p in fast_bowlers if p['Player'] == 'Jofra Archer'), None)
                atkinson = next((p for p in fast_bowlers if p['Player'] == 'Gus Atkinson'), None)
                if archer:
                    fast_bowlers = [p for p in fast_bowlers if p['Player'] != 'Jofra Archer']
                    fast_bowlers.insert(0, archer)
                if atkinson:
                    fast_bowlers = [p for p in fast_bowlers if p['Player'] != 'Gus Atkinson']
                    if archer:
                        fast_bowlers.insert(1, atkinson)
                    else:
                        fast_bowlers.insert(0, atkinson)
            
            fast_bowlers.sort(key=lambda x: (
                0 if (team.lower() == 'australia' and x['Player'] == 'Pat Cummins')
                else 1 if (team.lower() == 'australia' and x['Player'] == 'Mitchell Starc') 
                else 2 if (team.lower() == 'england' and x['Player'] == 'Jofra Archer')
                else 3 if (team.lower() == 'england' and x['Player'] == 'Gus Atkinson')
                else self.safe_float(x.get('Bowling_Avg', 50))
            ))
            for bowler in fast_bowlers[:remaining_slots]:
                if len(selected) < 11:
                    add_player(bowler)
        
        remaining_slots = 11 - len(selected)
        if remaining_slots > 0:
            if team.lower() == 'australia':
                remaining_players = [p for p in players if p['Player'] not in positions and p['Player'] != 'Mitchell Marsh']
            elif team.lower() == 'england':
                remaining_players = [p for p in players if p['Player'] not in positions and p['Player'] not in ['Chris Woakes', 'Ollie Robinson', 'Shoaib Bashir', 'Tom Hartley', 'Jack Leach']]
                archer = next((p for p in remaining_players if p['Player'] == 'Jofra Archer'), None)
                atkinson = next((p for p in remaining_players if p['Player'] == 'Gus Atkinson'), None)
                carse = next((p for p in remaining_players if p['Player'] == 'Brydon Carse'), None)
                if archer:
                    remaining_players = [p for p in remaining_players if p['Player'] != 'Jofra Archer']
                    remaining_players.insert(0, archer)
                if atkinson:
                    remaining_players = [p for p in remaining_players if p['Player'] != 'Gus Atkinson']
                    remaining_players.insert(1 if archer else 0, atkinson)
                if carse:
                    remaining_players = [p for p in remaining_players if p['Player'] != 'Brydon Carse']
                    insert_pos = 2 if (archer and atkinson) else (1 if (archer or atkinson) else 0)
                    remaining_players.insert(insert_pos, carse)
            else:
                remaining_players = [p for p in players if p['Player'] not in positions]
            
            remaining_players.sort(key=lambda x: (
                self.safe_float(x.get('Batting_Avg', 0)) * 0.5 + 
                (50 - self.safe_float(x.get('Bowling_Avg', 50))) * 0.5 if self.safe_float(x.get('Bowling_Avg', 50)) > 0 else self.safe_float(x.get('Batting_Avg', 0))
            ), reverse=True)
            
            for player in remaining_players[:remaining_slots]:
                if len(selected) < 11:
                    add_player(player)
        
        if team.lower() == 'australia':
            cummins_included = any(p['name'] == 'Pat Cummins' for p in selected)
            if not cummins_included:
                cummins = next((p for p in players if p['Player'] == 'Pat Cummins'), None)
                if cummins:
                    if len(selected) >= 11:
                        removed = selected.pop()
                        positions.discard(removed['name'])
                    add_player(cummins)
        
        if len(selected) < 11:
            if team.lower() == 'australia':
                all_available = [p for p in players if p['Player'] not in positions and p['Player'] != 'Mitchell Marsh']
                starc_included = any(p['name'] == 'Mitchell Starc' for p in selected)
                if not starc_included:
                    starc = next((p for p in players if p['Player'] == 'Mitchell Starc'), None)
                    if starc and len(selected) < 11:
                        add_player(starc)
            elif team.lower() == 'england':
                all_available = [p for p in players if p['Player'] not in positions and p['Player'] not in ['Chris Woakes', 'Ollie Robinson', 'Shoaib Bashir', 'Tom Hartley', 'Jack Leach']]
                archer_included = any(p['Player'] == 'Jofra Archer' for p in selected)
                atkinson_included = any(p['Player'] == 'Gus Atkinson' for p in selected)
                if not archer_included:
                    archer = next((p for p in players if p['Player'] == 'Jofra Archer'), None)
                    if archer and len(selected) < 11:
                        add_player(archer)
                if not atkinson_included:
                    atkinson = next((p for p in players if p['Player'] == 'Gus Atkinson'), None)
                    if atkinson and len(selected) < 11:
                        add_player(atkinson)
            else:
                all_available = [p for p in players if p['Player'] not in positions]
            
            for player in all_available[:11 - len(selected)]:
                if len(selected) < 11:
                    add_player(player)
        
        strategy_parts = []
        if pace_friendly:
            strategy_parts.append("Focus on pace bowling attack")
        if spin_friendly:
            strategy_parts.append("Include quality spinners")
        if batting_friendly in ['High', 'Very High']:
            strategy_parts.append("Prioritize strong batting lineup")
        if not strategy_parts:
            strategy_parts.append("Balanced approach for all conditions")
        
        strategy = ". ".join(strategy_parts) + "."
        
        return {
            'playing_11': selected[:11],
            'strategy': strategy,
            'captain': captain
        }

CSVTeamSelector = APITeamSelector