from flask import Flask, render_template, request, jsonify
from prediction_model import AshesPredictor, get_combined_xi, get_players_of_series
from pitch_conditions_ai import get_all_venue_conditions
from csv_team_selector import CSVTeamSelector
import os
import json

app = Flask(__name__)

# Initialize predictor
predictor = AshesPredictor()

@app.route('/')
def index():
    """Main page with series prediction"""
    # Always train model on startup (no pre-trained model file)
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_model()
    
    # Predict 2025-2026 series with correct schedule (Australia at home)
    venues_2025 = ["Perth", "Brisbane", "Adelaide", "Melbourne", "Sydney"]
    predictions = predictor.predict_series(venues_2025, home_team="Australia", away_team="England")
    series_summary = predictor.get_series_summary(predictions)
    
    # Get DeepSeek API key for pitch conditions
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    
    # Initialize CSV team selector
    team_selector = CSVTeamSelector()
    
    # Get pitch conditions using AI
    pitch_conditions = get_all_venue_conditions(deepseek_api_key)
    
    # Generate playing 11s for all venues using CSV data
    all_playing_11s = {}
    for venue in venues_2025:
        venue_conditions = pitch_conditions.get(venue, {})
        
        # Get playing 11s
        england_11 = team_selector.select_playing_11_for_venue('England', venue, venue_conditions)
        australia_11 = team_selector.select_playing_11_for_venue('Australia', venue, venue_conditions)
        
        # Generate strategies based on selected players
        england_strategy = generate_strategy_from_players('England', england_11, venue, venue_conditions)
        australia_strategy = generate_strategy_from_players('Australia', australia_11, venue, venue_conditions)
        
        all_playing_11s[venue] = {
            'england': {
                'playing_11': england_11,
                'strategy': england_strategy
            },
            'australia': {
                'playing_11': australia_11,
                'strategy': australia_strategy
            }
        }
    
    # Generate series combined 11 using CSV data
    series_combined_11 = team_selector.get_series_combined_11()
    
    # Get players of series data
    players_of_series = get_players_of_series()
    
    return render_template('index.html', 
                         predictions=predictions,
                         series_summary=series_summary,
                         players_of_series=players_of_series,
                         all_playing_11s=all_playing_11s,
                         series_combined_11=series_combined_11,
                         pitch_conditions=pitch_conditions)

@app.route('/predict_match', methods=['POST'])
def predict_match():
    """API endpoint for single match prediction"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['venue', 'home_team', 'away_team', 'toss_winner', 
                          'toss_decision', 'home_form', 'away_form', 
                          'venue_type', 'pitch_condition', 'weather']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        prediction = predictor.predict_match(
            venue=data['venue'],
            home_team=data['home_team'],
            away_team=data['away_team'],
            toss_winner=data['toss_winner'],
            toss_decision=data['toss_decision'],
            home_form=data['home_form'],
            away_form=data['away_form'],
            venue_type=data['venue_type'],
            pitch_condition=data['pitch_condition'],
            weather=data['weather']
        )
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/series_analysis')
def series_analysis():
    """Page showing detailed series analysis"""
    # Load historical data
    data = predictor.load_data()
    
    # Calculate historical statistics
    england_wins = len(data[data['Winning_Team'] == 'England'])
    australia_wins = len(data[data['Winning_Team'] == 'Australia'])
    draws = len(data[data['Winning_Team'] == 'Draw'])
    total_matches = len(data)
    
    # Venue analysis
    venue_stats = data.groupby('Venue')['Winning_Team'].value_counts().unstack(fill_value=0)
    
    # Recent form (last 3 series)
    recent_data = data[data['Year'] >= 2015]
    recent_england_wins = len(recent_data[recent_data['Winning_Team'] == 'England'])
    recent_australia_wins = len(recent_data[recent_data['Winning_Team'] == 'Australia'])
    recent_draws = len(recent_data[recent_data['Winning_Team'] == 'Draw'])
    
    stats = {
        'total_matches': total_matches,
        'england_wins': england_wins,
        'australia_wins': australia_wins,
        'draws': draws,
        'recent_england_wins': recent_england_wins,
        'recent_australia_wins': recent_australia_wins,
        'recent_draws': recent_draws,
        'venue_stats': venue_stats.to_dict() if not venue_stats.empty else {}
    }
    
    return render_template('series_analysis.html', stats=stats)

@app.route('/playing_11/<venue>')
def get_venue_playing_11(venue):
    """Get playing 11s for a specific venue"""
    try:
        from deepseek_team_selector import DeepSeekTeamSelector, get_venue_pitch_conditions
        
        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        selector = DeepSeekTeamSelector(deepseek_api_key)
        pitch_conditions = get_venue_pitch_conditions()
        
        pitch_condition = pitch_conditions.get(venue, "Normal")
        
        england_11 = selector.get_playing_11_for_venue(venue, "England", pitch_condition)
        australia_11 = selector.get_playing_11_for_venue(venue, "Australia", pitch_condition)
        
        return jsonify({
            'venue': venue,
            'pitch_condition': pitch_condition,
            'england_11': england_11,
            'australia_11': australia_11
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page explaining the prediction model"""
    return render_template('about.html')



def select_players_by_role(players, venue, pitch_condition):
    """Select 11 players based on roles and venue conditions"""
    selected_players = []
    selected_names = set()
    
    # Sort players by batting average for batters, bowling average for bowlers
    batters = [p for p in players if 'Batter' in p['Role'] or p['Role'] == 'Opener' or p['Role'] == 'Top_Order_Batter']
    allrounders = [p for p in players if 'Allrounder' in p['Role'] or p['Role'] == 'Allrounder']
    bowlers = [p for p in players if 'Bowler' in p['Role'] or p['Role'] == 'Fast_Bowler' or p['Role'] == 'Spinner']
    
    # Sort batters by batting average (highest first)
    batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
    
    # Sort allrounders by combined performance
    allrounders.sort(key=lambda x: (float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0) + (float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50), reverse=True)
    
    # Sort bowlers by bowling average (lowest first for better bowlers)
    bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
    
    # Always give preference to Starc over Boland for Australia
    if any('Starc' in p['Player'] for p in bowlers):
        starc_players = [p for p in bowlers if 'Starc' in p['Player']]
        other_bowlers = [p for p in bowlers if 'Starc' not in p['Player']]
        bowlers = starc_players + other_bowlers
    
    # Venue-specific adjustments (but keep Starc preference)
    if venue == "Perth":
        # Perth: Fast, bouncy pitch - prefer pace bowlers and aggressive batters
        # Keep Starc at the top, then sort others by bowling average
        starc_players = [p for p in bowlers if 'Starc' in p['Player']]
        other_bowlers = [p for p in bowlers if 'Starc' not in p['Player']]
        other_bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
        bowlers = starc_players + other_bowlers
    elif venue == "Brisbane":
        # Brisbane: Good batting surface - prefer better batters
        batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
        # Also prefer better bowlers for day-night conditions
        starc_players = [p for p in bowlers if 'Starc' in p['Player']]
        other_bowlers = [p for p in bowlers if 'Starc' not in p['Player']]
        other_bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
        bowlers = starc_players + other_bowlers
    elif venue == "Adelaide":
        # Adelaide: Balanced pitch - balanced approach
        batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
        starc_players = [p for p in bowlers if 'Starc' in p['Player']]
        other_bowlers = [p for p in bowlers if 'Starc' not in p['Player']]
        other_bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
        bowlers = starc_players + other_bowlers
    elif venue == "Melbourne":
        # Melbourne: Batting-friendly - prefer better batters
        batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
        # Also prefer better bowlers
        starc_players = [p for p in bowlers if 'Starc' in p['Player']]
        other_bowlers = [p for p in bowlers if 'Starc' not in p['Player']]
        other_bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
        bowlers = starc_players + other_bowlers
    elif venue == "Sydney":
        # Sydney: Spin-friendly - prefer spinners and better batters
        batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
        spinners = [p for p in bowlers if 'Spinner' in p['Role']]
        starc_players = [p for p in bowlers if 'Starc' in p['Player']]
        other_fast_bowlers = [p for p in bowlers if 'Fast_Bowler' in p['Role'] and 'Starc' not in p['Player']]
        bowlers = spinners + starc_players + other_fast_bowlers  # Spinners first, then Starc, then others
    
    # Default sorting if no venue-specific adjustments
    if not any(venue == v for v in ["Perth", "Brisbane", "Adelaide", "Melbourne", "Sydney"]):
        batters.sort(key=lambda x: float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0, reverse=True)
        allrounders.sort(key=lambda x: (float(x['Batting_Avg']) if x['Batting_Avg'] != 0 else 0) + (float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50), reverse=True)
        bowlers.sort(key=lambda x: float(x['Bowling_Avg']) if x['Bowling_Avg'] != 0 else 50)
    
    # Build playing 11 in correct batting order
    playing_11 = []
    
    # Select players in the correct order
    # 1. Openers (positions 1-2)
    openers_selected = 0
    for player in batters:
        if player['Role'] == 'Opener' and player['Player'] not in selected_names and openers_selected < 2:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"Opening batsman with {player['Batting_Avg']} average"
            })
            selected_names.add(player['Player'])
            openers_selected += 1
    
    # 2. Number 3 (Top Order Batter)
    for player in batters:
        if player['Role'] == 'Top_Order_Batter' and player['Player'] not in selected_names:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"Number 3 batsman with {player['Batting_Avg']} average"
            })
            selected_names.add(player['Player'])
            break
    
    # 3. Middle Order Batters (positions 4-5)
    middle_order_selected = 0
    for player in batters:
        if player['Role'] == 'Middle_Order_Batter' and player['Player'] not in selected_names and middle_order_selected < 2:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"Middle order batsman with {player['Batting_Avg']} average"
            })
            selected_names.add(player['Player'])
            middle_order_selected += 1
            if middle_order_selected >= 2:
                break
    
    # 4. All-rounder (position 6)
    for player in allrounders:
        if player['Player'] not in selected_names:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"All-rounder with {player['Batting_Avg']} batting and {player['Bowling_Avg']} bowling average"
            })
            selected_names.add(player['Player'])
            break
    
    # 5. Wicket-keeper (position 7)
    for player in batters:
        if player['Role'] == 'Wicketkeeper_Batter' and player['Player'] not in selected_names:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"Wicket-keeper batsman with {player['Batting_Avg']} average"
            })
            selected_names.add(player['Player'])
            break
    
    # 6. Bowlers (positions 8-11)
    # Fast bowlers first (3 positions)
    fast_bowlers_selected = 0
    for player in bowlers:
        if 'Fast_Bowler' in player['Role'] and player['Player'] not in selected_names and fast_bowlers_selected < 3:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"Fast bowler with {player['Bowling_Avg']} bowling average"
            })
            selected_names.add(player['Player'])
            fast_bowlers_selected += 1
            if fast_bowlers_selected >= 3:
                break
    
    # Spinner last (1 position)
    for player in bowlers:
        if 'Spinner' in player['Role'] and player['Player'] not in selected_names:
            playing_11.append({
                "name": player['Player'],
                "role": player['Role'],
                "position": str(len(playing_11) + 1),
                "reason": f"Spinner with {player['Bowling_Avg']} bowling average"
            })
            selected_names.add(player['Player'])
            break
    
    # If we don't have 11 players, fill with remaining best players
    while len(playing_11) < 11:
        for player in players:
            if player['Player'] not in selected_names:
                playing_11.append({
                    "name": player['Player'],
                    "role": player['Role'],
                    "position": str(len(playing_11) + 1),
                    "reason": f"Additional player with {player['Batting_Avg']} batting and {player['Bowling_Avg']} bowling average"
                })
                selected_names.add(player['Player'])
                break
    
    return playing_11

def generate_strategy_from_players(team, playing_11, venue, pitch_conditions):
    """Generate strategy based on selected players and venue conditions"""
    
    # Analyze selected players
    batters = [p for p in playing_11 if 'Batter' in p['role'] or p['role'] == 'Opener' or p['role'] == 'Top_Order_Batter']
    allrounders = [p for p in playing_11 if 'Allrounder' in p['role'] or p['role'] == 'Allrounder']
    bowlers = [p for p in playing_11 if 'Bowler' in p['role'] or p['role'] == 'Fast_Bowler' or p['role'] == 'Spinner']
    
    # Count different types of bowlers
    fast_bowlers = [p for p in bowlers if 'Fast_Bowler' in p['role']]
    spinners = [p for p in bowlers if 'Spinner' in p['role']]
    
    # Get pitch conditions
    pitch_type = pitch_conditions.get('pitch_type', 'Standard')
    pace_friendly = pitch_conditions.get('pace_friendly', True)
    spin_friendly = pitch_conditions.get('spin_friendly', False)
    batting_friendly = pitch_conditions.get('batting_friendly', 'Moderate')
    
    # Generate strategy based on team composition and conditions
    strategy_parts = []
    
    # Batting strategy
    if batting_friendly in ['High', 'Very High']:
        strategy_parts.append(f"Focus on aggressive batting with {len(batters)} specialist batters.")
    else:
        strategy_parts.append(f"Balanced batting approach with {len(batters)} batters.")
    
    # Bowling strategy
    if len(fast_bowlers) >= 3 and pace_friendly:
        strategy_parts.append(f"Pace-heavy attack with {len(fast_bowlers)} fast bowlers suits the {pitch_type} conditions.")
    elif len(spinners) >= 1 and spin_friendly:
        strategy_parts.append(f"Spin-friendly conditions with {len(spinners)} spinner(s) will be key.")
    else:
        strategy_parts.append(f"Mixed bowling attack with {len(fast_bowlers)} pace and {len(spinners)} spin options.")
    
    # All-rounder strategy
    if len(allrounders) > 0:
        strategy_parts.append(f"All-rounder(s) provide balance and flexibility.")
    
    # Venue-specific adjustments
    if venue == "Perth":
        strategy_parts.append("Perth's bouncy pitch requires aggressive pace bowling and solid batting.")
    elif venue == "Brisbane":
        strategy_parts.append("Day-night conditions require adaptability in both batting and bowling.")
    elif venue == "Adelaide":
        strategy_parts.append("Adelaide's balanced pitch requires a well-rounded approach.")
    elif venue == "Melbourne":
        strategy_parts.append("Melbourne's batting-friendly conditions require strong batting performance.")
    elif venue == "Sydney":
        strategy_parts.append("Sydney's spin-friendly conditions require effective spin bowling.")
    
    return " ".join(strategy_parts)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 