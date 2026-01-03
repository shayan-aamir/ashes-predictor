from flask import Flask, render_template, request, jsonify
from prediction_model import AshesPredictor, get_combined_xi, get_players_of_series
from pitch_conditions_ai import get_all_venue_conditions
from csv_team_selector import APITeamSelector, CSVTeamSelector
import os
import json

app = Flask(__name__)

# Initialize predictor (lazy loading to avoid cold start issues)
predictor = None

def get_predictor():
    """Get or create predictor instance (lazy initialization)"""
    global predictor
    if predictor is None:
        predictor = AshesPredictor()
    return predictor

@app.route('/')
def index():
    """Main index page with series predictions and recommendations"""
    try:
        # Get predictor instance
        pred = get_predictor()
        
        # Load model if not already loaded
        if pred.model is None:
            pred.load_model()
        
        # Get series predictions
        venues_2025 = ["Perth", "Brisbane", "Adelaide", "Melbourne", "Sydney"]
        predictions = pred.predict_series(venues_2025)
        series_summary = pred.get_series_summary(predictions)
        
        # Get players of the series
        players_of_series = get_players_of_series()
        
        # Get combined XI
        series_combined_11 = get_combined_xi()
        
        # Get all venue conditions and playing 11s
        api_key = os.getenv('DEEPSEEK_API_KEY')
        all_venue_conditions = get_all_venue_conditions(api_key)
        
        # Use Ashes-only, real-time squads for playing 11 creation
        team_selector = APITeamSelector(include_international=False)
        team_selector.refresh_data()
        all_playing_11s = {}
        
        for venue in venues_2025:
            pitch_conditions = all_venue_conditions.get(venue, {})
            pitch_condition = pitch_conditions.get('pitch_type', 'Standard')
            
            aus_11 = team_selector.select_playing_11_for_venue('Australia', venue, pitch_conditions)
            eng_11 = team_selector.select_playing_11_for_venue('England', venue, pitch_conditions)
            
            all_playing_11s[venue] = {
                'pitch_condition': pitch_condition,
                'australia': aus_11,
                'england': eng_11
            }
        
        return render_template('index.html',
                             predictions=predictions,
                             series_summary=series_summary,
                             players_of_series=players_of_series,
                             all_playing_11s=all_playing_11s,
                             series_combined_11=series_combined_11)
    except Exception as e:
        return f"Error loading page: {str(e)}", 500
    
@app.route('/predict_match', methods=['POST'])
def predict_match():
    """Predict a single match based on provided parameters"""
    try:
        data = request.json
        venue = data.get('venue', 'Perth')
        home_team = data.get('home_team', 'Australia')
        away_team = data.get('away_team', 'England')
        toss_winner = data.get('toss_winner', 'Australia')
        toss_decision = data.get('toss_decision', 'Bat')
        home_form = data.get('home_form', 'Good')
        away_form = data.get('away_form', 'Good')
        venue_type = data.get('venue_type', 'Modern')
        pitch_condition = data.get('pitch_condition', 'Hard')
        weather = data.get('weather', 'Sunny')
        
        pred = get_predictor()
        if pred.model is None:
            pred.load_model()
        
        prediction = pred.predict_match(
            venue, home_team, away_team, toss_winner, toss_decision,
            home_form, away_form, venue_type, pitch_condition, weather
        )
        
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/series_analysis')
def series_analysis():
    """Series analysis page with historical data"""
    try:
        pred = get_predictor()
        if pred.model is None:
            pred.load_model()
        
        # Load historical data
        if hasattr(pred, 'data') and pred.data is not None:
            historical_data = pred.data.to_dict('records')
        else:
            pred.load_data()
            historical_data = pred.data.to_dict('records')
        
        return render_template('series_analysis.html', historical_data=historical_data)
    except Exception as e:
        return f"Error loading analysis: {str(e)}", 500

@app.route('/playing_11/<venue>')
def get_venue_playing_11(venue):
    """Get playing 11 for a specific venue"""
    try:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        from pitch_conditions_ai import get_pitch_conditions_with_ai
        
        pitch_conditions = get_pitch_conditions_with_ai(venue, api_key)
        team_selector = APITeamSelector()
        
        aus_11 = team_selector.select_playing_11_for_venue('Australia', venue, pitch_conditions)
        eng_11 = team_selector.select_playing_11_for_venue('England', venue, pitch_conditions)
        
        return jsonify({
            'venue': venue,
            'pitch_conditions': pitch_conditions,
            'australia': aus_11,
            'england': eng_11
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

# ==================== RECOMMENDATION SYSTEM ENDPOINTS ====================
# These endpoints implement various recommendation system algorithms for the course

@app.route('/player_recommendations', methods=['GET'])
def player_recommendations():
    """
    Content-Based Recommendation Endpoint
    Recommends players based on user preference (batting, bowling, allrounder)
    Algorithm: Content-Based Filtering
    Supports players from any country
    """
    try:
        preference = request.args.get('preference', 'batting')
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        recommendations = team_selector.get_player_recommendations(preference, count, country)
        
        return jsonify({
            'algorithm': 'Content-Based Filtering',
            'preference': preference,
            'country': country,
            'count': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/collaborative_recommendations', methods=['GET'])
def collaborative_recommendations():
    """
    Collaborative Filtering Recommendation Endpoint
    Recommends players similar to a reference player
    Algorithm: User-Based Collaborative Filtering
    Works with players from any country
    """
    try:
        reference_player = request.args.get('player', '')
        if not reference_player:
            return jsonify({'error': 'Please provide a reference player name'}), 400
        
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        recommendations = team_selector.collaborative_filtering_recommendations(reference_player, count, country)
        
        return jsonify({
            'algorithm': 'Collaborative Filtering (User-Based)',
            'reference_player': reference_player,
            'country': country,
            'count': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hybrid_recommendations', methods=['GET'])
def hybrid_recommendations():
    """
    Hybrid Recommendation Endpoint
    Combines content-based and collaborative filtering
    Algorithm: Hybrid Recommendation System
    Supports players from any country
    """
    try:
        preference = request.args.get('preference', 'batting')
        venue = request.args.get('venue', None)
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        recommendations = team_selector.hybrid_recommendations(preference, venue, count, country)
        
        return jsonify({
            'algorithm': 'Hybrid Recommendation System',
            'preference': preference,
            'venue': venue,
            'country': country,
            'count': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/venue_recommendations', methods=['GET'])
def venue_recommendations():
    """
    Venue-Based Recommendation Endpoint
    Recommends players optimal for a specific venue
    Algorithm: Venue-Based Content Filtering
    Supports players from any country
    """
    try:
        venue = request.args.get('venue', '')
        if not venue:
            return jsonify({'error': 'Please provide a venue name'}), 400
        
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        recommendations = team_selector.venue_based_recommendations(venue, count, country)
        
        return jsonify({
            'algorithm': 'Venue-Based Recommendation',
            'venue': venue,
            'country': country,
            'count': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similarity_recommendations', methods=['GET'])
def similarity_recommendations():
    """
    Similarity-Based Recommendation Endpoint
    Finds players with similar profiles using item-based filtering
    Algorithm: Item-Based Collaborative Filtering
    Works with players from any country
    """
    try:
        player_name = request.args.get('player', '')
        if not player_name:
            return jsonify({'error': 'Please provide a player name'}), 400
        
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        recommendations = team_selector.similarity_based_recommendations(player_name, count, country)
        
        return jsonify({
            'algorithm': 'Item-Based Collaborative Filtering',
            'reference_player': player_name,
            'country': country,
            'count': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/popularity_recommendations', methods=['GET'])
def popularity_recommendations():
    """
    Popularity-Based Recommendation Endpoint
    Recommends most popular/successful players
    Algorithm: Popularity-Based Ranking
    Supports players from any country
    """
    try:
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        recommendations = team_selector.popularity_based_recommendations(count, country)
        
        return jsonify({
            'algorithm': 'Popularity-Based Recommendation',
            'country': country,
            'count': len(recommendations),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendation_comparison', methods=['GET'])
def recommendation_comparison():
    """
    Comparison Endpoint: Shows recommendations from multiple algorithms
    Useful for comparing different recommendation approaches
    Supports players from any country
    """
    try:
        preference = request.args.get('preference', 'batting')
        count = int(request.args.get('count', 5))
        country = request.args.get('country', None)  # Optional country filter
        team_selector = APITeamSelector(include_international=False)
        
        # Get recommendations from different algorithms
        content_based = team_selector.get_player_recommendations(preference, count, country)
        hybrid = team_selector.hybrid_recommendations(preference, None, count, country)
        popularity = team_selector.popularity_based_recommendations(count, country)
        
        return jsonify({
            'comparison': {
                'content_based': {
                    'algorithm': 'Content-Based Filtering',
                    'count': len(content_based),
                    'recommendations': content_based
                },
                'hybrid': {
                    'algorithm': 'Hybrid Recommendation',
                    'count': len(hybrid),
                    'recommendations': hybrid
                },
                'popularity_based': {
                    'algorithm': 'Popularity-Based',
                    'count': len(popularity),
                    'recommendations': popularity
                }
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_player', methods=['GET'])
def search_player():
    """
    Search for any player by name and country
    Returns player information if found
    """
    try:
        player_name = request.args.get('name', '')
        country = request.args.get('country', None)
        
        if not player_name:
            return jsonify({'error': 'Please provide a player name'}), 400
        
        team_selector = APITeamSelector(include_international=False)
        if not team_selector.fetcher:
            from cricket_data_fetcher import CricketDataFetcher
            team_selector.fetcher = CricketDataFetcher()
        
        player = team_selector.fetcher.find_player(player_name, country)
        
        if player:
            return jsonify({
                'found': True,
                'player': player
            })
        else:
            return jsonify({
                'found': False,
                'message': f'Player "{player_name}" not found' + (f' in {country}' if country else '')
            }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_players_by_country', methods=['GET'])
def get_players_by_country():
    """
    Get all players from a specific country
    """
    try:
        country = request.args.get('country', '')
        if not country:
            return jsonify({'error': 'Please provide a country name'}), 400
        
        team_selector = APITeamSelector(include_international=False)
        if not team_selector.fetcher:
            from cricket_data_fetcher import CricketDataFetcher
            team_selector.fetcher = CricketDataFetcher()
        
        players = team_selector.fetcher.get_players_by_country(country)
        
        return jsonify({
            'country': country,
            'count': len(players),
            'players': players
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== END OF RECOMMENDATION SYSTEM ENDPOINTS ====================

# Real-time Data Refresh Endpoint
@app.route('/refresh_squad_data', methods=['POST'])
def refresh_squad_data():
    """Refresh squad data from real-time APIs"""
    try:
        team_selector = APITeamSelector()
        success = team_selector.refresh_data()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Squad data refreshed successfully',
                'england_players': len(team_selector.england_players),
                'australia_players': len(team_selector.australia_players)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to refresh data. Using cached data.'
            }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_squad_info', methods=['GET'])
def get_squad_info():
    """Get current squad information"""
    try:
        team_selector = APITeamSelector(include_international=False)
        return jsonify({
            'england': {
                'count': len(team_selector.england_players),
                'players': [p['Player'] for p in team_selector.england_players]
            },
            'australia': {
                'count': len(team_selector.australia_players),
                'players': [p['Player'] for p in team_selector.australia_players]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def select_players_by_role(players, venue, pitch_condition):
    """Helper function to select players by role for a venue"""
    # This is handled by APITeamSelector class
    team_selector = APITeamSelector()
    return team_selector.select_playing_11_for_venue('', venue, {'pitch_type': pitch_condition})

def generate_strategy_from_players(team, playing_11, venue, pitch_conditions):
    """Generate strategy based on selected players and venue conditions"""
    strategy_parts = []
    
    if pitch_conditions.get('pace_friendly'):
        strategy_parts.append("Focus on pace bowling attack")
    if pitch_conditions.get('spin_friendly'):
        strategy_parts.append("Include quality spinners")
    if pitch_conditions.get('batting_friendly') in ['High', 'Very High']:
        strategy_parts.append("Prioritize strong batting lineup")
    
    if not strategy_parts:
        strategy_parts.append("Balanced approach for all conditions")
    
    return ". ".join(strategy_parts) + "."

# Add a simple health check route for Vercel
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Flask app is running'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)