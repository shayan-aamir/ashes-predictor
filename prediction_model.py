import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class AshesPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = [
            'Venue', 'Home_Team', 'Away_Team', 'Toss_Winner', 'Toss_Decision',
            'Home_Team_Form', 'Away_Team_Form', 'Venue_Type', 'Pitch_Condition', 'Weather'
        ]
        self.target_column = 'Winning_Team'
        
    def load_data(self, file_path=None):
        """Load and preprocess the Ashes data"""
        if file_path is None:
            # Use absolute path for Vercel compatibility
            file_path = os.path.join(os.path.dirname(__file__), 'ashes_data.csv')
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} matches from {self.data['Year'].min()} to {self.data['Year'].max()}")
        return self.data
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        df = self.data.copy()
        
        for column in self.feature_columns:
            if column in df.columns:
                le = LabelEncoder()
                df[f'{column}_encoded'] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le
        
        df['target'] = df.apply(self._create_target, axis=1)
        
        feature_cols = [f'{col}_encoded' for col in self.feature_columns]
        self.X = df[feature_cols]
        self.y = df['target']
        
        print(f"Preprocessed data shape: {self.X.shape}")
        print(f"Target distribution: {self.y.value_counts()}")
        
        return self.X, self.y
    
    def _create_target(self, row):
        """Create target variable based on winning team"""
        if row['Winning_Team'] == row['Home_Team']:
            return 1
        elif row['Winning_Team'] == row['Away_Team']:
            return 0
        else:
            return 2
    
    def train_model(self):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win', 'Draw']))
        
        return accuracy
    
    def predict_match(self, venue, home_team, away_team, toss_winner, toss_decision,
                     home_form, away_form, venue_type, pitch_condition, weather):
        """Predict the outcome of a single match"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        input_data = {
            'Venue': venue,
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Toss_Winner': toss_winner,
            'Toss_Decision': toss_decision,
            'Home_Team_Form': home_form,
            'Away_Team_Form': away_form,
            'Venue_Type': venue_type,
            'Pitch_Condition': pitch_condition,
            'Weather': weather
        }
        
        encoded_features = []
        for col in self.feature_columns:
            if col in input_data:
                try:
                    encoded_value = self.label_encoders[col].transform([input_data[col]])[0]
                    encoded_features.append(encoded_value)
                except ValueError:
                    encoded_features.append(0)
        
        prediction = self.model.predict([encoded_features])[0]
        probabilities = self.model.predict_proba([encoded_features])[0]
        
        if prediction == 1:
            result = home_team
        elif prediction == 0:
            result = away_team
        else:
            result = "Draw"
        
        return {
            'predicted_winner': result,
            'home_win_probability': probabilities[1],
            'away_win_probability': probabilities[0],
            'draw_probability': probabilities[2] if len(probabilities) > 2 else 0
        }
    
    def predict_series(self, venues, home_team="Australia", away_team="England", known_results=None):                                                          
        """
        Predict the entire Ashes series.
        known_results: optional dict keyed by match_number or venue -> fixed winner (e.g., {'Perth': 'Australia'})
        """
        predictions = []

        home_form = "Good"
        away_form = "Good"
        known_results = known_results or {}

        def update_form(last_winner):
            nonlocal home_form, away_form
            if last_winner == home_team:
                home_form = "Excellent"
                away_form = "Poor"
            elif last_winner == away_team:
                home_form = "Poor"
                away_form = "Excellent"
            else:
                home_form = "Good"
                away_form = "Good"

        for i, venue in enumerate(venues, 1):
            toss_winner = home_team if i % 2 == 1 else away_team
            toss_decision = "Bat" if i % 2 == 1 else "Bowl"
            venue_type = "Modern"
            pitch_condition = "Hard"
            weather = "Sunny"

            forced_winner = known_results.get(i) or known_results.get(venue)

            if forced_winner:
                match_prediction = {
                    'predicted_winner': forced_winner,
                    'home_win_probability': 1.0 if forced_winner == home_team else 0.0,
                    'away_win_probability': 1.0 if forced_winner == away_team else 0.0,
                    'draw_probability': 1.0 if forced_winner == "Draw" else 0.0
                }
            else:
                match_prediction = self.predict_match(
                    venue=venue,
                    home_team=home_team,
                    away_team=away_team,
                    toss_winner=toss_winner,
                    toss_decision=toss_decision,
                    home_form=home_form,
                    away_form=away_form,
                    venue_type=venue_type,
                    pitch_condition=pitch_condition,
                    weather=weather
                )

            predictions.append({
                'match_number': i,
                'venue': venue,
                'predicted_winner': match_prediction['predicted_winner'],       
                'home_win_probability': match_prediction['home_win_probability'],                                                                               
                'away_win_probability': match_prediction['away_win_probability'],                                                                               
                'draw_probability': match_prediction['draw_probability']
            })

            update_form(predictions[-1]['predicted_winner'])

        return predictions
    
    def get_series_summary(self, predictions):
        """Generate series summary and statistics"""
        home_wins = sum(1 for p in predictions if p['predicted_winner'] == 'Australia')
        away_wins = sum(1 for p in predictions if p['predicted_winner'] == 'England')
        draws = sum(1 for p in predictions if p['predicted_winner'] == 'Draw')
        
        series_winner = "Australia" if home_wins > away_wins else "England" if away_wins > home_wins else "Series Drawn"
        
        return {
            'series_winner': series_winner,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws,
            'total_matches': len(predictions)
        }
    
    def save_model(self, filepath='ashes_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            # Use absolute path for Vercel compatibility
            filepath = os.path.join(os.path.dirname(__file__), 'ashes_model.pkl')
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False

def get_combined_xi():
    """Get the combined XI for both teams based on current form"""
    try:
        from team_analyzer import get_updated_combined_xi
        return get_updated_combined_xi()
    except ImportError:
        combined_xi = [
            {'position': 1, 'name': 'Zak Crawley', 'team': 'ENG', 'role': 'Opener', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 2, 'name': 'Usman Khawaja', 'team': 'AUS', 'role': 'Opener', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 3, 'name': 'Marnus Labuschagne', 'team': 'AUS', 'role': 'Top_Order_Batter', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 4, 'name': 'Steven Smith', 'team': 'AUS', 'role': 'Middle_Order_Batter', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 5, 'name': 'Joe Root', 'team': 'ENG', 'role': 'Middle_Order_Batter', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 6, 'name': 'Ben Stokes', 'team': 'ENG', 'role': 'Allrounder', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 7, 'name': 'Alex Carey', 'team': 'AUS', 'role': 'Wicketkeeper_Batter', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 8, 'name': 'Pat Cummins', 'team': 'AUS', 'role': 'Fast_Bowler', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 9, 'name': 'Mitchell Starc', 'team': 'AUS', 'role': 'Fast_Bowler', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 10, 'name': 'Josh Hazlewood', 'team': 'AUS', 'role': 'Fast_Bowler', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'},
            {'position': 11, 'name': 'Nathan Lyon', 'team': 'AUS', 'role': 'Spinner', 'captain': 'Pat Cummins', 'vice_captain': 'Ben Stokes'}
        ]
        return combined_xi

def get_players_of_series():
    """Get predicted players of the series for both teams"""
    try:
        from team_analyzer import get_updated_players_of_series
        return get_updated_players_of_series()
    except ImportError:
        players = {
            'england_player_of_series': 'Ben Stokes',
            'australia_player_of_series': 'Pat Cummins',
            'overall_player_of_series': 'Pat Cummins'
        }
        return players

    def get_recommended_players_for_match(self, venue, home_team, away_team):
        """
        Recommendation System Integration: Recommends key players for a match
        Combines prediction model insights with recommendation algorithms
        """
        from csv_team_selector import APITeamSelector
        
        team_selector = APITeamSelector()
        venue_recommendations = team_selector.venue_based_recommendations(venue, count=3)
        hybrid_recommendations = team_selector.hybrid_recommendations('allrounder', venue, count=3)
        
        return {
            'venue': venue,
            'venue_based': venue_recommendations,
            'hybrid_recommendations': hybrid_recommendations,
            'recommendation_algorithm': 'Hybrid (Venue + Performance)'
        }

if __name__ == "__main__":
    predictor = AshesPredictor()
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_model()
    predictor.save_model()
    
    venues_2025 = ["Perth", "Brisbane", "Adelaide", "Melbourne", "Sydney"]
    predictions = predictor.predict_series(venues_2025)
    
    print("\n=== 2025-2026 Ashes Series Predictions ===")
    for pred in predictions:
        print(f"Test {pred['match_number']}: {pred['venue']} - {pred['predicted_winner']}")
    
    summary = predictor.get_series_summary(predictions)
    print(f"\nSeries Winner: {summary['series_winner']}")
    print(f"England Wins: {summary['home_wins']}")
    print(f"Australia Wins: {summary['away_wins']}")
    print(f"Draws: {summary['draws']}")
    
    print("\n=== Recommendation System Test ===")
    recs = predictor.get_recommended_players_for_match("Perth", "Australia", "England")
    print(f"Venue: {recs['venue']}")
    print(f"Algorithm: {recs['recommendation_algorithm']}") 