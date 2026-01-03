# 🏏 Ashes 2025-2026 Win Predictor

A comprehensive machine learning-based prediction system for the upcoming Ashes series between England and Australia, featuring AI-powered pitch analysis, dynamic team selection, and detailed match predictions.

## 🌟 Features

### 📊 Match Predictions
- **Series Outcome**: Predicts the overall winner of the 2025-2026 Ashes series
- **Individual Match Results**: Predicts winner, venue, and player of the match for each test
- **Historical Analysis**: Uses data from 2000-2023 Ashes series for accurate predictions
- **Probability Analysis**: Shows win probabilities for both teams

### 🏟️ Venue-Specific Playing 11s
- **AI-Powered Pitch Analysis**: Uses DeepSeek API to analyze pitch conditions for each venue
- **Dynamic Team Selection**: Generates different playing 11s for each match based on:
  - Pitch conditions (pace-friendly, spin-friendly, batting-friendly)
  - Venue characteristics (Perth's bounce, Sydney's spin, etc.)
  - Player performance statistics
- **Real-Time Squad Data**: Automatically fetches latest squad data from CricAPI/ESPN Cricinfo
- **Guaranteed 11 Players**: Selection algorithm ensures exactly 11 players are always selected
- **Captain Assignments**: 
  - **Australia**: Pat Cummins as captain
  - **England**: Ben Stokes as captain
- **Strategic Variations**: Different team compositions for each venue

### 🏆 Series Combined 11
- **Best of Both Teams**: Selects the ultimate playing 11 from both England and Australia
- **Performance-Based Selection**: Uses batting and bowling averages for player selection
- **Leadership Structure**:
  - **Captain**: Pat Cummins (Australia)
  - **Vice Captain**: Ben Stokes (England)
- **Role-Based Selection**: Ensures proper balance of openers, middle-order, all-rounders, and bowlers

### 🎯 Players of the Series
- **Best Batsman**: Top run-scorer prediction
- **Best Bowler**: Leading wicket-taker prediction
- **Best All-rounder**: Most valuable player prediction

### 🤖 Recommendation System (For Course)
This project implements multiple recommendation system algorithms for the recommendation system course:

#### 1. **Content-Based Filtering**
- Recommends players based on user preferences (batting, bowling, allrounder)
- Uses player attributes (batting average, bowling average) for similarity matching
- **Supports players from any country**
- **Endpoint**: `/player_recommendations?preference=batting&country=India&count=5`

#### 2. **Collaborative Filtering (User-Based)**
- Recommends players similar to a reference player
- Uses cosine similarity to find players with similar performance profiles
- **Works with any player from any country**
- **Endpoint**: `/collaborative_recommendations?player=Virat Kohli&country=India&count=5`

#### 3. **Hybrid Recommendation System**
- Combines content-based and collaborative filtering approaches
- Weighted combination (60% content-based, 40% collaborative)
- Considers venue performance and recent form
- **Supports players from any country**
- **Endpoint**: `/hybrid_recommendations?preference=batting&venue=Perth&country=Pakistan&count=5`

#### 4. **Venue-Based Recommendation**
- Recommends players optimal for specific venue conditions
- Considers pace-friendly, spin-friendly, and batting-friendly characteristics
- **Endpoint**: `/venue_recommendations?venue=Perth&count=5`

#### 5. **Item-Based Collaborative Filtering**
- Finds players with similar profiles using Euclidean distance
- Item-item similarity approach
- **Endpoint**: `/similarity_recommendations?player=Pat Cummins&count=5`

#### 6. **Popularity-Based Recommendation**
- Recommends most popular/successful players
- Based on match wins, player of match awards, and experience
- **Endpoint**: `/popularity_recommendations?count=5`

#### 7. **Recommendation Comparison**
- Compares recommendations from multiple algorithms
- Useful for evaluating different recommendation approaches
- **Endpoint**: `/recommendation_comparison?preference=batting&count=5`

### 🔄 Real-Time Data Management

#### Squad Data Refresh
- **Refresh Squad Data**: Updates team data from real-time APIs
- **Endpoint**: `POST /refresh_squad_data`
- **Get Squad Info**: View current squad information
- **Endpoint**: `GET /get_squad_info`
- **Search Player**: Find any player from any country
- **Endpoint**: `GET /search_player?name=Virat Kohli&country=India`
- **Get Players by Country**: Get all players from a specific country
- **Endpoint**: `GET /get_players_by_country?country=Pakistan`

#### Data Sources Priority
1. **CricAPI** (if API key provided) - Real-time cricket data
2. **ESPN Cricinfo** (if API key provided) - ESPN format support  
3. **Default Squad Data** - Built-in default Ashes 2025-2026 squads if APIs unavailable

**Note**: The system no longer uses CSV files for team data. All squad data is fetched from APIs or uses built-in default data.

#### Setting Up Real-Time Data
1. Get a free API key from [CricAPI](https://www.cricapi.com/)
2. Set environment variable: `export CRICAPI_KEY="your_key_here"`
3. The system will automatically fetch latest squad data on startup
4. Use `/refresh_squad_data` endpoint to manually update data

### 🏏 Match Schedule (2025-2026) - Pakistan Time (PKT)
1. **1st Test**: Perth (November 22-26, 2025) - 7:30 AM PKT
2. **2nd Test**: Brisbane (December 4-8, 2025) - 9:30 AM PKT (Day/Night)
3. **3rd Test**: Adelaide (December 12-16, 2025) - 5:00 AM PKT
4. **4th Test**: Melbourne (December 26-30, 2025) - 4:30 AM PKT
5. **5th Test**: Sydney (January 3-7, 2026) - 4:30 AM PKT

## 🛠️ Technical Stack

### Backend
- **Python 3.12**: Core programming language
- **Flask**: Web framework for the application
- **Scikit-learn**: Machine learning library for predictions
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Frontend
- **Bootstrap 5**: Modern, responsive UI framework
- **Font Awesome**: Icons and visual elements
- **HTML5/CSS3**: Structure and styling

### External APIs
- **DeepSeek API**: AI-powered pitch condition analysis
- **CricAPI**: Real-time cricket squad and player data (free tier available)
- **ESPN Cricinfo**: Cricket data format support (structure ready for API integration)
- **Environment Variables**: Secure API key management


## 📁 Project Structure

```
ASHES_PREDICTOR/
├── app.py                      # Main Flask application
├── prediction_model.py         # ML model for predictions
├── csv_team_selector.py        # Team selection logic (API-based, no CSV dependency)
├── cricket_data_fetcher.py     # Real-time cricket data fetcher (CricAPI/ESPN)
├── pitch_conditions_ai.py      # AI pitch analysis
├── ashes_data.csv             # Historical Ashes data (for ML model only)
├── requirements.txt           # Python dependencies
├── vercel.json               # Vercel deployment config
├── .vercelignore             # Vercel ignore rules
├── templates/
│   ├── base.html             # Base template
│   ├── index.html            # Main dashboard
│   ├── series_analysis.html  # Historical analysis
│   └── about.html            # About page
└── README.md                 # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ASHES_PREDICTOR
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**
   ```bash
   # Required for real-time cricket data (CricAPI)
   # Get free API key from https://www.cricapi.com/
   export CRICAPI_KEY="your_cricapi_key_here"
   
   # Optional: AI pitch analysis
   export DEEPSEEK_API_KEY="your_api_key_here"
   
   # Optional: ESPN Cricinfo API (if available)
   export ESPN_CRICINFO_API_KEY="your_espn_key_here"
   ```
   
   **Note**: If `CRICAPI_KEY` is not set, the system uses built-in default squad data.

4. **Set up API Keys (Recommended)**
   ```bash
   # Get free API key from https://www.cricapi.com/
   export CRICAPI_KEY="your_cricapi_key_here"
   
   # Optional: ESPN Cricinfo API key (if available)
   export ESPN_CRICINFO_API_KEY="your_espn_key_here"
   ```
   
   **Note**: If no API keys are provided, the system will use built-in default squad data.

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and go to `http://localhost:5000`


## 📊 Data Sources

### Historical Data
- **Ashes Series**: 2000-2023 match results and statistics
- **Venue Information**: Pitch characteristics and historical performance
- **Player Statistics**: Batting and bowling averages

### Current Squad Data (API-Based)
- **Real-Time Data**: Fetched from CricAPI or ESPN Cricinfo APIs
- **England Squad**: Updated Ashes 2025 squad (16 players) including Crawley, Duckett, Root, Stokes, Wood, Robinson, Bashir, and more
- **Australia Squad**: Updated Ashes 2025 squad (16 players) including Khawaja, Labuschagne, Smith, Head, Cummins, Starc, Lyon, and more
- **Player Roles**: Detailed role classifications (Opener, Middle Order, All-rounder, etc.)
- **Performance Metrics**: Recent form and career statistics from APIs
- **No CSV Dependency**: All team data comes from APIs or built-in defaults with updated 2025 squads

## 🎯 Prediction Methodology

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: Venue, home/away, historical performance, team composition
- **Training Data**: 2000-2023 Ashes series data
- **Validation**: Cross-validation and historical accuracy testing

### Team Selection Logic
- **Performance-Based**: Uses batting/bowling averages and recent form
- **Role-Specific**: Ensures proper team balance (openers, middle-order, bowlers)
- **Venue-Adaptive**: Adjusts selection based on pitch conditions
- **Captain Priority**: Ensures leadership roles are properly assigned

## 🤖 Recommendation System Algorithms

### Algorithm Details

#### 1. Content-Based Filtering
- **Type**: Feature-based similarity matching
- **Features Used**: Batting average, bowling average, player role
- **Similarity Metric**: Direct attribute comparison
- **Use Case**: When user has clear preference (batting/bowling/allrounder)

#### 2. Collaborative Filtering (User-Based)
- **Type**: User-based collaborative filtering
- **Similarity Metric**: Cosine similarity
- **Features**: Multi-dimensional feature vector (batting avg, bowling avg, experience, match wins)
- **Use Case**: Finding players similar to a known good player

#### 3. Hybrid Recommendation System
- **Type**: Weighted combination of multiple approaches
- **Components**: 
  - 60% Content-based (preference + venue + form)
  - 40% Collaborative (experience + match wins)
- **Use Case**: Most comprehensive recommendations considering multiple factors

#### 4. Venue-Based Recommendation
- **Type**: Content-based with venue-specific features
- **Features**: Pace-friendly, spin-friendly, batting-friendly characteristics
- **Use Case**: Optimal team selection for specific venues

#### 5. Item-Based Collaborative Filtering
- **Type**: Item-item similarity
- **Similarity Metric**: Euclidean distance (converted to similarity)
- **Features**: 5-dimensional feature space
- **Use Case**: Finding players with similar overall profiles

#### 6. Popularity-Based Recommendation
- **Type**: Ranking-based recommendation
- **Metrics**: Match wins, player of match awards, experience, performance
- **Use Case**: Recommending proven, successful players

### Recommendation System Evaluation
- **Diversity**: Different algorithms provide varied recommendations
- **Accuracy**: Based on historical performance data
- **Coverage**: Covers all player types and roles
- **Scalability**: Efficient algorithms suitable for real-time recommendations


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **Cricket Australia** and **England Cricket Board** for historical data
- **DeepSeek** for AI-powered pitch analysis
- **Vercel** for hosting and deployment
- **Bootstrap** for the beautiful UI framework


**Note**: This is a prediction system based on historical data and AI analysis. Actual match results may vary due to various factors including player form, weather conditions, and other unpredictable elements. 
