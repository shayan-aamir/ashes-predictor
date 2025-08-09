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

### 🏏 Match Schedule (2025-2026)
1. **1st Test**: Perth (November 22-26, 2025)
2. **2nd Test**: Brisbane (December 4-8, 2025)
3. **3rd Test**: Adelaide (December 12-16, 2025)
4. **4th Test**: Melbourne (December 26-30, 2025)
5. **5th Test**: Sydney (January 3-7, 2026)

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
- **Environment Variables**: Secure API key management

### Deployment
- **Vercel**: Cloud hosting platform
- **Gunicorn**: WSGI server for production

## 📁 Project Structure

```
ASHES_PREDICTOR/
├── app.py                      # Main Flask application
├── prediction_model.py         # ML model for predictions
├── csv_team_selector.py        # Team selection logic
├── pitch_conditions_ai.py      # AI pitch analysis
├── latest_teams_data.csv       # Current player statistics
├── ashes_data.csv             # Historical Ashes data
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

3. **Set up DeepSeek API (Optional)**
   ```bash
   # Set environment variable for AI pitch analysis
   export DEEPSEEK_API_KEY="your_api_key_here"
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:5000`

### Vercel Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Vercel**
   - Connect your GitHub repository to Vercel
   - Vercel will automatically detect the Flask app
   - Set environment variables in Vercel dashboard if using DeepSeek API

## 📊 Data Sources

### Historical Data
- **Ashes Series**: 2000-2023 match results and statistics
- **Venue Information**: Pitch characteristics and historical performance
- **Player Statistics**: Batting and bowling averages

### Current Squad Data
- **England Squad**: Latest players from ENG vs IND series
- **Australia Squad**: Latest players from AUS vs WI series
- **Player Roles**: Detailed role classifications (Opener, Middle Order, All-rounder, etc.)
- **Performance Metrics**: Recent form and career statistics

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

## 🌐 Live Demo

The application is deployed and accessible at: [Your Vercel URL]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Cricket Australia** and **England Cricket Board** for historical data
- **DeepSeek** for AI-powered pitch analysis
- **Vercel** for hosting and deployment
- **Bootstrap** for the beautiful UI framework

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This is a prediction system based on historical data and AI analysis. Actual match results may vary due to various factors including player form, weather conditions, and other unpredictable elements. 