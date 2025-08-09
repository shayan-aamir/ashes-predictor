#!/usr/bin/env python3
"""
AI module for getting pitch and ground conditions
"""

import os
import requests
import json

def get_pitch_conditions_with_ai(venue, api_key=None):
    """
    Get pitch conditions for a specific venue using AI
    """
    
    # Default pitch conditions if no API key
    default_conditions = {
        "Perth": {
            "pitch_type": "Fast and bouncy",
            "pace_friendly": True,
            "spin_friendly": False,
            "batting_friendly": "Moderate",
            "description": "Perth's WACA ground is known for its fast, bouncy pitch that favors pace bowlers. The ball comes onto the bat well but can be challenging for batters due to extra bounce."
        },
        "Brisbane": {
            "pitch_type": "Good batting surface",
            "pace_friendly": True,
            "spin_friendly": False,
            "batting_friendly": "High",
            "description": "Brisbane's Gabba traditionally offers a good batting surface with consistent bounce. Day-night conditions can add extra swing for fast bowlers."
        },
        "Adelaide": {
            "pitch_type": "Balanced",
            "pace_friendly": True,
            "spin_friendly": True,
            "batting_friendly": "High",
            "description": "Adelaide Oval provides a balanced pitch that offers something for both batters and bowlers. Good for stroke play but also assists spinners as the match progresses."
        },
        "Melbourne": {
            "pitch_type": "Batting-friendly",
            "pace_friendly": True,
            "spin_friendly": False,
            "batting_friendly": "Very High",
            "description": "Melbourne Cricket Ground typically offers a flat, batting-friendly surface. High-scoring matches are common here."
        },
        "Sydney": {
            "pitch_type": "Spin-friendly",
            "pace_friendly": False,
            "spin_friendly": True,
            "batting_friendly": "Moderate",
            "description": "Sydney Cricket Ground traditionally assists spinners more than other Australian venues. The pitch can deteriorate and offer turn as the match progresses."
        }
    }
    
    if not api_key:
        return default_conditions.get(venue, {
            "pitch_type": "Standard",
            "pace_friendly": True,
            "spin_friendly": False,
            "batting_friendly": "Moderate",
            "description": "Standard pitch conditions"
        })
    
    # Use AI API to get enhanced pitch conditions
    try:
        prompt = f"""
        Provide detailed pitch and ground conditions for {venue} cricket ground for Test matches. Include:
        1. Pitch type and characteristics
        2. Whether it favors pace or spin bowling
        3. Batting friendliness
        4. Specific conditions that affect team selection
        5. Historical performance patterns
        
        Format as JSON with keys: pitch_type, pace_friendly, spin_friendly, batting_friendly, description
        """
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse JSON from response
            try:
                # Extract JSON from the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    ai_conditions = json.loads(json_str)
                    return ai_conditions
            except:
                pass
            
            # If JSON parsing fails, return default with AI description
            default = default_conditions.get(venue, default_conditions["Perth"])
            default["description"] = content
            return default
            
    except Exception as e:
        print(f"Error getting AI pitch conditions: {e}")
    
    # Fallback to default conditions
    return default_conditions.get(venue, default_conditions["Perth"])

def get_all_venue_conditions(api_key=None):
    """
    Get pitch conditions for all Ashes venues
    """
    venues = ["Perth", "Brisbane", "Adelaide", "Melbourne", "Sydney"]
    conditions = {}
    
    for venue in venues:
        conditions[venue] = get_pitch_conditions_with_ai(venue, api_key)
    
    return conditions 