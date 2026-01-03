import os
import sys
import traceback

# Get the directory where this file is located
here = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to Python path so we can import app
if here not in sys.path:
    sys.path.insert(0, here)

# Import the Flask app with detailed error handling
try:
    from app import app
    # Vercel's @vercel/python automatically wraps WSGI apps
    # Just export the app directly
    handler = app
except Exception as e:
    # If import fails, create a simple error app that shows the actual error
    from flask import Flask
    error_app = Flask(__name__)
    
    error_details = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'traceback': traceback.format_exc(),
        'current_dir': here,
        'python_path': sys.path
    }
    
    @error_app.route('/', defaults={'path': ''})
    @error_app.route('/<path:path>')
    def catch_all(path):
        error_html = f"""
        <html>
        <head><title>Deployment Error</title></head>
        <body style="font-family: monospace; padding: 20px;">
            <h1>Application Failed to Load</h1>
            <h2>Error Type: {error_details['error_type']}</h2>
            <h3>Error Message:</h3>
            <pre>{error_details['error_message']}</pre>
            <h3>Full Traceback:</h3>
            <pre>{error_details['traceback']}</pre>
            <h3>Current Directory:</h3>
            <pre>{error_details['current_dir']}</pre>
            <h3>Python Path:</h3>
            <pre>{chr(10).join(error_details['python_path'])}</pre>
        </body>
        </html>
        """
        return error_html, 500
    
    handler = error_app

