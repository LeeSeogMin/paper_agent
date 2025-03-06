"""
Run script for the AI Paper Writing System web interface
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import Flask app
from web.app import app
from utils.logger import configure_logging

if __name__ == '__main__':
    # Configure logging
    configure_logging()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Determine if we're in development or production mode
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
