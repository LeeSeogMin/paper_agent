# AI Paper Writing System - Web Interface

This directory contains the Flask web application for the AI Paper Writing System.

## Directory Structure

- `app.py`: Main Flask application file
- `static/`: Static files (CSS, JavaScript)
- `templates/`: HTML templates

## Running the Web Interface

To run the web interface, execute the following command from the project root:

```bash
python run_web.py
```

The web interface will be available at http://localhost:8080

## Features

- Upload requirements files (txt, md, json)
- Specify research topic
- Select paper type
- Add additional instructions
- Track paper generation progress
- Download generated papers

## Requirements

- Flask
- Werkzeug
- Bootstrap (loaded from CDN)
- Bootstrap Icons (loaded from CDN)

## Development

To modify the web interface:

1. Edit HTML templates in the `templates/` directory
2. Edit CSS styles in the `static/style.css` file
3. Edit JavaScript functionality in the `static/script.js` file

## Integration with AI Paper Writing System

The web interface integrates with the AI Paper Writing System through the `CoordinatorAgent` class, which orchestrates the paper generation process. 