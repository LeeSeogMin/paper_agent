"""
AI Paper Writing System - Web Interface
Flask application for the AI Paper Writing System
"""

import os
import sys
from pathlib import Path
import datetime

# Add the utils directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import json
import uuid
from werkzeug.utils import secure_filename
import traceback
import requests  # HTTP 요청을 위한 라이브러리 추가

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from utils.logger import logger, configure_logging
from config.settings import OUTPUT_DIR
from utils.xai_client import XAIClient  # utils 폴더에서 가져오기

# XAI API 키 설정
XAI_API_KEY = os.environ.get('XAI_API_KEY')
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY 환경 변수가 설정되지 않았습니다. 설정 후 재시작 해주세요.")

# API 키가 'xai-' 접두사로 시작하는지 확인
if not XAI_API_KEY.startswith('xai-'):
    logger.warning("XAI_API_KEY가 'xai-' 접두사로 시작하지 않습니다. API 키 형식을 확인하세요.")

# 글로벌 XAI 클라이언트 초기화 - 모든 에이전트가 사용할 수 있도록 싱글톤 패턴 활용
try:
    # 싱글톤 인스턴스 초기화
    xai_client = XAIClient.get_instance()
    logger.info("XAI Client initialized successfully")
except Exception as e:
    logger.error(f"XAI Client 초기화 오류: {str(e)}")
    logger.error(traceback.format_exc())

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_key_for_paper_agent')

# Configure upload settings
UPLOAD_FOLDER = os.path.join(project_root, 'data', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize coordinator agent (with error handling)
coordinator_agent = None
try:
    from agents.coordinator_agent import CoordinatorAgent
    
    # CoordinatorAgent의 실제 지원하는 파라미터만 사용하여 초기화
    coordinator_agent = CoordinatorAgent(
        verbose=False
    )
    
    logger.info("CoordinatorAgent initialized successfully")
except Exception as e:
    error_msg = f"Error initializing CoordinatorAgent: {str(e)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    # We'll continue without the coordinator agent and handle it in the routes

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    # Check if coordinator agent is available
    if coordinator_agent is None:
        flash('시스템이 현재 점검 중입니다. 나중에 다시 시도해 주세요.', 'warning')
    return render_template('index.html', coordinator_agent=coordinator_agent)

@app.route('/generate_paper', methods=['POST'])
def generate_paper():
    """Handle paper generation request"""
    # Check if coordinator agent is available
    if coordinator_agent is None:
        flash('논문 생성 기능이 현재 사용할 수 없습니다. 시스템이 점검 중입니다.', 'error')
        return redirect(url_for('index'))
    
    # Get form data
    research_topic = request.form.get('research_topic', '')
    paper_type = request.form.get('paper_type', 'Literature Review')
    paper_format = request.form.get('paper_format', 'standard')
    additional_instructions = request.form.get('additional_instructions', '')
    
    # Check if a file was uploaded
    uploaded_file = None
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_file = file_path
    
    # Validate input
    if not research_topic:
        flash('연구 주제를 입력해 주세요', 'error')
        return redirect(url_for('index'))
    
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Create a task for the coordinator agent
    task = {
        'topic': research_topic,
        'paper_type': paper_type,
        'paper_format': paper_format,
        'additional_instructions': additional_instructions,
        'status': 'pending',
        'progress': 0,
        'result': None,
        'error': None,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Store task in session
    session['current_task'] = task
    
    try:
        # Start the paper generation workflow
        workflow_state = coordinator_agent.start_workflow(
            topic=research_topic,
            paper_type=paper_type,
            paper_format=paper_format,
            additional_instructions=additional_instructions,
            uploaded_file=uploaded_file
        )
        
        # Store workflow state ID in session
        session['workflow_state_id'] = request_id
        
        # Redirect to results page
        return redirect(url_for('results', request_id=request_id))
    
    except Exception as e:
        logger.error(f"Error generating paper: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"논문 생성 중 오류가 발생했습니다: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/results/<request_id>')
def results(request_id):
    """Show results page for a specific request"""
    # Check if coordinator agent is available
    if coordinator_agent is None:
        flash('결과 조회 기능이 현재 사용할 수 없습니다. 시스템이 점검 중입니다.', 'error')
        return redirect(url_for('index'))
    
    # Get workflow state ID from session
    workflow_state_id = session.get('workflow_state_id')
    
    if not workflow_state_id:
        flash('워크플로우 상태를 찾을 수 없습니다', 'error')
        return redirect(url_for('index'))
    
    # Get workflow status
    try:
        status = coordinator_agent.get_workflow_status()
        
        # Get paper summary if available
        paper_summary = coordinator_agent.get_paper_summary()
        
        return render_template(
            'results.html',
            request_id=request_id,
            status=status,
            paper_summary=paper_summary,
            coordinator_agent=coordinator_agent
        )
    
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f"결과 조회 중 오류가 발생했습니다: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/api/status/<request_id>')
def api_status(request_id):
    """API endpoint to get the status of a paper generation request"""
    # Check if coordinator agent is available
    if coordinator_agent is None:
        return jsonify({'error': '시스템이 점검 중입니다'}), 503
    
    try:
        # Get workflow status
        status = coordinator_agent.get_workflow_status()
        
        # Convert to dict if it's a Pydantic model
        if hasattr(status, 'dict'):
            return jsonify(status.dict())
        else:
            # If it's already a dict, return it directly
            return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/<request_id>')
def download_paper(request_id):
    """Download the generated paper"""
    # Check if coordinator agent is available
    if coordinator_agent is None:
        flash('다운로드 기능이 현재 사용할 수 없습니다. 시스템이 점검 중입니다.', 'error')
        return redirect(url_for('index'))
    
    # This would typically serve the generated paper file
    # For now, we'll just redirect to the results page
    flash('논문 다운로드 기능이 아직 구현되지 않았습니다', 'info')
    return redirect(url_for('results', request_id=request_id))

if __name__ == '__main__':
    # Configure logging
    configure_logging()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=True) 