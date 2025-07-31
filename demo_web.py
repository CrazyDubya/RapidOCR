#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Interactive Web Demo for RapidOCR
==================================

A Flask-based web interface that provides an interactive demonstration
of RapidOCR capabilities with real-time processing and visualization.

Features:
- File upload interface
- Real-time OCR processing
- Interactive result visualization
- Multiple output formats
- Performance metrics
- Batch processing interface

Author: Enhanced Demo for RapidOCR
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
from PIL import Image

try:
    from flask import Flask, render_template, request, jsonify, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Import our comprehensive demo
import sys
sys.path.append(str(Path(__file__).parent))
from demo_comprehensive import RapidOCRDemo, MockRapidOCR


class RapidOCRWebDemo:
    """Web-based demo interface for RapidOCR"""
    
    def __init__(self, use_mock=False):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for web demo. Install with: pip install flask")
            
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
        self.demo = RapidOCRDemo(use_mock=use_mock)
        self.setup_routes()
        
        # Create upload directory
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route("/")
        def index():
            return render_template("index.html")
        
        @self.app.route("/api/ocr", methods=["POST"])
        def api_ocr():
            try:
                if "file" not in request.files:
                    return jsonify({"error": "No file uploaded"}), 400
                
                file = request.files["file"]
                if file.filename == "":
                    return jsonify({"error": "Empty filename"}), 400
                
                # Save uploaded file
                filename = f"upload_{int(time.time())}_{file.filename}"
                file_path = self.upload_dir / filename
                file.save(str(file_path))
                
                # Process with OCR
                start_time = time.time()
                result = self.demo.engine(str(file_path))
                processing_time = time.time() - start_time
                
                # Convert result to JSON
                ocr_results = []
                if result and result.txts:
                    for i, txt in enumerate(result.txts):
                        ocr_result = {
                            "text": txt,
                            "confidence": result.scores[i] if result.scores and i < len(result.scores) else 0.5
                        }
                        if result.boxes and i < len(result.boxes):
                            ocr_result["box"] = result.boxes[i]
                        ocr_results.append(ocr_result)
                
                # Generate visualization
                vis_img = result.vis() if result else None
                vis_base64 = None
                if vis_img is not None:
                    _, buffer = cv2.imencode('.png', vis_img)
                    vis_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Clean up uploaded file
                file_path.unlink(missing_ok=True)
                
                return jsonify({
                    "success": True,
                    "processing_time": processing_time,
                    "results": ocr_results,
                    "visualization": vis_base64,
                    "summary": {
                        "text_count": len(ocr_results),
                        "total_confidence": sum(r.get("confidence", 0) for r in ocr_results),
                        "avg_confidence": sum(r.get("confidence", 0) for r in ocr_results) / len(ocr_results) if ocr_results else 0
                    }
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/batch", methods=["POST"])
        def api_batch():
            try:
                files = request.files.getlist("files")
                if not files:
                    return jsonify({"error": "No files uploaded"}), 400
                
                results = []
                total_start_time = time.time()
                
                for file in files[:10]:  # Limit to 10 files
                    if file.filename == "":
                        continue
                        
                    # Save file
                    filename = f"batch_{int(time.time())}_{file.filename}"
                    file_path = self.upload_dir / filename
                    file.save(str(file_path))
                    
                    # Process
                    start_time = time.time()
                    try:
                        result = self.demo.engine(str(file_path))
                        processing_time = time.time() - start_time
                        
                        file_result = {
                            "filename": file.filename,
                            "processing_time": processing_time,
                            "text_count": len(result) if result else 0,
                            "texts": result.txts if result and result.txts else [],
                            "success": True
                        }
                    except Exception as e:
                        file_result = {
                            "filename": file.filename,
                            "error": str(e),
                            "success": False
                        }
                    
                    results.append(file_result)
                    
                    # Clean up
                    file_path.unlink(missing_ok=True)
                
                total_time = time.time() - total_start_time
                
                return jsonify({
                    "success": True,
                    "total_time": total_time,
                    "file_count": len(results),
                    "successful_count": len([r for r in results if r.get("success", False)]),
                    "results": results
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/info")
        def api_info():
            return jsonify({
                "engine_type": "Mock" if self.demo.use_mock else "RapidOCR",
                "capabilities": [
                    "Text Detection",
                    "Text Recognition", 
                    "Multi-language Support",
                    "Batch Processing",
                    "Visualization",
                    "Multiple Output Formats"
                ],
                "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
                "max_file_size": "10MB",
                "max_batch_size": 10
            })
    
    def create_html_template(self):
        """Create HTML template for the web interface"""
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RapidOCR Interactive Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 30px;
        }
        
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
        }
        
        .tab {
            padding: 15px 25px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 50px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        
        .upload-icon {
            font-size: 4em;
            color: #ddd;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 15px;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .result-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .confidence {
            display: inline-block;
            padding: 3px 8px;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 12px;
            font-size: 0.9em;
            margin-left: 10px;
        }
        
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        
        .visualization img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .loading {
            text-align: center;
            padding: 50px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .batch-results {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #c62828;
        }
        
        .success {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2e7d32;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 RapidOCR Interactive Demo</h1>
            <p>Experience the power of optical character recognition with real-time processing and visualization</p>
        </div>
        
        <div class="main-content">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('single')">Single Image OCR</div>
                <div class="tab" onclick="switchTab('batch')">Batch Processing</div>
                <div class="tab" onclick="switchTab('info')">Information</div>
            </div>
            
            <!-- Single Image Tab -->
            <div id="single" class="tab-content active">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Click here or drag and drop an image file</div>
                    <div style="font-size: 0.9em; color: #999;">Supported formats: JPG, PNG, BMP, TIFF</div>
                </div>
                <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="processFile(this.files[0])">
                
                <div id="singleLoading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <div>Processing image with OCR...</div>
                </div>
                
                <div id="singleResults" class="results" style="display: none;">
                    <!-- Results will be populated here -->
                </div>
            </div>
            
            <!-- Batch Processing Tab -->
            <div id="batch" class="tab-content">
                <div class="upload-area" onclick="document.getElementById('batchFileInput').click()">
                    <div class="upload-icon">📚</div>
                    <div class="upload-text">Select multiple images for batch processing</div>
                    <div style="font-size: 0.9em; color: #999;">Maximum 10 files, up to 10MB each</div>
                </div>
                <input type="file" id="batchFileInput" class="file-input" multiple accept="image/*" onchange="processBatch(this.files)">
                
                <div id="batchLoading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <div>Processing batch...</div>
                </div>
                
                <div id="batchResults" class="results" style="display: none;">
                    <!-- Batch results will be populated here -->
                </div>
            </div>
            
            <!-- Information Tab -->
            <div id="info" class="tab-content">
                <div id="infoContent">
                    <!-- Info will be loaded here -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        function switchTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Load info if info tab is selected
            if (tabName === 'info') {
                loadInfo();
            }
        }
        
        // Drag and drop functionality
        function setupDragDrop() {
            const uploadAreas = document.querySelectorAll('.upload-area');
            
            uploadAreas.forEach(area => {
                area.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    area.classList.add('dragover');
                });
                
                area.addEventListener('dragleave', () => {
                    area.classList.remove('dragover');
                });
                
                area.addEventListener('drop', (e) => {
                    e.preventDefault();
                    area.classList.remove('dragover');
                    
                    const files = e.dataTransfer.files;
                    if (area.onclick.toString().includes('fileInput')) {
                        if (files.length > 0) {
                            processFile(files[0]);
                        }
                    } else {
                        processBatch(files);
                    }
                });
            });
        }
        
        // Process single file
        function processFile(file) {
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('singleLoading').style.display = 'block';
            document.getElementById('singleResults').style.display = 'none';
            
            fetch('/api/ocr', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('singleLoading').style.display = 'none';
                displaySingleResults(data);
            })
            .catch(error => {
                document.getElementById('singleLoading').style.display = 'none';
                displayError('Error processing file: ' + error.message);
            });
        }
        
        // Display single file results
        function displaySingleResults(data) {
            const resultsDiv = document.getElementById('singleResults');
            
            if (data.error) {
                resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                resultsDiv.style.display = 'block';
                return;
            }
            
            let html = `
                <div class="success">✅ OCR processing completed successfully!</div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">${data.summary.text_count}</div>
                        <div class="stat-label">Text Regions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${data.processing_time.toFixed(3)}s</div>
                        <div class="stat-label">Processing Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${(data.summary.avg_confidence * 100).toFixed(1)}%</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                </div>
            `;
            
            if (data.visualization) {
                html += `
                    <div class="visualization">
                        <h3>🖼️ Visualization</h3>
                        <img src="data:image/png;base64,${data.visualization}" alt="OCR Visualization">
                    </div>
                `;
            }
            
            if (data.results && data.results.length > 0) {
                html += '<h3>📝 Detected Text</h3>';
                data.results.forEach((result, index) => {
                    html += `
                        <div class="result-item">
                            <strong>Text ${index + 1}:</strong> ${result.text}
                            <span class="confidence">${(result.confidence * 100).toFixed(1)}% confidence</span>
                        </div>
                    `;
                });
            } else {
                html += '<div class="result-item">No text detected in the image.</div>';
            }
            
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
        
        // Process batch files
        function processBatch(files) {
            if (files.length === 0) return;
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            
            document.getElementById('batchLoading').style.display = 'block';
            document.getElementById('batchResults').style.display = 'none';
            
            fetch('/api/batch', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('batchLoading').style.display = 'none';
                displayBatchResults(data);
            })
            .catch(error => {
                document.getElementById('batchLoading').style.display = 'none';
                displayError('Error processing batch: ' + error.message);
            });
        }
        
        // Display batch results
        function displayBatchResults(data) {
            const resultsDiv = document.getElementById('batchResults');
            
            if (data.error) {
                resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                resultsDiv.style.display = 'block';
                return;
            }
            
            let html = `
                <div class="success">✅ Batch processing completed!</div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">${data.file_count}</div>
                        <div class="stat-label">Total Files</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${data.successful_count}</div>
                        <div class="stat-label">Successful</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${data.total_time.toFixed(2)}s</div>
                        <div class="stat-label">Total Time</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${(data.total_time / data.file_count).toFixed(2)}s</div>
                        <div class="stat-label">Avg Time/File</div>
                    </div>
                </div>
                
                <h3>📋 Individual Results</h3>
                <div class="batch-results">
            `;
            
            data.results.forEach((result, index) => {
                if (result.success) {
                    html += `
                        <div class="result-item">
                            <strong>📄 ${result.filename}</strong>
                            <div>⏱ Processing time: ${result.processing_time.toFixed(3)}s</div>
                            <div>📝 Text regions: ${result.text_count}</div>
                            ${result.texts.length > 0 ? 
                                '<div>📋 Texts: ' + result.texts.join(', ') + '</div>' : 
                                '<div>No text detected</div>'
                            }
                        </div>
                    `;
                } else {
                    html += `
                        <div class="result-item error">
                            <strong>📄 ${result.filename}</strong>
                            <div>❌ Error: ${result.error}</div>
                        </div>
                    `;
                }
            });
            
            html += '</div>';
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
        
        // Load system information
        function loadInfo() {
            fetch('/api/info')
            .then(response => response.json())
            .then(data => {
                const infoDiv = document.getElementById('infoContent');
                
                let html = `
                    <h2>ℹ️ System Information</h2>
                    
                    <div class="result-item">
                        <h3>🔧 Engine Type</h3>
                        <p>${data.engine_type}</p>
                    </div>
                    
                    <div class="result-item">
                        <h3>✨ Capabilities</h3>
                        <ul>
                `;
                
                data.capabilities.forEach(cap => {
                    html += `<li>✓ ${cap}</li>`;
                });
                
                html += `
                        </ul>
                    </div>
                    
                    <div class="result-item">
                        <h3>📁 Supported Formats</h3>
                        <p>${data.supported_formats.join(', ')}</p>
                    </div>
                    
                    <div class="result-item">
                        <h3>⚙️ Limitations</h3>
                        <ul>
                            <li>Maximum file size: ${data.max_file_size}</li>
                            <li>Maximum batch size: ${data.max_batch_size} files</li>
                        </ul>
                    </div>
                    
                    <div class="result-item">
                        <h3>📚 About This Demo</h3>
                        <p>This interactive web demo showcases the capabilities of RapidOCR, a fast and accurate OCR engine. 
                        Upload images to see real-time text detection and recognition in action!</p>
                    </div>
                `;
                
                infoDiv.innerHTML = html;
            })
            .catch(error => {
                document.getElementById('infoContent').innerHTML = 
                    `<div class="error">Error loading info: ${error.message}</div>`;
            });
        }
        
        // Display error
        function displayError(message) {
            const activeTab = document.querySelector('.tab-content.active');
            const resultsDiv = activeTab.querySelector('.results') || activeTab;
            
            resultsDiv.innerHTML = `<div class="error">${message}</div>`;
            resultsDiv.style.display = 'block';
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            setupDragDrop();
            loadInfo();
        });
    </script>
</body>
</html>
        """
        
        template_path = template_dir / "index.html"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML template created: {template_path}")
    
    def run(self, host="127.0.0.1", port=5000, debug=True):
        """Run the web demo"""
        self.create_html_template()
        
        print(f"🚀 Starting RapidOCR Web Demo...")
        print(f"🌐 Server running at: http://{host}:{port}")
        print(f"🔧 Engine: {'Mock' if self.demo.use_mock else 'RapidOCR'}")
        print("📝 Open the URL in your browser to access the interactive demo")
        
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function for web demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RapidOCR Interactive Web Demo")
    parser.add_argument("--mock", action="store_true", help="Use mock engine")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    
    args = parser.parse_args()
    
    if not FLASK_AVAILABLE:
        print("❌ Flask is not available. Install with: pip install flask")
        return 1
    
    try:
        web_demo = RapidOCRWebDemo(use_mock=args.mock)
        web_demo.run(
            host=args.host, 
            port=args.port, 
            debug=not args.no_debug
        )
    except KeyboardInterrupt:
        print("\n⚠ Web demo stopped by user")
    except Exception as e:
        print(f"❌ Web demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())