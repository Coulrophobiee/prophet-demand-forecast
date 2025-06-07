#!/usr/bin/env python3
"""
Web Server for Prophet Food Demand Forecasting Dashboard
Fixed version with proper model key handling
"""

import http.server
import socketserver
import json
import os
import webbrowser
import threading
import time
from urllib.parse import urlparse, parse_qs
from forecaster import FoodDemandForecaster

# Global forecaster instance
forecaster = FoodDemandForecaster()

class ForecastingHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/' or parsed_path.path == '/index.html':
            self.serve_dashboard()
        elif parsed_path.path == '/api/train':
            self.handle_train_model(parsed_path.query)
        elif parsed_path.path == '/api/forecast':
            self.handle_forecast(parsed_path.query)
        elif parsed_path.path == '/api/evaluate':
            self.handle_evaluate(parsed_path.query)
        elif parsed_path.path == '/api/training_performance':
            self.handle_training_performance(parsed_path.query)
        elif parsed_path.path == '/api/data_availability':
            self.handle_data_availability()
        elif parsed_path.path == '/api/summary':
            self.handle_summary()
        else:
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Read dashboard HTML file
        try:
            with open('dashboard.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.wfile.write(html_content.encode())
        except FileNotFoundError:
            error_html = '''
            <!DOCTYPE html>
            <html>
            <head><title>Error</title></head>
            <body>
                <h1>Dashboard Not Found</h1>
                <p>Please make sure dashboard.html is in the same directory as server.py</p>
            </body>
            </html>
            '''
            self.wfile.write(error_html.encode())
    
    def handle_train_model(self, query):
        """Handle model training API endpoint with proper key handling"""
        params = parse_qs(query)
        region_id = params.get('region', [None])[0]
        meal_id = params.get('meal', [None])[0]
        
        print(f"🎯 API Training request:")
        print(f"📋 Raw parameters: region={region_id}, meal={meal_id}")
        
        # Train the model
        model, message = forecaster.train_prophet_model(region_id, meal_id)
        
        # Get the actual model key from forecaster
        actual_model_key = forecaster._normalize_model_key(region_id, meal_id)
        print(f"🔑 Actual model key: '{actual_model_key}'")
        print(f"🔍 Available models after training: {list(forecaster.models.keys())}")
        
        if model:
            # Try to evaluate the model using the actual key
            try:
                print(f"🔍 Attempting evaluation with key: '{actual_model_key}'")
                metrics, eval_message = forecaster.evaluate_model_performance(actual_model_key)
                
                if metrics:
                    response = {
                        'success': True,
                        'message': message,
                        'evaluation': eval_message,
                        'metrics': metrics,
                        'model_key': actual_model_key
                    }
                    print(f"✅ Training and evaluation successful")
                else:
                    response = {
                        'success': True,
                        'message': message,
                        'evaluation': f"Model trained successfully, but evaluation failed: {eval_message}",
                        'metrics': None,
                        'model_key': actual_model_key
                    }
                    print(f"⚠️ Training successful, evaluation failed")
                    
            except Exception as e:
                print(f"❌ Evaluation error: {e}")
                response = {
                    'success': True,
                    'message': message,
                    'evaluation': f"Model trained successfully, but evaluation failed: {str(e)}",
                    'metrics': None,
                    'model_key': actual_model_key
                }
        else:
            response = {
                'success': False,
                'message': message,
                'metrics': None
            }
            print(f"❌ Training failed: {message}")
        
        self.send_json_response(response)
    
    def handle_forecast(self, query):
        """Handle forecast generation API endpoint"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        weeks = int(params.get('weeks', [10])[0])
        confidence = float(params.get('confidence', [95])[0]) / 100  # Convert % to decimal
        
        print(f"🔍 Forecast request for model: '{model_key}'")
        print(f"📋 Available models: {list(forecaster.models.keys())}")
        print(f"📊 Confidence level: {confidence*100}%")
        
        # Generate forecast with the provided model key
        forecast, message = forecaster.generate_forecast(model_key, weeks, confidence)
        
        if forecast is not None:
            # Get the actual model info to extract meal_id
            model_info = None
            for key, info in forecaster.models.items():
                if key == model_key or key.replace('meal_', 'all_') == model_key or key.replace('all_', 'meal_') == model_key:
                    model_info = info
                    break
            
            # Convert forecast to JSON-serializable format
            forecast_data = {
                'weeks': list(range(1, weeks + 1)),
                'predicted': forecast['yhat'].round().astype(int).tolist(),
                'upper': forecast['yhat_upper'].round().astype(int).tolist(),
                'lower': forecast['yhat_lower'].round().astype(int).tolist(),
                'trend': forecast['trend'].round().astype(int).tolist(),
                'confidence_level': confidence * 100,
                'meal_id': model_info['meal_id'] if model_info else None  # Add meal_id to response
            }
            
            response = {
                'success': True,
                'message': message,
                'forecast': forecast_data,
                'model_used': model_key
            }
            print(f"✅ Forecast generated successfully")
        else:
            response = {
                'success': False,
                'message': message
            }
            print(f"❌ Forecast failed: {message}")
        
        self.send_json_response(response)
    
    def handle_training_performance(self, query):
        """Handle training performance data request"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        print(f"📊 Training performance request for model: '{model_key}'")
        print(f"🔍 Available training performance data: {list(forecaster.training_performance.keys())}")
        
        # Get training performance data
        performance_data = forecaster.get_training_performance(model_key)
        
        if performance_data:
            response = {
                'success': True,
                'message': f'Training performance data retrieved for {model_key}',
                'performance': performance_data
            }
            print(f"✅ Training performance data found")
        else:
            response = {
                'success': False,
                'message': f'No training performance data available for model {model_key}'
            }
            print(f"❌ No training performance data available")
        
        self.send_json_response(response)
    
    def handle_data_availability(self):
        """Handle data availability request"""
        availability = forecaster.get_data_availability()
        
        response = {
            'success': True,
            'message': 'Data availability information retrieved',
            'availability': availability
        }
        
        self.send_json_response(response)
    
    def handle_evaluate(self, query):
        """Handle model evaluation API endpoint"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        print(f"🔍 Evaluation request for model: '{model_key}'")
        
        metrics, message = forecaster.evaluate_model_performance(model_key)
        
        response = {
            'success': metrics is not None,
            'message': message,
            'metrics': metrics
        }
        
        if metrics:
            print(f"✅ Evaluation successful")
        else:
            print(f"❌ Evaluation failed: {message}")
        
        self.send_json_response(response)
    
    def handle_summary(self):
        """Handle summary API endpoint"""
        summary = forecaster.get_model_summary()
        self.send_json_response(summary)
    
    def send_json_response(self, data):
        """Send JSON response with proper headers"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:8000')

def start_server(port=8000):
    """Start the web server"""
    try:
        # Check if data files exist
        if not os.path.exists('train.csv'):
            print("❌ train.csv not found in current directory")
            print("📥 Please download train.csv from the Kaggle dataset")
            return
        
        # test.csv is optional - not required for forecasting
        if not os.path.exists('test.csv'):
            print("ℹ️ test.csv not found - continuing without it (not required for forecasting)")
            
        if not os.path.exists('dashboard.html'):
            print("❌ dashboard.html not found in current directory")
            print("📥 Please ensure dashboard.html is in the same directory")
            return
        
        # Load the data
        if not forecaster.load_data():
            print("❌ Failed to load data")
            return
        
        # Start the server
        with socketserver.TCPServer(("", port), ForecastingHTTPRequestHandler) as httpd:
            print(f"🧠 Prophet Food Forecasting Dashboard Started!")
            print(f"📊 Open your browser and go to: http://localhost:{port}")
            print(f"🗺️ Features: Regional analysis, Prophet training, model evaluation")
            print(f"🎯 Performance metrics: Prophet accuracy, MAE, RMSE, R² scores")
            print(f"🛑 Press Ctrl+C to stop the server")
            print("-" * 60)
            
            # Open browser in a separate thread
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            start_server(port + 1)
        else:
            print(f"❌ Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")

if __name__ == "__main__":
    print("🧠 Facebook Prophet Food Demand Forecasting")
    print("🗺️ Regional Analysis Version with Data Availability Analysis")
    print("=" * 60)
    print("📋 Requirements:")
    print("   - pip install prophet pandas numpy scikit-learn scipy")
    print("   - train.csv and test.csv in current directory")
    print("   - dashboard.html in current directory")
    print("=" * 60)
    start_server()