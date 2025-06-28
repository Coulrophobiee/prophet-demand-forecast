#!/usr/bin/env python3
"""
Enhanced Web Server for Prophet Food Demand Forecasting Dashboard
Production version with business cost analysis functionality
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
        elif parsed_path.path == '/api/model_comparison':
            self.handle_model_comparison(parsed_path.query)
        elif parsed_path.path == '/api/business_cost_comparison':
            self.handle_business_cost_comparison(parsed_path.query)
        elif parsed_path.path == '/api/business_cost_analysis':
            self.handle_business_cost_analysis(parsed_path.query)
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
        """Handle model training API endpoint"""
        params = parse_qs(query)
        region_id = params.get('region', [None])[0]
        meal_id = params.get('meal', [None])[0]
        
        model, message = forecaster.train_prophet_model(region_id, meal_id)
        actual_model_key = forecaster._normalize_model_key(region_id, meal_id)
        
        if model:
            try:
                metrics, eval_message = forecaster.evaluate_model_performance(actual_model_key)
                
                if metrics:
                    response = {
                        'success': True,
                        'message': message,
                        'evaluation': eval_message,
                        'metrics': metrics,
                        'model_key': actual_model_key
                    }
                else:
                    response = {
                        'success': True,
                        'message': message,
                        'evaluation': f"Model trained successfully, but evaluation failed: {eval_message}",
                        'metrics': None,
                        'model_key': actual_model_key
                    }
                    
            except Exception as e:
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
        
        self.send_json_response(response)
    
    def handle_forecast(self, query):
        """Handle forecast generation API endpoint with automatic business cost analysis"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        weeks = int(params.get('weeks', [10])[0])
        confidence = float(params.get('confidence', [95])[0]) / 100
        
        forecast, message = forecaster.generate_forecast(model_key, weeks, confidence)
        
        if forecast is not None:
            model_info = None
            for key, info in forecaster.models.items():
                if key == model_key or key.replace('meal_', 'all_') == model_key or key.replace('all_', 'meal_') == model_key:
                    model_info = info
                    break
            
            meal_id = model_info['meal_id'] if model_info else None
            meal_name = forecaster.get_meal_display_name(meal_id) if meal_id else None
            
            forecast_data = {
                'weeks': list(range(1, weeks + 1)),
                'predicted': forecast['yhat'].round().astype(int).tolist(),
                'upper': forecast['yhat_upper'].round().astype(int).tolist(),
                'lower': forecast['yhat_lower'].round().astype(int).tolist(),
                'trend': forecast['trend'].round().astype(int).tolist(),
                'confidence_level': confidence * 100,
                'meal_id': meal_id,
                'meal_name': meal_name
            }
            
            # Generate business cost analysis
            business_cost_analysis = None
            try:
                business_comparison, cost_message = forecaster.compare_business_costs(model_key)
                if business_comparison:
                    business_cost_analysis = business_comparison
            except Exception as e:
                print("Error generating business cost analysis:", e)
            
            response = {
                'success': True,
                'message': message,
                'forecast': forecast_data,
                'model_used': model_key,
                'business_cost_analysis': business_cost_analysis
            }
            
        else:
            response = {
                'success': False,
                'message': message
            }
        
        self.send_json_response(response)
    
    def handle_training_performance(self, query):
        """Handle training performance data request"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        performance_data = forecaster.get_training_performance(model_key)
        
        if performance_data:
            response = {
                'success': True,
                'message': f'Training performance data retrieved for {model_key}',
                'performance': performance_data
            }
        else:
            response = {
                'success': False,
                'message': f'No training performance data available for model {model_key}'
            }
        
        self.send_json_response(response)
    
    def handle_model_comparison(self, query):
        """Handle model comparison API endpoint"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        try:
            comparison_data, message = forecaster.evaluate_baseline_vs_prophet(model_key)
            
            if comparison_data:
                response = {
                    'success': True,
                    'message': message,
                    'comparison': comparison_data
                }
            else:
                response = {
                    'success': False,
                    'message': message
                }
                
        except Exception as e:
            response = {
                'success': False,
                'message': f"Error generating model comparison: {str(e)}"
            }
        
        self.send_json_response(response)
    
    def handle_business_cost_comparison(self, query):
        """Handle business cost comparison API endpoint"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        try:
            business_comparison, message = forecaster.compare_business_costs(model_key)
            
            if business_comparison:
                response = {
                    'success': True,
                    'message': message,
                    'business_comparison': business_comparison
                }
            else:
                response = {
                    'success': False,
                    'message': message
                }
                
        except Exception as e:
            print("Business cost comparison error:", e)
            response = {
                'success': False,
                'message': f"Error generating business cost comparison: {str(e)}"
            }
        
        self.send_json_response(response)
    
    def handle_business_cost_analysis(self, query):
        """Handle business cost analysis API endpoint"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        meal_price = params.get('meal_price', [None])[0]
        
        if meal_price:
            try:
                meal_price = float(meal_price)
            except ValueError:
                meal_price = None
        
        try:
            business_costs, message = forecaster.evaluate_business_cost(model_key, meal_price)
            
            if business_costs:
                response = {
                    'success': True,
                    'message': message,
                    'business_costs': business_costs
                }
            else:
                response = {
                    'success': False,
                    'message': message
                }
                
        except Exception as e:
            print("Business cost analysis error:", e)
            response = {
                'success': False,
                'message': f"Error generating business cost analysis: {str(e)}"
            }
        
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
        
        metrics, message = forecaster.evaluate_model_performance(model_key)
        
        response = {
            'success': metrics is not None,
            'message': message,
            'metrics': metrics
        }
        
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
        if not os.path.exists('train.csv'):
            print("ERROR: train.csv not found in current directory")
            print("Please download train.csv from the Kaggle dataset")
            return
            
        if not os.path.exists('dashboard.html'):
            print("ERROR: dashboard.html not found in current directory")
            print("Please ensure dashboard.html is in the same directory")
            return
        
        if not forecaster.load_data():
            print("ERROR: Failed to load data")
            return
        
        with socketserver.TCPServer(("", port), ForecastingHTTPRequestHandler) as httpd:
            print("Prophet Food Forecasting Dashboard Started!")
            print(f"Open your browser and go to: http://localhost:{port}")
            print("Features: Regional analysis, Prophet training, model evaluation")
            print("Business cost analysis in Euro terms")
            print("Press Ctrl+C to stop the server")
            print("-" * 60)
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 98:
            print(f"ERROR: Port {port} is already in use. Trying port {port + 1}...")
            start_server(port + 1)
        else:
            print(f"ERROR: Error starting server: {e}")
    except KeyboardInterrupt:
        print("\nServer stopped by user")

if __name__ == "__main__":
    print("Facebook Prophet Food Demand Forecasting")
    print("Regional Analysis Version with Business Cost Analysis")
    print("=" * 60)
    print("=" * 60)
    start_server()