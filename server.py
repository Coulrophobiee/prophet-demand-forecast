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
        elif parsed_path.path == '/api/business_cost_comparison':  # NEW
            self.handle_business_cost_comparison(parsed_path.query)
        elif parsed_path.path == '/api/business_cost_analysis':    # NEW
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
        """Handle forecast generation API endpoint with automatic business cost analysis"""
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
            # Get the actual model info to extract meal_id and meal_name
            model_info = None
            for key, info in forecaster.models.items():
                if key == model_key or key.replace('meal_', 'all_') == model_key or key.replace('all_', 'meal_') == model_key:
                    model_info = info
                    break
            
            meal_id = model_info['meal_id'] if model_info else None
            meal_name = forecaster.get_meal_display_name(meal_id) if meal_id else None
            
            # Convert forecast to JSON-serializable format
            forecast_data = {
                'weeks': list(range(1, weeks + 1)),
                'predicted': forecast['yhat'].round().astype(int).tolist(),
                'upper': forecast['yhat_upper'].round().astype(int).tolist(),
                'lower': forecast['yhat_lower'].round().astype(int).tolist(),
                'trend': forecast['trend'].round().astype(int).tolist(),
                'confidence_level': confidence * 100,
                'meal_id': meal_id,
                'meal_name': meal_name  # Add meal name to response
            }
            
            # **NEW: Automatically generate business cost analysis**
            business_cost_analysis = None
            try:
                print(f"💰 Automatically generating business cost analysis for forecast...")
                business_comparison, cost_message = forecaster.compare_business_costs(model_key)
                
                if business_comparison:
                    business_cost_analysis = business_comparison
                    print(f"✅ Business cost analysis included in forecast response")
                    print(f"💰 Weekly savings: €{business_comparison['avg_weekly_savings_euro']:.2f}")
                    print(f"📅 Annual savings: €{business_comparison['annual_savings_estimate_euro']:.2f}")
                else:
                    print(f"⚠️ Business cost analysis failed: {cost_message}")
                    
            except Exception as e:
                print(f"❌ Error generating business cost analysis: {e}")
            
            response = {
                'success': True,
                'message': message,
                'forecast': forecast_data,
                'model_used': model_key,
                'business_cost_analysis': business_cost_analysis  # **NEW: Include business cost analysis**
            }
            print(f"✅ Forecast generated successfully for {meal_name or f'Meal {meal_id}'}")
            
            if business_cost_analysis:
                print(f"💰 Business cost analysis included with forecast")
            
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
    
    def handle_model_comparison(self, query):
        """Handle model comparison API endpoint"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        print(f"🥊 Model comparison request for: '{model_key}'")
        print(f"🔍 Available models: {list(forecaster.models.keys())}")
        
        try:
            # Generate comparison data
            comparison_data, message = forecaster.evaluate_baseline_vs_prophet(model_key)
            
            if comparison_data:
                response = {
                    'success': True,
                    'message': message,
                    'comparison': comparison_data
                }
                print(f"✅ Model comparison generated successfully")
                print(f"📊 Prophet accuracy: {comparison_data['prophet_metrics']['accuracy']:.1f}%")
                print(f"📊 Baseline accuracy: {comparison_data['baseline_metrics']['accuracy']:.1f}%")
                print(f"📈 Improvement: {comparison_data['accuracy_improvement']:+.1f} percentage points")
            else:
                response = {
                    'success': False,
                    'message': message
                }
                print(f"❌ Model comparison failed: {message}")
                
        except Exception as e:
            print(f"❌ Model comparison error: {e}")
            response = {
                'success': False,
                'message': f"Error generating model comparison: {str(e)}"
            }
        
        self.send_json_response(response)
    
    def handle_business_cost_comparison(self, query):
        """Handle business cost comparison API endpoint - NEW"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        
        print(f"💰 Business cost comparison request for: '{model_key}'")
        print(f"🔍 Available models: {list(forecaster.models.keys())}")
        
        try:
            # Generate business cost comparison
            business_comparison, message = forecaster.compare_business_costs(model_key)
            
            if business_comparison:
                response = {
                    'success': True,
                    'message': message,
                    'business_comparison': business_comparison
                }
                print(f"✅ Business cost comparison generated successfully")
                print(f"💰 Prophet cost: €{business_comparison['prophet_total_penalty']:.2f}")
                print(f"📊 Baseline cost: €{business_comparison['baseline_total_penalty']:.2f}")
                print(f"📈 Savings: €{business_comparison['cost_savings_euro']:.2f}")
                print(f"🎯 Annual impact: €{business_comparison['annual_savings_estimate_euro']:.2f}")
            else:
                response = {
                    'success': False,
                    'message': message
                }
                print(f"❌ Business cost comparison failed: {message}")
                
        except Exception as e:
            print(f"❌ Business cost comparison error: {e}")
            import traceback
            traceback.print_exc()
            response = {
                'success': False,
                'message': f"Error generating business cost comparison: {str(e)}"
            }
        
        self.send_json_response(response)
    
    def handle_business_cost_analysis(self, query):
        """Handle business cost analysis API endpoint - NEW"""
        params = parse_qs(query)
        model_key = params.get('model', ['overall'])[0]
        meal_price = params.get('meal_price', [None])[0]
        
        if meal_price:
            try:
                meal_price = float(meal_price)
            except ValueError:
                meal_price = None
        
        print(f"💰 Business cost analysis request for: '{model_key}'")
        if meal_price:
            print(f"💵 Custom meal price: €{meal_price:.2f}")
        
        try:
            # Generate business cost analysis
            business_costs, message = forecaster.evaluate_business_cost(model_key, meal_price)
            
            if business_costs:
                response = {
                    'success': True,
                    'message': message,
                    'business_costs': business_costs
                }
                print(f"✅ Business cost analysis generated successfully")
                print(f"💰 Total penalty: €{business_costs['total_penalty_euro']:.2f}")
                print(f"📊 Penalty as % of revenue: {business_costs['penalty_percentage_of_revenue']:.2f}%")
            else:
                response = {
                    'success': False,
                    'message': message
                }
                print(f"❌ Business cost analysis failed: {message}")
                
        except Exception as e:
            print(f"❌ Business cost analysis error: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"🥊 Model comparison: Prophet vs Baseline evaluation")
            print(f"💰 NEW: Business cost analysis in Euro terms")
            print(f"💸 NEW: Food waste vs lost sales cost breakdown")
            print(f"📈 NEW: Weekly/annual savings projections")
            print(f"🍽️ NEW: AI-generated meal database with ingredients")
            print(f"⚖️ NEW: Perishable vs non-perishable cost modeling")
            print(f"📊 NEW: ROI analysis for Prophet implementation")
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
    print("🗺️ Regional Analysis Version with Business Cost Analysis")
    print("=" * 60)
    print("📋 Requirements:")
    print("   - pip install prophet pandas numpy scikit-learn scipy")
    print("   - train.csv and test.csv in current directory")
    print("   - dashboard.html in current directory")
    print("   - forecaster.py in current directory")
    print("=" * 60)
    print("🆕 New Features:")
    print("   - Prophet vs Baseline model comparison")
    print("   - Business cost analysis in Euro terms")
    print("   - Food waste vs lost sales breakdown")
    print("   - Weekly/annual savings projections")
    print("   - Real ROI calculation for forecasting models")
    print("   - AI-generated meal database with ingredients")
    print("   - Perishable vs non-perishable cost modeling")
    print("   - Annual business impact projections")
    print("=" * 60)
    start_server()