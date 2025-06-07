🧠 Prophet Food Demand Forecasting Dashboard
A comprehensive web-based forecasting system using Facebook Prophet for regional food demand analysis.
✨ Features

🗺️ Regional Analysis: Automatic center-to-region mapping and regional demand forecasting
📊 Smart Data Availability: Analyzes which region/meal combinations have sufficient data
🎯 Interactive Selection: Click-to-select viable forecasting options
📈 Prophet Integration: Advanced time series forecasting with seasonality detection
📋 Performance Metrics: Comprehensive model evaluation (Accuracy, MAE, RMSE, R², MAPE)
🔮 Custom Forecasting: Configurable forecast periods and confidence intervals
📊 Real-time Visualization: Training performance charts with actual vs predicted data

🚀 Quick Start
Prerequisites
bash: pip install -r requirements.txt
Required Data
Place your train.csv file in the same directory. The file should contain:

center_id: Food center identifier
meal_id: Meal category identifier
week: Week number
num_orders: Number of orders (target variable)

Running the Application
bashpython server.py
Open your browser to http://localhost:8000

📊 How It Works
1. Data Processing

Regional Mapping: Automatically divides centers into 4 balanced regions
Data Aggregation: Aggregates demand by week, region, and meal category
Availability Analysis: Identifies combinations with sufficient data (30+ weeks)

2. Model Training

Facebook Prophet: Handles seasonality, trends, and changepoints automatically
Flexible Filtering: Train models for specific regions, meals, or overall demand
Performance Evaluation: Uses train/test split for honest performance assessment

3. Forecasting

Custom Periods: Forecast 1-52 weeks ahead
Confidence Intervals: Adjustable confidence levels (50-99%)
Business Insights: Actionable recommendations for inventory and planning

📈 Dashboard Features
Smart Selection Interface

Green Cards: Show viable region/meal combinations with sufficient data
Real-time Feedback: ✅/❌ indicators show data availability as you select
Auto-suggestions: Intelligent recommendations when insufficient data

Performance Visualization

Training Performance: Actual vs predicted on historical data
Forecast Charts: Future predictions with confidence bands
Model Quality Radar: Multi-dimensional performance assessment

Business Intelligence

Regional Insights: Compare performance across different regions
Demand Patterns: Identify seasonal trends and growth opportunities
Planning Recommendations: Inventory, staffing, and capacity suggestions

📁 File Structure
├── forecaster.py      # Core Prophet forecasting engine
├── server.py          # Web server and API endpoints
├── dashboard.html     # Interactive web interface
├── requirements.txt   # Python dependencies
├── train.csv         # Training data (your data file)
└── README.md         # This file

🎯 API Endpoints

/api/train - Train Prophet models
/api/forecast - Generate forecasts
/api/training_performance - Get training vs actual data
/api/data_availability - Get viable option analysis
/api/evaluate - Model performance metrics
/api/summary - System status and data info

🔧 Configuration
Prophet Parameters
Models use optimized settings for food demand:

Yearly seasonality: Enabled
Weekly seasonality: Enabled
Seasonality mode: Multiplicative
Changepoint prior scale: 0.05

Data Requirements

Minimum 30 weeks of data per combination
Consistent week numbering
Non-negative demand values

📊 Performance Metrics

Accuracy: 100 - MAPE (higher is better)
MAE: Mean Absolute Error
RMSE: Root Mean Square Error
R²: Coefficient of determination
MAPE: Mean Absolute Percentage Error
Bias: Average prediction bias

🎯 Use Cases

Inventory Planning: Optimize stock levels by region
Capacity Management: Plan staffing and kitchen capacity
Financial Forecasting: Budget for regional operations
Marketing Strategy: Target high-demand periods and regions
Expansion Planning: Identify underserved regions

🛠️ Troubleshooting
"Insufficient Data" Errors

Use the data availability analysis to see viable options
Try broader combinations (all regions or all meals)
Ensure your data has consistent week coverage

Performance Issues

Check for data quality issues (missing weeks, outliers)
Consider longer training periods for better seasonality detection
Verify regional mapping makes business sense

🤝 Contributing
Feel free to submit issues and enhancement requests!

📄 License
This project is licensed under the MIT License.