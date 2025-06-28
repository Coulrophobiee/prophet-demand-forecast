# Prophet Food Demand Forecasting Dashboard

A comprehensive web-based application for food demand forecasting using Facebook Prophet with business cost analysis and warehouse capacity planning.

## Overview

This application provides AI-powered meal demand forecasting for food service operations, enabling accurate inventory planning and cost optimization. Built with Facebook Prophet, it offers superior prediction accuracy compared to traditional forecasting methods while providing detailed business impact analysis.

The system analyzes historical meal order data across multiple regions and generates forecasts with confidence intervals, helping businesses optimize inventory, reduce waste, and improve operational efficiency.

## Features

- Prophet Forecasting: Train meal-specific demand prediction models with adaptive configuration
- Regional Analysis: Forecast demand across different regions with intelligent data filtering
- Model Comparison: Compare Prophet accuracy against baseline ("same week last year") methods
- Financial Impact: Calculate cost differences between forecasting approaches in Euro terms
- Interactive Charts: Visualize forecasts with confidence intervals and trend components
- Business Metrics: MAE, accuracy percentages, and comprehensive cost analysis
- Confidence Intervals: Customizable prediction confidence levels (50-99%)
- Ingredient Cost Modeling: Uses meal database with perishable/non-perishable cost structures
- Warehouse Planning: Optional storage capacity forecasting for logistics optimization

## Key Benefits

- Improved Accuracy: Typically 5-15% improvement over traditional forecasting methods
- Cost Reduction: Optimize inventory costs through better demand prediction
- Waste Minimization: Reduce food waste by avoiding overproduction
- Staff Planning: Accurate preparation time estimates based on forecasted demand
- Regional Insights: Compare performance across different operational regions

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download `train.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting) and place it in the application directory

3. Run the server:
```bash
python server.py
```

4. Open your browser to `http://localhost:8000`

The `meal_database.json` file is included in the repository.

## Usage

### Basic Forecasting Workflow

1. Select a Meal: Choose from the dropdown menu (required)
   - Meals are automatically loaded from your training data
   - Display names are loaded from the included meal database
   
2. Choose Region: Select specific region or leave as "All Regions"
   - Region-Kassel, Region-Luzern, Region-Wien available
   - Region-Lyon excluded due to data quality issues
   
3. Set Parameters: 
   - Forecast weeks (1-52): How many weeks ahead to predict
   - Confidence level (50-99%): Statistical confidence for prediction intervals
   
4. Generate Forecast: Click "Train & Show Forecast + Financial Impact"
   - System automatically trains Prophet model
   - Generates forecasts with confidence intervals
   - Compares against baseline predictions
   - Calculates financial impact analysis

### Understanding the Results

#### Forecast Visualization
- Blue line: Prophet predictions showing expected demand
- Orange area: Confidence interval band showing prediction uncertainty
- Purple dashed line: Underlying trend component
- Interactive tooltips: Hover for detailed values

#### Model Performance Metrics
- Prophet Accuracy: Percentage accuracy on holdout test data
- Baseline Accuracy: "Same week last year" comparison method
- MAE (Mean Absolute Error): Average prediction error in number of orders
- Improvement: How much better Prophet performs vs baseline

#### Financial Impact Analysis
- Weekly Savings/Loss: Average cost difference per week using Prophet vs baseline
- Cost Improvement: Percentage improvement in cost efficiency
- Weeks Tested: Number of weeks used for validation

### Advanced Features

#### Confidence Intervals
Adjust confidence levels to match your risk tolerance:
- 95%: Conservative planning with wider intervals
- 80%: Balanced approach for normal operations  
- 50%: Aggressive planning with tighter intervals

#### Regional Filtering
Compare performance across regions:
- Some meals may only be available in certain regions
- Regional models can provide more accurate local forecasts
- System automatically validates data availability

#### Business Planning Integration
For short-term forecasts (1-3 weeks):
- Ingredient recommendations with safety buffers
- Staff planning with estimated preparation hours
- Budget projections based on forecasted demand
- Action items for operational planning

## Data Source

This application uses the [Food Demand Forecasting dataset from Kaggle](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting). Download the `train.csv` file from this dataset and place it in the application directory.

## Data Format

The required `train.csv` contains:
- `week`: Week number
- `center_id`: Fulfillment center ID
- `meal_id`: Unique meal identifier  
- `num_orders`: Number of orders (target variable)
- `checkout_price`: Meal price (used as regressor)

## How It Works

### Data Processing Pipeline

1. Data Loading: System loads and validates your CSV training data
2. Regional Mapping: Centers are automatically mapped to operational regions
3. Data Filtering: Removes meals with insufficient historical data (<20 weeks)
4. Feature Engineering: Extracts price regressors and seasonal patterns

### Model Training Process

1. Adaptive Configuration: Prophet parameters automatically adjust based on data characteristics
   - High variance data: Multiplicative seasonality with flexible changepoints
   - Medium variance data: Standard additive model with balanced parameters
   - Low variance data: Conservative model with reduced seasonality

2. Regressor Integration: Incorporates checkout price as an external regressor when available
3. Cross-Validation: Uses time-series aware validation (80% training, 20% testing)
4. Performance Evaluation: Calculates robust metrics handling outliers and seasonality

### Forecasting Engine

- Prophet Algorithm: Uses Facebook's advanced time series forecasting
- Seasonality Detection: Automatically identifies weekly and yearly patterns
- Trend Analysis: Detects and projects underlying demand trends
- Uncertainty Quantification: Provides statistically valid confidence intervals

### Business Cost Calculation

The financial analysis considers realistic operational costs:

Overprediction Penalties:
- Storage costs (15% of revenue per excess order)
- Opportunity costs (2% of revenue for tied-up capital)
- Perishable ingredient waste (based on meal composition from database)

Underprediction Penalties:
- Emergency procurement costs (150% of normal ingredient cost)
- Lost sales and reduced customer satisfaction

## Technical Details

### Performance Metrics

The system uses two primary metrics to evaluate forecast quality:

Accuracy (%):
```
Accuracy = 100 - MAPE
where MAPE = (1/n) × Σ|actual - predicted|/actual × 100
```
Higher accuracy percentages indicate better predictions (closer to 100% is better).

Mean Absolute Error (MAE):
```
MAE = (1/n) × Σ|actual - predicted|
```
Lower MAE values indicate better predictions (closer to 0 is better). MAE is expressed in the same units as your data (number of orders).

### Model Performance

Accuracy Expectations:
- 90%+: Excellent performance, suitable for automated planning
- 80-90%: Good performance, reliable for most use cases  
- 70-80%: Fair performance, useful with human oversight
- <70%: Poor performance, investigate data quality issues

Validation Methodology:
- Time-series split maintaining temporal order
- Out-of-sample testing on most recent 20% of data
- Comparison against "same week last year" baseline

## Troubleshooting

### Common Issues

"Meal selection required"
- Solution: Select a specific meal from the dropdown before training

"Insufficient data" error  
- Cause: Selected meal has fewer than 20 weeks of historical data
- Solution: Choose a different meal or remove regional filter

"No data found for meal in region"
- Cause: Meal not available in selected region
- Solution: Try "All Regions" or select different region

Low forecast accuracy
- Check for irregular demand patterns
- Verify data quality and completeness
- Consider seasonal business factors not captured in data

### Performance Optimization

For large datasets:
- Start with regional filtering to reduce computation time
- Process meals individually rather than all at once
- Monitor system memory usage during training

For improved accuracy:
- Ensure at least 52 weeks of training data
- Include price information when available
- Remove meals with erratic or discontinued demand patterns

## Files

- `server.py` - Web server and API endpoints
- `forecaster.py` - Prophet forecasting engine and business logic
- `dashboard.html` - Frontend interface
- `requirements.txt` - Python dependencies
- `train.csv` - Training data (download from Kaggle)
- `meal_database.json` - Meal information with ingredient costs (included)
- `warehouse_forecaster.py` - Optional warehouse capacity forecasting (logistics use case)
- `warehouse_data.json` - Generated warehouse storage requirements (if using warehouse forecaster)

## Warehouse Forecasting (Optional)

The application includes an additional warehouse capacity forecasting feature for logistics planning:

```bash
python warehouse_forecaster.py
```

This generates `warehouse_data.json` with storage requirements across regions, useful for:
- Warehouse capacity planning
- Storage utilization forecasting  
- Regional logistics optimization

The warehouse forecaster uses the same Prophet models to predict storage needs based on meal demand forecasts.

## Dependencies

- prophet
- pandas
- numpy
- scikit-learn
- scipy

## License

This project is released under the MIT License.

What does MIT License mean?
The MIT License is one of the most permissive open-source licenses. It means you can:
- ✅ Use this code for any purpose (personal, commercial, etc.)
- ✅ Modify and distribute the code
- ✅ Include it in proprietary software
- ✅ Sell products that use this code

The only requirement is that you include the original copyright notice and license text if you redistribute the code. There's no warranty - you use the code "as is" at your own risk. This makes it very business-friendly since there are minimal restrictions on how you can use the software.