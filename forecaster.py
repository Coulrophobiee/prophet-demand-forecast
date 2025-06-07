#!/usr/bin/env python3
"""
Food Demand Forecaster - Prophet Model Implementation
Enhanced version with data availability analysis and smart filtering
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from prophet import Prophet # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
import warnings
import os

warnings.filterwarnings('ignore')

class FoodDemandForecaster:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        self.training_performance = {}
        self.regional_mapping = {}
        self.column_mapping = {}
        self.train_aggregated = None
        self.weekly_demand = None
        self.data_availability = {}  # Track data availability
        
    def load_data(self):
        """Load train and test datasets"""
        try:
            if os.path.exists('train.csv'):
                self.train_data = pd.read_csv('train.csv')
                print(f"✅ Loaded train.csv: {len(self.train_data)} records")
                print(f"📋 Train columns: {list(self.train_data.columns)}")
            else:
                print("❌ train.csv not found in current directory")
                return False
                
            # OPTIONAL: test.csv is not required for forecasting
            if os.path.exists('test.csv'):
                self.test_data = pd.read_csv('test.csv')
                print(f"✅ Loaded test.csv: {len(self.test_data)} records")
                print(f"📋 Test columns: {list(self.test_data.columns)}")
            else:
                self.test_data = None
                print("ℹ️ test.csv not found - continuing without it (not required for forecasting)")
                
            self.preprocess_data()
            self.analyze_data_availability()  # Analyze what combinations work
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def detect_column_names(self):
        """Detect the correct column names from the data"""
        if self.train_data is None:
            return None
        
        columns = self.train_data.columns.tolist()
        
        # Common variations for columns
        target_candidates = ['num_orders', 'orders', 'demand', 'quantity', 'qty', 'sales']
        week_candidates = ['week', 'time', 'period', 'date']
        center_candidates = ['center_id', 'center', 'location_id', 'location', 'store_id', 'store']
        meal_candidates = ['meal_id', 'meal', 'product_id', 'product', 'item_id', 'item']
        
        detected = {}
        
        # Find columns by matching candidates
        for col in columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in target_candidates):
                detected['target'] = col
            elif any(candidate in col_lower for candidate in week_candidates):
                detected['week'] = col
            elif any(candidate in col_lower for candidate in center_candidates):
                detected['center'] = col
            elif any(candidate in col_lower for candidate in meal_candidates):
                detected['meal'] = col
        
        print(f"🔍 Detected columns: {detected}")
        return detected

    def create_regional_mapping(self):
        """Create regional mapping for centers"""
        if self.train_data is None:
            return {}
        
        # Get all unique centers
        centers = sorted(self.train_data['center_id'].unique())
        total_centers = len(centers)
        
        print(f"📍 Found {total_centers} centers: {centers}")
        
        # Divide centers into 4 regions
        centers_per_region = total_centers // 4
        remainder = total_centers % 4
        
        regional_mapping = {}
        current_index = 0
        
        # Create 4 regions
        for region_num in range(1, 5):
            region_name = f"Region_{region_num}"
            region_size = centers_per_region + (1 if region_num <= remainder else 0)
            region_centers = centers[current_index:current_index + region_size]
            
            for center in region_centers:
                regional_mapping[center] = region_name
            
            current_index += region_size
            print(f"🗺️ {region_name}: Centers {region_centers} ({len(region_centers)} centers)")
        
        return regional_mapping

    def preprocess_data(self):
        """Preprocess data for Prophet model"""
        if self.train_data is None:
            return
        
        # Create regional mapping
        self.regional_mapping = self.create_regional_mapping()
        
        # Add region column
        self.train_data['region'] = self.train_data['center_id'].map(self.regional_mapping)
        if self.test_data is not None:
            self.test_data['region'] = self.test_data['center_id'].map(self.regional_mapping)
        
        # Detect column names
        self.column_mapping = self.detect_column_names()
        
        if not self.column_mapping or 'target' not in self.column_mapping:
            print("❌ Could not detect target column")
            return
        
        target_col = self.column_mapping['target']
        week_col = self.column_mapping.get('week', 'week')
        meal_col = self.column_mapping.get('meal', 'meal_id')
        
        print(f"📊 Using columns: target={target_col}, week={week_col}, meal={meal_col}")
        
        try:
            # Aggregate weekly demand by region and meal
            group_cols = [week_col, 'region']
            if meal_col in self.train_data.columns:
                group_cols.append(meal_col)
            
            self.train_aggregated = self.train_data.groupby(group_cols).agg({
                target_col: 'sum'
            }).reset_index()
            
            # Rename columns to standard names
            rename_dict = {target_col: 'num_orders'}
            if week_col != 'week':
                rename_dict[week_col] = 'week'
            if meal_col != 'meal_id' and meal_col in self.train_aggregated.columns:
                rename_dict[meal_col] = 'meal_id'
            
            self.train_aggregated = self.train_aggregated.rename(columns=rename_dict)
            
            # Create overall weekly aggregation
            self.weekly_demand = self.train_data.groupby(week_col)[target_col].sum().reset_index()
            self.weekly_demand = self.weekly_demand.rename(columns={week_col: 'week', target_col: 'num_orders'})
            self.weekly_demand['ds'] = pd.to_datetime('2010-01-01') + pd.to_timedelta(self.weekly_demand['week'] * 7, unit='D')
            self.weekly_demand['y'] = self.weekly_demand['num_orders']
            
            # Log actual week range
            min_week = self.weekly_demand['week'].min()
            max_week = self.weekly_demand['week'].max()
            total_weeks = len(self.weekly_demand)
            
            print(f"📊 Data preprocessed: {total_weeks} weeks of data")
            print(f"📅 Week range: {min_week} - {max_week}")
            print(f"🗺️ Regional aggregation complete with {len(self.regional_mapping)} centers in 4 regions")
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")

    def analyze_data_availability(self):
        """Analyze data availability for different region/meal combinations"""
        print("\n🔍 Analyzing data availability for different combinations...")
        
        if self.train_aggregated is None:
            return
        
        availability = {}
        min_weeks_required = 30  # Minimum weeks needed for Prophet
        
        # Check overall
        overall_weeks = len(self.weekly_demand)
        availability['overall'] = {
            'weeks': overall_weeks,
            'sufficient': overall_weeks >= min_weeks_required,
            'avg_demand': self.weekly_demand['num_orders'].mean(),
            'description': f"All regions, all meals ({overall_weeks} weeks)"
        }
        
        # Check each region (all meals)
        for region in ['Region_1', 'Region_2', 'Region_3', 'Region_4']:
            region_data = self.train_aggregated[self.train_aggregated['region'] == region]
            if len(region_data) > 0:
                region_weekly = region_data.groupby('week')['num_orders'].sum().reset_index()
                weeks_count = len(region_weekly)
                availability[f"{region}_all"] = {
                    'weeks': weeks_count,
                    'sufficient': weeks_count >= min_weeks_required,
                    'avg_demand': region_weekly['num_orders'].mean() if weeks_count > 0 else 0,
                    'description': f"{region}, all meals ({weeks_count} weeks)"
                }
        
        # Check each meal (all regions)
        if 'meal_id' in self.train_aggregated.columns:
            for meal in sorted(self.train_aggregated['meal_id'].unique()):
                meal_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal]
                if len(meal_data) > 0:
                    meal_weekly = meal_data.groupby('week')['num_orders'].sum().reset_index()
                    weeks_count = len(meal_weekly)
                    availability[f"all_{meal}"] = {
                        'weeks': weeks_count,
                        'sufficient': weeks_count >= min_weeks_required,
                        'avg_demand': meal_weekly['num_orders'].mean() if weeks_count > 0 else 0,
                        'description': f"All regions, meal {meal} ({weeks_count} weeks)"
                    }
        
        # Check specific region + meal combinations (sample the most promising ones)
        promising_combinations = []
        if 'meal_id' in self.train_aggregated.columns:
            for region in ['Region_1', 'Region_2', 'Region_3', 'Region_4']:
                for meal in sorted(self.train_aggregated['meal_id'].unique()):
                    combo_data = self.train_aggregated[
                        (self.train_aggregated['region'] == region) & 
                        (self.train_aggregated['meal_id'] == meal)
                    ]
                    if len(combo_data) > 0:
                        combo_weekly = combo_data.groupby('week')['num_orders'].sum().reset_index()
                        weeks_count = len(combo_weekly)
                        if weeks_count >= min_weeks_required:
                            promising_combinations.append({
                                'key': f"{region}_{meal}",
                                'weeks': weeks_count,
                                'avg_demand': combo_weekly['num_orders'].mean(),
                                'description': f"{region}, meal {meal} ({weeks_count} weeks)"
                            })
        
        # Add top 5 promising combinations to availability
        promising_combinations.sort(key=lambda x: x['avg_demand'], reverse=True)
        for combo in promising_combinations[:5]:
            availability[combo['key']] = {
                'weeks': combo['weeks'],
                'sufficient': True,
                'avg_demand': combo['avg_demand'],
                'description': combo['description']
            }
        
        self.data_availability = availability
        
        # Print summary
        print(f"\n📊 Data Availability Summary:")
        print(f"{'Key':<20} {'Sufficient':<12} {'Weeks':<8} {'Avg Demand':<12} {'Description'}")
        print("-" * 80)
        
        for key, info in availability.items():
            sufficient_icon = "✅" if info['sufficient'] else "❌"
            print(f"{key:<20} {sufficient_icon:<12} {info['weeks']:<8} {info['avg_demand']:<12.0f} {info['description']}")
        
        # Show recommendations
        viable_options = [k for k, v in availability.items() if v['sufficient']]
        print(f"\n✅ Viable forecasting options: {len(viable_options)}")
        for option in viable_options[:10]:  # Show top 10
            print(f"   • {option}: {availability[option]['description']}")
    
    def get_data_availability(self):
        """Get data availability information for the frontend"""
        return self.data_availability
    
    def train_prophet_model(self, region_id=None, meal_id=None):
        """Train Prophet model for specific region/meal combination or overall"""
        try:
            if region_id:
                region_id = str(region_id)  # Ensure string
            if meal_id:
                # Try both string and int versions
                try:
                    meal_id_int = int(meal_id)
                except:
                    meal_id_int = meal_id
                meal_id = meal_id_int
            
            # Create model key
            if region_id and meal_id:
                key = f"{region_id}_{meal_id}"
                print(f"🎯 Training for Region: {region_id}, Meal: {meal_id}")
                
                # DEBUG: Check data filtering step by step
                print(f"🔍 Debug: Starting with {len(self.train_aggregated)} total aggregated records")
                
                region_filtered = self.train_aggregated[self.train_aggregated['region'] == region_id]
                print(f"🔍 Debug: After region filter ({region_id}): {len(region_filtered)} records")
                
                if 'meal_id' in self.train_aggregated.columns:
                    # Try both string and int versions of meal_id
                    meal_filtered = region_filtered[region_filtered['meal_id'] == meal_id]
                    if len(meal_filtered) == 0 and isinstance(meal_id, int):
                        # Try string version
                        meal_filtered = region_filtered[region_filtered['meal_id'] == str(meal_id)]
                        print(f"🔍 Debug: Tried string version of meal_id")
                    elif len(meal_filtered) == 0 and isinstance(meal_id, str):
                        # Try int version
                        try:
                            meal_filtered = region_filtered[region_filtered['meal_id'] == int(meal_id)]
                            print(f"🔍 Debug: Tried int version of meal_id")
                        except:
                            pass
                    
                    print(f"🔍 Debug: After meal filter ({meal_id}): {len(meal_filtered)} records")
                    filtered_data = meal_filtered
                else:
                    print(f"🔍 Debug: No meal_id column found")
                    filtered_data = region_filtered
                
                print(f"🔍 Debug: Final filtered data: {len(filtered_data)} records")
                if len(filtered_data) > 0:
                    print(f"🔍 Debug: Sample weeks: {sorted(filtered_data['week'].unique())[:10]}")
                
                demand_data = filtered_data.groupby('week')['num_orders'].sum().reset_index()
                
                # Show which centers are in this region
                region_centers = [center for center, region in self.regional_mapping.items() if region == region_id]
                print(f"📍 {region_id} includes centers: {region_centers}")
                
            elif region_id:
                key = f"{region_id}_all"
                print(f"🎯 Training for Region: {region_id} (all meals)")
                
                # DEBUG: Check region filtering
                print(f"🔍 Debug: Starting with {len(self.train_aggregated)} total aggregated records")
                filtered_data = self.train_aggregated[self.train_aggregated['region'] == region_id]
                print(f"🔍 Debug: After region filter ({region_id}): {len(filtered_data)} records")
                
                if len(filtered_data) == 0:
                    print(f"🔍 Debug: Available regions in data: {sorted(self.train_aggregated['region'].unique())}")
                    print(f"🔍 Debug: Looking for region: '{region_id}' (type: {type(region_id)})")
                    print(f"🔍 Debug: Region types in data: {[type(r) for r in self.train_aggregated['region'].unique()[:3]]}")
                
                if len(filtered_data) > 0:
                    print(f"🔍 Debug: Sample weeks: {sorted(filtered_data['week'].unique())[:10]}")
                
                demand_data = filtered_data.groupby('week')['num_orders'].sum().reset_index()
                
                region_centers = [center for center, region in self.regional_mapping.items() if region == region_id]
                print(f"📍 {region_id} includes centers: {region_centers}")
                
            elif meal_id:
                key = f"all_{meal_id}"
                print(f"🎯 Training for all regions, Meal: {meal_id}")
                
                # DEBUG: Check meal filtering
                print(f"🔍 Debug: Starting with {len(self.train_aggregated)} total aggregated records")
                if 'meal_id' in self.train_aggregated.columns:
                    filtered_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]
                    print(f"🔍 Debug: After meal filter ({meal_id}): {len(filtered_data)} records")
                    
                    if len(filtered_data) == 0:
                        print(f"🔍 Debug: Available meals: {sorted(self.train_aggregated['meal_id'].unique())[:10]}")
                        print(f"🔍 Debug: Looking for meal: '{meal_id}' (type: {type(meal_id)})")
                        print(f"🔍 Debug: Meal types in data: {[type(m) for m in self.train_aggregated['meal_id'].unique()[:3]]}")
                else:
                    print(f"🔍 Debug: No meal_id column found!")
                    filtered_data = pd.DataFrame()  # Empty dataframe
                
                demand_data = filtered_data.groupby('week')['num_orders'].sum().reset_index()
            else:
                key = "overall"
                demand_data = self.weekly_demand.copy()
                print(f"🎯 Training overall model (all regions, all meals)")
            
            print(f"🔍 Debug: Final demand_data shape: {demand_data.shape}")
            print(f"🔍 Debug: Final demand_data columns: {list(demand_data.columns)}")
            
            # ENHANCED: Better insufficient data check with helpful message
            min_weeks_required = 30
            if len(demand_data) < min_weeks_required:
                # Provide helpful suggestion
                available_weeks = len(demand_data)
                
                # DEBUG: Show what went wrong
                print(f"❌ Debug: Insufficient data - only {available_weeks} weeks found")
                print(f"🔍 Debug: Expected at least {min_weeks_required} weeks")
                
                # Check if it's a data type issue
                if region_id:
                    print(f"🔍 Debug: Looking for region '{region_id}' (type: {type(region_id)})")
                    unique_regions = self.train_aggregated['region'].unique()
                    print(f"🔍 Debug: Available regions: {unique_regions} (types: {[type(r) for r in unique_regions]})")
                
                if meal_id:
                    print(f"🔍 Debug: Looking for meal '{meal_id}' (type: {type(meal_id)})")
                    if 'meal_id' in self.train_aggregated.columns:
                        unique_meals = self.train_aggregated['meal_id'].unique()
                        print(f"🔍 Debug: Available meals: {sorted(unique_meals)[:10]} (types: {[type(m) for m in unique_meals[:3]]})")
                
                suggestion = self._get_alternative_suggestion(region_id, meal_id, available_weeks)
                return None, f"Insufficient data: only {available_weeks} weeks available (need {min_weeks_required}+). {suggestion}"
            
            # Log the actual week range being used
            min_week = demand_data['week'].min()
            max_week = demand_data['week'].max()
            total_weeks = len(demand_data)
            print(f"📊 Training data: {total_weeks} weeks (weeks {min_week}-{max_week})")
            print(f"📈 Demand range: {demand_data['num_orders'].min()}-{demand_data['num_orders'].max()}")
            
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime('2010-01-01') + pd.to_timedelta(demand_data['week'] * 7, unit='D'),
                'y': demand_data['num_orders']
            })
            
            # Create and train Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            self.models[key] = model
            
            # Generate predictions on training data for performance visualization
            self._generate_training_performance(key, model, prophet_data, demand_data)
            
            return model, f"Model trained successfully for {key}"
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error training model: {str(e)}"
    
    def _get_alternative_suggestion(self, region_id, meal_id, available_weeks):
        """Provide helpful suggestions when data is insufficient"""
        suggestions = []
        
        if region_id and meal_id:
            # Try region-only
            region_key = f"{region_id}_all"
            if region_key in self.data_availability and self.data_availability[region_key]['sufficient']:
                suggestions.append(f"Try '{region_id}' with all meals")
            
            # Try meal-only
            meal_key = f"all_{meal_id}"
            if meal_key in self.data_availability and self.data_availability[meal_key]['sufficient']:
                suggestions.append(f"Try 'All Regions' with meal {meal_id}")
        
        elif region_id:
            # Try overall
            if self.data_availability.get('overall', {}).get('sufficient', False):
                suggestions.append("Try 'All Regions, All Meals'")
        
        elif meal_id:
            # Try overall
            if self.data_availability.get('overall', {}).get('sufficient', False):
                suggestions.append("Try 'All Regions, All Meals'")
        
        if suggestions:
            return "Suggestions: " + " or ".join(suggestions)
        else:
            return "Try 'All Regions, All Meals' for maximum data coverage"
    
    def _generate_training_performance(self, model_key, model, training_data, original_demand_data):
        """Generate predictions on training data for performance visualization"""
        try:
            # Make predictions on the training data
            predictions = model.predict(training_data[['ds']])
            
            # Use original week numbers and take last 20 for visualization
            actual_weeks = original_demand_data['week'].values
            actual_orders = original_demand_data['num_orders'].values
            predicted_orders = predictions['yhat'].values
            
            # Take last 20 weeks for cleaner visualization
            display_count = min(20, len(actual_weeks))
            start_idx = len(actual_weeks) - display_count
            
            # Store actual vs predicted data for visualization
            self.training_performance[model_key] = {
                'weeks': [f"Week {int(w)}" for w in actual_weeks[start_idx:]],
                'actual': actual_orders[start_idx:].round().astype(int).tolist(),
                'predicted': predicted_orders[start_idx:].round().astype(int).tolist(),
                'upper': predictions['yhat_upper'].iloc[start_idx:].round().astype(int).tolist(),
                'lower': predictions['yhat_lower'].iloc[start_idx:].round().astype(int).tolist(),
                'week_numbers': actual_weeks[start_idx:].tolist(),
                'total_weeks_available': len(actual_weeks),
                'week_range': f"{int(actual_weeks.min())}-{int(actual_weeks.max())}"
            }
            
            print(f"📊 Training performance data generated for {model_key}")
            print(f"📅 Full dataset: weeks {int(actual_weeks.min())}-{int(actual_weeks.max())} ({len(actual_weeks)} weeks)")
            print(f"📈 Visualization: last {display_count} weeks ({int(actual_weeks[start_idx])}-{int(actual_weeks[-1])})")
            
        except Exception as e:
            print(f"❌ Error generating training performance data: {e}")
    
    def get_training_performance(self, model_key):
        """Get training performance data for visualization"""
        return self.training_performance.get(model_key, None)
    
    def generate_forecast(self, model_key, weeks_ahead=10, confidence_level=0.95):
        """Generate forecast using trained Prophet model with custom confidence level"""
        try:
            if model_key not in self.models:
                return None, "Model not found"
            
            model = self.models[model_key]
            
            # Create future dataframe
            last_date = model.history['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=weeks_ahead,
                freq='W'
            )
            
            future = pd.DataFrame({'ds': future_dates})
            
            # Generate forecast with custom confidence interval
            forecast = model.predict(future)
            
            # Calculate custom confidence intervals based on user input
            if confidence_level != 0.8:  # Prophet's default is 80%
                print(f"📊 Adjusting confidence intervals to {confidence_level*100}%")
                
                # Get the standard Prophet uncertainty (difference from yhat to upper bound)
                default_uncertainty = forecast['yhat_upper'] - forecast['yhat']
                
                # Calculate scaling factor for different confidence levels
                # Prophet uses 80% by default, so we scale from that
                from scipy import stats # type: ignore
                default_z = stats.norm.ppf(0.9)  # 80% CI uses z-score for 90th percentile
                custom_z = stats.norm.ppf((1 + confidence_level) / 2)  # User's confidence level
                scaling_factor = custom_z / default_z
                
                print(f"📈 Scaling factor: {scaling_factor:.3f}")
                
                # Apply custom confidence intervals
                custom_uncertainty = default_uncertainty * scaling_factor
                forecast['yhat_upper'] = forecast['yhat'] + custom_uncertainty
                forecast['yhat_lower'] = forecast['yhat'] - custom_uncertainty
                
                # Ensure lower bound is not negative (orders can't be negative)
                forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
            
            # Store forecast
            self.forecasts[model_key] = {
                'forecast': forecast,
                'model_key': model_key,
                'weeks_ahead': weeks_ahead,
                'confidence_level': confidence_level
            }
            
            return forecast, f"Forecast generated successfully with {confidence_level*100}% confidence intervals"
            
        except Exception as e:
            print(f"❌ Forecast error: {e}")
            return None, f"Error generating forecast: {str(e)}"
    
    def evaluate_model_performance(self, model_key):
        """Evaluate Prophet model performance using train data split"""
        try:
            if model_key not in self.models:
                return None, "Model not found"
            
            model = self.models[model_key]
            target_col = self.column_mapping.get('target', 'num_orders')
            week_col = self.column_mapping.get('week', 'week')
            
            print(f"🔍 Evaluating model: {model_key}")
            print(f"⚠️ Using train data split for evaluation (test.csv has no target column)")
            
            # Get the correct filtered data based on model type
            if model_key == "overall":
                # Use the full weekly demand data
                train_weekly = self.weekly_demand[['week', 'num_orders']].copy()
                print(f"📊 Overall model: using full dataset")
            else:
                parts = model_key.split('_')
                region_id = parts[0] if parts[0] not in ['all', 'overall'] else None
                meal_id = parts[1] if len(parts) > 1 and parts[1] != 'all' else None
                
                print(f"🎯 Filtering for region: {region_id}, meal: {meal_id}")
                
                # Start with the aggregated data
                if region_id and meal_id:
                    filtered_data = self.train_aggregated[
                        (self.train_aggregated['region'] == region_id) & 
                        (self.train_aggregated['meal_id'] == meal_id)
                    ]
                elif region_id:
                    filtered_data = self.train_aggregated[self.train_aggregated['region'] == region_id]
                elif meal_id:
                    filtered_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]
                else:
                    filtered_data = self.train_aggregated
                
                # Aggregate by week
                train_weekly = filtered_data.groupby('week')['num_orders'].sum().reset_index()
                
                if region_id:
                    region_centers = [center for center, region in self.regional_mapping.items() if region == region_id]
                    print(f"📍 Region {region_id} includes centers: {region_centers}")
            
            # Use last 20% of available data for evaluation
            total_weeks = len(train_weekly)
            eval_size = max(5, int(total_weeks * 0.2))
            eval_data = train_weekly.tail(eval_size).copy()
            
            if len(eval_data) == 0:
                return None, "No evaluation data available"
            
            # Log the actual evaluation period
            eval_min_week = eval_data['week'].min()
            eval_max_week = eval_data['week'].max()
            print(f"📊 Total available weeks: {total_weeks} (weeks {train_weekly['week'].min()}-{train_weekly['week'].max()})")
            print(f"📈 Using {len(eval_data)} weeks for evaluation (weeks {eval_min_week}-{eval_max_week})")
            
            # Create evaluation dates
            eval_data_prophet = pd.DataFrame({
                'ds': pd.to_datetime('2010-01-01') + pd.to_timedelta(eval_data['week'] * 7, unit='D'),
                'y': eval_data['num_orders']
            })
            
            # Make predictions
            predictions = model.predict(eval_data_prophet[['ds']])
            
            # Calculate metrics
            y_true = eval_data_prophet['y'].values
            y_pred = np.maximum(predictions['yhat'].values, 0)  # Handle negative predictions
            
            print(f"📊 Predictions range: {y_pred.min():.1f} - {y_pred.max():.1f}")
            print(f"📊 Actual range: {y_true.min()} - {y_true.max()}")
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape_values = np.abs((y_true - y_pred) / np.maximum(y_true, 1))
            mape = np.mean(mape_values) * 100
            r2 = r2_score(y_true, y_pred)
            bias = np.mean(y_pred - y_true)
            accuracy = max(0, 100 - mape)
            
            metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape),
                'bias': float(bias),
                'accuracy': float(accuracy),
                'test_samples': len(y_true),
                'predictions_positive': int(np.sum(y_pred > 0)),
                'total_weeks_used': total_weeks,
                'evaluation_weeks': f"{eval_min_week}-{eval_max_week}"
            }
            
            print(f"✅ Metrics calculated:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            
            self.performance_metrics[model_key] = metrics
            
            return metrics, f"Model evaluation completed using weeks {eval_min_week}-{eval_max_week}"
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error evaluating model: {str(e)}"
    
    def get_model_summary(self):
        """Get summary of all trained models and their performance"""
        summary = {
            'models_trained': len(self.models),
            'models': {},
            'data_info': {
                'train_records': len(self.train_data) if self.train_data is not None else 0,
                'test_records': len(self.test_data) if self.test_data is not None else 0,
                'regions': ['Region_1', 'Region_2', 'Region_3', 'Region_4'],
                'meals': list(self.train_data['meal_id'].unique()) if self.train_data is not None else [],
                'weeks_range': [
                    int(self.train_data['week'].min()) if self.train_data is not None else 0,
                    int(self.train_data['week'].max()) if self.train_data is not None else 0
                ]
            },
            'data_availability': self.data_availability  # Include availability info
        }
        
        for key in self.models.keys():
            summary['models'][key] = {
                'performance': self.performance_metrics.get(key, {}),
                'has_forecast': key in self.forecasts
            }
        
        return summary