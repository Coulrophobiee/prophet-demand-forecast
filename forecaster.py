#!/usr/bin/env python3
"""
Food Demand Forecaster - Prophet Model Implementation
Fixed version with proper model key handling
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
        self.data_availability = {}
        self.meal_price_data = {}
        
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
                
            # test.csv is optional
            if os.path.exists('test.csv'):
                self.test_data = pd.read_csv('test.csv')
                print(f"✅ Loaded test.csv: {len(self.test_data)} records")
                print(f"📋 Test columns: {list(self.test_data.columns)}")
            else:
                self.test_data = None
                print("ℹ️ test.csv not found - continuing without it (not required for forecasting)")
                
            self.preprocess_data()
            self.analyze_data_availability()
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
        checkout_price_candidates = ['checkout_price', 'final_price', 'selling_price', 'price']
        base_price_candidates = ['base_price', 'original_price', 'list_price', 'standard_price']
        
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
            elif any(candidate in col_lower for candidate in checkout_price_candidates):
                detected['checkout_price'] = col
            elif any(candidate in col_lower for candidate in base_price_candidates):
                detected['base_price'] = col
        
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

    def calculate_discount_metrics(self):
        """Calculate price metrics for Prophet regressors - simplified to checkout_price only"""
        checkout_col = self.column_mapping.get('checkout_price')
        
        if checkout_col:
            print(f"💰 Price analysis completed:")
            print(f"   • Using checkout_price as regressor")
            print(f"   • Average checkout price: ${self.train_data[checkout_col].mean():.2f}")
            print(f"   • Price range: ${self.train_data[checkout_col].min():.2f} - ${self.train_data[checkout_col].max():.2f}")
        else:
            print("⚠️ checkout_price column not found - skipping price regression")

    def preprocess_data(self):
        """Preprocess data for Prophet model with checkout_price regressor"""
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
        
        # Calculate price metrics
        self.calculate_discount_metrics()
        
        target_col = self.column_mapping['target']
        week_col = self.column_mapping.get('week', 'week')
        meal_col = self.column_mapping.get('meal', 'meal_id')
        checkout_price_col = self.column_mapping.get('checkout_price')
        
        print(f"📊 Using columns: target={target_col}, week={week_col}, meal={meal_col}")
        print(f"💰 Price regressor: checkout_price={checkout_price_col}")
        
        try:
            # Aggregate weekly demand by region and meal with price information
            agg_dict = {target_col: 'sum'}
            
            # Add price aggregation if available
            if checkout_price_col:
                agg_dict[checkout_price_col] = 'mean'
            
            group_cols = [week_col, 'region', meal_col] if meal_col in self.train_data.columns else [week_col, 'region']
            
            self.train_aggregated = self.train_data.groupby(group_cols).agg(agg_dict).reset_index()
            
            # Rename columns to standard names
            rename_dict = {target_col: 'num_orders'}
            if week_col != 'week':
                rename_dict[week_col] = 'week'
            if meal_col != 'meal_id' and meal_col in self.train_aggregated.columns:
                rename_dict[meal_col] = 'meal_id'
            
            self.train_aggregated = self.train_aggregated.rename(columns=rename_dict)
            
            # Create overall weekly aggregation with price data
            overall_agg_dict = {target_col: 'sum'}
            if checkout_price_col:
                overall_agg_dict[checkout_price_col] = 'mean'
            
            self.weekly_demand = self.train_data.groupby(week_col).agg(overall_agg_dict).reset_index()
            self.weekly_demand = self.weekly_demand.rename(columns={week_col: 'week', target_col: 'num_orders'})
            self.weekly_demand['ds'] = pd.to_datetime('2010-01-01') + pd.to_timedelta(self.weekly_demand['week'] * 7, unit='D')
            self.weekly_demand['y'] = self.weekly_demand['num_orders']
            
            # Store meal price data for future forecasting
            self.calculate_meal_price_stats()
            
            # Log actual week range
            min_week = self.weekly_demand['week'].min()
            max_week = self.weekly_demand['week'].max()
            total_weeks = len(self.weekly_demand)
            
            print(f"📊 Data preprocessed: {total_weeks} weeks of data")
            print(f"📅 Week range: {min_week} - {max_week}")
            print(f"🗺️ Regional aggregation complete with {len(self.regional_mapping)} centers in 4 regions")
            print(f"🍽️ Meal-specific forecasting enabled with checkout_price regressor")
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")

    def calculate_meal_price_stats(self):
        """Calculate price statistics per meal for forecasting - simplified to checkout_price only"""
        if 'meal_id' not in self.train_aggregated.columns:
            return
        
        checkout_col = self.column_mapping.get('checkout_price')
        
        self.meal_price_data = {}
        
        for meal_id in self.train_aggregated['meal_id'].unique():
            meal_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]
            
            price_stats = {'meal_id': meal_id}
            
            if checkout_col and checkout_col in meal_data.columns:
                price_stats['avg_checkout_price'] = meal_data[checkout_col].mean()
                price_stats['recent_checkout_price'] = meal_data[checkout_col].tail(12).mean()
            
            self.meal_price_data[meal_id] = price_stats
        
        print(f"💰 Checkout price statistics calculated for {len(self.meal_price_data)} meals")

    def analyze_data_availability(self):
        """Analyze data availability for meal-specific forecasting"""
        print("\n🔍 Analyzing data availability for meal-specific forecasting...")
        
        if self.train_aggregated is None or 'meal_id' not in self.train_aggregated.columns:
            print("❌ No meal data available for analysis")
            return
        
        availability = {}
        min_weeks_required = 20  # Lower threshold for meal-specific (more realistic)
        
        # Get total weeks for coverage calculation
        overall_weeks = len(self.weekly_demand)
        
        # Check each meal individually (all regions)
        for meal in sorted(self.train_aggregated['meal_id'].unique()):
            meal_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal]
            if len(meal_data) > 0:
                meal_weekly = meal_data.groupby('week')['num_orders'].sum().reset_index()
                weeks_count = len(meal_weekly)
                coverage = weeks_count / overall_weeks
                
                availability[f"meal_{meal}"] = {
                    'weeks': weeks_count,
                    'sufficient': weeks_count >= min_weeks_required,
                    'avg_demand': meal_weekly['num_orders'].mean() if weeks_count > 0 else 0,
                    'description': f"Meal {meal} - All regions ({weeks_count} weeks, {coverage:.1%} coverage)",
                    'type': 'meal',
                    'coverage': coverage
                }
        
        # Check meal + region combinations for top meals
        top_meals = sorted(
            [k for k, v in availability.items() if k.startswith('meal_') and v['sufficient']], 
            key=lambda x: availability[x]['avg_demand'], 
            reverse=True
        )[:5]  # Top 5 meals only
        
        for meal_key in top_meals:
            meal_id = int(meal_key.split('_')[1])
            for region in ['Region_1', 'Region_2', 'Region_3', 'Region_4']:
                combo_data = self.train_aggregated[
                    (self.train_aggregated['meal_id'] == meal_id) & 
                    (self.train_aggregated['region'] == region)
                ]
                if len(combo_data) > 0:
                    combo_weekly = combo_data.groupby('week')['num_orders'].sum().reset_index()
                    weeks_count = len(combo_weekly)
                    
                    if weeks_count >= min_weeks_required:  # Only show viable combinations
                        availability[f"meal_{meal_id}_{region}"] = {
                            'weeks': weeks_count,
                            'sufficient': True,
                            'avg_demand': combo_weekly['num_orders'].mean(),
                            'description': f"Meal {meal_id} - {region} ({weeks_count} weeks)",
                            'type': 'meal_region'
                        }
        
        self.data_availability = availability
        
        # Print summary
        print(f"\n📊 Meal-Specific Forecasting Options:")
        print(f"{'Option':<25} {'Sufficient':<12} {'Weeks':<8} {'Avg Demand':<12} {'Description'}")
        print("-" * 85)
        
        # Group by type for better display
        for category in ['meal', 'meal_region']:
            category_items = [(k, v) for k, v in availability.items() if v.get('type') == category]
            if category_items:
                for key, info in sorted(category_items, key=lambda x: x[1]['avg_demand'], reverse=True):
                    sufficient_icon = "✅" if info['sufficient'] else "❌"
                    print(f"{key:<25} {sufficient_icon:<12} {info['weeks']:<8} {info['avg_demand']:<12.0f} {info['description']}")
        
        # Show summary
        viable_meals = len([k for k, v in availability.items() if v['sufficient'] and v['type'] == 'meal'])
        viable_combinations = len([k for k, v in availability.items() if v['sufficient'] and v['type'] == 'meal_region'])
        
        print(f"\n✅ Meal Forecasting Summary:")
        print(f"   • Individual meals available: {viable_meals}")
        print(f"   • Meal + region combinations: {viable_combinations}")
        print(f"   • Price regressors available: {'Yes' if self.column_mapping.get('checkout_price') else 'No'}")
        print(f"🎯 Focus: Meal-specific demand for ingredient ordering")

    def _normalize_model_key(self, region_id=None, meal_id=None):
        """Create consistent model key"""
        if meal_id:
            try:
                meal_id = int(meal_id)
            except:
                meal_id = str(meal_id)
        
        if region_id and meal_id:
            return f"meal_{meal_id}_{region_id}"
        elif meal_id:
            return f"meal_{meal_id}"
        elif region_id:
            return f"{region_id}_all"
        else:
            return "overall"

    def train_prophet_model(self, region_id=None, meal_id=None):
        """Train Prophet model for specific meal/region with price regressors"""
        try:
            # Convert inputs
            if region_id:
                region_id = str(region_id)
            if meal_id:
                try:
                    meal_id = int(meal_id)
                except:
                    meal_id = str(meal_id)
            
            # Create consistent model key
            model_key = self._normalize_model_key(region_id, meal_id)
            print(f"🎯 Training model with key: '{model_key}'")
            
            # Validate meal selection
            if not meal_id:
                return None, "❌ Meal selection required for forecasting. Please select a specific meal."
            
            print(f"🎯 Training for Meal {meal_id}" + (f" in {region_id}" if region_id else " (all regions)"))
            
            # Filter data
            if 'meal_id' not in self.train_aggregated.columns:
                return None, "❌ No meal data available in dataset"
            
            filtered_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]
            
            if region_id:
                filtered_data = filtered_data[filtered_data['region'] == region_id]
                region_centers = [center for center, region in self.regional_mapping.items() if region == region_id]
                print(f"📍 {region_id} includes centers: {region_centers}")
            
            print(f"🔍 Debug: Filtered data: {len(filtered_data)} records")
            
            if len(filtered_data) == 0:
                available_meals = sorted(self.train_aggregated['meal_id'].unique())
                return None, f"❌ No data found for Meal {meal_id}. Available meals: {available_meals[:10]}..."
            
            # Aggregate by week
            demand_data = filtered_data.groupby('week').agg({
                'num_orders': 'sum',
                **{col: 'mean' for col in filtered_data.columns 
                   if col == 'checkout_price' and col in filtered_data.columns}
            }).reset_index()
            
            print(f"🔍 Debug: Weekly data: {len(demand_data)} weeks")
            
            # Check for sufficient data
            min_weeks_required = 20
            if len(demand_data) < min_weeks_required:
                available_weeks = len(demand_data)
                coverage = available_weeks / len(self.weekly_demand) if len(self.weekly_demand) > 0 else 0
                
                suggestion = "Try a different meal or remove region filter"
                return None, f"❌ Insufficient data: only {available_weeks} weeks available ({coverage:.1%} coverage). Need {min_weeks_required}+. {suggestion}"
            
            # Log the actual week range being used
            min_week = demand_data['week'].min()
            max_week = demand_data['week'].max()
            total_weeks = len(demand_data)
            coverage = total_weeks / len(self.weekly_demand)
            
            print(f"📊 Training data: {total_weeks} weeks (weeks {min_week}-{max_week})")
            print(f"📈 Coverage: {coverage:.1%} of total timeline")
            print(f"📦 Demand range: {demand_data['num_orders'].min()}-{demand_data['num_orders'].max()}")
            
            # Prepare data for Prophet with checkout_price regressor
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime('2010-01-01') + pd.to_timedelta(demand_data['week'] * 7, unit='D'),
                'y': demand_data['num_orders']
            })
            
            # Add checkout_price regressor if available
            regressors_added = []
            
            if 'checkout_price' in demand_data.columns:
                prophet_data['checkout_price'] = demand_data['checkout_price'].fillna(demand_data['checkout_price'].mean())
                regressors_added.append('checkout_price')
            
            print(f"💰 Price regressor added: {regressors_added}")
            
            # Create and configure Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',  # Better for meal-specific data
                changepoint_prior_scale=0.05,
                uncertainty_samples=1000
            )
            
            # Add checkout_price regressor
            for regressor in regressors_added:
                model.add_regressor(regressor)
                print(f"   📊 Added regressor: {regressor}")
            
            # Train model
            model.fit(prophet_data)
            self.models[model_key] = {
                'model': model,
                'regressors': regressors_added,
                'meal_id': meal_id,
                'region_id': region_id,
                'training_weeks': total_weeks,
                'coverage': coverage,
                'training_data': prophet_data,  # Store for evaluation
                'original_data': demand_data     # Store original weekly data
            }
            
            # Generate predictions on training data for performance visualization
            self._generate_training_performance(model_key, model, prophet_data, demand_data)
            
            print(f"✅ Model stored with key: '{model_key}'")
            return model, f"✅ Meal-specific model trained for {model_key} with checkout_price regressor"
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error training model: {str(e)}"
    
    def _generate_training_performance(self, model_key, model, training_data, original_demand_data):
        """Generate predictions on training data for performance visualization"""
        try:
            # Make predictions on the training data
            predictions = model.predict(training_data[['ds'] + [col for col in training_data.columns if col not in ['ds', 'y']]])
            
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
        # Normalize the model key before lookup
        normalized_key = model_key
        if normalized_key not in self.training_performance:
            # Try different variations
            alternatives = []
            if model_key.startswith('all_'):
                alternatives.append(model_key.replace('all_', 'meal_'))
            elif model_key.startswith('meal_'):
                alternatives.append(model_key.replace('meal_', 'all_'))
            
            for alt_key in alternatives:
                if alt_key in self.training_performance:
                    print(f"🔍 Found training performance with alternative key: {alt_key}")
                    return self.training_performance[alt_key]
        
        return self.training_performance.get(normalized_key, None)
    
    def generate_forecast(self, model_key, weeks_ahead=10, confidence_level=0.95):
        """Generate forecast using trained Prophet model with price regressors"""
        try:
            # Normalize model key and find matching model
            print(f"🔍 Looking for model: '{model_key}'")
            print(f"🔍 Available models: {list(self.models.keys())}")
            
            if model_key not in self.models:
                # Try different key formats
                alternatives = []
                if model_key.startswith('all_'):
                    alternatives.append(model_key.replace('all_', 'meal_'))
                elif model_key.startswith('meal_'):
                    alternatives.append(model_key.replace('meal_', 'all_'))
                
                for alt_key in alternatives:
                    if alt_key in self.models:
                        print(f"✅ Found model with alternative key: '{alt_key}'")
                        model_key = alt_key
                        break
                else:
                    return None, f"Model not found. Available: {list(self.models.keys())}"
            
            model_info = self.models[model_key]
            model = model_info['model']
            regressors = model_info['regressors']
            meal_id = model_info['meal_id']
            
            # Create future dataframe
            last_date = model.history['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=weeks_ahead,
                freq='W'
            )
            
            future = pd.DataFrame({'ds': future_dates})
            
            # Add regressor values for future periods
            if meal_id in self.meal_price_data:
                meal_prices = self.meal_price_data[meal_id]
                
                # Use recent price trends for future forecasting
                if 'checkout_price' in regressors:
                    future['checkout_price'] = meal_prices.get('recent_checkout_price', meal_prices.get('avg_checkout_price', 0))
                
                print(f"💰 Using checkout_price regressor for forecast:")
                for regressor in regressors:
                    if regressor in future.columns:
                        print(f"   • {regressor}: ${future[regressor].iloc[0]:.2f}")
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Calculate custom confidence intervals if needed
            if confidence_level != 0.8:
                print(f"📊 Adjusting confidence intervals to {confidence_level*100}%")
                
                from scipy import stats
                default_z = stats.norm.ppf(0.9)
                custom_z = stats.norm.ppf((1 + confidence_level) / 2)
                scaling_factor = custom_z / default_z
                
                default_uncertainty = forecast['yhat_upper'] - forecast['yhat']
                custom_uncertainty = default_uncertainty * scaling_factor
                forecast['yhat_upper'] = forecast['yhat'] + custom_uncertainty
                forecast['yhat_lower'] = forecast['yhat'] - custom_uncertainty
                forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
            
            # Store forecast
            self.forecasts[model_key] = {
                'forecast': forecast,
                'model_key': model_key,
                'weeks_ahead': weeks_ahead,
                'confidence_level': confidence_level,
                'meal_id': meal_id,
                'regressors_used': regressors
            }
            
            return forecast, f"✅ Meal-specific forecast generated for Meal {meal_id} with checkout_price regressor ({confidence_level*100}% CI)"
            
        except Exception as e:
            print(f"❌ Forecast error: {e}")
            return None, f"Error generating forecast: {str(e)}"
    
    def evaluate_model_performance(self, model_key):
        """Evaluate Prophet model performance using train data split"""
        try:
            print(f"🔍 Evaluating model: '{model_key}'")
            print(f"🔍 Available models: {list(self.models.keys())}")
            
            # Normalize model key and find matching model
            if model_key not in self.models:
                # Try different key formats
                alternatives = []
                if model_key.startswith('all_'):
                    alternatives.append(model_key.replace('all_', 'meal_'))
                elif model_key.startswith('meal_'):
                    alternatives.append(model_key.replace('meal_', 'all_'))
                
                for alt_key in alternatives:
                    if alt_key in self.models:
                        print(f"✅ Found model with alternative key: '{alt_key}'")
                        model_key = alt_key
                        break
                else:
                    return None, f"Model not found. Available: {list(self.models.keys())}"
            
            model_info = self.models[model_key]
            model = model_info['model']
            meal_id = model_info['meal_id']
            region_id = model_info.get('region_id')
            
            print(f"🔍 Evaluating model: {model_key}")
            
            # Get stored training data
            training_data = model_info.get('training_data')
            original_data = model_info.get('original_data')
            
            if training_data is None or original_data is None:
                return None, "No training data available for evaluation"
            
            # Use last 20% for evaluation
            total_weeks = len(original_data)
            eval_size = max(5, int(total_weeks * 0.2))
            eval_indices = list(range(total_weeks - eval_size, total_weeks))
            
            eval_data_prophet = training_data.iloc[eval_indices].copy()
            eval_data_original = original_data.iloc[eval_indices].copy()
            
            if len(eval_data_prophet) == 0:
                return None, "No evaluation data available"
            
            # Log the actual evaluation period
            eval_min_week = eval_data_original['week'].min()
            eval_max_week = eval_data_original['week'].max()
            print(f"📊 Total available weeks: {total_weeks} (weeks {original_data['week'].min()}-{original_data['week'].max()})")
            print(f"📈 Using {len(eval_data_prophet)} weeks for evaluation (weeks {eval_min_week}-{eval_max_week})")
            
            # Add regressors for evaluation
            regressors = model_info['regressors']
            for regressor in regressors:
                if regressor in eval_data_original.columns:
                    eval_data_prophet[regressor] = eval_data_original[regressor].fillna(eval_data_original[regressor].mean())
            
            # Make predictions
            predictions = model.predict(eval_data_prophet[['ds'] + regressors])
            
            # Calculate metrics
            y_true = eval_data_prophet['y'].values
            y_pred = np.maximum(predictions['yhat'].values, 0)
            
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
                'evaluation_weeks': f"{eval_min_week}-{eval_max_week}",
                'meal_id': meal_id,
                'region_id': region_id,
                'regressors_used': len(regressors)
            }
            
            print(f"✅ Metrics calculated for Meal {meal_id}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            
            self.performance_metrics[model_key] = metrics
            
            return metrics, f"Model evaluation completed for Meal {meal_id} using weeks {eval_min_week}-{eval_max_week}"
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error evaluating model: {str(e)}"
    
    def get_data_availability(self):
        """Get data availability information for the frontend"""
        return self.data_availability
    
    def get_model_summary(self):
        """Get summary of all trained models and their performance"""
        summary = {
            'models_trained': len(self.models),
            'models': {},
            'data_info': {
                'train_records': len(self.train_data) if self.train_data is not None else 0,
                'test_records': len(self.test_data) if self.test_data is not None else 0,
                'regions': ['Region_1', 'Region_2', 'Region_3', 'Region_4'],
                'meals': list(self.train_data['meal_id'].unique()) if self.train_data is not None and 'meal_id' in self.train_data.columns else [],
                'weeks_range': [
                    int(self.train_data['week'].min()) if self.train_data is not None else 0,
                    int(self.train_data['week'].max()) if self.train_data is not None else 0
                ],
                'has_price_data': bool(self.column_mapping.get('checkout_price')),
                'price_regressors_available': ['checkout_price'] if self.column_mapping.get('checkout_price') else []
            },
            'data_availability': self.data_availability
        }
        
        for key in self.models.keys():
            model_info = self.models[key]
            summary['models'][key] = {
                'performance': self.performance_metrics.get(key, {}),
                'has_forecast': key in self.forecasts,
                'meal_id': model_info.get('meal_id'),
                'region_id': model_info.get('region_id'),
                'regressors': model_info.get('regressors', []),
                'training_weeks': model_info.get('training_weeks', 0),
                'coverage': model_info.get('coverage', 0)
            }
        
        return summary