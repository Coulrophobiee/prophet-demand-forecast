#!/usr/bin/env python3
"""
Food Demand Forecaster - Prophet Model Implementation with Business Cost Analysis
Enhanced version with Euro-based cost evaluation for business impact assessment
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import json
import random

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
        self.meal_names = {}
        self.meal_database = {}
        self.business_costs = {}
            
    def load_meal_names(self):
        """Load meal names and data from existing JSON file"""
        try:
            if os.path.exists('data/meal_database.json'):
                with open('data/meal_database.json', 'r', encoding='utf-8') as f:
                    raw_meal_data = json.load(f)
                
                self.meal_database = {}
                self.meal_names = {}
                
                for meal_id_str, meal_info in raw_meal_data.items():
                    meal_id = int(meal_id_str)
                    meal_name = meal_info.get('name', f'Meal {meal_id}')
                    
                    price_dist = meal_info.get('price_distribution', [])
                    perishable_flags = meal_info.get('perishable', [])
                    
                    perishable_percentage = 0.0
                    if len(price_dist) == len(perishable_flags):
                        for i, is_perishable in enumerate(perishable_flags):
                            if is_perishable and i < len(price_dist):
                                perishable_percentage += price_dist[i]
                    
                    kg_per_10 = meal_info.get('kg_per_10_persons', [])
                    total_kg_per_portion = sum(kg_per_10) / 10.0 if kg_per_10 else 0.5
                    
                    processed_meal_data = {
                        'name': meal_name,
                        'ingredients': meal_info.get('ingredients', []),
                        'perishable_percentage': perishable_percentage,
                        'non_perishable_percentage': 100.0 - perishable_percentage,
                        'total_kg_per_portion': total_kg_per_portion,
                        'type': meal_info.get('type', 'Unknown'),
                        'price_tier': meal_info.get('price_tier', 'Standard'),
                        'original_data': meal_info
                    }
                    
                    self.meal_database[str(meal_id)] = processed_meal_data
                    self.meal_database[int(meal_id)] = processed_meal_data
                    self.meal_database[meal_id] = processed_meal_data
                    
                    self.meal_names[meal_id] = meal_name
                
                print(f"🍽️ Loaded {len(self.meal_names)} meals from existing meal_database.json")
                
        except Exception as e:
            print(f"❌ Error loading meal database: {e}")
    
    def get_meal_display_name(self, meal_id):
        """Get display name for a meal"""
        try:
            meal_id_int = int(meal_id)
            return self.meal_names.get(meal_id_int, f"Meal {meal_id}")
        except:
            return f"Meal {meal_id}"
        
    def calculate_business_cost(self, actual_orders, predicted_orders, meal_id, avg_meal_price=15.0):
        """Calculate business costs for given actual vs predicted orders"""
        try:
            INGREDIENT_COST_RATE = 0.40
            STORAGE_COST_RATE = 0.15
            EMERGENCY_COST_RATE = 0.60
            OPPORTUNITY_COST_RATE = 0.02
            
            meal_info = None
            for key_variant in [str(meal_id), int(meal_id), meal_id]:
                if key_variant in self.meal_database:
                    meal_info = self.meal_database[key_variant]
                    break
            
            if not meal_info:
                return None
                
            meal_name = meal_info.get('name', f"Meal {meal_id}")
            meal_type = meal_info.get('type', 'Unknown')
            
            original_data = meal_info.get('original_data', meal_info)
            ingredients = original_data.get('ingredients', [])
            kg_per_10_persons = original_data.get('kg_per_10_persons', [])
            perishable = original_data.get('perishable', [])
            price_distribution = original_data.get('price_distribution', [])
            storage_m3 = original_data.get('storage_m3', [])
            
            if not all(len(arr) == len(ingredients) for arr in [kg_per_10_persons, perishable, price_distribution, storage_m3]):
                return None
            
            ingredient_cost_per_order = avg_meal_price * INGREDIENT_COST_RATE
            storage_cost_per_order = avg_meal_price * STORAGE_COST_RATE
            
            difference = predicted_orders - actual_orders
            
            overprediction_penalty = 0.0
            underprediction_penalty = 0.0
            
            if difference > 0:  # Overprediction
                excess_orders = difference
                
                storage_cost = excess_orders * (avg_meal_price * STORAGE_COST_RATE)
                opportunity_cost = excess_orders * (avg_meal_price * OPPORTUNITY_COST_RATE)
                overprediction_penalty += storage_cost + opportunity_cost

                for i, ingredient_name in enumerate(ingredients):
                    is_perishable = perishable[i]
                    if is_perishable:
                        cost_percentage = price_distribution[i] / 100
                        ingredient_cost = excess_orders * avg_meal_price * INGREDIENT_COST_RATE * cost_percentage
                        overprediction_penalty += ingredient_cost
                                        
            elif difference < 0:  # Underprediction
                shortage_orders = abs(difference)
                underprediction_penalty = shortage_orders * ((avg_meal_price * INGREDIENT_COST_RATE) / 2)
            
            total_penalty = overprediction_penalty + underprediction_penalty
            total_revenue = actual_orders * avg_meal_price
            penalty_percentage = (total_penalty / total_revenue * 100) if total_revenue > 0 else 0
            
            total_kg_per_portion = sum(kg_per_10_persons) / 10
            perishable_cost_percentage = sum(
                price_distribution[i] for i, is_perish in enumerate(perishable) if is_perish
            )
            
            return {
                'actual_orders': int(actual_orders),
                'predicted_orders': int(predicted_orders),
                'difference': int(difference),
                'meal_id': meal_id,
                'meal_name': meal_name,
                'meal_type': meal_type,
                'avg_meal_price': avg_meal_price,
                'total_kg_per_portion': total_kg_per_portion,
                'perishable_cost_percentage': perishable_cost_percentage,
                'ingredient_count': len(ingredients),
                'overprediction_penalty': overprediction_penalty,
                'underprediction_penalty': underprediction_penalty,
                'total_penalty': total_penalty,
                'total_revenue': total_revenue,
                'penalty_percentage': penalty_percentage,
                'ingredient_cost_per_order': ingredient_cost_per_order,
                'storage_cost_per_order': storage_cost_per_order,
                'ingredients_breakdown': [
                    {
                        'name': ingredients[i],
                        'kg_per_portion': kg_per_10_persons[i] / 10,
                        'is_perishable': perishable[i],
                        'cost_percentage': price_distribution[i],
                        'storage_m3_per_portion': storage_m3[i] / 10
                    }
                    for i in range(len(ingredients))
                ]
            }
            
        except Exception as e:
            print(f"❌ Error calculating business cost for meal {meal_id}: {e}")
            return None

    def evaluate_business_cost(self, model_key, custom_meal_price=None):
        """Evaluate business costs for a trained model using test weeks"""
        try:
            if model_key not in self.models:
                alternatives = []
                if model_key.startswith('all_'):
                    alternatives.append(model_key.replace('all_', 'meal_'))
                elif model_key.startswith('meal_'):
                    alternatives.append(model_key.replace('meal_', 'all_'))
                
                for alt_key in alternatives:
                    if alt_key in self.models:
                        model_key = alt_key
                        break
                else:
                    return None, f"Model not found: {model_key}"
            
            model_info = self.models[model_key]
            meal_id = model_info['meal_id']
            
            comparison_data = self.get_model_comparison(model_key)
            if not comparison_data:
                return None, "No comparison data available for business cost analysis"
            
            if custom_meal_price:
                avg_meal_price = custom_meal_price
            else:
                avg_meal_price = 15.0
                if meal_id in self.meal_price_data and 'avg_checkout_price' in self.meal_price_data[meal_id]:
                    avg_meal_price = self.meal_price_data[meal_id]['avg_checkout_price']
            
            actual_orders = comparison_data['actual']
            prophet_predictions = comparison_data['prophet_predicted']
            weeks = comparison_data['weeks']
            
            weekly_costs = []
            total_penalty = 0.0
            total_revenue = 0.0
            
            for i, week in enumerate(weeks):
                actual = actual_orders[i]
                predicted = prophet_predictions[i]
                
                cost_analysis = self.calculate_business_cost(actual, predicted, meal_id, avg_meal_price)
                if cost_analysis:
                    weekly_costs.append({
                        'week': week,
                        'actual': actual,
                        'predicted': predicted,
                        'penalty_euro': cost_analysis['total_penalty'],
                        'revenue_euro': cost_analysis['total_revenue'],
                        'penalty_percentage': cost_analysis['penalty_percentage']
                    })
                    
                    total_penalty += cost_analysis['total_penalty']
                    total_revenue += cost_analysis['total_revenue']
            
            avg_weekly_penalty = total_penalty / len(weekly_costs) if weekly_costs else 0
            penalty_percentage_of_revenue = (total_penalty / total_revenue * 100) if total_revenue > 0 else 0
            annual_penalty_estimate = avg_weekly_penalty * 52
            annual_revenue_estimate = (total_revenue / len(weekly_costs)) * 52 if weekly_costs else 0
            
            meal_name = self.get_meal_display_name(meal_id)
            
            business_cost_analysis = {
                'model_key': model_key,
                'meal_id': meal_id,
                'meal_name': meal_name,
                'evaluation_period': comparison_data['eval_period'],
                'weeks_analyzed': len(weekly_costs),
                'avg_meal_price_euro': avg_meal_price,
                'weekly_costs': weekly_costs,
                'total_penalty_euro': total_penalty,
                'total_revenue_euro': total_revenue,
                'avg_weekly_penalty_euro': avg_weekly_penalty,
                'penalty_percentage_of_revenue': penalty_percentage_of_revenue,
                'annual_penalty_estimate_euro': annual_penalty_estimate,
                'annual_revenue_estimate_euro': annual_revenue_estimate,
                'prophet_accuracy': comparison_data['prophet_metrics']['accuracy']
            }
            
            return business_cost_analysis, f"Business cost analysis completed for {meal_name}"
            
        except Exception as e:
            print(f"❌ Error in business cost evaluation: {e}")
            return None, f"Error evaluating business costs: {str(e)}"
    
    def compare_business_costs(self, model_key):
        """Compare business costs between Prophet and Baseline models"""
        try:
            comparison_data = self.get_model_comparison(model_key)
            if not comparison_data:
                return None, "No comparison data available for business cost analysis"
            
            model_info = self.models[model_key]
            meal_id = model_info['meal_id']
            
            meal_in_db = False
            for key_variant in [str(meal_id), int(meal_id), meal_id]:
                if key_variant in self.meal_database:
                    meal_in_db = True
                    break
            
            avg_meal_price = 15.0
            if meal_id in self.meal_price_data and 'avg_checkout_price' in self.meal_price_data[meal_id]:
                avg_meal_price = self.meal_price_data[meal_id]['avg_checkout_price']
            
            actual_orders = comparison_data['actual']
            prophet_predictions = comparison_data['prophet_predicted']
            baseline_predictions = comparison_data['baseline_predicted']
            weeks = comparison_data['weeks']
            
            prophet_weekly_penalties = []
            baseline_weekly_penalties = []
            weekly_revenues = []
            
            weekly_comparison = []
            successful_weeks = 0
            
            for i, week in enumerate(weeks):
                actual = actual_orders[i]
                prophet_pred = prophet_predictions[i]
                baseline_pred = baseline_predictions[i]
                
                prophet_cost = self.calculate_business_cost(actual, prophet_pred, meal_id, avg_meal_price)
                baseline_cost = self.calculate_business_cost(actual, baseline_pred, meal_id, avg_meal_price)
                
                if prophet_cost and baseline_cost:
                    prophet_penalty = prophet_cost['total_penalty']
                    baseline_penalty = baseline_cost['total_penalty']
                    revenue = prophet_cost['total_revenue']
                    
                    prophet_weekly_penalties.append(prophet_penalty)
                    baseline_weekly_penalties.append(baseline_penalty)
                    weekly_revenues.append(revenue)
                    
                    weekly_comparison.append({
                        'week': week,
                        'actual': actual,
                        'prophet_predicted': prophet_pred,
                        'baseline_predicted': baseline_pred,
                        'prophet_penalty': prophet_penalty,
                        'baseline_penalty': baseline_penalty,
                        'savings': baseline_penalty - prophet_penalty,
                        'revenue': revenue
                    })
                    successful_weeks += 1
            
            if not weekly_comparison:
                return None, "Could not calculate business costs for any weeks"
            
            avg_weekly_prophet_penalty = sum(prophet_weekly_penalties) / len(prophet_weekly_penalties)
            avg_weekly_baseline_penalty = sum(baseline_weekly_penalties) / len(baseline_weekly_penalties)
            avg_weekly_savings = avg_weekly_baseline_penalty - avg_weekly_prophet_penalty
            avg_weekly_revenue = sum(weekly_revenues) / len(weekly_revenues)
            
            prophet_total_penalty = sum(prophet_weekly_penalties)
            baseline_total_penalty = sum(baseline_weekly_penalties)
            cost_savings_euro = baseline_total_penalty - prophet_total_penalty
            total_revenue = sum(weekly_revenues)
            
            cost_savings_percentage = (cost_savings_euro / baseline_total_penalty * 100) if baseline_total_penalty > 0 else 0
            
            annual_savings_estimate = avg_weekly_savings * 52
            annual_revenue_estimate = avg_weekly_revenue * 52
            annual_savings_percentage = (annual_savings_estimate / annual_revenue_estimate * 100) if annual_revenue_estimate > 0 else 0
            
            prophet_implementation_cost = 50000
            roi_ratio = annual_savings_estimate / prophet_implementation_cost if prophet_implementation_cost > 0 else 0
            payback_months = (prophet_implementation_cost / avg_weekly_savings * (1/4.33)) if avg_weekly_savings > 0 else float('inf')
            
            meal_name = self.get_meal_display_name(meal_id)
            
            business_comparison = {
                'model_key': model_key,
                'meal_id': meal_id,
                'meal_name': meal_name,
                'evaluation_period': comparison_data['eval_period'],
                'weeks_analyzed': len(weekly_comparison),
                'avg_meal_price_euro': avg_meal_price,
                'weekly_comparison': weekly_comparison,
                
                'prophet_total_penalty': prophet_total_penalty,
                'baseline_total_penalty': baseline_total_penalty,
                'cost_savings_euro': cost_savings_euro,
                'cost_savings_percentage': cost_savings_percentage,
                'total_revenue_euro': total_revenue,
                
                'avg_weekly_savings_euro': avg_weekly_savings,
                'avg_weekly_prophet_penalty': avg_weekly_prophet_penalty,
                'avg_weekly_baseline_penalty': avg_weekly_baseline_penalty,
                'avg_weekly_revenue': avg_weekly_revenue,
                
                'annual_savings_estimate_euro': annual_savings_estimate,
                'annual_revenue_estimate_euro': annual_revenue_estimate,
                'annual_savings_percentage': annual_savings_percentage,
                
                'prophet_implementation_cost_euro': prophet_implementation_cost,
                'roi_ratio': roi_ratio,
                'payback_months': min(payback_months, 999),
                
                'prophet_accuracy': comparison_data['prophet_metrics']['accuracy'],
                'baseline_accuracy': comparison_data['baseline_metrics']['accuracy'],
                'accuracy_improvement': comparison_data['accuracy_improvement'],
                
                'business_recommendation': {
                    'implement_prophet': cost_savings_euro > 0,
                    'savings_per_euro_invested': roi_ratio,
                    'payback_period_months': min(payback_months, 999),
                    'annual_impact_estimate': annual_savings_estimate,
                    'confidence_level': 'High' if comparison_data['accuracy_improvement'] > 5 else 'Medium' if comparison_data['accuracy_improvement'] > 0 else 'Low'
                }
            }
            
            return business_comparison, f"Business cost comparison completed for {meal_name}"
            
        except Exception as e:
            print(f"❌ Error in business cost comparison: {e}")
            return None, f"Error comparing business costs: {str(e)}"
        
    def load_data(self):
        """Load train and test datasets"""
        try:
            self.load_meal_names()
            
            if os.path.exists('train.csv'):
                self.train_data = pd.read_csv('train.csv')
                print(f"✅ Loaded train.csv: {len(self.train_data)} records")
            else:
                print("❌ train.csv not found in current directory")
                return False
                
            if os.path.exists('test.csv'):
                self.test_data = pd.read_csv('test.csv')
                print(f"✅ Loaded test.csv: {len(self.test_data)} records")
            else:
                self.test_data = None
                print("ℹ️ test.csv not found - continuing without it (not required for forecasting)")
            
            if not self.meal_database:
                print("⚠️ No meal database loaded - business cost analysis may use default values")
                
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
        
        target_candidates = ['num_orders', 'orders', 'demand', 'quantity', 'qty', 'sales']
        week_candidates = ['week', 'time', 'period', 'date']
        center_candidates = ['center_id', 'center', 'location_id', 'location', 'store_id', 'store']
        meal_candidates = ['meal_id', 'meal', 'product_id', 'product', 'item_id', 'item']
        checkout_price_candidates = ['checkout_price', 'final_price', 'selling_price', 'price']
        base_price_candidates = ['base_price', 'original_price', 'list_price', 'standard_price']
        
        detected = {}
        
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
        
        return detected

    def create_regional_mapping(self):
        """Create robust regional mapping for centers - excluding Region-Lyon"""
        if self.train_data is None:
            return {}
        
        regional_mapping = {}
        
        try:
            if os.path.exists('fulfilment_center_info.csv'):
                center_info = pd.read_csv('fulfilment_center_info.csv')
                
                for _, row in center_info.iterrows():
                    center_id = row['center_id']
                    region_code = row['region_code']
                    
                    if region_code == 56:
                        final_region = "Region-Kassel"
                    elif region_code == 34:
                        final_region = "Region-Luzern"  
                    elif region_code == 77:
                        final_region = "Region-Wien"
                    elif region_code == 85:
                        continue  # Skip Region-Lyon
                    else:
                        final_region = "Region-Kassel"  # Default fallback
                    
                    regional_mapping[center_id] = final_region
                
                print(f"✅ Regional mapping created (excluding Region-Lyon)")
                
        except Exception as e:
            print(f"❌ Error loading fulfilment center info: {e}")
            centers = sorted(self.train_data['center_id'].unique())
            for i, center in enumerate(centers):
                region_names = ["Region-Kassel", "Region-Wien", "Region-Luzern"]
                regional_mapping[center] = region_names[i % 3]
        
        return regional_mapping

    def calculate_discount_metrics(self):
        """Calculate price metrics for Prophet regressors"""
        checkout_col = self.column_mapping.get('checkout_price')
        
        if checkout_col:
            print(f"💰 Price analysis completed - using checkout_price as regressor")
        else:
            print("⚠️ checkout_price column not found - skipping price regression")

    def filter_meals(self):
        """Remove meals with insufficient coverage"""
        meals_to_remove = [1571, 2104, 2956, 2490, 2569, 2664]
        
        if self.train_data is not None and 'meal_id' in self.train_data.columns:
            initial_count = len(self.train_data)
            meals_found = [meal for meal in meals_to_remove if meal in self.train_data['meal_id'].unique()]
            
            if meals_found:
                self.train_data = self.train_data[~self.train_data['meal_id'].isin(meals_to_remove)]
                
                if self.test_data is not None and 'meal_id' in self.test_data.columns:
                    self.test_data = self.test_data[~self.test_data['meal_id'].isin(meals_to_remove)]
                
                final_count = len(self.train_data)
                removed_count = initial_count - final_count
                
                print(f"✅ Filtered out {len(meals_found)} meals with insufficient coverage")

    def preprocess_data(self):
        """Preprocess data for Prophet model with checkout_price regressor - excluding Region-Lyon"""
        if self.train_data is None:
            return
        
        self.filter_meals()
        self.regional_mapping = self.create_regional_mapping()
        
        self.train_data['region'] = self.train_data['center_id'].map(self.regional_mapping)
        
        initial_count = len(self.train_data)
        self.train_data = self.train_data.dropna(subset=['region'])
        excluded_count = initial_count - len(self.train_data)
        
        if excluded_count > 0:
            print(f"🚫 Excluded {excluded_count:,} records from Region-Lyon")
        
        if self.test_data is not None:
            self.test_data['region'] = self.test_data['center_id'].map(self.regional_mapping)
            self.test_data = self.test_data.dropna(subset=['region'])
        
        self.column_mapping = self.detect_column_names()
        
        if not self.column_mapping or 'target' not in self.column_mapping:
            print("❌ Could not detect target column")
            return
        
        self.calculate_discount_metrics()
        
        target_col = self.column_mapping['target']
        week_col = self.column_mapping.get('week', 'week')
        meal_col = self.column_mapping.get('meal', 'meal_id')
        checkout_price_col = self.column_mapping.get('checkout_price')
        
        try:
            agg_dict = {target_col: 'sum'}
            
            if checkout_price_col:
                agg_dict[checkout_price_col] = 'mean'
            
            group_cols = [week_col, 'region', meal_col] if meal_col in self.train_data.columns else [week_col, 'region']
            
            self.train_aggregated = self.train_data.groupby(group_cols).agg(agg_dict).reset_index()
            
            rename_dict = {target_col: 'num_orders'}
            if week_col != 'week':
                rename_dict[week_col] = 'week'
            if meal_col != 'meal_id' and meal_col in self.train_aggregated.columns:
                rename_dict[meal_col] = 'meal_id'
            
            self.train_aggregated = self.train_aggregated.rename(columns=rename_dict)
            
            overall_agg_dict = {target_col: 'sum'}
            if checkout_price_col:
                overall_agg_dict[checkout_price_col] = 'mean'
            
            self.weekly_demand = self.train_data.groupby(week_col).agg(overall_agg_dict).reset_index()
            self.weekly_demand = self.weekly_demand.rename(columns={week_col: 'week', target_col: 'num_orders'})
            self.weekly_demand['ds'] = pd.to_datetime('2010-01-01') + pd.to_timedelta(self.weekly_demand['week'] * 7, unit='D')
            self.weekly_demand['y'] = self.weekly_demand['num_orders']
            
            self.calculate_meal_price_stats()
            
            min_week = self.weekly_demand['week'].min()
            max_week = self.weekly_demand['week'].max()
            total_weeks = len(self.weekly_demand)
            
            print(f"📊 Data preprocessed: {total_weeks} weeks ({min_week} - {max_week})")
            print(f"🗺️ Regional aggregation complete - {len(set(self.regional_mapping.values()))} regions")
            print(f"🚫 Region-Lyon excluded from all analysis")
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")

    def calculate_meal_price_stats(self):
        """Calculate price statistics per meal for forecasting"""
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

    def analyze_data_availability(self):
        """Analyze data availability for meal-specific forecasting"""
        if self.train_aggregated is None or 'meal_id' not in self.train_aggregated.columns:
            print("❌ No meal data available for analysis")
            return
        
        availability = {}
        min_weeks_required = 20
        overall_weeks = len(self.weekly_demand)
        
        for meal in sorted(self.train_aggregated['meal_id'].unique()):
            meal_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal]
            if len(meal_data) > 0:
                meal_weekly = meal_data.groupby('week')['num_orders'].sum().reset_index()
                weeks_count = len(meal_weekly)
                coverage = weeks_count / overall_weeks
                avg_demand = meal_weekly['num_orders'].mean()
                meal_name = self.get_meal_display_name(meal)
                
                availability[f"meal_{meal}"] = {
                    'weeks': weeks_count,
                    'sufficient': weeks_count >= min_weeks_required,
                    'avg_demand': avg_demand,
                    'description': f"{meal_name} - All regions ({weeks_count} weeks, {coverage:.1%} coverage)",
                    'type': 'meal',
                    'coverage': coverage
                }
        
        viable_meals = [k for k, v in availability.items() if k.startswith('meal_') and v['sufficient']]
        
        for meal_key in viable_meals[:10]:
            meal_id = int(meal_key.split('_')[1])
            
            for region in sorted(set(self.regional_mapping.values())):
                combo_data = self.train_aggregated[
                    (self.train_aggregated['meal_id'] == meal_id) & 
                    (self.train_aggregated['region'] == region)
                ]
                
                if len(combo_data) > 0:
                    combo_weekly = combo_data.groupby('week')['num_orders'].sum().reset_index()
                    weeks_count = len(combo_weekly)
                    avg_demand = combo_weekly['num_orders'].mean()
                    
                    if weeks_count >= min_weeks_required:
                        meal_name = self.get_meal_display_name(meal_id)
                        availability[f"meal_{meal_id}_{region}"] = {
                            'weeks': weeks_count,
                            'sufficient': True,
                            'avg_demand': avg_demand,
                            'description': f"{meal_name} - {region} ({weeks_count} weeks)",
                            'type': 'meal_region'
                        }
        
        self.data_availability = availability
        
        viable_individual_meals = len([k for k, v in availability.items() if v['sufficient'] and v['type'] == 'meal'])
        viable_combinations = len([k for k, v in availability.items() if v['sufficient'] and v['type'] == 'meal_region'])
        
        print(f"✅ Meal forecasting available: {viable_individual_meals} meals, {viable_combinations} combinations")

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

    def calculate_robust_metrics(self, y_true, y_pred):
        """Calculate more robust metrics that handle outliers better"""
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        def robust_mape(actual, predicted):
            mask = np.abs(actual) > np.percentile(np.abs(actual), 5)
            actual_masked = actual[mask]
            predicted_masked = predicted[mask]
            
            if len(actual_masked) == 0:
                return 100.0
            
            percentage_errors = np.abs((actual_masked - predicted_masked) / actual_masked) * 100
            percentage_errors = np.minimum(percentage_errors, 200)
            
            return np.mean(percentage_errors)
        
        def symmetric_mape(actual, predicted):
            denominator = (np.abs(actual) + np.abs(predicted)) / 2
            mask = denominator > np.percentile(denominator, 5)
            
            if np.sum(mask) == 0:
                return 100.0
            
            actual_masked = actual[mask]
            predicted_masked = predicted[mask]
            denominator_masked = denominator[mask]
            
            smape = np.mean(np.abs(actual_masked - predicted_masked) / denominator_masked) * 100
            return np.minimum(smape, 200)
        
        median_absolute_error = np.median(np.abs(y_true - y_pred))
        
        standard_mape = robust_mape(y_true, y_pred)
        symmetric_mape_val = symmetric_mape(y_true, y_pred)
        
        bias = np.mean(y_pred - y_true)
        
        accuracy_standard = max(0, 100 - standard_mape)
        accuracy_symmetric = max(0, 100 - symmetric_mape_val)
        
        data_variance = np.var(y_true)
        data_mean = np.mean(y_true)
        cv = (np.sqrt(data_variance) / data_mean) * 100 if data_mean > 0 else 100
        
        if cv > 50:
            primary_accuracy = accuracy_symmetric
            primary_mape = symmetric_mape_val
            accuracy_note = "Using Symmetric MAPE due to high variance"
        else:
            primary_accuracy = accuracy_standard
            primary_mape = standard_mape
            accuracy_note = "Using standard MAPE"
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(primary_mape),
            'mape_standard': float(standard_mape),
            'mape_symmetric': float(symmetric_mape_val),
            'median_ae': float(median_absolute_error),
            'bias': float(bias),
            'accuracy': float(primary_accuracy),
            'cv': float(cv),
            'accuracy_note': accuracy_note
        }

    def detect_data_characteristics(self, data):
        """Analyze data to determine optimal Prophet configuration"""
        if len(data) == 0:
            return 'medium'
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 100
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR
        outlier_count = np.sum(data > outlier_threshold)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        if cv > 60 or outlier_percentage > 10:
            variance_level = 'high'
        elif cv > 30 or outlier_percentage > 5:
            variance_level = 'medium'
        else:
            variance_level = 'low'
        
        return variance_level

    def create_robust_prophet_model(self, data_variance_level='medium'):
        """Create Prophet model with parameters optimized for data characteristics"""
        
        if data_variance_level == 'high':
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=0.01,
                uncertainty_samples=1000,
                interval_width=0.8
            )
        elif data_variance_level == 'medium':
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,
                uncertainty_samples=1000
            )
        else:  # low variance
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.01,
                uncertainty_samples=1000
            )
        
        return model

    def train_prophet_model(self, region_id=None, meal_id=None):
        """Train Prophet model for specific meal/region with adaptive configuration"""
        try:
            if region_id:
                region_id = str(region_id)
            if meal_id:
                try:
                    meal_id = int(meal_id)
                except:
                    meal_id = str(meal_id)
            
            model_key = self._normalize_model_key(region_id, meal_id)
            
            if not meal_id:
                return None, "❌ Meal selection required for forecasting. Please select a specific meal."
            
            if 'meal_id' not in self.train_aggregated.columns:
                return None, "❌ No meal data available in dataset"
            
            available_meals = set(self.train_aggregated['meal_id'].unique())
            if meal_id not in available_meals:
                return None, f"❌ Meal {meal_id} not found. Available meals: {sorted(list(available_meals))[:10]}..."
            
            filtered_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]
            
            if region_id:
                available_regions = set(self.train_aggregated['region'].unique())
                if region_id not in available_regions:
                    return None, f"❌ Region {region_id} not found. Available regions: {sorted(list(available_regions))}"
                
                filtered_data = filtered_data[filtered_data['region'] == region_id]
            
            if len(filtered_data) == 0:
                if region_id:
                    meal_regions = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]['region'].unique()
                    return None, f"❌ No data found for Meal {meal_id} in {region_id}. Available regions: {' and '.join(sorted(meal_regions))}"
                else:
                    return None, f"❌ No data found for Meal {meal_id}"
            
            demand_data = filtered_data.groupby('week').agg({
                'num_orders': 'sum',
                **{col: 'mean' for col in filtered_data.columns 
                   if col == 'checkout_price' and col in filtered_data.columns}
            }).reset_index()
            
            min_weeks_required = 20
            if len(demand_data) < min_weeks_required:
                available_weeks = len(demand_data)
                coverage = available_weeks / len(self.weekly_demand) if len(self.weekly_demand) > 0 else 0
                
                if region_id:
                    suggestion = f"Try removing region filter for Meal {meal_id} or select a different region"
                else:
                    suggestion = "Try a different meal with more data"
                
                return None, f"❌ Insufficient data: only {available_weeks} weeks available ({coverage:.1%} coverage). Need {min_weeks_required}+. {suggestion}"
            
            min_week = demand_data['week'].min()
            max_week = demand_data['week'].max()
            total_weeks = len(demand_data)
            coverage = total_weeks / len(self.weekly_demand)
            
            demand_values = demand_data['num_orders'].values
            variance_level = self.detect_data_characteristics(demand_values)
            
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime('2010-01-01') + pd.to_timedelta(demand_data['week'] * 7, unit='D'),
                'y': demand_data['num_orders']
            })
            
            regressors_added = []
            
            if 'checkout_price' in demand_data.columns:
                prophet_data['checkout_price'] = demand_data['checkout_price'].fillna(demand_data['checkout_price'].mean())
                regressors_added.append('checkout_price')
            
            model = self.create_robust_prophet_model(variance_level)
            
            for regressor in regressors_added:
                model.add_regressor(regressor)
            
            model.fit(prophet_data)
            self.models[model_key] = {
                'model': model,
                'regressors': regressors_added,
                'meal_id': meal_id,
                'region_id': region_id,
                'training_weeks': total_weeks,
                'coverage': coverage,
                'training_data': prophet_data,
                'original_data': demand_data,
                'variance_level': variance_level,
                'data_cv': (np.std(demand_values) / np.mean(demand_values)) * 100
            }
            
            self._generate_training_performance(model_key, model, prophet_data, demand_data)
            
            print(f"✅ Model trained for {model_key} - {total_weeks} weeks ({variance_level} variance)")
            return model, f"✅ Model trained for {model_key}"
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return None, f"Error training model: {str(e)}"
    
    def _generate_training_performance(self, model_key, model, training_data, original_demand_data):
        """Generate predictions on training data for performance visualization"""
        try:
            predictions = model.predict(training_data[['ds'] + [col for col in training_data.columns if col not in ['ds', 'y']]])
            
            actual_weeks = original_demand_data['week'].values
            actual_orders = original_demand_data['num_orders'].values
            predicted_orders = predictions['yhat'].values
            
            display_count = min(20, len(actual_weeks))
            start_idx = len(actual_weeks) - display_count
            
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
            
        except Exception as e:
            print(f"❌ Error generating training performance data: {e}")
    
    def get_training_performance(self, model_key):
        """Get training performance data for visualization"""
        normalized_key = model_key
        if normalized_key not in self.training_performance:
            alternatives = []
            if model_key.startswith('all_'):
                alternatives.append(model_key.replace('all_', 'meal_'))
            elif model_key.startswith('meal_'):
                alternatives.append(model_key.replace('meal_', 'all_'))
            
            for alt_key in alternatives:
                if alt_key in self.training_performance:
                    return self.training_performance[alt_key]
        
        return self.training_performance.get(normalized_key, None)
    
    def generate_forecast(self, model_key, weeks_ahead=10, confidence_level=0.95):
        """Generate forecast using trained Prophet model with price regressors"""
        try:
            if model_key not in self.models:
                alternatives = []
                if model_key.startswith('all_'):
                    alternatives.append(model_key.replace('all_', 'meal_'))
                elif model_key.startswith('meal_'):
                    alternatives.append(model_key.replace('meal_', 'all_'))
                
                for alt_key in alternatives:
                    if alt_key in self.models:
                        model_key = alt_key
                        break
                else:
                    return None, f"Model not found. Available: {list(self.models.keys())}"
            
            model_info = self.models[model_key]
            model = model_info['model']
            regressors = model_info['regressors']
            meal_id = model_info['meal_id']
            
            last_date = model.history['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),
                periods=weeks_ahead,
                freq='W'
            )
            
            future = pd.DataFrame({'ds': future_dates})
            
            if meal_id in self.meal_price_data:
                meal_prices = self.meal_price_data[meal_id]
                
                if 'checkout_price' in regressors:
                    future['checkout_price'] = meal_prices.get('recent_checkout_price', meal_prices.get('avg_checkout_price', 0))
            
            forecast = model.predict(future)
            
            if confidence_level != 0.8:
                from scipy import stats # type: ignore
                default_z = stats.norm.ppf(0.9)
                custom_z = stats.norm.ppf((1 + confidence_level) / 2)
                scaling_factor = custom_z / default_z
                
                default_uncertainty = forecast['yhat_upper'] - forecast['yhat']
                custom_uncertainty = default_uncertainty * scaling_factor
                forecast['yhat_upper'] = forecast['yhat'] + custom_uncertainty
                forecast['yhat_lower'] = forecast['yhat'] - custom_uncertainty
                forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
            
            self.forecasts[model_key] = {
                'forecast': forecast,
                'model_key': model_key,
                'weeks_ahead': weeks_ahead,
                'confidence_level': confidence_level,
                'meal_id': meal_id,
                'regressors_used': regressors
            }
            
            meal_name = self.get_meal_display_name(meal_id)
            return forecast, f"✅ Forecast generated for {meal_name}"
            
        except Exception as e:
            return None, f"Error generating forecast: {str(e)}"
    
    def evaluate_model_performance(self, model_key):
        """Evaluate Prophet model performance using robust metrics"""
        try:
            if model_key not in self.models:
                alternatives = []
                if model_key.startswith('all_'):
                    alternatives.append(model_key.replace('all_', 'meal_'))
                elif model_key.startswith('meal_'):
                    alternatives.append(model_key.replace('meal_', 'all_'))
                
                for alt_key in alternatives:
                    if alt_key in self.models:
                        model_key = alt_key
                        break
                else:
                    return None, f"Model not found. Available: {list(self.models.keys())}"
            
            model_info = self.models[model_key]
            model = model_info['model']
            meal_id = model_info['meal_id']
            region_id = model_info.get('region_id')
            variance_level = model_info.get('variance_level', 'medium')
            data_cv = model_info.get('data_cv', 0)
            
            training_data = model_info.get('training_data')
            original_data = model_info.get('original_data')
            
            if training_data is None or original_data is None:
                return None, "No training data available for evaluation"
            
            total_weeks = len(original_data)
            eval_size = max(5, int(total_weeks * 0.2))
            eval_indices = list(range(total_weeks - eval_size, total_weeks))
            
            eval_data_prophet = training_data.iloc[eval_indices].copy()
            eval_data_original = original_data.iloc[eval_indices].copy()
            
            if len(eval_data_prophet) == 0:
                return None, "No evaluation data available"
            
            eval_min_week = eval_data_original['week'].min()
            eval_max_week = eval_data_original['week'].max()
            
            regressors = model_info['regressors']
            for regressor in regressors:
                if regressor in eval_data_original.columns:
                    eval_data_prophet[regressor] = eval_data_original[regressor].fillna(eval_data_original[regressor].mean())
            
            predictions = model.predict(eval_data_prophet[['ds'] + regressors])
            
            y_true = eval_data_prophet['y'].values
            y_pred = np.maximum(predictions['yhat'].values, 0)
            
            robust_metrics = self.calculate_robust_metrics(y_true, y_pred)
            
            metrics = {
                **robust_metrics,
                'test_samples': len(y_true),
                'predictions_positive': int(np.sum(y_pred > 0)),
                'total_weeks_used': total_weeks,
                'evaluation_weeks': f"{eval_min_week}-{eval_max_week}",
                'meal_id': meal_id,
                'meal_name': self.get_meal_display_name(meal_id),
                'region_id': region_id,
                'regressors_used': len(regressors),
                'variance_level': variance_level,
                'data_cv': data_cv
            }
            
            self.performance_metrics[model_key] = metrics
            
            return metrics, f"Model evaluation completed for Meal {meal_id}"
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return None, f"Error evaluating model: {str(e)}"
    
    def get_data_availability(self):
        """Get data availability information for the frontend"""
        return self.data_availability
    
    def get_model_summary(self):
        """Get summary of all trained models and their performance"""
        actual_regions = sorted(set(self.regional_mapping.values())) if self.regional_mapping else ['Region-Kassel', 'Region-Wien', 'Region-Luzern']
        
        meals_list = []
        if self.train_data is not None and 'meal_id' in self.train_data.columns:
            unique_meals = sorted(self.train_data['meal_id'].unique())
            for meal_id in unique_meals:
                meals_list.append({
                    'id': meal_id,
                    'name': self.get_meal_display_name(meal_id),
                    'display': f"{meal_id}: {self.get_meal_display_name(meal_id)}"
                })
        
        summary = {
            'models_trained': len(self.models),
            'models': {},
            'data_info': {
                'train_records': len(self.train_data) if self.train_data is not None else 0,
                'test_records': len(self.test_data) if self.test_data is not None else 0,
                'regions': actual_regions,
                'meals': [meal['id'] for meal in meals_list],
                'meals_with_names': meals_list,
                'weeks_range': [
                    int(self.train_data['week'].min()) if self.train_data is not None and len(self.train_data) > 0 and not pd.isna(self.train_data['week'].min()) else 0,
                    int(self.train_data['week'].max()) if self.train_data is not None and len(self.train_data) > 0 and not pd.isna(self.train_data['week'].max()) else 0
                ],
                'has_price_data': bool(self.column_mapping.get('checkout_price')),
                'price_regressors_available': ['checkout_price'] if self.column_mapping.get('checkout_price') else [],
                'business_cost_analysis_enabled': len(self.meal_database) > 0,
                'excluded_regions': ['Region-Lyon']
            },
            'data_availability': self.data_availability
        }
        
        for key in self.models.keys():
            model_info = self.models[key]
            summary['models'][key] = {
                'performance': self.performance_metrics.get(key, {}),
                'has_forecast': key in self.forecasts,
                'meal_id': model_info.get('meal_id'),
                'meal_name': self.get_meal_display_name(model_info.get('meal_id')) if model_info.get('meal_id') else None,
                'region_id': model_info.get('region_id'),
                'regressors': model_info.get('regressors', []),
                'training_weeks': model_info.get('training_weeks', 0),
                'coverage': model_info.get('coverage', 0)
            }
        
        return summary
    
    def create_naive_baseline_model(self, model_key):
        """Create a naive baseline model that uses 'same week last year' predictions"""
        try:
            if model_key not in self.models:
                return None, "Prophet model not found for baseline comparison"
            
            model_info = self.models[model_key]
            original_data = model_info.get('original_data')
            
            if original_data is None:
                return None, "No original training data available for baseline"
            
            sorted_data = original_data.sort_values('week').reset_index(drop=True)
            weeks = sorted_data['week'].values
            demands = sorted_data['num_orders'].values
            
            weeks_per_year = 52
            
            total_weeks = len(sorted_data)
            eval_size = max(5, int(total_weeks * 0.2))
            eval_start_idx = total_weeks - eval_size
            
            baseline_predictions = []
            baseline_weeks = []
            
            for i in range(eval_start_idx, total_weeks):
                current_week = weeks[i]
                target_week = current_week - weeks_per_year
                
                week_diffs = np.abs(weeks[:i] - target_week)
                
                if len(week_diffs) > 0:
                    closest_idx = np.argmin(week_diffs)
                    closest_week = weeks[closest_idx]
                    baseline_pred = demands[closest_idx]
                    
                    if abs(closest_week - target_week) <= 4:
                        baseline_predictions.append(baseline_pred)
                    else:
                        recent_avg = np.mean(demands[max(0, i-12):i])
                        baseline_predictions.append(recent_avg)
                else:
                    baseline_predictions.append(np.mean(demands[:i]) if i > 0 else demands[0])
                
                baseline_weeks.append(current_week)
            
            baseline_model = {
                'predictions': np.array(baseline_predictions),
                'weeks': np.array(baseline_weeks),
                'eval_start_idx': eval_start_idx,
                'method': 'same_week_last_year',
                'weeks_per_year': weeks_per_year
            }
            
            self.baseline_models = getattr(self, 'baseline_models', {})
            self.baseline_models[model_key] = baseline_model
            
            return baseline_model, "Baseline model created"
            
        except Exception as e:
            print(f"❌ Error creating baseline model: {e}")
            return None, f"Error creating baseline: {str(e)}"

    def evaluate_baseline_vs_prophet(self, model_key):
        """Compare Prophet model performance against naive baseline"""
        try:
            if model_key not in self.models:
                return None, "Prophet model not found"
            
            if not hasattr(self, 'baseline_models') or model_key not in self.baseline_models:
                baseline_model, message = self.create_naive_baseline_model(model_key)
                if baseline_model is None:
                    return None, message
            
            model_info = self.models[model_key]
            original_data = model_info.get('original_data')
            training_data = model_info.get('training_data')
            
            if original_data is None or training_data is None:
                return None, "No evaluation data available"
            
            baseline_model = self.baseline_models[model_key]
            baseline_predictions = baseline_model['predictions']
            eval_weeks = baseline_model['weeks']
            
            total_weeks = len(original_data)
            eval_size = len(baseline_predictions)
            eval_start_idx = total_weeks - eval_size
            
            eval_data_prophet = training_data.iloc[eval_start_idx:eval_start_idx + eval_size].copy()
            eval_data_original = original_data.iloc[eval_start_idx:eval_start_idx + eval_size].copy()
            
            regressors = model_info['regressors']
            for regressor in regressors:
                if regressor in eval_data_original.columns:
                    eval_data_prophet[regressor] = eval_data_original[regressor].fillna(eval_data_original[regressor].mean())
            
            prophet_model = model_info['model']
            prophet_predictions = prophet_model.predict(eval_data_prophet[['ds'] + regressors])
            prophet_pred_values = np.maximum(prophet_predictions['yhat'].values, 0)
            
            actual_values = eval_data_prophet['y'].values
            
            prophet_metrics = self.calculate_robust_metrics(actual_values, prophet_pred_values)
            baseline_metrics = self.calculate_robust_metrics(actual_values, baseline_predictions)
            
            accuracy_improvement = prophet_metrics['accuracy'] - baseline_metrics['accuracy']
            mae_improvement = ((baseline_metrics['mae'] - prophet_metrics['mae']) / baseline_metrics['mae']) * 100
            
            comparison_data = {
                'weeks': [f"Week {int(w)}" for w in eval_weeks],
                'week_numbers': eval_weeks.tolist(),
                'actual': actual_values.round().astype(int).tolist(),
                'prophet_predicted': prophet_pred_values.round().astype(int).tolist(),
                'baseline_predicted': baseline_predictions.round().astype(int).tolist(),
                'prophet_upper': prophet_predictions['yhat_upper'].round().astype(int).tolist(),
                'prophet_lower': prophet_predictions['yhat_lower'].round().astype(int).tolist(),
                'prophet_metrics': prophet_metrics,
                'baseline_metrics': baseline_metrics,
                'accuracy_improvement': accuracy_improvement,
                'mae_improvement': mae_improvement,
                'eval_period': f"{int(eval_weeks.min())}-{int(eval_weeks.max())}",
                'meal_id': model_info.get('meal_id'),
                'region_id': model_info.get('region_id')
            }
            
            self.model_comparisons = getattr(self, 'model_comparisons', {})
            self.model_comparisons[model_key] = comparison_data
            
            return comparison_data, f"Model comparison completed"
            
        except Exception as e:
            print(f"❌ Error in baseline comparison: {e}")
            return None, f"Error comparing models: {str(e)}"

    def get_model_comparison(self, model_key):
        """Get stored model comparison data"""
        if not hasattr(self, 'model_comparisons') or model_key not in self.model_comparisons:
            comparison_data, message = self.evaluate_baseline_vs_prophet(model_key)
            return comparison_data
        
        return self.model_comparisons.get(model_key, None)