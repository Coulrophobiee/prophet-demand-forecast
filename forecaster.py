#!/usr/bin/env python3
"""
Food Demand Forecaster - Prophet Model Implementation with Business Cost Analysis
Enhanced version with Euro-based cost evaluation for business impact assessment
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from prophet import Prophet # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
import warnings
import os
import json

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
        self.meal_names = {}  # Store meal names database
        self.meal_database = {}  # Store meal ingredients and costs
        self.business_costs = {}  # Store business cost analysis
        
    def generate_meal_database(self):
        """Generate AI-like meal database with ingredients and cost structure"""
        print("🍽️ Generating meal database with ingredients and cost structure...")
        
        # Define ingredient categories
        perishable_ingredients = [
            {'name': 'Fresh Chicken Breast', 'cost_percentage': 35, 'kg_per_portion': 0.15, 'perishable': True},
            {'name': 'Fresh Salmon Fillet', 'cost_percentage': 45, 'kg_per_portion': 0.12, 'perishable': True},
            {'name': 'Fresh Vegetables Mix', 'cost_percentage': 20, 'kg_per_portion': 0.20, 'perishable': True},
            {'name': 'Fresh Herbs', 'cost_percentage': 8, 'kg_per_portion': 0.02, 'perishable': True},
            {'name': 'Fresh Dairy Cream', 'cost_percentage': 15, 'kg_per_portion': 0.05, 'perishable': True},
            {'name': 'Fresh Mushrooms', 'cost_percentage': 12, 'kg_per_portion': 0.08, 'perishable': True},
            {'name': 'Fresh Tomatoes', 'cost_percentage': 10, 'kg_per_portion': 0.10, 'perishable': True},
            {'name': 'Fresh Lettuce', 'cost_percentage': 6, 'kg_per_portion': 0.05, 'perishable': True}
        ]
        
        non_perishable_ingredients = [
            {'name': 'Rice', 'cost_percentage': 15, 'kg_per_portion': 0.08, 'perishable': False},
            {'name': 'Pasta', 'cost_percentage': 12, 'kg_per_portion': 0.09, 'perishable': False},
            {'name': 'Olive Oil', 'cost_percentage': 8, 'kg_per_portion': 0.01, 'perishable': False},
            {'name': 'Spices Mix', 'cost_percentage': 5, 'kg_per_portion': 0.005, 'perishable': False},
            {'name': 'Canned Tomatoes', 'cost_percentage': 10, 'kg_per_portion': 0.12, 'perishable': False},
            {'name': 'Quinoa', 'cost_percentage': 18, 'kg_per_portion': 0.06, 'perishable': False},
            {'name': 'Dried Beans', 'cost_percentage': 8, 'kg_per_portion': 0.07, 'perishable': False},
            {'name': 'Flour', 'cost_percentage': 6, 'kg_per_portion': 0.05, 'perishable': False},
            {'name': 'Coconut Milk', 'cost_percentage': 12, 'kg_per_portion': 0.04, 'perishable': False}
        ]
        
        all_ingredients = perishable_ingredients + non_perishable_ingredients
        
        # Generate meals from existing meal IDs in the dataset
        if self.train_data is not None and 'meal_id' in self.train_data.columns:
            unique_meals = sorted(self.train_data['meal_id'].unique())
            
            for meal_id in unique_meals:
                # Generate 5 random ingredients for each meal
                selected_ingredients = random.sample(all_ingredients, 5)
                
                # Normalize cost percentages to sum to 100%
                total_percentage = sum(ing['cost_percentage'] for ing in selected_ingredients)
                normalized_ingredients = []
                
                for ing in selected_ingredients:
                    normalized_ing = ing.copy()
                    normalized_ing['cost_percentage'] = (ing['cost_percentage'] / total_percentage) * 100
                    normalized_ingredients.append(normalized_ing)
                
                # Calculate perishable and non-perishable percentages
                perishable_percentage = sum(ing['cost_percentage'] for ing in normalized_ingredients if ing['perishable'])
                non_perishable_percentage = sum(ing['cost_percentage'] for ing in normalized_ingredients if not ing['perishable'])
                total_kg_per_portion = sum(ing['kg_per_portion'] for ing in normalized_ingredients)
                
                # Generate meal name if not available
                meal_name = self.get_meal_display_name(meal_id)
                if meal_name == f"Meal {meal_id}":
                    # Generate a more descriptive name based on main ingredients
                    main_protein = next((ing['name'] for ing in normalized_ingredients if 'Chicken' in ing['name'] or 'Salmon' in ing['name']), 'Protein')
                    main_carb = next((ing['name'] for ing in normalized_ingredients if ing['name'] in ['Rice', 'Pasta', 'Quinoa']), 'Rice')
                    meal_name = f"{main_protein.replace('Fresh ', '')} with {main_carb}"
                
                # Store with both string and int keys for compatibility
                meal_data = {
                    'name': meal_name,
                    'ingredients': normalized_ingredients,
                    'total_kg_per_portion': total_kg_per_portion,
                    'perishable_percentage': perishable_percentage,
                    'non_perishable_percentage': non_perishable_percentage
                }
                
                # Store with multiple key formats for robustness
                self.meal_database[str(meal_id)] = meal_data
                self.meal_database[int(meal_id)] = meal_data
                self.meal_database[meal_id] = meal_data
            
            print(f"✅ Generated meal database for {len(unique_meals)} meals")
            print(f"📊 Average ingredients per meal: 5")
            print(f"🥗 Perishable ingredients: {len(perishable_ingredients)}")
            print(f"🍚 Non-perishable ingredients: {len(non_perishable_ingredients)}")
            
            # Debug: Print a few examples
            sample_ids = list(unique_meals)[:3]
            for sample_id in sample_ids:
                sample_data = self.meal_database[sample_id]
                print(f"   • Meal {sample_id}: {sample_data['perishable_percentage']:.1f}% perishable, {sample_data['total_kg_per_portion']:.2f}kg/portion")
            
            # Save meal database to JSON file for persistence
            try:
                # Convert keys to strings for JSON compatibility
                json_compatible_db = {}
                for meal_id in unique_meals:
                    json_compatible_db[str(meal_id)] = self.meal_database[meal_id]
                
                with open('meal_database.json', 'w', encoding='utf-8') as f:
                    json.dump(json_compatible_db, f, indent=2, ensure_ascii=False)
                print(f"💾 Saved meal database to meal_database.json")
            except Exception as e:
                print(f"⚠️ Could not save meal database: {e}")
                
        else:
            print("❌ No meal data available for database generation")
    
    def load_meal_names(self):
        """Load meal names and data from existing JSON file"""
        try:
            if os.path.exists('meal_database.json'):
                with open('meal_database.json', 'r', encoding='utf-8') as f:
                    raw_meal_data = json.load(f)
                
                # Process the existing meal database format
                self.meal_database = {}
                self.meal_names = {}
                
                for meal_id_str, meal_info in raw_meal_data.items():
                    meal_id = int(meal_id_str)
                    meal_name = meal_info.get('name', f'Meal {meal_id}')
                    
                    # Calculate perishable percentage from price_distribution and perishable arrays
                    price_dist = meal_info.get('price_distribution', [])
                    perishable_flags = meal_info.get('perishable', [])
                    
                    perishable_percentage = 0.0
                    if len(price_dist) == len(perishable_flags):
                        for i, is_perishable in enumerate(perishable_flags):
                            if is_perishable and i < len(price_dist):
                                perishable_percentage += price_dist[i]
                    
                    # Calculate total kg per portion from kg_per_10_persons
                    kg_per_10 = meal_info.get('kg_per_10_persons', [])
                    total_kg_per_portion = sum(kg_per_10) / 10.0 if kg_per_10 else 0.5
                    
                    # Store processed meal data with multiple key formats
                    processed_meal_data = {
                        'name': meal_name,
                        'ingredients': meal_info.get('ingredients', []),
                        'perishable_percentage': perishable_percentage,
                        'non_perishable_percentage': 100.0 - perishable_percentage,
                        'total_kg_per_portion': total_kg_per_portion,
                        'type': meal_info.get('type', 'Unknown'),
                        'price_tier': meal_info.get('price_tier', 'Standard'),
                        'original_data': meal_info  # Keep original for reference
                    }
                    
                    # Store with multiple key formats for robustness
                    self.meal_database[str(meal_id)] = processed_meal_data
                    self.meal_database[int(meal_id)] = processed_meal_data
                    self.meal_database[meal_id] = processed_meal_data
                    
                    # Store meal name
                    self.meal_names[meal_id] = meal_name
                
                print(f"🍽️ Loaded {len(self.meal_names)} meals from existing meal_database.json")
                print(f"📊 Processed meal data with perishable percentages and portion weights")
                
                # Show some examples
                sample_meals = list(self.meal_names.items())[:3]
                for meal_id, name in sample_meals:
                    meal_data = self.meal_database[meal_id]
                    print(f"   • {meal_id}: {name}")
                    print(f"     └ {meal_data['perishable_percentage']:.1f}% perishable, {meal_data['total_kg_per_portion']:.2f}kg/portion, {meal_data['type']}")
                if len(self.meal_names) > 3:
                    print(f"   • ... and {len(self.meal_names) - 3} more")
                
        except Exception as e:
            print(f"❌ Error loading meal database: {e}")
            print(f"🔄 Falling back to generated meal database")
            self.meal_names = {}
            self.meal_database = {}
            if self.train_data is not None:
                self.generate_meal_database()
    
    def get_meal_display_name(self, meal_id):
        """Get display name for a meal (name if available, otherwise ID)"""
        try:
            meal_id_int = int(meal_id)
            return self.meal_names.get(meal_id_int, f"Meal {meal_id}")
        except:
            return f"Meal {meal_id}"
        
    def calculate_business_cost(self, actual_orders, predicted_orders, meal_id, avg_meal_price=15.0):
        """
        Calculate business costs for given actual vs predicted orders
        
        Parameters:
        - actual_orders: actual number of orders
        - predicted_orders: predicted number of orders  
        - meal_id: meal identifier
        - avg_meal_price: average price per meal in Euro
        
        Returns:
        - Dictionary with detailed cost breakdown
        """
        try:
            # Cost parameters (as percentages of revenue)
            INGREDIENT_COST_RATE = 0.40  # 40% of revenue
            STORAGE_COST_RATE = 0.15     # 15% of revenue
            EMERGENCY_COST_RATE = 0.60   # 60% of revenue (150% of normal 40%)
            OPPORTUNITY_COST_RATE = 0.02  # 2% opportunity cost for tied-up capital
            
            # Get meal information with proper key handling
            meal_info = None
            for key_variant in [str(meal_id), int(meal_id), meal_id]:
                if key_variant in self.meal_database:
                    meal_info = self.meal_database[key_variant]
                    break
            
            if not meal_info:
                print(f"❌ Meal {meal_id} not found in database")
                return None
                
            meal_name = meal_info.get('name', f"Meal {meal_id}")
            meal_type = meal_info.get('type', 'Unknown')
            
            # Extract ingredient data from database
            original_data = meal_info.get('original_data', meal_info)
            ingredients = original_data.get('ingredients', [])
            kg_per_10_persons = original_data.get('kg_per_10_persons', [])
            perishable = original_data.get('perishable', [])
            price_distribution = original_data.get('price_distribution', [])
            storage_m3 = original_data.get('storage_m3', [])
            print(f"🔍 Debugging meal {meal_id}:")
            print(f"   ingredients: {len(ingredients)} elements")
            print(f"   kg_per_10_persons: {len(kg_per_10_persons)} elements") 
            print(f"   perishable: {len(perishable)} elements")
            print(f"   price_distribution: {len(price_distribution)} elements")
            print(f"   storage_m3: {len(storage_m3)} elements")
            print(f"   All arrays: {[len(ingredients), len(kg_per_10_persons), len(perishable), len(price_distribution), len(storage_m3)]}")

            
            # Validate arrays have same length
            if not all(len(arr) == len(ingredients) for arr in [kg_per_10_persons, perishable, price_distribution, storage_m3]):
                print(f"❌ Inconsistent ingredient data arrays for meal {meal_id}")
                return None
                
            
            # Calculate base costs
            ingredient_cost_per_order = avg_meal_price * INGREDIENT_COST_RATE
            storage_cost_per_order = avg_meal_price * STORAGE_COST_RATE
            
            # Calculate prediction difference
            difference = predicted_orders - actual_orders
            
            overprediction_penalty = 0.0
            underprediction_penalty = 0.0
            
            if difference > 0:  # Overprediction
                excess_orders = difference
                print(f"   📦 Overprediction: {excess_orders:.0f} orders")
                
                storage_cost = excess_orders * (avg_meal_price * STORAGE_COST_RATE)
                opportunity_cost = excess_orders * (avg_meal_price * OPPORTUNITY_COST_RATE)
                overprediction_penalty += storage_cost + opportunity_cost

                # Process each ingredient individually
                for i, ingredient_name in enumerate(ingredients):

                    is_perishable = perishable[i]

                    if is_perishable:
                        cost_percentage = price_distribution[i] / 100  # Convert to decimal
                        ingredient_cost = excess_orders * avg_meal_price * INGREDIENT_COST_RATE * cost_percentage
                        overprediction_penalty += ingredient_cost
                                        
            elif difference < 0:  # Underprediction
                shortage_orders = abs(difference)
                
                # Emergency procurement at 150% cost (60% instead of 40%)
                underprediction_penalty = shortage_orders * ((avg_meal_price * INGREDIENT_COST_RATE) / 2)
                
                print(f"   📦 Underprediction: {shortage_orders:.0f} orders")
                print(f"   🚨 Emergency cost penalty: €{underprediction_penalty:.2f}")
            
            total_penalty = overprediction_penalty + underprediction_penalty
            
            # Calculate revenue impact
            total_revenue = actual_orders * avg_meal_price
            penalty_percentage = (total_penalty / total_revenue * 100) if total_revenue > 0 else 0
            
            # Calculate meal-level metrics for reporting
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
            import traceback
            traceback.print_exc()
            return None

    
    def evaluate_business_cost(self, model_key, custom_meal_price=None):
        """
        Evaluate business costs for a trained model using test weeks
        
        Parameters:
        - model_key: identifier for the trained model
        - custom_meal_price: optional custom meal price in Euro
        
        Returns:
        - Dictionary with business cost analysis
        """
        try:
            print(f"💰 Evaluating business costs for model: {model_key}")
            
            # Find the model
            if model_key not in self.models:
                # Try alternative key formats
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
            
            # Get the comparison data (same period used for Prophet vs Baseline comparison)
            comparison_data = self.get_model_comparison(model_key)
            if not comparison_data:
                return None, "No comparison data available for business cost analysis"
            
            # Use custom price or estimate from data
            if custom_meal_price:
                avg_meal_price = custom_meal_price
            else:
                # Estimate meal price (assume €15 average)
                avg_meal_price = 15.0
                if meal_id in self.meal_price_data and 'avg_checkout_price' in self.meal_price_data[meal_id]:
                    avg_meal_price = self.meal_price_data[meal_id]['avg_checkout_price']
            
            print(f"💵 Using meal price: €{avg_meal_price:.2f}")
            
            # Calculate costs for each week in the comparison period
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
            
            # Calculate summary metrics
            avg_weekly_penalty = total_penalty / len(weekly_costs) if weekly_costs else 0
            penalty_percentage_of_revenue = (total_penalty / total_revenue * 100) if total_revenue > 0 else 0
            
            # Project annual impact (52 weeks)
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
            
            print(f"✅ Business cost analysis completed")
            print(f"💰 Total penalty over {len(weekly_costs)} weeks: €{total_penalty:.2f}")
            print(f"📊 Penalty as % of revenue: {penalty_percentage_of_revenue:.2f}%")
            print(f"📅 Estimated annual penalty: €{annual_penalty_estimate:.2f}")
            
            return business_cost_analysis, f"Business cost analysis completed for {meal_name}"
            
        except Exception as e:
            print(f"❌ Error in business cost evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error evaluating business costs: {str(e)}"
    
    def compare_business_costs(self, model_key):
        """
        Compare business costs between Prophet and Baseline models
        
        Parameters:
        - model_key: identifier for the trained model
        
        Returns:
        - Dictionary with comparative business cost analysis
        """
        try:
            print(f"💰 Comparing business costs: Prophet vs Baseline for {model_key}")
            
            # Get comparison data
            comparison_data = self.get_model_comparison(model_key)
            if not comparison_data:
                return None, "No comparison data available for business cost analysis"
            
            model_info = self.models[model_key]
            meal_id = model_info['meal_id']
            
            print(f"🍽️ Analyzing meal {meal_id} business costs")
            
            # Check if meal exists in database
            meal_in_db = False
            for key_variant in [str(meal_id), int(meal_id), meal_id]:
                if key_variant in self.meal_database:
                    meal_in_db = True
                    break
            
            if meal_in_db:
                print(f"✅ Meal {meal_id} found in meal database")
            else:
                print(f"⚠️ Meal {meal_id} not found in meal database - will use defaults")
            
            # Use checkout price if available, otherwise default to €15
            avg_meal_price = 15.0
            if meal_id in self.meal_price_data and 'avg_checkout_price' in self.meal_price_data[meal_id]:
                avg_meal_price = self.meal_price_data[meal_id]['avg_checkout_price']
                print(f"💵 Using meal-specific price: €{avg_meal_price:.2f}")
            else:
                print(f"💵 Using default price: €{avg_meal_price:.2f}")
            
            # Get data for comparison
            actual_orders = comparison_data['actual']
            prophet_predictions = comparison_data['prophet_predicted']
            baseline_predictions = comparison_data['baseline_predicted']
            weeks = comparison_data['weeks']
            
            print(f"📊 Processing {len(weeks)} weeks of comparison data")
            
            # Calculate costs for both models
            prophet_weekly_penalties = []
            baseline_weekly_penalties = []
            weekly_revenues = []
            
            weekly_comparison = []
            successful_weeks = 0
            
            for i, week in enumerate(weeks):
                actual = actual_orders[i]
                prophet_pred = prophet_predictions[i]
                baseline_pred = baseline_predictions[i]
                
                # Calculate Prophet costs
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
                else:
                    print(f"⚠️ Could not calculate costs for week {week}")
            
            print(f"✅ Successfully processed {successful_weeks}/{len(weeks)} weeks")
            
            if not weekly_comparison:
                return None, "Could not calculate business costs for any weeks"
            
            # Calculate weekly averages (as requested)
            avg_weekly_prophet_penalty = sum(prophet_weekly_penalties) / len(prophet_weekly_penalties)
            avg_weekly_baseline_penalty = sum(baseline_weekly_penalties) / len(baseline_weekly_penalties)
            avg_weekly_savings = avg_weekly_baseline_penalty - avg_weekly_prophet_penalty
            avg_weekly_revenue = sum(weekly_revenues) / len(weekly_revenues)
            
            # Calculate totals for test period
            prophet_total_penalty = sum(prophet_weekly_penalties)
            baseline_total_penalty = sum(baseline_weekly_penalties)
            cost_savings_euro = baseline_total_penalty - prophet_total_penalty
            total_revenue = sum(weekly_revenues)
            
            # Calculate percentages
            cost_savings_percentage = (cost_savings_euro / baseline_total_penalty * 100) if baseline_total_penalty > 0 else 0
            
            # Project annual impact based on weekly averages
            annual_savings_estimate = avg_weekly_savings * 52
            annual_revenue_estimate = avg_weekly_revenue * 52
            annual_savings_percentage = (annual_savings_estimate / annual_revenue_estimate * 100) if annual_revenue_estimate > 0 else 0
            
            # Determine ROI calculation
            # Assume Prophet implementation cost of €50,000 annually
            prophet_implementation_cost = 50000
            roi_ratio = annual_savings_estimate / prophet_implementation_cost if prophet_implementation_cost > 0 else 0
            payback_months = (prophet_implementation_cost / avg_weekly_savings * (1/4.33)) if avg_weekly_savings > 0 else float('inf')  # 4.33 weeks per month
            
            meal_name = self.get_meal_display_name(meal_id)
            
            business_comparison = {
                'model_key': model_key,
                'meal_id': meal_id,
                'meal_name': meal_name,
                'evaluation_period': comparison_data['eval_period'],
                'weeks_analyzed': len(weekly_comparison),
                'avg_meal_price_euro': avg_meal_price,
                'weekly_comparison': weekly_comparison,
                
                # Total costs for test period
                'prophet_total_penalty': prophet_total_penalty,
                'baseline_total_penalty': baseline_total_penalty,
                'cost_savings_euro': cost_savings_euro,
                'cost_savings_percentage': cost_savings_percentage,
                'total_revenue_euro': total_revenue,
                
                # Weekly averages (as requested)
                'avg_weekly_savings_euro': avg_weekly_savings,
                'avg_weekly_prophet_penalty': avg_weekly_prophet_penalty,
                'avg_weekly_baseline_penalty': avg_weekly_baseline_penalty,
                'avg_weekly_revenue': avg_weekly_revenue,
                
                # Annual projections based on weekly averages
                'annual_savings_estimate_euro': annual_savings_estimate,
                'annual_revenue_estimate_euro': annual_revenue_estimate,
                'annual_savings_percentage': annual_savings_percentage,
                
                # ROI analysis
                'prophet_implementation_cost_euro': prophet_implementation_cost,
                'roi_ratio': roi_ratio,
                'payback_months': min(payback_months, 999),  # Cap at 999 months for display
                
                # Model performance context
                'prophet_accuracy': comparison_data['prophet_metrics']['accuracy'],
                'baseline_accuracy': comparison_data['baseline_metrics']['accuracy'],
                'accuracy_improvement': comparison_data['accuracy_improvement'],
                
                # Business recommendation
                'business_recommendation': {
                    'implement_prophet': cost_savings_euro > 0,
                    'savings_per_euro_invested': roi_ratio,
                    'payback_period_months': min(payback_months, 999),
                    'annual_impact_estimate': annual_savings_estimate,
                    'confidence_level': 'High' if comparison_data['accuracy_improvement'] > 5 else 'Medium' if comparison_data['accuracy_improvement'] > 0 else 'Low'
                }
            }
            
            print(f"✅ Business cost comparison completed")
            print(f"💰 Average weekly Prophet penalty: €{avg_weekly_prophet_penalty:.2f}")
            print(f"📊 Average weekly Baseline penalty: €{avg_weekly_baseline_penalty:.2f}")
            print(f"💡 Average weekly savings with Prophet: €{avg_weekly_savings:.2f}")
            print(f"📅 Estimated annual savings: €{annual_savings_estimate:.2f}")
            print(f"📈 ROI ratio: {roi_ratio:.2f}x")
            print(f"⏱️ Payback period: {payback_months:.1f} months")
            
            return business_comparison, f"Business cost comparison completed for {meal_name}"
            
        except Exception as e:
            print(f"❌ Error in business cost comparison: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error comparing business costs: {str(e)}"
        
    def load_data(self):
        """Load train and test datasets"""
        try:
            # Load meal names and database first
            self.load_meal_names()
            
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
            
            # Validate that we have meal database
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
        """Create robust regional mapping for centers - excluding Region-Lyon"""
        if self.train_data is None:
            return {}
        
        regional_mapping = {}
        
        try:
            if os.path.exists('fulfilment_center_info.csv'):
                print("📍 Loading regional mapping from fulfilment_center_info.csv")
                center_info = pd.read_csv('fulfilment_center_info.csv')
                
                print(f"📋 Fulfilment center data loaded: {len(center_info)} centers")
                print(f"📋 Columns: {list(center_info.columns)}")
                
                # Create mapping based on region_code - EXCLUDING Region-Lyon (85)
                for _, row in center_info.iterrows():
                    center_id = row['center_id']
                    region_code = row['region_code']
                    
                    # Map region codes to standardized region names - SKIP Lyon
                    if region_code == 56:
                        final_region = "Region-Kassel"
                    elif region_code == 34:
                        final_region = "Region-Luzern"  
                    elif region_code == 77:
                        final_region = "Region-Wien"
                    elif region_code == 85:
                        # SKIP Region-Lyon completely
                        print(f"   ⚠️ Skipping center {center_id} from Region-Lyon (poor data quality)")
                        continue
                    else:
                        # Any other region codes also get assigned to a valid region
                        final_region = "Region-Kassel"  # Default fallback
                    
                    regional_mapping[center_id] = final_region
                
                # Print detailed mapping
                print(f"🗺️ Regional mapping created (excluding Region-Lyon):")
                
                # Group by final region for summary
                region_summary = {}
                for center_id, region in regional_mapping.items():
                    if region not in region_summary:
                        region_summary[region] = []
                    region_summary[region].append(center_id)
                
                for region, centers in sorted(region_summary.items()):
                    print(f"   {region}: Centers {sorted(centers)} ({len(centers)} centers)")
                
                # Check data availability per region
                train_centers = set(self.train_data['center_id'].unique())
                mapped_centers = set(regional_mapping.keys())
                excluded_centers = train_centers - mapped_centers
                
                if excluded_centers:
                    print(f"⚠️ Excluded centers (Region-Lyon): {sorted(excluded_centers)}")
                
                # Check actual data availability per region
                print(f"\n📊 Data availability analysis per region (excluding Lyon):")
                for region in sorted(set(regional_mapping.values())):
                    region_centers = [c for c, r in regional_mapping.items() if r == region]
                    region_data = self.train_data[self.train_data['center_id'].isin(region_centers)]
                    
                    if len(region_data) > 0:
                        unique_meals = len(region_data['meal_id'].unique()) if 'meal_id' in region_data.columns else 0
                        total_records = len(region_data)
                        week_range = f"{region_data['week'].min()}-{region_data['week'].max()}" if 'week' in region_data.columns else "N/A"
                        print(f"   ✅ {region}: {total_records:,} records, {unique_meals} meals, weeks {week_range}")
                    else:
                        print(f"   ❌ {region}: NO DATA AVAILABLE")
                
                print(f"✅ Final regional mapping: {len(regional_mapping)} centers across {len(set(regional_mapping.values()))} regions")
                print(f"🚫 Region-Lyon excluded due to data quality issues")
                
            else:
                print("📍 fulfilment_center_info.csv not found, using automatic regional mapping")
                centers = sorted(self.train_data['center_id'].unique())
                total_centers = len(centers)
                
                print(f"📍 Found {total_centers} centers: {centers}")
                
                # Divide centers into 3 regions (excluding Lyon)
                centers_per_region = total_centers // 3
                remainder = total_centers % 3
                
                current_index = 0
                region_names = ["Region-Kassel", "Region-Wien", "Region-Luzern"]  # Removed Lyon
                
                for i, region_name in enumerate(region_names):
                    region_size = centers_per_region + (1 if i < remainder else 0)
                    region_centers = centers[current_index:current_index + region_size]
                    
                    for center in region_centers:
                        regional_mapping[center] = region_name
                    
                    current_index += region_size
                    print(f"🗺️ {region_name}: Centers {region_centers} ({len(region_centers)} centers)")
                
        except Exception as e:
            print(f"❌ Error loading fulfilment center info: {e}")
            print("📍 Falling back to minimal regional mapping (excluding Lyon)")
            
            # Fallback: simple mapping without Lyon
            centers = sorted(self.train_data['center_id'].unique())
            for i, center in enumerate(centers):
                region_names = ["Region-Kassel", "Region-Wien", "Region-Luzern"]  # No Lyon
                regional_mapping[center] = region_names[i % 3]
        
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

    def filter_meals(self):
        """Remove meals with insufficient coverage"""
        meals_to_remove = [1571, 2104, 2956, 2490, 2569, 2664]
        
        if self.train_data is not None and 'meal_id' in self.train_data.columns:
            initial_meals = set(self.train_data['meal_id'].unique())
            initial_count = len(self.train_data)
            
            print(f"🍽️ Filtering meals with insufficient coverage...")
            print(f"📊 Initial meals: {len(initial_meals)} total")
            print(f"📊 Initial records: {initial_count:,}")
            
            # Filter out specified meals
            meals_found = [meal for meal in meals_to_remove if meal in initial_meals]
            meals_not_found = [meal for meal in meals_to_remove if meal not in initial_meals]
            
            if meals_found:
                print(f"🗑️ Removing meals: {meals_found}")
                self.train_data = self.train_data[~self.train_data['meal_id'].isin(meals_to_remove)]
                
                # Also filter test data if it exists
                if self.test_data is not None and 'meal_id' in self.test_data.columns:
                    self.test_data = self.test_data[~self.test_data['meal_id'].isin(meals_to_remove)]
                
                final_meals = set(self.train_data['meal_id'].unique())
                final_count = len(self.train_data)
                removed_count = initial_count - final_count
                
                print(f"✅ Filtering complete:")
                print(f"   • Meals removed: {len(meals_found)}")
                print(f"   • Records removed: {removed_count:,}")
                print(f"   • Remaining meals: {len(final_meals)}")
                print(f"   • Remaining records: {final_count:,}")
                print(f"   • Data reduction: {(removed_count/initial_count)*100:.1f}%")
                
            else:
                print(f"ℹ️ No meals to remove found in dataset")
            
            if meals_not_found:
                print(f"⚠️ Meals not found in dataset: {meals_not_found}")
        else:
            print(f"⚠️ No meal data available for filtering")

    def preprocess_data(self):
        """Preprocess data for Prophet model with checkout_price regressor - excluding Region-Lyon"""
        if self.train_data is None:
            return
        
        # Filter meals with insufficient coverage FIRST
        self.filter_meals()
        
        # Create regional mapping (excludes Lyon)
        self.regional_mapping = self.create_regional_mapping()
        
        # Add region column and filter out unmapped centers (Lyon centers)
        self.train_data['region'] = self.train_data['center_id'].map(self.regional_mapping)
        
        # Remove rows where region is NaN (these are Lyon centers that were excluded)
        initial_count = len(self.train_data)
        self.train_data = self.train_data.dropna(subset=['region'])
        excluded_count = initial_count - len(self.train_data)
        
        if excluded_count > 0:
            print(f"🚫 Excluded {excluded_count:,} records from Region-Lyon (poor data quality)")
            print(f"✅ Remaining data: {len(self.train_data):,} records")
        
        if self.test_data is not None:
            self.test_data['region'] = self.test_data['center_id'].map(self.regional_mapping)
            self.test_data = self.test_data.dropna(subset=['region'])
        
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
            print(f"🗺️ Regional aggregation complete with {len(self.regional_mapping)} centers in {len(set(self.regional_mapping.values()))} regions")
            print(f"🍽️ Meal-specific forecasting enabled with checkout_price regressor")
            print(f"🚫 Region-Lyon excluded from all analysis")
            
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
        """Analyze data availability for meal-specific forecasting with improved regional analysis"""
        print("\n🔍 Analyzing data availability for meal-specific forecasting...")
        
        if self.train_aggregated is None or 'meal_id' not in self.train_aggregated.columns:
            print("❌ No meal data available for analysis")
            return
        
        availability = {}
        min_weeks_required = 20
        
        # Get total weeks for coverage calculation
        overall_weeks = len(self.weekly_demand)
        
        # Check each meal individually (all regions)
        print(f"\n📊 Analyzing meal availability across all regions:")
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
                
                status = "✅" if weeks_count >= min_weeks_required else "❌"
                print(f"   {status} Meal {meal} ({meal_name}): {weeks_count} weeks, avg demand: {avg_demand:.0f}")
        
        # Check meal + region combinations for viable meals only
        print(f"\n📊 Analyzing meal + region combinations:")
        viable_meals = [k for k, v in availability.items() if k.startswith('meal_') and v['sufficient']]
        
        for meal_key in viable_meals[:10]:  # Limit to top 10 meals to avoid too much output
            meal_id = int(meal_key.split('_')[1])
            meal_name = self.get_meal_display_name(meal_id)
            
            print(f"\n   🍽️ {meal_name} (ID: {meal_id}) regional breakdown:")
            
            # Check each region for this meal
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
                        availability[f"meal_{meal_id}_{region}"] = {
                            'weeks': weeks_count,
                            'sufficient': True,
                            'avg_demand': avg_demand,
                            'description': f"{meal_name} - {region} ({weeks_count} weeks)",
                            'type': 'meal_region'
                        }
                        print(f"      ✅ {region}: {weeks_count} weeks, avg demand: {avg_demand:.0f}")
                    else:
                        print(f"      ❌ {region}: {weeks_count} weeks (insufficient)")
                else:
                    print(f"      ❌ {region}: No data available")
        
        self.data_availability = availability
        
        # Print summary
        viable_individual_meals = len([k for k, v in availability.items() if v['sufficient'] and v['type'] == 'meal'])
        viable_combinations = len([k for k, v in availability.items() if v['sufficient'] and v['type'] == 'meal_region'])
        
        print(f"\n✅ Meal Forecasting Summary:")
        print(f"   • Individual meals available: {viable_individual_meals}")
        print(f"   • Meal + region combinations: {viable_combinations}")
        print(f"   • Price regressors available: {'Yes' if self.column_mapping.get('checkout_price') else 'No'}")
        print(f"🎯 Focus: Meal-specific demand for ingredient ordering")
        print(f"💰 Business cost analysis: {'Enabled' if self.meal_database else 'Limited (using defaults)'}")
        if self.meal_database:
            print(f"🍽️ Meal database: {len(self.meal_database)} meals with ingredient cost structures")

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
        
        # Standard metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # More robust MAPE calculation with outlier handling
        def robust_mape(actual, predicted):
            """Calculate MAPE with outlier protection"""
            # Avoid division by zero and very small numbers
            mask = np.abs(actual) > np.percentile(np.abs(actual), 5)  # Remove bottom 5%
            actual_masked = actual[mask]
            predicted_masked = predicted[mask]
            
            if len(actual_masked) == 0:
                return 100.0  # If no valid data, return high error
            
            # Calculate percentage errors
            percentage_errors = np.abs((actual_masked - predicted_masked) / actual_masked) * 100
            
            # Cap extreme percentage errors at 200% to prevent outliers from dominating
            percentage_errors = np.minimum(percentage_errors, 200)
            
            return np.mean(percentage_errors)
        
        # Symmetric MAPE (more robust alternative)
        def symmetric_mape(actual, predicted):
            """Calculate symmetric MAPE - less biased than regular MAPE"""
            denominator = (np.abs(actual) + np.abs(predicted)) / 2
            mask = denominator > np.percentile(denominator, 5)  # Remove very small denominators
            
            if np.sum(mask) == 0:
                return 100.0
            
            actual_masked = actual[mask]
            predicted_masked = predicted[mask]
            denominator_masked = denominator[mask]
            
            smape = np.mean(np.abs(actual_masked - predicted_masked) / denominator_masked) * 100
            return np.minimum(smape, 200)  # Cap at 200%
        
        # Median-based metrics (more robust to outliers)
        median_absolute_error = np.median(np.abs(y_true - y_pred))
        
        # Calculate different MAPE versions
        standard_mape = robust_mape(y_true, y_pred)
        symmetric_mape_val = symmetric_mape(y_true, y_pred)
        
        # Bias calculation
        bias = np.mean(y_pred - y_true)
        
        # Alternative accuracy measures
        accuracy_standard = max(0, 100 - standard_mape)
        accuracy_symmetric = max(0, 100 - symmetric_mape_val)
        
        # Weighted accuracy based on data characteristics
        data_variance = np.var(y_true)
        data_mean = np.mean(y_true)
        cv = (np.sqrt(data_variance) / data_mean) * 100 if data_mean > 0 else 100
        
        # If coefficient of variation is high (>50%), use symmetric MAPE
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
        
        # Calculate coefficient of variation
        mean_val = np.mean(data)
        std_val = np.std(data)
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 100
        
        # Detect outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR
        outlier_count = np.sum(data > outlier_threshold)
        outlier_percentage = (outlier_count / len(data)) * 100
        
        print(f"📊 Data characteristics analysis:")
        print(f"   • Coefficient of Variation: {cv:.1f}%")
        print(f"   • Outliers detected: {outlier_count} ({outlier_percentage:.1f}%)")
        print(f"   • Data range: {np.min(data):.0f} - {np.max(data):.0f}")
        
        # Determine variance level
        if cv > 60 or outlier_percentage > 10:
            variance_level = 'high'
            print(f"   🔺 High variance data detected - using robust configuration")
        elif cv > 30 or outlier_percentage > 5:
            variance_level = 'medium'
            print(f"   📊 Medium variance data - using standard configuration")
        else:
            variance_level = 'low'
            print(f"   📈 Low variance data - using conservative configuration")
        
        return variance_level

    def create_robust_prophet_model(self, data_variance_level='medium'):
        """Create Prophet model with parameters optimized for data characteristics"""
        
        if data_variance_level == 'high':
            # High variance data - more flexible model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',  # Better for high variance
                changepoint_prior_scale=0.1,  # More flexible changepoints
                seasonality_prior_scale=0.01,  # Less aggressive seasonality
                uncertainty_samples=1000,
                interval_width=0.8  # Narrower intervals for stability
            )
            print(f"   🔺 Using HIGH variance Prophet configuration")
        elif data_variance_level == 'medium':
            # Standard configuration
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,
                uncertainty_samples=1000
            )
            print(f"   📊 Using MEDIUM variance Prophet configuration")
        else:  # low variance
            # Conservative model for stable data
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.01,  # Less flexible
                uncertainty_samples=1000
            )
            print(f"   📈 Using LOW variance Prophet configuration")
        
        return model

    def train_prophet_model(self, region_id=None, meal_id=None):
        """Train Prophet model for specific meal/region with adaptive configuration"""
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
            
            # Check if meal exists
            available_meals = set(self.train_aggregated['meal_id'].unique())
            if meal_id not in available_meals:
                return None, f"❌ Meal {meal_id} not found. Available meals: {sorted(list(available_meals))[:10]}..."
            
            filtered_data = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]
            
            if region_id:
                # Check if region exists
                available_regions = set(self.train_aggregated['region'].unique())
                if region_id not in available_regions:
                    return None, f"❌ Region {region_id} not found. Available regions: {sorted(list(available_regions))}"
                
                filtered_data = filtered_data[filtered_data['region'] == region_id]
                region_centers = [center for center, region in self.regional_mapping.items() if region == region_id]
                print(f"📍 {region_id} includes centers: {region_centers}")
            
            print(f"🔍 Debug: Filtered data: {len(filtered_data)} records")
            
            if len(filtered_data) == 0:
                if region_id:
                    # Check what regions this meal is available in
                    meal_regions = self.train_aggregated[self.train_aggregated['meal_id'] == meal_id]['region'].unique()
                    return None, f"❌ No data found for Meal {meal_id} in {region_id}. Available regions: {' and '.join(sorted(meal_regions))}"
                else:
                    return None, f"❌ No data found for Meal {meal_id}"
            
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
                
                # Provide specific suggestions
                if region_id:
                    suggestion = f"Try removing region filter for Meal {meal_id} or select a different region"
                else:
                    suggestion = "Try a different meal with more data"
                
                return None, f"❌ Insufficient data: only {available_weeks} weeks available ({coverage:.1%} coverage). Need {min_weeks_required}+. {suggestion}"
            
            # Log the actual week range being used
            min_week = demand_data['week'].min()
            max_week = demand_data['week'].max()
            total_weeks = len(demand_data)
            coverage = total_weeks / len(self.weekly_demand)
            
            print(f"📊 Training data: {total_weeks} weeks (weeks {min_week}-{max_week})")
            print(f"📈 Coverage: {coverage:.1%} of total timeline")
            print(f"📦 Demand range: {demand_data['num_orders'].min()}-{demand_data['num_orders'].max()}")
            
            # Analyze data characteristics and choose appropriate model
            demand_values = demand_data['num_orders'].values
            variance_level = self.detect_data_characteristics(demand_values)
            
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
            
            # Create adaptive Prophet model based on data characteristics
            model = self.create_robust_prophet_model(variance_level)
            
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
                'original_data': demand_data,   # Store original weekly data
                'variance_level': variance_level,  # Store variance level
                'data_cv': (np.std(demand_values) / np.mean(demand_values)) * 100  # Store CV
            }
            
            # Generate predictions on training data for performance visualization
            self._generate_training_performance(model_key, model, prophet_data, demand_data)
            
            print(f"✅ Model stored with key: '{model_key}'")
            return model, f"✅ Adaptive model trained for {model_key} (variance: {variance_level})"
            
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
                
                from scipy import stats # type: ignore
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
            
            meal_name = self.get_meal_display_name(meal_id)
            return forecast, f"✅ Forecast generated successfully for {meal_name}"
            
        except Exception as e:
            print(f"❌ Forecast error: {e}")
            return None, f"Error generating forecast: {str(e)}"
    
    def evaluate_model_performance(self, model_key):
        """Evaluate Prophet model performance using robust metrics for high-variance data"""
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
            variance_level = model_info.get('variance_level', 'medium')
            data_cv = model_info.get('data_cv', 0)
            
            print(f"🔍 Evaluating model: {model_key} (variance: {variance_level}, CV: {data_cv:.1f}%)")
            
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
            
            # Use robust metrics calculation
            robust_metrics = self.calculate_robust_metrics(y_true, y_pred)
            
            # Add standard fields for compatibility
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
            
            print(f"✅ Robust metrics calculated for Meal {meal_id}:")
            print(f"   📊 {metrics['accuracy_note']}")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['accuracy_note', 'meal_name']:
                    if 'mape' in key or 'accuracy' in key or 'cv' in key:
                        print(f"   {key}: {value:.1f}%")
                    else:
                        print(f"   {key}: {value:.2f}")
            
            self.performance_metrics[model_key] = metrics
            
            return metrics, f"Robust model evaluation completed for Meal {meal_id} using weeks {eval_min_week}-{eval_max_week}"
            
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
        # Get actual regions from the regional mapping (excluding Lyon)
        actual_regions = sorted(set(self.regional_mapping.values())) if self.regional_mapping else ['Region-Kassel', 'Region-Wien', 'Region-Luzern']
        
        # Create meals list with names for frontend
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
                'meals': [meal['id'] for meal in meals_list],  # Keep backward compatibility
                'meals_with_names': meals_list,  # Full meal info
                'weeks_range': [
                    int(self.train_data['week'].min()) if self.train_data is not None else 0,
                    int(self.train_data['week'].max()) if self.train_data is not None else 0
                ],
                'has_price_data': bool(self.column_mapping.get('checkout_price')),
                'price_regressors_available': ['checkout_price'] if self.column_mapping.get('checkout_price') else [],
                'business_cost_analysis_enabled': len(self.meal_database) > 0,
                'excluded_regions': ['Region-Lyon']  # Track excluded regions
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
            
            print(f"🔄 Creating naive baseline for {model_key}")
            
            # Sort data by week to ensure proper ordering
            sorted_data = original_data.sort_values('week').reset_index(drop=True)
            weeks = sorted_data['week'].values
            demands = sorted_data['num_orders'].values
            
            # Calculate weeks per year (assuming roughly 52 weeks)
            weeks_per_year = 52
            
            # Create baseline predictions for evaluation period
            total_weeks = len(sorted_data)
            eval_size = max(5, int(total_weeks * 0.2))  # Same as Prophet evaluation
            eval_start_idx = total_weeks - eval_size
            
            baseline_predictions = []
            baseline_weeks = []
            
            print(f"📊 Creating baseline for last {eval_size} weeks")
            
            for i in range(eval_start_idx, total_weeks):
                current_week = weeks[i]
                
                # Look for the same week last year (52 weeks ago)
                target_week = current_week - weeks_per_year
                
                # Find the closest available week to target_week
                week_diffs = np.abs(weeks[:i] - target_week)  # Only look at past data
                
                if len(week_diffs) > 0:
                    closest_idx = np.argmin(week_diffs)
                    closest_week = weeks[closest_idx]
                    baseline_pred = demands[closest_idx]
                    
                    # If we found a reasonably close match (within 4 weeks of target)
                    if abs(closest_week - target_week) <= 4:
                        baseline_predictions.append(baseline_pred)
                        print(f"   Week {current_week}: Using week {closest_week} demand ({baseline_pred}) [target: {target_week}]")
                    else:
                        # Fallback: use recent average if no good historical match
                        recent_avg = np.mean(demands[max(0, i-12):i])  # Last 12 weeks average
                        baseline_predictions.append(recent_avg)
                        print(f"   Week {current_week}: Using recent avg ({recent_avg:.0f}) [no good historical match]")
                else:
                    # Fallback for edge cases
                    baseline_predictions.append(np.mean(demands[:i]) if i > 0 else demands[0])
                
                baseline_weeks.append(current_week)
            
            # Store baseline model results
            baseline_model = {
                'predictions': np.array(baseline_predictions),
                'weeks': np.array(baseline_weeks),
                'eval_start_idx': eval_start_idx,
                'method': 'same_week_last_year',
                'weeks_per_year': weeks_per_year
            }
            
            self.baseline_models = getattr(self, 'baseline_models', {})
            self.baseline_models[model_key] = baseline_model
            
            print(f"✅ Naive baseline created: {len(baseline_predictions)} predictions")
            return baseline_model, f"Baseline model created using same-week-last-year approach"
            
        except Exception as e:
            print(f"❌ Error creating baseline model: {e}")
            return None, f"Error creating baseline: {str(e)}"

    def evaluate_baseline_vs_prophet(self, model_key):
        """Compare Prophet model performance against naive baseline"""
        try:
            if model_key not in self.models:
                return None, "Prophet model not found"
            
            # Create baseline model if it doesn't exist
            if not hasattr(self, 'baseline_models') or model_key not in self.baseline_models:
                baseline_model, message = self.create_naive_baseline_model(model_key)
                if baseline_model is None:
                    return None, message
            
            # Get Prophet evaluation data
            model_info = self.models[model_key]
            original_data = model_info.get('original_data')
            training_data = model_info.get('training_data')
            
            if original_data is None or training_data is None:
                return None, "No evaluation data available"
            
            # Get baseline predictions
            baseline_model = self.baseline_models[model_key]
            baseline_predictions = baseline_model['predictions']
            eval_weeks = baseline_model['weeks']
            
            # Get Prophet predictions for the same period
            total_weeks = len(original_data)
            eval_size = len(baseline_predictions)
            eval_start_idx = total_weeks - eval_size
            
            # Prepare Prophet evaluation data
            eval_data_prophet = training_data.iloc[eval_start_idx:eval_start_idx + eval_size].copy()
            eval_data_original = original_data.iloc[eval_start_idx:eval_start_idx + eval_size].copy()
            
            # Add regressors for Prophet evaluation
            regressors = model_info['regressors']
            for regressor in regressors:
                if regressor in eval_data_original.columns:
                    eval_data_prophet[regressor] = eval_data_original[regressor].fillna(eval_data_original[regressor].mean())
            
            # Get Prophet predictions
            prophet_model = model_info['model']
            prophet_predictions = prophet_model.predict(eval_data_prophet[['ds'] + regressors])
            prophet_pred_values = np.maximum(prophet_predictions['yhat'].values, 0)
            
            # Get actual values
            actual_values = eval_data_prophet['y'].values
            
            print(f"📊 Comparing Prophet vs Baseline for {model_key}")
            print(f"📅 Evaluation period: {len(actual_values)} weeks")
            
            prophet_metrics = self.calculate_robust_metrics(actual_values, prophet_pred_values) #calculate_comparison_metrics(actual_values, prophet_pred_values, "Prophet")
            baseline_metrics = self.calculate_robust_metrics(actual_values, baseline_predictions)#calculate_comparison_metrics(actual_values, baseline_predictions, "Baseline")
            
            # Calculate improvement
            accuracy_improvement = prophet_metrics['accuracy'] - baseline_metrics['accuracy']
            mae_improvement = ((baseline_metrics['mae'] - prophet_metrics['mae']) / baseline_metrics['mae']) * 100
            
            print(f"🎯 Prophet vs Baseline Performance:")
            print(f"   Accuracy improvement: {accuracy_improvement:+.1f} percentage points")
            print(f"   MAE improvement: {mae_improvement:+.1f}%")
            
            # Prepare comparison data for visualization
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
            
            # Store comparison for later retrieval
            self.model_comparisons = getattr(self, 'model_comparisons', {})
            self.model_comparisons[model_key] = comparison_data
            
            return comparison_data, f"Model comparison completed for evaluation period {comparison_data['eval_period']}"
            
        except Exception as e:
            print(f"❌ Error in baseline comparison: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error comparing models: {str(e)}"

    def get_model_comparison(self, model_key):
        """Get stored model comparison data"""
        if not hasattr(self, 'model_comparisons') or model_key not in self.model_comparisons:
            # Generate comparison if it doesn't exist
            comparison_data, message = self.evaluate_baseline_vs_prophet(model_key)
            return comparison_data
        
        return self.model_comparisons.get(model_key, None)