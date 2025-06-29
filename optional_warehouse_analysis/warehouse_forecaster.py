#!/usr/bin/env python3
"""
Simple Warehouse Forecaster - SIMPLIFIED VERSION
Uses the existing forecaster.py to generate ONE data file: warehouse_data.json
Just the essentials: absolute storage needed + percentage utilization per week.
"""

import json
from datetime import datetime

from forecaster import FoodDemandForecaster


class SimpleWarehouseForecaster:
    def __init__(self):
        self.forecaster = FoodDemandForecaster()
        
        # Warehouse capacities (m³)
        self.warehouse_capacities = {
            'Region-Kassel': {'warehouse': 1163.00, 'cool_storage': 5412.00},
            'Region-Luzern': {'warehouse': 366.00, 'cool_storage': 2157.00},
            'Region-Wien': {'warehouse': 328.00, 'cool_storage': 1832.00}
        }
    
    def calculate_meal_storage(self, meal_id, portions):
        """Calculate storage requirements using your meal database - CORRECTED FOR PER-ORDER"""
        try:
            # Use your existing meal_database
            meal_info = None
            for key in [str(meal_id), int(meal_id), meal_id]:
                if hasattr(self.forecaster, 'meal_database') and key in self.forecaster.meal_database:
                    meal_info = self.forecaster.meal_database[key]
                    break
            
            if not meal_info:
                # Default if not found
                return {'warehouse': portions * 0.01, 'cool_storage': portions * 0.015}
            
            # Extract from your database
            original_data = meal_info.get('original_data', meal_info)
            storage_m3 = original_data.get('storage_m3', [])
            perishable = original_data.get('perishable', [])
            
            if not storage_m3 or not perishable:
                return {'warehouse': portions * 0.01, 'cool_storage': portions * 0.015}
            
            warehouse_needed = 0.0
            cool_storage_needed = 0.0
            
            for i, storage_per_order in enumerate(storage_m3):
                if i < len(perishable):
                    # SIMPLE: storage_m3 is per single order, so multiply by number of orders
                    total_storage = portions * storage_per_order
                    
                    if perishable[i]:
                        cool_storage_needed += total_storage
                    else:
                        warehouse_needed += total_storage
            
            return {'warehouse': warehouse_needed, 'cool_storage': cool_storage_needed}
            
        except Exception:
            return {'warehouse': portions * 0.01, 'cool_storage': portions * 0.015}
    
    def forecast_region(self, region_id, weeks_ahead=10):
        """Calculate total warehouse requirements for all meals in a region - SIMPLIFIED"""
        print(f"📍 Processing {region_id}...")
        
        # Get meals in region using your data
        region_data = self.forecaster.train_aggregated[self.forecaster.train_aggregated['region'] == region_id]
        meals = sorted(region_data['meal_id'].unique())
        
        print(f"🍽️ Found {len(meals)} meals")
        
        # Initialize weekly totals
        weekly_warehouse = [0.0] * weeks_ahead
        weekly_cool_storage = [0.0] * weeks_ahead
        successful = 0
        
        # Loop through all meals and sum up requirements
        for meal_id in meals:
            try:
                # Train and forecast using YOUR methods
                model, _ = self.forecaster.train_prophet_model(region_id, meal_id)
                if not model:
                    continue
                
                model_key = self.forecaster._normalize_model_key(region_id, meal_id)
                forecast, _ = self.forecaster.generate_forecast(model_key, weeks_ahead)
                if forecast is None:
                    continue
                
                # Get predictions
                predicted_orders = forecast['yhat'].round().astype(int).tolist()
                meal_name = self.forecaster.get_meal_display_name(meal_id) if hasattr(self.forecaster, 'get_meal_display_name') else f"Meal {meal_id}"
                
                # Calculate storage for each week and add to totals
                for week_idx, orders in enumerate(predicted_orders):
                    storage = self.calculate_meal_storage(meal_id, orders)
                    
                    # Add to regional totals
                    weekly_warehouse[week_idx] += storage['warehouse']
                    weekly_cool_storage[week_idx] += storage['cool_storage']
                
                successful += 1
                print(f"  ✅ {meal_name}: {sum(predicted_orders)} orders")
                
            except Exception as e:
                print(f"  ❌ Meal {meal_id}: {e}")
                continue
        
        if successful == 0:
            return None
        
        # Calculate utilization percentages
        capacity = self.warehouse_capacities[region_id]
        warehouse_utilization = [(req / capacity['warehouse']) * 100 for req in weekly_warehouse]
        cool_utilization = [(req / capacity['cool_storage']) * 100 for req in weekly_cool_storage]
        
        # BUILD SIMPLIFIED TIMELINE - JUST THE ESSENTIALS
        timeline = []
        for week_idx in range(weeks_ahead):
            timeline.append({
                'week': week_idx + 1,
                'warehouse_needed_m3': round(weekly_warehouse[week_idx], 2),
                'cool_storage_needed_m3': round(weekly_cool_storage[week_idx], 2),
                'warehouse_utilization_percent': round(warehouse_utilization[week_idx], 1),
                'cool_storage_utilization_percent': round(cool_utilization[week_idx], 1)
            })
        
        return {
            'region_name': region_id,
            'region_display_name': region_id.replace('Region-', ''),
            'max_warehouse_capacity_m3': capacity['warehouse'],
            'max_cool_storage_capacity_m3': capacity['cool_storage'],
            'weekly_timeline': timeline
        }
    
    def generate_warehouse_data(self, weeks_ahead=10, regions=None):
        """Generate simplified warehouse data with timeline focus"""
                
        # Load data using your forecaster
        if not self.forecaster.load_data():
            print("❌ Failed to load data")
            return False
        
        # Use all regions if none specified
        if regions is None:
            regions = list(self.warehouse_capacities.keys())
        
        # Process each region
        warehouse_data = {
            'timestamp': datetime.now().isoformat(),
            'weeks_ahead': weeks_ahead,
            'regions': {}
        }
        
        # Process each region
        for region_id in regions:
            if region_id not in self.warehouse_capacities:
                print(f"❌ Unknown region: {region_id}")
                continue
            
            region_data = self.forecast_region(region_id, weeks_ahead)
            if region_data:
                warehouse_data['regions'][region_id] = region_data
                
                # Find peak utilization for simple summary
                timeline = region_data['weekly_timeline']
                peak_warehouse = max(week['warehouse_utilization_percent'] for week in timeline)
                peak_cool = max(week['cool_storage_utilization_percent'] for week in timeline)
                
                print(f"✅ {region_data['region_display_name']}: Peak {peak_warehouse:.1f}% warehouse, {peak_cool:.1f}% cool")
        
        if not warehouse_data['regions']:
            print("❌ No regional data generated")
            return False
        
        # Save to ONE file
        try:
            with open('warehouse_data.json', 'w') as f:
                json.dump(warehouse_data, f, indent=2, default=str)
            
            print(f"\n🎉 SUCCESS!")
            print(f"📁 Generated: warehouse_data.json")
            print(f"📊 Contains {len(warehouse_data['regions'])} regions with {weeks_ahead}-week timelines")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False
        
    def forecast_all_regions_x_weeks(self, x_weeks=20):
        """
        Simple method to forecast warehouse capacity for all regions for x weeks.
        Just call this method and get the complete warehouse data.
        
        Inputs:
            int: the weeks that should be forecasted defaults to 20

        Returns:
            dict: Complete warehouse forecast data for all regions (xweeks)
            bool: Success status
        """
        print("🚀 Starting 20-week warehouse capacity forecast for all regions...")
        
        # Load data using your forecaster
        if not self.forecaster.load_data():
            print("❌ Failed to load data")
            return None, False
        
        # X weeks, all regions
        weeks_ahead = x_weeks
        regions = list(self.warehouse_capacities.keys())
        
        # Process each region
        warehouse_data = {
            'timestamp': datetime.now().isoformat(),
            'weeks_ahead': weeks_ahead,
            'regions': {}
        }
        
        print(f"📊 Processing {len(regions)} regions for {weeks_ahead} weeks...")
        
        for region_id in regions:
            print(f"\n📍 Processing {region_id}...")
            
            # Get meals in region using your data
            region_data = self.forecaster.train_aggregated[self.forecaster.train_aggregated['region'] == region_id]
            meals = sorted(region_data['meal_id'].unique())
            
            print(f"🍽️ Found {len(meals)} meals")
            
            # Initialize weekly totals
            weekly_warehouse = [0.0] * weeks_ahead
            weekly_cool_storage = [0.0] * weeks_ahead
            successful = 0
            
            # Loop through all meals and sum up requirements
            for meal_id in meals:
                try:
                    # Train and forecast using YOUR methods
                    model, _ = self.forecaster.train_prophet_model(region_id, meal_id)
                    if not model:
                        continue
                    
                    model_key = self.forecaster._normalize_model_key(region_id, meal_id)
                    forecast, _ = self.forecaster.generate_forecast(model_key, weeks_ahead)
                    if forecast is None:
                        continue
                    
                    # Get predictions
                    predicted_orders = forecast['yhat'].round().astype(int).tolist()
                    meal_name = self.forecaster.get_meal_display_name(meal_id) if hasattr(self.forecaster, 'get_meal_display_name') else f"Meal {meal_id}"
                    
                    # Calculate storage for each week and add to totals
                    for week_idx, orders in enumerate(predicted_orders):
                        storage = self.calculate_meal_storage(meal_id, orders)
                        
                        # Add to regional totals
                        weekly_warehouse[week_idx] += storage['warehouse']
                        weekly_cool_storage[week_idx] += storage['cool_storage']
                    
                    successful += 1
                    print(f"  ✅ {meal_name}: {sum(predicted_orders)} orders")
                    
                except Exception as e:
                    print(f"  ❌ Meal {meal_id}: {e}")
                    continue
            
            if successful == 0:
                print(f"❌ No successful forecasts for {region_id}")
                continue
            
            # Calculate utilization percentages
            capacity = self.warehouse_capacities[region_id]
            warehouse_utilization = [(req / capacity['warehouse']) * 100 for req in weekly_warehouse]
            cool_utilization = [(req / capacity['cool_storage']) * 100 for req in weekly_cool_storage]
            
            # BUILD TIMELINE - 20 weeks of essential data
            timeline = []
            for week_idx in range(weeks_ahead):
                timeline.append({
                    'week': week_idx + 1,
                    'warehouse_needed_m3': round(weekly_warehouse[week_idx], 2),
                    'cool_storage_needed_m3': round(weekly_cool_storage[week_idx], 2),
                    'warehouse_utilization_percent': round(warehouse_utilization[week_idx], 1),
                    'cool_storage_utilization_percent': round(cool_utilization[week_idx], 1)
                })
            
            # Store region data
            warehouse_data['regions'][region_id] = {
                'region_name': region_id,
                'region_display_name': region_id.replace('Region-', ''),
                'max_warehouse_capacity_m3': capacity['warehouse'],
                'max_cool_storage_capacity_m3': capacity['cool_storage'],
                'weekly_timeline': timeline
            }
            
            # Show peak utilization summary
            peak_warehouse = max(warehouse_utilization)
            peak_cool = max(cool_utilization)
            print(f"✅ {region_id.replace('Region-', '')}: Peak {peak_warehouse:.1f}% warehouse, {peak_cool:.1f}% cool")
        
        if not warehouse_data['regions']:
            print("❌ No regional data generated")
            return None, False
        
        # Save to file
        try:
            with open('warehouse_data.json', 'w') as f:
                json.dump(warehouse_data, f, indent=2, default=str)
            
            print(f"\n🎉 SUCCESS!")
            print(f"📁 Generated: warehouse_data.json")
            print(f"📊 Contains {len(warehouse_data['regions'])} regions with 20-week timelines")
            print(f"📅 Total datapoints: {len(warehouse_data['regions']) * 20 * 4}")  # regions * weeks * 4 metrics
            
            return warehouse_data, True
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return None, False
    
def main():
    """Main execution - calls the MVP core functionality"""
    print("📦 MVP Warehouse Forecaster")
    print("🎯 Core: 20-week capacity forecast for all regions")
    print()
    
    forecaster = SimpleWarehouseForecaster()
    
    # 🎯 MVP CORE CALL 🎯
    warehouse_data, success = forecaster.forecast_all_regions_x_weeks(20)
    
    if success:
        print("\n✅ MVP execution completed successfully!")
        return 0
    else:
        print("\n❌ MVP execution failed!")
        return 1


if __name__ == "__main__":
    exit(main())