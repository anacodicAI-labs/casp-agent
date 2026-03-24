"""
Verify consistency: Code → CSV → Paper
Run from code/ directory.
"""
import os
import sys
import csv
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from ml.predictive_analytics import PredictiveAnalytics
from analytics.carbon_intelligence import CarbonCostOfIntelligence

from config.agent_mapping import get_agent_config
from config.vehicle_emissions import calculate_transport_carbon

out_dir = os.path.join(script_dir, "..", "outputs", "figdata")

print('=' * 80)
print('VERIFICATION: Code → CSV → Paper Consistency')
print('=' * 80)

# Load dataset
PA = PredictiveAnalytics
pa = PA()
pa.load_data()

# 1. Verify pareto_9types.csv
print('\n1. pareto_9types.csv (Code → CSV → Paper)')
print('-' * 80)

# Get India AI carbon once
cci_india = CarbonCostOfIntelligence(default_country='india')
ai_result_india = cci_india.calculate_optimization_carbon(
    num_routes=3,
    model_type='gradient-boosting',
    country='india',
    include_llm_reasoning=True
)
code_ai_india = ai_result_india['total_ai_carbon_gco2']

with open(os.path.join(out_dir, "pareto_9types.csv"), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pkg = row['package_type']
        csv_carbon = int(row['carbon_gco2'])
        csv_service = int(row['service_pct'])
        
        # Compute from code
        pkg_df = pa.df[pa.df['package_type'].str.lower() == pkg.lower()].copy()
        if len(pkg_df) == 0:
            print(f'{pkg:20s}: NO DATA')
            continue
        
        config = get_agent_config(pkg.lower())
        cold_mult = config.get('cold_chain_multiplier', 1.0)
        pkg_df['transport_carbon'] = pkg_df.apply(
            lambda r: calculate_transport_carbon(
                distance_km=r['distance_km'],
                vehicle_type=r['vehicle_type'],
                cold_chain_multiplier=cold_mult
            ), axis=1
        )
        code_transport = pkg_df['transport_carbon'].mean()
        code_total = code_transport + code_ai_india
        
        # CASP calculation
        service_perf = csv_service / 100
        code_casp = (service_perf / code_total) * 1e6
        
        transport_match = abs(code_transport - csv_carbon) < 1
        total_match = abs(code_total - csv_carbon) < 1
        
        print(f'{pkg:20s}:')
        print(f'  Code Transport: {code_transport:8.0f} | CSV: {csv_carbon:8d} | Match: {transport_match}')
        print(f'  Code Total:     {code_total:8.0f} | CSV Total: {csv_carbon:8d} | Match: {total_match}')
        print(f'  Code CASP:       {code_casp:6.2f} | Service: {csv_service}%')
        if not transport_match or not total_match:
            print(f'  ⚠️  MISMATCH DETECTED!')
        print()

# 2. Verify casp_by_country.csv
print('\n2. casp_by_country.csv (Code → CSV → Paper)')
print('-' * 80)
pharmacy_df = pa.df[pa.df['package_type'].str.lower() == 'pharmacy'].copy()
pharmacy_config = get_agent_config('pharmacy')
cold_mult = pharmacy_config.get('cold_chain_multiplier', 2.5)
pharmacy_df['transport_carbon'] = pharmacy_df.apply(
    lambda r: calculate_transport_carbon(
        distance_km=r['distance_km'],
        vehicle_type=r['vehicle_type'],
        cold_chain_multiplier=cold_mult
    ), axis=1
)
code_transport_pharmacy = pharmacy_df['transport_carbon'].mean()
service_pct = pharmacy_config.get('on_time_threshold', 0.99) * 100
service_perf = service_pct / 100

with open(os.path.join(out_dir, "casp_by_country.csv"), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        country = row['country']
        csv_casp = float(row['casp_e6'])
        
        # Compute from code
        cci = CarbonCostOfIntelligence(default_country=country.lower())
        ai_result = cci.calculate_optimization_carbon(
            num_routes=3,
            model_type='gradient-boosting',
            country=country.lower(),
            include_llm_reasoning=True
        )
        code_ai = ai_result['total_ai_carbon_gco2']
        code_total = code_transport_pharmacy + code_ai
        code_casp = (service_perf / code_total) * 1e6
        
        match = abs(code_casp - csv_casp) < 0.01
        print(f'{country:10s}: Code CASP={code_casp:6.2f} | CSV={csv_casp:6.2f} | Match: {match}')
        if not match:
            print(f'  ⚠️  MISMATCH DETECTED!')

# 3. Verify paper table values
print('\n3. Paper Table tab:category_carbon (Code → Paper)')
print('-' * 80)
paper_values = {
    'Pharmacy': {'transport': 148437, 'ai': 0.003, 'total': 148437, 'casp': 6.67},
    'Groceries': {'transport': 110190, 'ai': 0.003, 'total': 110190, 'casp': 8.98},
    'Electronics': {'transport': 57355, 'ai': 0.003, 'total': 57355, 'casp': 16.56},
    'Automobile parts': {'transport': 55888, 'ai': 0.003, 'total': 55888, 'casp': 17.00},
    'Furniture': {'transport': 55457, 'ai': 0.003, 'total': 55457, 'casp': 17.13},
    'Documents': {'transport': 56819, 'ai': 0.003, 'total': 56819, 'casp': 16.72},
    'Fragile items': {'transport': 56163, 'ai': 0.003, 'total': 56163, 'casp': 16.92},
    'Clothing': {'transport': 57922, 'ai': 0.003, 'total': 57922, 'casp': 14.67},
    'Cosmetics': {'transport': 57285, 'ai': 0.003, 'total': 57285, 'casp': 14.84}
}

paper_ai = code_ai_india  # Use same AI carbon

mismatches = []

for pkg_name, paper in paper_values.items():
    pkg_lower = pkg_name.lower()
    
    pkg_df = pa.df[pa.df['package_type'].str.lower() == pkg_lower].copy()
    if len(pkg_df) == 0:
        print(f'{pkg_name:20s}: NO DATA')
        continue
    
    config = get_agent_config(pkg_lower)
    cold_mult = config.get('cold_chain_multiplier', 1.0)
    pkg_df['transport_carbon'] = pkg_df.apply(
        lambda r: calculate_transport_carbon(
            distance_km=r['distance_km'],
            vehicle_type=r['vehicle_type'],
            cold_chain_multiplier=cold_mult
        ), axis=1
    )
    code_transport = pkg_df['transport_carbon'].mean()
    code_total = code_transport + paper_ai
    
    service_pct_val = config.get('on_time_threshold', 0.95) * 100
    service_perf_val = service_pct_val / 100
    code_casp = (service_perf_val / code_total) * 1e6
    
    transport_match = abs(code_transport - paper['transport']) < 1
    total_match = abs(code_total - paper['total']) < 1
    casp_match = abs(code_casp - paper['casp']) < 0.1
    
    print(f'{pkg_name:20s}:')
    print(f'  Transport: Code={code_transport:8.0f} | Paper={paper["transport"]:8d} | Match: {transport_match}')
    print(f'  Total:     Code={code_total:8.0f} | Paper={paper["total"]:8d} | Match: {total_match}')
    print(f'  CASP:      Code={code_casp:6.2f} | Paper={paper["casp"]:6.2f} | Match: {casp_match}')
    
    if not transport_match or not total_match or not casp_match:
        mismatches.append(pkg_name)
        print(f'  ⚠️  MISMATCH DETECTED!')
    print()

print('\n' + '=' * 80)
if mismatches:
    print(f'⚠️  MISMATCHES FOUND: {", ".join(mismatches)}')
    print('Action needed: Update paper or regenerate CSVs')
else:
    print('✅ ALL VALUES CONSISTENT: Code → CSV → Paper')
print('=' * 80)
