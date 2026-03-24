"""
Agent Mapping Configuration
Maps package types to appropriate logistics agents based on CRITICALITY-BASED CLASSIFICATION

================================================================================
CLASSIFICATION FRAMEWORK JUSTIFICATION (For Research Paper)
================================================================================

This classification is based on industry Service Level Agreements (SLAs) and 
regulatory requirements, NOT arbitrary thresholds.

TIER 1 - CRITICAL (≥99% On-Time Required)
-----------------------------------------
Regulatory Basis:
  - WHO Good Distribution Practice (GDP) guidelines (WHO Technical Report Series No. 961)
  - FDA Drug Supply Chain Security Act (DSCSA)
  - EU Guidelines 2013/C 343/01 for medicinal products
  - ISPE Good Practice Guide: Cold Chain Management

Industry Benchmarks:
  - "Pharmaceutical & Critical Medical Supplies: 99%+" (Service Club, 2025)
  - World Courier reports 99.6% on-time for pharma cold chain (World Courier, 2025)
  - "Courier & Last-Mile Parcel: 95%-98%" (Service Club, 2025)

TIER 2 - HIGH-VALUE (≥95% On-Time Required)
-------------------------------------------
Justification:
  - High economic impact of delivery failure
  - Time-sensitive (production line dependencies)
  - Business-critical components

Industry Benchmarks:
  - "E-commerce (non-perishable goods): 94%-97%" (Service Club, 2025)
  - "Best-in-class eCommerce operations typically maintain OTD rates above 95%" (Opensend, 2025)
  - "If on-time delivery rate dips below 95%, that's cause for concern" (Amazon MCF, 2025)
  - "Healthy benchmarks are 95%+ for domestic shipments" (Fashion Benchmarks, 2025)

TIER 3 - STANDARD (≥85% On-Time Required)
-----------------------------------------
Justification:
  - Flexible delivery windows acceptable to consumers
  - Lower unit value, lower impact of delay
  - Non-perishable, non-critical

Industry Benchmarks:
  - "Grocery Delivery: 85%-90%" (Service Club, 2025)
  - "Industry average hovers between 85-90%" (Opensend, 2025)
  - "Advanced fashion fill rate: 70%-80%" (FCBCO Benchmarking, 2024)
  - "OTD rate below 90% can significantly harm repeat purchase rates" (SmartRoutes, 2025)

================================================================================
REFERENCES (For Paper Citation) — with links
================================================================================

[1] WHO Technical Report Series, No. 961, 2011 - Annex 9: Model Guidance for
    Storage and Transport of Time- and Temperature-Sensitive Pharmaceutical Products
    https://www.who.int/docs/default-source/medicines/norms-and-standards/guidelines/distribution/trs961-annex9-modelguidanceforstoragetransport.pdf

[2] EU Guidelines for Good Distribution Practice of Medicinal Products for
    Human Use (2013/C 343/01)
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52013XC0117%2801%29

[3] FDA Drug Supply Chain Security Act (DSCSA), 2013
    https://www.fda.gov/drugs/drug-supply-chain-integrity/drug-supply-chain-security-act-dscsa

[4] ISPE Good Practice Guide: Cold Chain Management, 2011 (industry standard; no public URL)

[5] Service Club (2025). "On-Time Delivery Rate Benchmarks: How Your Business 
    Stacks Up In 2025". https://serviceclub.com/on-time-delivery-rate-benchmarks/

[6] Opensend (2025). "7 On-time Delivery Rate Statistics For eCommerce Stores".
    https://www.opensend.com/post/on-time-delivery-rate-statistics-ecommerce

[7] Amazon Multi-Channel Fulfillment (2025). "Guide to the 5 most important 
    ecommerce fulfillment KPIs". https://supplychain.amazon.com/learn/

[8] FCBCO (2024). "Benchmarking Metrics for Warehouse Operations and 
    Fulfillment Centers". https://www.fcbco.com/blog/

[9] SmartRoutes (2025). "Delivery Success Rates: Key Retail & eCommerce Stats".
    https://smartroutes.io/blogs/delivery-success-rates/

[10] World Courier (2025). "Pharmaceutical Cold Chain Solutions".
     https://www.worldcourier.com/solutions/pharmaceutical-cold-chain

================================================================================
"""

AGENT_MAPPING = {
    # =========================================================================
    # TIER 1: CRITICAL - Life-Safety Products (≥99% on-time required)
    # =========================================================================
    # Regulatory basis: WHO GDP, FDA DSCSA, EU 2013/C 343/01
    # Industry benchmark: 99%+ for pharmaceutical logistics [5,10]
    
    'pharmacy': {
        'agent': 'critical',
        'tier': 1,
        'tier_name': 'CRITICAL',
        'on_time_threshold': 0.99,  # WHO/FDA regulated - industry achieves 99%+
        'cold_chain_multiplier': 2.5,  # Temperature-controlled required
        'priority': 'safety',
        'impact_multiplier': 10.0,  # Lives at risk - highest impact
        'regulatory_basis': 'WHO GDP, FDA DSCSA, EU 2013/C 343/01',
        'benchmark_source': 'Service Club 2025: Pharma 99%+; World Courier: 99.6%'
    },
    'groceries': {
        'agent': 'critical',
        'tier': 1,
        'tier_name': 'CRITICAL',
        'on_time_threshold': 0.99,  # Perishable - food safety critical
        'cold_chain_multiplier': 2.0,  # Refrigeration often required
        'priority': 'safety',
        'impact_multiplier': 3.0,  # Food safety concerns
        'regulatory_basis': 'Food safety regulations, cold chain requirements',
        'benchmark_source': 'Perishable goods require highest reliability'
    },
    
    # =========================================================================
    # TIER 2: HIGH-VALUE - Business-Critical Products (≥95% on-time required)
    # =========================================================================
    # Industry benchmark: 94-97% for e-commerce non-perishable [5,6,7]
    
    'automobile parts': {
        'agent': 'high_value',
        'tier': 2,
        'tier_name': 'HIGH_VALUE',
        'on_time_threshold': 0.95,  # Production line dependent
        'cold_chain_multiplier': 1.0,
        'priority': 'reliability',
        'impact_multiplier': 3.0,  # Can halt production lines
        'regulatory_basis': 'Industry SLA standards',
        'benchmark_source': 'Service Club 2025: E-commerce 94-97%'
    },
    'furniture': {
        'agent': 'high_value',
        'tier': 2,
        'tier_name': 'HIGH_VALUE',
        'on_time_threshold': 0.95,  # High-value, scheduled delivery
        'cold_chain_multiplier': 1.0,
        'priority': 'cost',
        'impact_multiplier': 2.0,  # High-value items
        'regulatory_basis': 'Industry SLA standards',
        'benchmark_source': 'Opensend 2025: Best-in-class 95%+'
    },
    'documents': {
        'agent': 'high_value',
        'tier': 2,
        'tier_name': 'HIGH_VALUE',
        'on_time_threshold': 0.95,  # Time-sensitive legal/business docs
        'cold_chain_multiplier': 1.0,
        'priority': 'speed',
        'impact_multiplier': 2.5,  # Business-critical timing
        'regulatory_basis': 'Industry SLA standards',
        'benchmark_source': 'Amazon MCF 2025: Below 95% is concern'
    },
    'fragile items': {
        'agent': 'high_value',
        'tier': 2,
        'tier_name': 'HIGH_VALUE',
        'on_time_threshold': 0.95,  # Special handling, high care
        'cold_chain_multiplier': 1.0,
        'priority': 'safety',
        'impact_multiplier': 2.0,  # Damage risk
        'regulatory_basis': 'Industry SLA standards',
        'benchmark_source': 'FCBCO 2024: Gifts/home 85-95%'
    },
    'electronics': {
        'agent': 'high_value',
        'tier': 2,
        'tier_name': 'HIGH_VALUE',
        'on_time_threshold': 0.95,  # High-value, time-sensitive; benchmark aligns with Tier 2
        'cold_chain_multiplier': 1.0,
        'priority': 'value',
        'impact_multiplier': 5.0,  # High economic value
        'regulatory_basis': 'Industry SLA standards',
        'benchmark_source': 'Service Club 2025: Courier 95-98%'
    },
    
    # =========================================================================
    # TIER 3: STANDARD - Consumer Goods (≥85% on-time required)
    # =========================================================================
    # Industry benchmark: 85-90% average, fashion 70-80% fill rate [5,6,8,9]
    
    'clothing': {
        'agent': 'standard',
        'tier': 3,
        'tier_name': 'STANDARD',
        'on_time_threshold': 0.85,  # Flexible delivery acceptable
        'cold_chain_multiplier': 1.0,
        'priority': 'carbon',  # Can optimize for sustainability
        'impact_multiplier': 1.0,  # Lower urgency
        'regulatory_basis': 'Consumer goods - no regulatory mandate',
        'benchmark_source': 'FCBCO 2024: Fashion 70-80%; Service Club: 85-90% avg'
    },
    'cosmetics': {
        'agent': 'standard',
        'tier': 3,
        'tier_name': 'STANDARD',
        'on_time_threshold': 0.85,  # Non-urgent consumer goods
        'cold_chain_multiplier': 1.0,
        'priority': 'carbon',  # Can optimize for sustainability
        'impact_multiplier': 1.0,  # Lower urgency
        'regulatory_basis': 'Consumer goods - no regulatory mandate',
        'benchmark_source': 'Opensend 2025: Industry average 85-90%'
    }
}


def get_agent_config(package_type: str) -> dict:
    """
    Get agent configuration for a package type.
    
    Args:
        package_type: Type of package (pharmacy, clothing, etc.)
    
    Returns:
        Configuration dictionary with thresholds and parameters
    """
    return AGENT_MAPPING.get(package_type.lower(), {
        'agent': 'high_value',
        'tier': 2,
        'tier_name': 'HIGH_VALUE',
        'on_time_threshold': 0.95,
        'cold_chain_multiplier': 1.0,
        'priority': 'reliability',
        'impact_multiplier': 1.5,
        'regulatory_basis': 'Default - Industry SLA standards',
        'benchmark_source': 'Service Club 2025: E-commerce 94-97%'
    })


def get_stakes_level(package_type: str) -> str:
    """
    Get stakes level for a package type.
    
    Args:
        package_type: Type of package
    
    Returns:
        Stakes level string: 'critical', 'high_value', or 'standard'
    """
    config = get_agent_config(package_type)
    return config['agent']


def get_tier_info(package_type: str) -> dict:
    """
    Get tier information with justification for research paper.
    
    Args:
        package_type: Type of package
    
    Returns:
        Dictionary with tier details and citations
    """
    config = get_agent_config(package_type)
    return {
        'tier': config.get('tier', 2),
        'tier_name': config.get('tier_name', 'HIGH_VALUE'),
        'on_time_threshold': config.get('on_time_threshold', 0.95),
        'regulatory_basis': config.get('regulatory_basis', ''),
        'benchmark_source': config.get('benchmark_source', '')
    }


def get_classification_summary() -> str:
    """
    Get a summary of the classification framework for documentation.
    
    Returns:
        Formatted string with classification details
    """
    summary = """
================================================================================
CRITICALITY-BASED PRODUCT CLASSIFICATION FRAMEWORK
================================================================================

TIER 1: CRITICAL (≥99% On-Time)
  Products: pharmacy, groceries
  Regulatory: WHO GDP, FDA DSCSA, EU 2013/C 343/01
  Benchmark: Pharmaceutical logistics achieves 99%+ (World Courier)

TIER 2: HIGH-VALUE (≥95% On-Time)  
  Products: automobile parts, furniture, documents, fragile items, electronics
  Benchmark: E-commerce best-in-class 94-97% (Service Club 2025)

TIER 3: STANDARD (≥85% On-Time)
  Products: clothing, cosmetics
  Benchmark: Industry average 85-90% (Opensend 2025)

================================================================================
"""
    return summary


# Print classification summary when module is run directly
if __name__ == "__main__":
    print(get_classification_summary())
    
    # Show all package types with their tiers
    print("\nPackage Type Classifications:")
    print("-" * 60)
    for pkg_type, config in AGENT_MAPPING.items():
        print(f"{pkg_type:20} | Tier {config['tier']}: {config['tier_name']:12} | "
              f"≥{config['on_time_threshold']*100:.0f}%")
