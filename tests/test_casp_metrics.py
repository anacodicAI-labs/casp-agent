from services.carbon_service import CarbonService


def test_casp_formula_uses_pct_scale():
    # 99% service on 148000 gCO2 => 99 / 148000 (paper scale)
    score = CarbonService._compute_casp_score(99.0, 148000.0)
    assert abs(score - (99.0 / 148000.0)) < 1e-12


def test_casp_tier_thresholds():
    assert CarbonService._casp_tier_from_score(0.0005) == "CRITICAL"
    assert CarbonService._casp_tier_from_score(0.0010) == "HIGH"
    assert CarbonService._casp_tier_from_score(0.0016) == "STANDARD"


def test_casp_zero_carbon_guard():
    assert CarbonService._compute_casp_score(99.0, 0.0) == 0.0
