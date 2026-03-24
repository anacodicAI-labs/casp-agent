from __future__ import annotations

import json

from tools._common import _get_orchestrator, _json_safe, tool


@tool
def carbon_analysis_tool(carrier_options_json: str, package_type: str, optimization_result_json: str) -> str:
    carrier_options = json.loads(carrier_options_json)
    opt_result = json.loads(optimization_result_json)

    orch = _get_orchestrator()
    carbon_result = orch.carbon_service.analyze(
        carrier_options=carrier_options,
        package_type=package_type,
        optimization_result=opt_result,
        country="india",
    )
    return json.dumps(_json_safe(carbon_result))

