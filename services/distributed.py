#!/usr/bin/env python3
# =============================================================================
# distributed.py
# Run the distributed algorithm workflow for a selected Data-v2 bus dataset.
#
# This service submits the per-timestep dispatch problem for a chosen bus system,
# collects the operation results, and writes them to results/<bus>/operation_results.csv
# for downstream plotting and analysis.
#
# Usage:
#   python services/distributed.py --bus 13bus_base --use_chronic chronic_1 --horizon 24
# =============================================================================

# Author:      Talha Rehman                  (Incheon National University)
# Co-authors:  Muhammad Ahsan Khan           (Incheon National University)
#               Woon-Gyu Lee                  (Incheon National University)
#               Hyeong-Jun Yoo                (KERI)
#               Hak-Man Kim (Corresponding)   (Incheon National University)

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_API_ADDRESS = "https://tie6e8nzmi.execute-api.us-east-1.amazonaws.com/algo1"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the distributed algorithm for a selected Data-v2 bus.")
    parser.add_argument("--bus", required=True, help="Bus dataset folder name under Data-v2, for example 13bus_base")
    parser.add_argument("--use_chronic", required=True, help="Chronic folder name, for example chronic_1")
    parser.add_argument("--horizon", type=int, default=24, help="Number of timesteps to simulate (default: 24)")
    return parser.parse_args(argv)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_bus_dir(bus_name: str, repo_root: Path | None = None) -> Path:
    base = repo_root or resolve_repo_root()
    candidate = base / "Data-v2" / bus_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unable to locate bus dataset for '{bus_name}' under {base / 'Data-v2'}")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    header = [column.strip().lstrip("\ufeff") for column in rows[0]]
    return [dict(zip(header, row)) for row in rows[1:]]


def load_profile_multipliers(csv_path: Path) -> Dict[str, List[float]]:
    if not csv_path.exists():
        return {}

    rows = read_csv_rows(csv_path)
    if not rows:
        return {}

    profile_values: Dict[str, List[float]] = {}
    for key in rows[0].keys():
        if key == "":
            continue
        profile_values[key] = []

    for row in rows:
        for profile_name, value in row.items():
            if profile_name in profile_values:
                profile_values[profile_name].append(float(value))

    return profile_values


def validate_agent_limit(agents: Dict[str, Dict[str, Any]], max_agents: int = 150) -> None:
    if len(agents) > max_agents:
        raise ValueError(
            f"This workflow supports at most {max_agents} agents for resource constraints, but {len(agents)} were loaded."
        )


def load_bus_dataset(bus_dir: Path, chronic_name: str) -> Dict[str, Any]:
    chronic_dir = bus_dir / "chronics" / chronic_name
    if not chronic_dir.exists():
        chronic_dir = bus_dir

    demand_rows = read_csv_rows(bus_dir / "demand.csv")
    res_rows = read_csv_rows(bus_dir / "res.csv")
    dg_rows = read_csv_rows(bus_dir / "dg.csv")
    ess_rows = read_csv_rows(bus_dir / "ess.csv")

    demand_multipliers = load_profile_multipliers(chronic_dir / "demandmultiplier.csv")
    renewable_multipliers = load_profile_multipliers(chronic_dir / "renewablemultiplier.csv")
    prices = load_profile_multipliers(chronic_dir / "prices.csv") if (chronic_dir / "prices.csv").exists() else {}

    agents: Dict[str, Dict[str, Any]] = {}

    for row in dg_rows:
        agent_name = f"DG{row['element_id']}"
        agents[agent_name] = {
            "type": "DG",
            "element_id": row["element_id"],
            "max": float(row.get("max_p_kw", 0.0)),
            "alpha": float(row.get("alpha", 0.0)),
            "beta": float(row.get("beta", 0.0)),
            "gamma": float(row.get("gamma", 0.0)),
            "value": 0.0,
        }

    for row in res_rows:
        agent_name = f"RES{row['element_id']}"
        agents[agent_name] = {
            "type": "RES",
            "element_id": row["element_id"],
            "max": float(row.get("max_p_kw", 0.0)),
            "alpha": float(row.get("alpha", 0.0)),
            "beta": float(row.get("beta", 0.0)),
            "gamma": 0.0,
            "profile": row.get("profile", ""),
            "profile_values": [],
            "value": 0.0,
        }

    for row in demand_rows:
        agent_name = f"LOAD{row['element_id']}"
        agents[agent_name] = {
            "type": "LOAD",
            "element_id": row["element_id"],
            "max": float(row.get("max_p_kw", 0.0)),
            "alpha": float(row.get("alpha", 0.0)),
            "beta": float(row.get("beta", 0.0)),
            "gamma": 0.0,
            "profile": row.get("profile", ""),
            "profile_values": [],
            "loss_factor": float(row.get("loss_factor", 0.0)),
            "value": 0.0,
        }

    for row in ess_rows:
        agent_name = f"ESS{row['element_id']}"
        agents[agent_name] = {
            "type": "ESS",
            "element_id": row["element_id"],
            "capacity": float(row.get("max_e_kwh", 0.0)),
            "efficiency": float(row.get("efficiency", 1.0)),
            "soc_min": float(row.get("soc_min", 0.0)),
            "soc_max": float(row.get("soc_max", 1.0)),
            "soc_current": float(row.get("soc_init", 0.3)),
            "alpha": float(row.get("alpha", 0.0)),
            "beta": float(row.get("beta", 0.0)),
            "gamma": 0.0,
            "value": 0.0,
        }

    for agent_name, agent_config in agents.items():
        if agent_config["type"] in {"RES", "LOAD"}:
            profile_name = agent_config.get("profile", "")
            if profile_name in demand_multipliers and agent_config["type"] == "LOAD":
                agent_config["profile_values"] = [float(value) for value in demand_multipliers[profile_name]]
            elif profile_name in renewable_multipliers and agent_config["type"] == "RES":
                agent_config["profile_values"] = [float(value) for value in renewable_multipliers[profile_name]]
            else:
                agent_config["profile_values"] = [1.0] * 24

    validate_agent_limit(agents)

    return {
        "bus_dir": bus_dir,
        "chronic_name": chronic_name,
        "agents": agents,
        "demand_multipliers": demand_multipliers,
        "renewable_multipliers": renewable_multipliers,
        "prices": prices,
    }


def build_agent_payload(agent_name: str, agent_type: str, agent_config: Dict[str, Any], timestep_value: float) -> Dict[str, Any]:
    if agent_type == "DG":
        return {
            "type": "DG",
            "max": agent_config.get("max", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
        }

    if agent_type == "RES":
        return {
            "type": "RES",
            "value": round(timestep_value, 6),
            "max": agent_config.get("max", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
        }

    if agent_type == "LOAD":
        return {
            "type": "LOAD",
            "value": round(timestep_value, 6),
            "loss_factor": agent_config.get("loss_factor", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
        }

    if agent_type == "ESS":
        return {
            "type": "ESS",
            "soc_current": round(agent_config.get("soc_current", 0.3), 6),
            "capacity": agent_config.get("capacity", 0.0),
            "efficiency": agent_config.get("efficiency", 1.0),
            "soc_min": agent_config.get("soc_min", 0.0),
            "soc_max": agent_config.get("soc_max", 1.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
        }

    return {"type": agent_type, "value": round(timestep_value, 6)}


def build_payload_for_timestep(bus_data: Dict[str, Any], timestep_index: int, ess_states: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for agent_name, agent_config in bus_data["agents"].items():
        agent_type = agent_config["type"]
        if agent_type in {"RES", "LOAD"}:
            profile_values = agent_config.get("profile_values", [1.0] * 24)
            multiplier = profile_values[timestep_index] if timestep_index < len(profile_values) else profile_values[-1]
            base_value = agent_config.get("max", 0.0)
            timestep_value = base_value * multiplier
        elif agent_type == "DG":
            timestep_value = agent_config.get("value", 0.0)
        else:
            timestep_value = 0.0

        if agent_type == "ESS":
            agent_config = dict(agent_config)
            agent_config["soc_current"] = ess_states.get(agent_name, agent_config.get("soc_current", 0.3))

        payload[agent_name] = build_agent_payload(agent_name, agent_type, agent_config, timestep_value)

    return payload


def send_step_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        DEFAULT_API_ADDRESS,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response_handle:
            body = response_handle.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc


def normalize_results(payload_results: Any) -> Dict[str, float]:
    if not isinstance(payload_results, dict):
        return {}

    normalized: Dict[str, float] = {}
    for agent_name, value in payload_results.items():
        try:
            normalized[str(agent_name)] = float(value)
        except (TypeError, ValueError):
            continue

    return normalized


def update_ess_soc(soc_current: float, setpoint: float, efficiency: float, e_max: float) -> float:
    if setpoint > 0:
        return soc_current - ((setpoint / efficiency) / e_max)
    return soc_current - ((setpoint * efficiency) / e_max)


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def build_operation_columns(agents: Dict[str, Dict[str, Any]]) -> List[str]:
    columns = ["timestep", "convergence_iteration"]
    for agent_name, agent_config in agents.items():
        agent_type = agent_config["type"]
        if agent_type == "DG":
            columns.append(f"{agent_name}_value")
        elif agent_type == "RES":
            columns.extend([f"{agent_name}_value", f"{agent_name}_curtail"])
        elif agent_type == "LOAD":
            columns.extend([f"{agent_name}_value", f"{agent_name}_loss", f"{agent_name}_shed"])
        elif agent_type == "ESS":
            columns.extend([f"{agent_name}_value", f"{agent_name}_soc"])
    return columns


def build_operation_row(
    timestep_number: int,
    convergence_iteration: int,
    agents: Dict[str, Dict[str, Any]],
    profile_values: Dict[str, float],
    normalized_results: Dict[str, float],
    ess_states: Dict[str, float],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"timestep": timestep_number, "convergence_iteration": convergence_iteration}
    for agent_name, agent_config in agents.items():
        agent_type = agent_config["type"]
        if agent_type == "DG":
            row[f"{agent_name}_value"] = normalized_results.get(agent_name, 0.0)
        elif agent_type == "RES":
            row[f"{agent_name}_value"] = profile_values.get(agent_name, 0.0)
            curtailment = normalized_results.get(agent_name, 0.0)
            row[f"{agent_name}_curtail"] = round(-abs(curtailment), 6)
        elif agent_type == "LOAD":
            base_value = profile_values.get(agent_name, 0.0)
            row[f"{agent_name}_value"] = base_value
            loss_adjusted_value = base_value * (1.0 + agent_config.get("loss_factor", 0.0))
            row[f"{agent_name}_loss"] = round(loss_adjusted_value, 6)
            shed_value = normalized_results.get(agent_name, 0.0)
            row[f"{agent_name}_shed"] = round(-abs(shed_value), 6)
        elif agent_type == "ESS":
            row[f"{agent_name}_value"] = normalized_results.get(agent_name, 0.0)
            row[f"{agent_name}_soc"] = round(ess_states.get(agent_name, 0.0), 6)
    return row


def write_operation_results(output_dir: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    operation_path = output_dir / "operation_results.csv"
    with operation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_horizon_simulation(bus_data: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    repo_root = resolve_repo_root()
    output_dir = repo_root / "results" / bus_data["bus_dir"].name

    agents = bus_data["agents"]
    ess_states = {agent_name: agent_config.get("soc_current", 0.3) for agent_name, agent_config in agents.items() if agent_config["type"] == "ESS"}
    profile_values: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    for timestep in range(1, horizon + 1):
        payload = build_payload_for_timestep(bus_data, timestep - 1, ess_states)
        response = send_step_request(payload)
        print(f"[timestep {timestep}] response: {json.dumps(response, indent=2)}")

        status_value = response.get("status")
        if status_value is False or str(status_value).lower() == "false":
            message = response.get("message") or "Simulation aborted by API."
            print(message)
            return {"output_dir": output_dir, "rows": rows, "status": False, "message": message}

        normalized_results = normalize_results(response.get("results", {}))
        convergence_iteration_value = response.get("convergence_iteration")
        if isinstance(convergence_iteration_value, (int, float)):
            convergence_iteration = int(convergence_iteration_value)
        else:
            convergence_iteration = 0

        for agent_name, agent_config in agents.items():
            if agent_config["type"] in {"RES", "LOAD"}:
                profile_values[agent_name] = payload[agent_name]["value"]

        for agent_name, agent_config in agents.items():
            if agent_config["type"] == "ESS" and agent_name in normalized_results:
                ess_states[agent_name] = update_ess_soc(
                    ess_states[agent_name],
                    normalized_results[agent_name],
                    agent_config.get("efficiency", 1.0),
                    agent_config.get("capacity", 1.0),
                )

        rows.append(
            build_operation_row(
                timestep,
                convergence_iteration,
                agents,
                profile_values,
                normalized_results,
                ess_states,
            )
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_output_dirs(output_dir)

    columns = build_operation_columns(agents)
    write_operation_results(output_dir, rows, columns)

    return {"output_dir": output_dir, "rows": rows, "status": True}


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_root()
    bus_dir = resolve_bus_dir(args.bus, repo_root)
    bus_data = load_bus_dataset(bus_dir, args.use_chronic)
    result = run_horizon_simulation(bus_data, args.horizon)
    if result.get("status") is False:
        return 0
    print(f"Simulation completed. Results written to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
