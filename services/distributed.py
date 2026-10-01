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
import urllib.error
import urllib.request
from itertools import zip_longest
from pathlib import Path
from typing import Any

API_ADDRESS = "https://tie6e8nzmi.execute-api.us-east-1.amazonaws.com/algo1"
REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_AGENTS = 150
AGENT_FILES = {"DG": "dg.csv", "RES": "res.csv", "LOAD": "demand.csv", "ESS": "ess.csv"}
PROFILE_FILES = {"RES": "renewablemultiplier.csv", "LOAD": "demandmultiplier.csv"}
ITERATION_SERIES = ("lambda", "delta", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the distributed algorithm for a selected Data-v2 bus."
    )
    parser.add_argument(
        "--bus",
        required=True,
        help="Bus dataset folder name under Data-v2, for example 13bus_base",
    )
    parser.add_argument(
        "--use_chronic",
        required=True,
        help="Chronic folder name, for example chronic_1",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of timesteps to simulate (default: 24)",
    )
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    return args


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_profiles(csv_path: Path) -> dict[str, list[float]]:
    if not csv_path.exists():
        return {}
    rows = read_csv_rows(csv_path)
    names = rows[0].keys() if rows else ()
    return {name: [float(row[name]) for row in rows] for name in names if name}


def build_agent(
    agent_type: str, row: dict[str, str], profiles: dict[str, dict[str, list[float]]]
) -> dict[str, Any]:
    def number(column: str, default: float) -> float:
        return float(row.get(column, default))

    agent: dict[str, Any] = {"type": agent_type}
    if agent_type == "ESS":
        agent["soc_init"] = number("soc_init", 0.3)
        params = {
            "capacity": number("max_e_kwh", 0.0),
            "efficiency": number("efficiency", 1.0),
            "soc_min": number("soc_min", 0.0),
            "soc_max": number("soc_max", 1.0),
            "soc_frequency_reserve": number("soc_frequency_reserve", 0.0),
        }
    elif agent_type == "LOAD":
        params = {"loss_factor": number("loss_factor", 0.0)}
    else:
        params = {"max": number("max_p_kw", 0.0)}

    if agent_type in profiles:
        multipliers = profiles[agent_type].get(row.get("profile", ""), [1.0])
        agent["values"] = [round(number("max_p_kw", 0.0) * m, 6) for m in multipliers]

    agent["params"] = params | {
        "alpha": number("alpha", 0.0),
        "beta": number("beta", 0.0),
        "gamma": number("gamma", 0.0) if agent_type == "DG" else 0.0,
        "scale_cost": number("scale_cost", 1.0),
    }
    return agent


def load_agents(bus_dir: Path, chronic_name: str) -> dict[str, dict[str, Any]]:
    chronic_dir = bus_dir / "chronics" / chronic_name
    if not chronic_dir.exists():
        chronic_dir = bus_dir

    profiles = {
        agent_type: load_profiles(chronic_dir / file_name)
        for agent_type, file_name in PROFILE_FILES.items()
    }
    agents = {
        f"{agent_type}{row['element_id']}": build_agent(agent_type, row, profiles)
        for agent_type, file_name in AGENT_FILES.items()
        for row in read_csv_rows(bus_dir / file_name)
    }
    if len(agents) > MAX_AGENTS:
        raise ValueError(
            f"This workflow supports at most {MAX_AGENTS} agents for resource "
            f"constraints, but {len(agents)} were loaded."
        )
    return agents


def build_payload(
    agents: dict[str, dict[str, Any]], timestep_index: int, ess_states: dict[str, float]
) -> dict[str, dict[str, Any]]:
    payload = {}
    for name, agent in agents.items():
        if "values" in agent:
            values = agent["values"]
            state = {"value": values[min(timestep_index, len(values) - 1)]}
        elif name in ess_states:
            state = {"soc_current": round(ess_states[name], 6)}
        else:
            state = {}
        payload[name] = {"type": agent["type"], **state, **agent["params"]}
    return payload


def send_step_request(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        API_ADDRESS,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(
            f"API request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc


def save_iteration_results(
    iteration_results: dict[str, dict[str, list[float]]] | None,
    output_dir: Path,
    timestep: int,
) -> None:
    if not iteration_results:
        print(f"[timestep {timestep}] iteration_results not received.")
        return

    iteration_dir = output_dir / "iteration_results"
    iteration_dir.mkdir(exist_ok=True)
    csv_path = iteration_dir / f"timestep_{timestep}.csv"
    columns = [
        f"{name}_{key}" for name in iteration_results for key in ITERATION_SERIES
    ]
    series = [
        agent_series.get(key, [])
        for agent_series in iteration_results.values()
        for key in ITERATION_SERIES
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", *columns])
        writer.writerows(
            [index, *values]
            for index, values in enumerate(zip_longest(*series, fillvalue=""), 1)
        )
    print(f"[timestep {timestep}] iteration_results received and saved to {csv_path}")


def normalize_results(results: Any) -> dict[str, float]:
    if not isinstance(results, dict):
        return {}
    normalized = {}
    for name, value in results.items():
        try:
            normalized[str(name)] = float(value)
        except (TypeError, ValueError):
            pass
    return normalized


def update_ess_soc(
    soc: float, setpoint: float, efficiency: float, capacity: float
) -> float:
    energy = setpoint / efficiency if setpoint > 0 else setpoint * efficiency
    return soc - energy / capacity


def build_operation_row(
    timestep: int,
    convergence_iteration: int,
    agents: dict[str, dict[str, Any]],
    payload: dict[str, dict[str, Any]],
    results: dict[str, float],
    ess_states: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestep": timestep,
        "convergence_iteration": convergence_iteration,
    }
    for name, agent in agents.items():
        result = results.get(name, 0.0)
        if agent["type"] == "DG":
            row[f"{name}_value"] = result
        elif agent["type"] == "RES":
            row[f"{name}_value"] = payload[name]["value"]
            row[f"{name}_curtail"] = round(-abs(result), 6)
        elif agent["type"] == "LOAD":
            value = payload[name]["value"]
            row[f"{name}_value"] = value
            row[f"{name}_loss"] = round(
                value * (1.0 + agent["params"]["loss_factor"]), 6
            )
            row[f"{name}_shed"] = round(-abs(result), 6)
        else:
            row[f"{name}_value"] = result
            row[f"{name}_soc"] = round(ess_states[name], 6)
    return row


def run_horizon_simulation(
    agents: dict[str, dict[str, Any]], output_dir: Path, horizon: int
) -> bool:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    ess_states = {
        name: agent["soc_init"]
        for name, agent in agents.items()
        if agent["type"] == "ESS"
    }
    rows = []
    for timestep in range(1, horizon + 1):
        payload = build_payload(agents, timestep - 1, ess_states)
        response = send_step_request(payload)
        summary = {
            key: value for key, value in response.items() if key != "iteration_results"
        }
        print(f"[timestep {timestep}] response: {json.dumps(summary, indent=2)}")
        save_iteration_results(response.get("iteration_results"), output_dir, timestep)

        if str(response.get("status")).lower() == "false":
            print(response.get("message") or "Simulation aborted by API.")
            return False

        results = normalize_results(response.get("results"))
        for name, soc in ess_states.items():
            if name in results:
                params = agents[name]["params"]
                ess_states[name] = update_ess_soc(
                    soc, results[name], params["efficiency"], params["capacity"]
                )

        convergence = response.get("convergence_iteration")
        convergence = int(convergence) if isinstance(convergence, (int, float)) else 0
        rows.append(
            build_operation_row(
                timestep, convergence, agents, payload, results, ess_states
            )
        )

    with (output_dir / "operation_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return True


def main() -> None:
    args = parse_args()
    bus_dir = REPO_ROOT / "Data-v2" / args.bus
    if not bus_dir.exists():
        raise FileNotFoundError(
            f"Unable to locate bus dataset for '{args.bus}' under {REPO_ROOT / 'Data-v2'}"
        )
    output_dir = REPO_ROOT / "results" / bus_dir.name
    if run_horizon_simulation(
        load_agents(bus_dir, args.use_chronic), output_dir, args.horizon
    ):
        print(f"Simulation completed. Results written to {output_dir}")


if __name__ == "__main__":
    main()
