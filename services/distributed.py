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
    """Parse command-line arguments for the distributed algorithm CLI.

    Args:
        argv (List[str] | None): Argument strings to parse instead of
            sys.argv, or None to parse sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments with attributes ``bus``,
        ``use_chronic``, and ``horizon``.

    Raises:
        SystemExit: If required arguments are missing or invalid, as
            raised internally by argparse.
    """
    parser = argparse.ArgumentParser(description="Run the distributed algorithm for a selected Data-v2 bus.")
    parser.add_argument("--bus", required=True, help="Bus dataset folder name under Data-v2, for example 13bus_base")
    parser.add_argument("--use_chronic", required=True, help="Chronic folder name, for example chronic_1")
    parser.add_argument("--horizon", type=int, default=24, help="Number of timesteps to simulate (default: 24)")
    return parser.parse_args(argv)


def resolve_repo_root() -> Path:
    """Resolve the repository root directory relative to this script.

    Returns:
        Path: Absolute path to the repository root (the parent of the
        directory containing this file).
    """
    return Path(__file__).resolve().parent.parent


def resolve_bus_dir(bus_name: str, repo_root: Path | None = None) -> Path:
    """Resolve the dataset directory for a given bus name under Data-v2.

    Args:
        bus_name (str): Bus dataset folder name under Data-v2, for
            example "13bus_base".
        repo_root (Path | None): Repository root to resolve the dataset
            under, or None to auto-detect via resolve_repo_root().

    Returns:
        Path: Absolute path to the resolved bus dataset directory.

    Raises:
        FileNotFoundError: If no matching directory exists under
            <repo_root>/Data-v2.
    """
    base = repo_root or resolve_repo_root()
    candidate = base / "Data-v2" / bus_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unable to locate bus dataset for '{bus_name}' under {base / 'Data-v2'}")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of header-keyed row dictionaries.

    Args:
        csv_path (Path): Path to the CSV file to read.

    Returns:
        List[Dict[str, str]]: One dictionary per data row, keyed by the
        (stripped, BOM-free) header column names. Returns an empty list
        if the file has no rows.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        OSError: If the file cannot be opened or read.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    header = [column.strip().lstrip("\ufeff") for column in rows[0]]
    return [dict(zip(header, row)) for row in rows[1:]]


def load_profile_multipliers(csv_path: Path) -> Dict[str, List[float]]:
    """Load per-profile multiplier series from a chronic CSV file.

    Args:
        csv_path (Path): Path to a profile-multiplier CSV (for example
            demandmultiplier.csv, renewablemultiplier.csv, or
            prices.csv), where each non-empty column header is a
            profile name and each row holds one timestep's values.

    Returns:
        Dict[str, List[float]]: Mapping of profile name to its list of
        multiplier values in row order. Returns an empty dict if the
        file does not exist or has no rows.

    Raises:
        ValueError: If a value in the CSV cannot be converted to float.
    """
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
    """Validate that the number of loaded agents does not exceed a limit.

    Args:
        agents (Dict[str, Dict[str, Any]]): Mapping of agent name to
            agent configuration.
        max_agents (int): Maximum number of agents supported by the
            workflow. Defaults to 150.

    Returns:
        None

    Raises:
        ValueError: If len(agents) exceeds max_agents.
    """
    if len(agents) > max_agents:
        raise ValueError(
            f"This workflow supports at most {max_agents} agents for resource constraints, but {len(agents)} were loaded."
        )


def load_bus_dataset(bus_dir: Path, chronic_name: str) -> Dict[str, Any]:
    """Load a bus dataset's static topology data and chronic profiles.

    Reads demand/res/dg/ess CSVs from bus_dir, builds an agent
    configuration dictionary for every DG, RES, LOAD, and ESS element,
    attaches demand/renewable profile multiplier series to RES/LOAD
    agents, and validates the total agent count.

    Args:
        bus_dir (Path): Path to the bus dataset directory (as returned
            by resolve_bus_dir).
        chronic_name (str): Chronic folder name under bus_dir/chronics
            to load profile multipliers and prices from. Falls back to
            bus_dir itself if the chronic folder does not exist.

    Returns:
        Dict[str, Any]: Dataset dictionary with keys "bus_dir",
        "chronic_name", "agents", "demand_multipliers",
        "renewable_multipliers", and "prices".

    Raises:
        FileNotFoundError: If a required demand.csv, res.csv, dg.csv,
            or ess.csv file is missing under bus_dir.
        ValueError: If the number of loaded agents exceeds the limit
            enforced by validate_agent_limit, or if a numeric field in
            the source CSVs cannot be converted to float.
    """
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
            "scale_cost": float(row.get("scale_cost", 1.0)),
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
            "scale_cost": float(row.get("scale_cost", 1.0)),
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
            "scale_cost": float(row.get("scale_cost", 1.0)),
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
            "soc_frequency_reserve": float(row.get("soc_frequency_reserve", 0.0)),
            "alpha": float(row.get("alpha", 0.0)),
            "beta": float(row.get("beta", 0.0)),
            "gamma": 0.0,
            "scale_cost": float(row.get("scale_cost", 1.0)),
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
    """Build the per-agent request payload for a single API timestep call.

    Args:
        agent_name (str): Name of the agent (unused directly, kept for
            call-site symmetry with other per-agent helpers).
        agent_type (str): Agent category, one of "DG", "RES", "LOAD",
            "ESS", or another custom type.
        agent_config (Dict[str, Any]): Agent configuration dictionary
            as produced by load_bus_dataset.
        timestep_value (float): Value relevant to this agent at the
            current timestep (for example, forecast RES/LOAD power or
            DG setpoint), used depending on agent_type.

    Returns:
        Dict[str, Any]: Payload fields for this agent to send to the
        distributed algorithm API. Unrecognized agent_type values fall
        back to a minimal payload with "type" and rounded "value".
    """
    if agent_type == "DG":
        return {
            "type": "DG",
            "max": agent_config.get("max", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
            "scale_cost": agent_config.get("scale_cost", 1.0),
        }

    if agent_type == "RES":
        return {
            "type": "RES",
            "value": round(timestep_value, 6),
            "max": agent_config.get("max", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
            "scale_cost": agent_config.get("scale_cost", 1.0),
        }

    if agent_type == "LOAD":
        return {
            "type": "LOAD",
            "value": round(timestep_value, 6),
            "loss_factor": agent_config.get("loss_factor", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
            "scale_cost": agent_config.get("scale_cost", 1.0),
        }

    if agent_type == "ESS":
        return {
            "type": "ESS",
            "soc_current": round(agent_config.get("soc_current", 0.3), 6),
            "capacity": agent_config.get("capacity", 0.0),
            "efficiency": agent_config.get("efficiency", 1.0),
            "soc_min": agent_config.get("soc_min", 0.0),
            "soc_max": agent_config.get("soc_max", 1.0),
            "soc_frequency_reserve": agent_config.get("soc_frequency_reserve", 0.0),
            "alpha": agent_config.get("alpha", 0.0),
            "beta": agent_config.get("beta", 0.0),
            "gamma": agent_config.get("gamma", 0.0),
            "scale_cost": agent_config.get("scale_cost", 1.0),
        }

    return {"type": agent_type, "value": round(timestep_value, 6)}


def build_payload_for_timestep(bus_data: Dict[str, Any], timestep_index: int, ess_states: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Build the full API request payload for every agent at one timestep.

    Computes each RES/LOAD agent's forecast value from its profile
    multiplier at timestep_index, applies the current ESS state of
    charge, and delegates per-agent payload construction to
    build_agent_payload.

    Args:
        bus_data (Dict[str, Any]): Dataset dictionary as returned by
            load_bus_dataset.
        timestep_index (int): Zero-based index into each agent's
            profile_values series for the current timestep.
        ess_states (Dict[str, float]): Mapping of ESS agent name to its
            current state of charge, used to override each ESS agent's
            "soc_current" for this timestep.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of agent name to its request
        payload for this timestep.
    """
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
    """Send one timestep's payload to the distributed algorithm API.

    Args:
        payload (Dict[str, Any]): Per-agent request payload for a
            single timestep, as built by build_payload_for_timestep.

    Returns:
        Dict[str, Any]: Parsed JSON response body from the API.

    Raises:
        RuntimeError: If the request fails with an HTTP error status or
            a lower-level URL/connection error.
    """
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


def print_response_summary(response: Dict[str, Any], timestep: int) -> None:
    """Print the API response for a timestep, excluding iteration_results.

    iteration_results can be large (or already stripped server-side when
    it would push the response over the API's size limit), so it is
    never dumped to the console -- only status/message/results/
    convergence_iteration (and any other top-level fields) are printed.
    Whether iteration_results was actually received is reported
    separately by handle_iteration_results().

    Args:
        response (Dict[str, Any]): Parsed JSON response from the API
            for this timestep.
        timestep (int): 1-based timestep number, used in the printed
            log prefix.

    Returns:
        None
    """
    printable = {key: value for key, value in response.items() if key != "iteration_results"}
    print(f"[timestep {timestep}] response: {json.dumps(printable, indent=2)}")


def save_iteration_results_csv(iteration_results: Dict[str, Dict[str, List[float]]], output_dir: Path, timestep: int) -> Path:
    """Write one CSV of per-iteration lambda/delta/p history for every agent.

    Output is written to
    results/<bus>/iteration_results/timestep_<n>.csv with columns
    "iteration", "<agent_id>_lambda", "<agent_id>_delta",
    "<agent_id>_p", ... for each agent in iteration_results.

    Args:
        iteration_results (Dict[str, Dict[str, List[float]]]): Mapping
            of agent name to a dict of series name ("lambda", "delta",
            "p") to per-iteration value lists.
        output_dir (Path): Base results directory for this bus dataset
            (results/<bus>).
        timestep (int): 1-based timestep number, used to name the
            output CSV file.

    Returns:
        Path: Path to the written CSV file.

    Raises:
        OSError: If the iteration_results directory cannot be created
            or the CSV file cannot be written.
    """
    iteration_dir = output_dir / "iteration_results"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    csv_path = iteration_dir / f"timestep_{timestep}.csv"

    agent_names = list(iteration_results.keys())
    series_keys = ("lambda", "delta", "p")

    n_rows = 0
    for agent_series in iteration_results.values():
        for key in series_keys:
            n_rows = max(n_rows, len(agent_series.get(key, [])))

    columns = ["iteration"]
    for agent_name in agent_names:
        for key in series_keys:
            columns.append(f"{agent_name}_{key}")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(n_rows):
            row: List[Any] = [i + 1]
            for agent_name in agent_names:
                agent_series = iteration_results[agent_name]
                for key in series_keys:
                    values = agent_series.get(key, [])
                    row.append(values[i] if i < len(values) else "")
            writer.writerow(row)

    return csv_path


def handle_iteration_results(response: Dict[str, Any], output_dir: Path, timestep: int) -> Path | None:
    """Report and, if present, persist a timestep's iteration_results.

    Checks whether the API response included iteration_results for
    this timestep and, if so, saves it to CSV via
    save_iteration_results_csv.

    Args:
        response (Dict[str, Any]): Parsed JSON response from the API
            for this timestep.
        output_dir (Path): Base results directory for this bus dataset
            (results/<bus>).
        timestep (int): 1-based timestep number.

    Returns:
        Path | None: Path to the saved CSV file, or None if
        iteration_results was missing or empty (for example, when the
        API omitted it for response-size reasons, as explained in the
        "message" field already printed above).

    Raises:
        OSError: If the iteration_results directory cannot be created
            or the CSV file cannot be written.
    """
    iteration_results = response.get("iteration_results")
    if not iteration_results:
        print(f"[timestep {timestep}] iteration_results not received.")
        return None

    csv_path = save_iteration_results_csv(iteration_results, output_dir, timestep)
    print(f"[timestep {timestep}] iteration_results received and saved to {csv_path}")
    return csv_path


def normalize_results(payload_results: Any) -> Dict[str, float]:
    """Coerce an API "results" payload into a clean agent-name-to-float map.

    Args:
        payload_results (Any): Value of the "results" field from an API
            response, expected to be a dict mapping agent name to a
            numeric-like value.

    Returns:
        Dict[str, float]: Mapping of agent name to float value. Entries
        that are not dict-convertible to float are skipped. Returns an
        empty dict if payload_results is not a dict.
    """
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
    """Update an ESS agent's state of charge after a dispatch setpoint.

    Discharging (setpoint > 0) reduces SOC by the setpoint scaled by
    the round-trip efficiency; charging (setpoint <= 0) reduces SOC by
    the setpoint scaled inversely by efficiency (charging setpoints are
    expected to be negative, so this increases SOC).

    Args:
        soc_current (float): State of charge before applying setpoint,
            as a fraction of capacity.
        setpoint (float): Dispatched power for this timestep; positive
            for discharge, non-positive for charge.
        efficiency (float): Round-trip efficiency factor used to derate
            energy delivered/absorbed.
        e_max (float): ESS energy capacity used to convert power to a
            fractional SOC change.

    Returns:
        float: Updated state of charge as a fraction of capacity.

    Raises:
        ZeroDivisionError: If efficiency or e_max is zero.
    """
    if setpoint > 0:
        return soc_current - ((setpoint / efficiency) / e_max)
    return soc_current - ((setpoint * efficiency) / e_max)


def ensure_output_dirs(output_dir: Path) -> None:
    """Create the results output directory if it does not already exist.

    Args:
        output_dir (Path): Directory to create, including any missing
            parent directories.

    Returns:
        None

    Raises:
        OSError: If the directory cannot be created (for example, due
            to insufficient permissions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def build_operation_columns(agents: Dict[str, Dict[str, Any]]) -> List[str]:
    """Build the operation_results.csv column header list for all agents.

    Args:
        agents (Dict[str, Dict[str, Any]]): Mapping of agent name to
            agent configuration, as produced by load_bus_dataset.

    Returns:
        List[str]: Ordered column names starting with "timestep" and
        "convergence_iteration", followed by type-specific value/loss/
        shed/curtail/soc columns per agent.
    """
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
    """Build one operation_results.csv row for a single timestep.

    Args:
        timestep_number (int): 1-based timestep number for this row.
        convergence_iteration (int): Number of iterations the
            distributed algorithm took to converge at this timestep.
        agents (Dict[str, Dict[str, Any]]): Mapping of agent name to
            agent configuration, as produced by load_bus_dataset.
        profile_values (Dict[str, float]): Mapping of RES/LOAD agent
            name to their dispatched forecast value for this timestep.
        normalized_results (Dict[str, float]): Mapping of agent name to
            the API-returned setpoint/curtailment/shed value for this
            timestep, as produced by normalize_results.
        ess_states (Dict[str, float]): Mapping of ESS agent name to its
            state of charge after this timestep's dispatch.

    Returns:
        Dict[str, Any]: Row dictionary keyed by the columns produced by
        build_operation_columns for this timestep.
    """
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
    """Write the full-horizon operation results table to CSV.

    Args:
        output_dir (Path): Base results directory for this bus dataset
            (results/<bus>); the file is written to
            output_dir/operation_results.csv.
        rows (List[Dict[str, Any]]): Per-timestep row dictionaries, as
            produced by build_operation_row.
        columns (List[str]): Ordered column names to use as the CSV
            header, as produced by build_operation_columns.

    Returns:
        None

    Raises:
        OSError: If the output file cannot be written.
    """
    operation_path = output_dir / "operation_results.csv"
    with operation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_horizon_simulation(bus_data: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    """Run the distributed dispatch algorithm across a full time horizon.

    Recreates the bus's results directory, then for each timestep
    builds and sends the API payload, records the response summary and
    any iteration_results, updates ESS states of charge, and
    accumulates operation rows. Writes operation_results.csv on
    successful completion of all timesteps.

    Args:
        bus_data (Dict[str, Any]): Dataset dictionary as returned by
            load_bus_dataset.
        horizon (int): Number of timesteps to simulate.

    Returns:
        Dict[str, Any]: Result dictionary with keys "output_dir" (Path
        to the results directory), "rows" (accumulated operation rows),
        "status" (bool, True on full success), and "message" (str,
        present only when status is False).

    Raises:
        RuntimeError: If an API request fails for any timestep, as
            raised by send_step_request.
    """
    repo_root = resolve_repo_root()
    output_dir = repo_root / "results" / bus_data["bus_dir"].name

    # Cleared and recreated up front (rather than only on a fully successful
    # run) so that per-timestep iteration_results CSVs can be written to
    # output_dir as the horizon loop progresses below.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_output_dirs(output_dir)

    agents = bus_data["agents"]
    ess_states = {agent_name: agent_config.get("soc_current", 0.3) for agent_name, agent_config in agents.items() if agent_config["type"] == "ESS"}
    profile_values: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    for timestep in range(1, horizon + 1):
        payload = build_payload_for_timestep(bus_data, timestep - 1, ess_states)
        response = send_step_request(payload)
        print_response_summary(response, timestep)
        handle_iteration_results(response, output_dir, timestep)

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

    columns = build_operation_columns(agents)
    write_operation_results(output_dir, rows, columns)

    return {"output_dir": output_dir, "rows": rows, "status": True}


def main(argv: List[str] | None = None) -> int:
    """Entry point: parse arguments and run the distributed algorithm workflow.

    Args:
        argv (List[str] | None): Argument strings to parse instead of
            sys.argv, or None to parse sys.argv.

    Returns:
        int: Process exit code; always 0 on completion, whether the
        simulation succeeded or was aborted by the API.

    Raises:
        FileNotFoundError: If the requested bus dataset cannot be
            located, as raised by resolve_bus_dir.
        ValueError: If the bus dataset exceeds the supported agent
            count, as raised by validate_agent_limit.
        RuntimeError: If an API request fails, as raised by
            send_step_request.
    """
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