from pathlib import Path

_LOGO = str(Path(__file__).parent / "logo" / "logo.png")

import streamlit as st
import pandas as pd
import torch
import plotly.graph_objects as go

from energy_estimator import list_boards, list_layer_keys
from energy_estimator.layer_energy_interpolation import estimate_energy


# ─── Session state ────────────────────────────────────────────────────────────

def init_state():
    if "layers" not in st.session_state:
        st.session_state.layers = []
    if "next_id" not in st.session_state:
        st.session_state.next_id = 1


def reset_layers():
    st.session_state.layers = []
    st.session_state.next_id = 1


def get_last_output_dim(default: int = 256) -> int:
    layers = st.session_state.get("layers", [])
    if not layers:
        return default
    last = layers[-1]
    if last["type"] == "Linear":
        return int(last["output dimension"])
    if last["type"] == "Conv2d":
        return int(last["cout"])
    return default


def add_linear(input_dim: int | None = None, output_dim: int = 1024):
    if input_dim is None:
        input_dim = get_last_output_dim(default=256)
    i = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.layers.append({
        "id": i,
        "type": "Linear",
        "name": f"fc{i}",
        "input dimension": int(input_dim),
        "output dimension": int(output_dim),
        "pruning_rate": 0.0,
    })


def add_conv2d():
    i = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.layers.append({
        "id": i,
        "type": "Conv2d",
        "name": f"conv{i}",
        "cin": 3,
        "cout": 16,
        "kh": 3,
        "kw": 3,
        "padding": 0,
        "pruning_rate": 0.0,
    })


def remove_layer(layer_id: int):
    st.session_state.layers = [l for l in st.session_state.layers if l["id"] != layer_id]


# ─── Model loading ────────────────────────────────────────────────────────────

def extract_layers_from_model(uploaded_file) -> list[dict]:
    uploaded_file.seek(0)
    obj = torch.load(uploaded_file, map_location="cpu")

    state_dict = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state_dict, dict):
        raise ValueError("File does not contain a valid state_dict.")

    layers = []
    layer_id = 1

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or not name.endswith(".weight"):
            continue

        shape = tensor.shape
        layer_name = name.replace(".weight", "")

        if len(shape) == 2:
            out_features, in_features = shape
            layers.append({
                "id": layer_id,
                "type": "Linear",
                "name": layer_name,
                "input dimension": int(in_features),
                "output dimension": int(out_features),
                "pruning_rate": 0.0,
            })
            layer_id += 1

        elif len(shape) == 4:
            out_channels, in_channels, kh, kw = shape
            layers.append({
                "id": layer_id,
                "type": "Conv2d",
                "name": layer_name,
                "cin": int(in_channels),
                "cout": int(out_channels),
                "kh": int(kh),
                "kw": int(kw),
                "padding": 0,
                "pruning_rate": 0.0,
            })
            layer_id += 1

    if not layers:
        raise ValueError("No Linear or Conv2d layers found in the state_dict.")
    return layers


# ─── Energy helpers ───────────────────────────────────────────────────────────

def get_layer_key(layer: dict) -> str:
    if layer["type"] == "Linear":
        return "linear"
    k = int(layer["kh"])
    p = int(layer.get("padding", 0))
    return f"conv_k{k}_p{p}"


def get_in_out_dim(layer: dict) -> tuple[int, int]:
    if layer["type"] == "Linear":
        return int(layer["input dimension"]), int(layer["output dimension"])
    return int(layer["cin"]), int(layer["cout"])


# ─── Charts ───────────────────────────────────────────────────────────────────

def plot_per_layer(res: pd.DataFrame, unit: str):
    fig = go.Figure(go.Bar(
        x=res["Used Energy"],
        y=res["Name"],
        orientation="h",
        marker_color="#2ecc71",
        text=[f"{e:.2f} {unit}" for e in res["Used Energy"]],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title=f"Energy ({unit})",
        yaxis_title="Layer",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_cumulative(res: pd.DataFrame, unit: str):
    fig = go.Figure(go.Scatter(
        x=res["Name"],
        y=res["Used Energy"].cumsum(),
        mode="lines+markers",
        marker=dict(size=8, color="#2ecc71"),
        line=dict(width=3, color="#27ae60"),
    ))
    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title=f"Cumulative Energy ({unit})",
        template="plotly_white",
        width=900,
        height=450,
        xaxis=dict(tickangle=-45),
    )
    return fig


# ─── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Energy Estimator", page_icon=_LOGO, layout="wide")
st.title("NeuroEnergy — AI Model Energy Estimation Tool")

init_state()

# Sidebar
st.sidebar.image(_LOGO, width=250)

board = st.sidebar.selectbox("Hardware board", list_boards())

st.sidebar.header("Add layers")
c1, c2 = st.sidebar.columns(2)
with c1:
    st.button("➕ Add Linear", on_click=add_linear)
with c2:
    st.button("➕ Add Conv2d", on_click=add_conv2d)

units = st.sidebar.selectbox("Display units", ["Joules", "mJoules"], index=1)

uploaded_model = st.sidebar.file_uploader("Upload PyTorch model", type=["pt", "pth"])

if st.sidebar.button("Load layers from model"):
    try:
        layers = extract_layers_from_model(uploaded_model)
        st.session_state.layers = layers
        st.session_state.next_id = len(layers) + 1
        st.sidebar.success(f"Loaded {len(layers)} layers.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")

if st.sidebar.button("Clear architecture"):
    reset_layers()
    st.rerun()

# Architecture editor
st.subheader("Architecture")

if not st.session_state.layers:
    st.info("Add layers from the sidebar, or upload a model and load its layers.")
else:
    for layer in st.session_state.layers:
        with st.container(border=True):
            top = st.columns([2, 2, 2, 3, 1])
            layer["name"] = top[0].text_input("Name", layer["name"], key=f"name_{layer['id']}")
            top[1].write(f"**Type:** {layer['type']}")
            layer["pruning_rate"] = top[2].number_input(
                "Pruning rate (0..1)", min_value=0.0, max_value=1.0,
                value=float(layer.get("pruning_rate", 0.0)), step=0.01,
                key=f"prate_{layer['id']}",
            )
            if top[4].button("🗑️ Remove", key=f"rm_{layer['id']}"):
                remove_layer(layer["id"])
                st.rerun()

            if layer["type"] == "Linear":
                col = st.columns(4)
                layer["input dimension"] = col[0].number_input(
                    "Input dimension", min_value=1, value=int(layer["input dimension"]),
                    step=1, key=f"in_{layer['id']}",
                )
                layer["output dimension"] = col[1].number_input(
                    "Output dimension", min_value=1, value=int(layer["output dimension"]),
                    step=1, key=f"out_{layer['id']}",
                )

            elif layer["type"] == "Conv2d":
                col = st.columns(5)
                layer["cin"]     = col[0].number_input("Cin",     min_value=1, value=int(layer["cin"]),              step=1, key=f"cin_{layer['id']}")
                layer["cout"]    = col[1].number_input("Cout",    min_value=1, value=int(layer["cout"]),             step=1, key=f"cout_{layer['id']}")
                layer["kh"]      = col[2].number_input("Kh",      min_value=1, value=int(layer["kh"]),               step=1, key=f"kh_{layer['id']}")
                layer["kw"]      = col[3].number_input("Kw",      min_value=1, value=int(layer["kw"]),               step=1, key=f"kw_{layer['id']}")
                layer["padding"] = col[4].number_input("Padding", min_value=0, value=int(layer.get("padding", 0)),   step=1, key=f"pad_{layer['id']}")

st.divider()

# Energy estimation
st.subheader("Energy estimation")

if st.button("Estimate energy", type="primary"):
    rows, errors = [], []
    total_energy = 0.0
    available_keys = list_layer_keys(board)

    for layer in st.session_state.layers:
        try:
            layer_key = get_layer_key(layer)
            if layer_key not in available_keys:
                raise ValueError(f"'{layer_key}' not available on {board}. Available: {available_keys}")

            in_dim, out_dim = get_in_out_dim(layer)
            p = max(0.0, min(1.0, float(layer.get("pruning_rate", 0.0))))
            eff_out = out_dim * (1.0 - p)

            E_dense = float(estimate_energy(board, layer_key, in_dim, out_dim))
            E_used  = float(estimate_energy(board, layer_key, in_dim, eff_out))
            total_energy += E_used

            rows.append({
                "Type": layer["type"],
                "Name": layer["name"],
                "In Dim": in_dim,
                "Out Dim": out_dim,
                "Pruning Rate": p,
                "Dense Energy": E_dense,
                "Used Energy": E_used,
            })

        except Exception as e:
            errors.append({"name": layer.get("name", "?"), "error": str(e)})

    if rows:
        res = pd.DataFrame(rows)
        scale, unit = (1000.0, "mJ") if units == "mJoules" else (1.0, "J")
        res["Dense Energy"] *= scale
        res["Used Energy"]  *= scale
        total_display = total_energy * scale

        st.metric("Total energy estimate", f"{total_display:.6f} {unit}")

        st.subheader("Energy Consumption per Layer")
        st.plotly_chart(plot_per_layer(res, unit), width="stretch")

        st.subheader("Cumulative Energy Consumption")
        st.plotly_chart(plot_cumulative(res, unit), width="stretch")

        st.subheader("Layer Energy Details")
        st.dataframe(res, width="stretch")

    if errors:
        st.warning("Some layers could not be estimated.")
        st.dataframe(pd.DataFrame(errors), width="stretch")
