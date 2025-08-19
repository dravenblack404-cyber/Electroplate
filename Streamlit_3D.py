import streamlit as st
from collections import namedtuple
from ortools.sat.python import cp_model
import plotly.graph_objects as go

# --- 1. Definisi Parameter ---
PackedItem = namedtuple('PackedItem', ['name', 'position', 'dimension'])

# --- UI ---
st.title("📦 Electroplating Bath Packing Optimizer")

st.sidebar.header("⚙️ Input Parameters")

# Clearance inputs
WALL_CLEARANCE = st.sidebar.number_input("Wall Clearance", min_value=0, value=10, step=1)
PART_CLEARANCE = st.sidebar.number_input("Part Clearance", min_value=0, value=5, step=1)

# Container dims
st.sidebar.subheader("Bath Dimensions")
bath_length = st.sidebar.number_input("Bath Length", min_value=10, value=300, step=10)
bath_height = st.sidebar.number_input("Bath Height", min_value=10, value=200, step=10)
bath_width  = st.sidebar.number_input("Bath Width", min_value=10, value=100, step=10)

electroplating_bath_dims = {
    'name': 'Electroplating-Bath',
    'width': bath_width,
    'height': bath_height,
    'length': bath_length
}

# Part definitions (dynamic table)
st.sidebar.subheader("Part Definitions")
num_part_types = st.sidebar.number_input("Number of Part Types", min_value=1, max_value=10, value=3)

part_definitions = []
for i in range(num_part_types):
    with st.sidebar.expander(f"Part Type {i+1}"):
        name = st.text_input(f"Name Part {i+1}", value=f"Part-{i+1}")
        l = st.number_input(f"Length Part {i+1}", min_value=1, value=20, step=1, key=f"l{i}")
        h = st.number_input(f"Height Part {i+1}", min_value=1, value=20, step=1, key=f"h{i}")
        w = st.number_input(f"Width Part {i+1}", min_value=1, value=20, step=1, key=f"w{i}")
        qty = st.number_input(f"Quantity Part {i+1}", min_value=1, value=5, step=1, key=f"q{i}")
        part_definitions.append({'name': name, 'dims': (l, h, w), 'quantity': qty})

# --- Helper ---
def get_rotations(dims):
    w, h, d = dims
    rotations = {
        (w, h, d), (w, d, h), (h, w, d),
        (h, d, w), (d, w, h), (d, h, w)
    }
    return list(rotations)

def visualize_with_plotly(packed_items, container_dims, container_name):
    data = []
    def get_cuboid_vertices(position, dimension):
        x, y, z = position
        w, h, d = dimension
        return [
            (x, y, z), (x + w, y, z), (x + w, y + h, z), (x, y + h, z),
            (x, y, z + d), (x + w, y, z + d), (x + w, y + h, z + d), (x, y + h, z + d)
        ]

    for item in packed_items:
        verts = get_cuboid_vertices(item.position, item.dimension)
        data.append(go.Mesh3d(
            x=[v[0] for v in verts], y=[v[1] for v in verts], z=[v[2] for v in verts],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 5, 6],
            opacity=0.8, name=item.name, hoverinfo='name'
        ))

    container_verts = get_cuboid_vertices(
        (0,0,0),
        (container_dims['length'], container_dims['height'], container_dims['width'])
    )
    cx, cy, cz = zip(*container_verts)
    data.append(go.Scatter3d(
        x=cx, y=cy, z=cz, mode='lines',
        line=dict(color='black', width=3), name='Container'
    ))

    layout = go.Layout(
        title=f"Interactive View: {container_name}",
        scene=dict(
            xaxis=dict(title='Length (X)'),
            yaxis=dict(title='Height (Y)'),
            zaxis=dict(title='Width (Z)')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return go.Figure(data=data, layout=layout)

# --- Solver ---
def solve_single_batch(parts_available, bath):
    model = cp_model.CpModel()

    num_parts = len(parts_available)
    is_packed = [model.NewBoolVar(f'is_packed_{p}') for p in range(num_parts)]
    x_vars = [model.NewIntVar(0, bath['length'], f'x_{p}') for p in range(num_parts)]
    y_vars = [model.NewIntVar(0, bath['height'], f'y_{p}') for p in range(num_parts)]
    dx_vars = [model.NewIntVar(0, bath['length'], f'dx_{p}') for p in range(num_parts)]
    dy_vars = [model.NewIntVar(0, bath['height'], f'dy_{p}') for p in range(num_parts)]
    dz_vars = [model.NewIntVar(0, bath['width'], f'dz_{p}') for p in range(num_parts)]
    max_depth_of_batch = model.NewIntVar(0, bath['width'], 'max_depth_of_batch')

    dx_eff = [model.NewIntVar(0, bath['length'], f'dx_eff_{p}') for p in range(num_parts)]
    dy_eff = [model.NewIntVar(0, bath['height'], f'dy_eff_{p}') for p in range(num_parts)]

    end_x_vars = [model.NewIntVar(0, bath['length'], f'end_x_{p}') for p in range(num_parts)]
    end_y_vars = [model.NewIntVar(0, bath['height'], f'end_y_{p}') for p in range(num_parts)]

    x_intervals = [model.NewOptionalIntervalVar(
        x_vars[p], dx_eff[p], end_x_vars[p], is_packed[p], f'x_interval_{p}'
    ) for p in range(num_parts)]
    y_intervals = [model.NewOptionalIntervalVar(
        y_vars[p], dy_eff[p], end_y_vars[p], is_packed[p], f'y_interval_{p}'
    ) for p in range(num_parts)]
    
    part_volumes = []
    for p in range(num_parts):
        # rotasi
        rotations = parts_available[p]['rotations']
        l_p_r = [model.NewBoolVar(f'l_{p}_{r}') for r in range(len(rotations))]
        model.Add(sum(l_p_r) == is_packed[p])
        for r, rot in enumerate(rotations):
            model.Add(dx_vars[p] == rot[0]).OnlyEnforceIf(l_p_r[r])
            model.Add(dy_vars[p] == rot[1]).OnlyEnforceIf(l_p_r[r])
            model.Add(dz_vars[p] == rot[2]).OnlyEnforceIf(l_p_r[r])

        model.Add(dx_eff[p] == dx_vars[p] + PART_CLEARANCE)
        model.Add(dy_eff[p] == dy_vars[p] + PART_CLEARANCE)

        model.Add(end_x_vars[p] == x_vars[p] + dx_eff[p])
        model.Add(end_y_vars[p] == y_vars[p] + dy_eff[p])

        model.Add(x_vars[p] >= WALL_CLEARANCE).OnlyEnforceIf(is_packed[p])
        model.Add(y_vars[p] >= WALL_CLEARANCE).OnlyEnforceIf(is_packed[p])
        model.Add(x_vars[p] + dx_vars[p] + PART_CLEARANCE <= bath['length'] - WALL_CLEARANCE).OnlyEnforceIf(is_packed[p])
        model.Add(y_vars[p] + dy_vars[p] + PART_CLEARANCE <= bath['height'] - WALL_CLEARANCE).OnlyEnforceIf(is_packed[p])

        model.Add(dz_vars[p] <= max_depth_of_batch).OnlyEnforceIf(is_packed[p])

        part_volumes.append(parts_available[p]['volume'])

    model.Add(max_depth_of_batch <= bath['width'] - (2 * WALL_CLEARANCE))
    model.AddNoOverlap2D(x_intervals, y_intervals)

    model.Maximize(sum(is_packed[p] * part_volumes[p] for p in range(num_parts)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    status = solver.Solve(model)

    packed_items = []
    used_indices = []
    total_volume = 0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        bar_position_z = bath['width'] / 2
        for p in range(num_parts):
            if solver.BooleanValue(is_packed[p]):
                pos_x = solver.Value(x_vars[p])
                pos_y = solver.Value(y_vars[p])
                dim_dx = solver.Value(dx_vars[p])
                dim_dy = solver.Value(dy_vars[p])
                dim_dz = solver.Value(dz_vars[p])
                pos_z = bar_position_z - (dim_dz / 2)
                packed_items.append(PackedItem(
                    name=parts_available[p]['name'],
                    position=(pos_x, pos_y, pos_z),
                    dimension=(dim_dx, dim_dy, dim_dz)
                ))
                total_volume += parts_available[p]['volume']
                used_indices.append(p)

    return packed_items, used_indices, total_volume


def solve_multi_batch(part_definitions, bath):
    # flatten part list
    all_parts = []
    for part_type in part_definitions:
        dims = part_type['dims']
        volume = dims[0]*dims[1]*dims[2]
        for i in range(part_type['quantity']):
            all_parts.append({
                'name': f"{part_type['name']}-{i+1}",
                'rotations': get_rotations(dims),
                'volume': volume
            })

    batch_id = 1
    results = []

    while all_parts:
        packed_items, used_indices, total_volume = solve_single_batch(all_parts, bath)
        if not packed_items:
            break
        results.append(packed_items)
        all_parts = [p for idx, p in enumerate(all_parts) if idx not in used_indices]
        batch_id += 1

    return results


# --- Run ---
if st.button("🚀 Solve Packing"):
    results = solve_multi_batch(part_definitions, electroplating_bath_dims)
    if not results:
        st.error("Tidak ada part yang bisa dipacking dalam bath ini ❌")
    else:
        for batch_id, packed_items in enumerate(results, start=1):
            st.subheader(f"Batch {batch_id} ({len(packed_items)} parts)")
            st.plotly_chart(visualize_with_plotly(packed_items, electroplating_bath_dims, f"Batch {batch_id}"))
