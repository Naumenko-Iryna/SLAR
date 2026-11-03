import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

def get_triangle_data(pts):
    A, B, C = pts[0], pts[1], pts[2]
    a, b, c = np.linalg.norm(B - C), np.linalg.norm(A - C), np.linalg.norm(A - B)
    
    area = 0.5 * abs(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1]))
    
    cos_vals = [
        np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0),
        np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0),
        np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)
    ]
    min_deg = float(np.min(np.degrees(np.arccos(cos_vals))))

    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if abs(D) < 1e-9:
        return min_deg, area, np.mean(pts, axis=0)
    
    ux = ((A[0]**2 + A[1]**2)*(B[1] - C[1]) + (B[0]**2 + B[1]**2)*(C[1] - A[1]) + (C[0]**2 + C[1]**2)*(A[1] - B[1])) / D
    uy = ((A[0]**2 + A[1]**2)*(C[0] - B[0]) + (B[0]**2 + B[1]**2)*(A[0] - C[0]) + (C[0]**2 + C[1]**2)*(B[0] - A[0])) / D
    return min_deg, area, np.array([ux, uy])

def check_convexity(pts):
    n = len(pts)
    signs = []
    for i in range(n):
        v1 = pts[(i + 1) % n] - pts[i]
        v2 = pts[(i + 2) % n] - pts[(i + 1) % n]
        cross_2d = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross_2d) > 1e-7:
            signs.append(cross_2d > 0)
    return len(set(signs)) <= 1

def generate_mesh(poly, edge_types, target_angle, max_area, max_iter=150):
    path = Path(poly)
    n_v = len(poly)
    pts = []
    seg_len = sum(np.linalg.norm(poly[(i+1)%n_v] - poly[i]) for i in range(n_v)) / (n_v * 3)
    
    for i in range(n_v):
        p1, p2 = poly[i], poly[(i+1)%n_v]
        n_segs = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / seg_len)))
        for j in range(n_segs):
            pts.append(p1 + (p2 - p1) * (j / n_segs))
    pts = np.array(pts)

    for _ in range(max_iter):
        tri = Delaunay(pts)
        simplices = tri.simplices[path.contains_points(np.mean(pts[tri.simplices], axis=1), radius=-1e-5)]
        
        bad_elems = []
        for s in simplices:
            deg, area, cc = get_triangle_data(pts[s])
            if deg < target_angle or area > max_area:
                error = max(0, target_angle - deg) + max(0, (area - max_area) / max_area * 10)
                bad_elems.append((error, cc, np.mean(pts[s], axis=0)))
                
        if not bad_elems:
            break
            
        bad_elems.sort(key=lambda x: x[0], reverse=True)
        added = False
        for _, cc, centroid in bad_elems:
            candidate = cc if path.contains_point(cc) else centroid
            if np.min(np.linalg.norm(pts - candidate, axis=1)) > 1e-3:
                pts = np.vstack([pts, candidate])
                added = True
                break
        if not added:
            break

    tri = Delaunay(pts)
    simplices = tri.simplices[path.contains_points(np.mean(pts[tri.simplices], axis=1), radius=-1e-5)]
    
    boundaries = []
    for p in pts:
        mark = 0
        for i in range(n_v):
            p1, p2 = poly[i], poly[(i+1)%n_v]
            if abs(np.linalg.norm(p - p1) + np.linalg.norm(p2 - p) - np.linalg.norm(p2 - p1)) < 1e-4:
                mark = edge_types[i]
                break
        boundaries.append(mark)
        
    return pts, simplices, boundaries

def get_boundary_edges(elements, nodes, verts, edge_types):
    edges = {}
    for elem in elements:
        for i in range(3):
            n1, n2 = elem[i], elem[(i+1)%3]
            edge = tuple(sorted((n1, n2)))
            edges[edge] = edges.get(edge, 0) + 1

    bnd_edges = []
    n_v = len(verts)
    for edge, count in edges.items():
        if count == 1: 
            n1, n2 = edge
            midpoint = (nodes[n1] + nodes[n2]) / 2.0
            edge_type = 0
            for i in range(n_v):
                p1, p2 = verts[i], verts[(i+1)%n_v]
                d1, d2, d3 = np.linalg.norm(midpoint - p1), np.linalg.norm(p2 - midpoint), np.linalg.norm(p2 - p1)
                if abs(d1 + d2 - d3) < 1e-4:
                    edge_type = edge_types[i]
                    break
            bnd_edges.append({'nodes': [n1, n2], 'type': edge_type})
    return bnd_edges

# ПОБУДОВА МАТРИЦЬ МСЕ ТА СЛАР

def build_fem_matrices(nodes, elements, boundary_edges, a11, a22, d_c, f_c, sigma, beta, psi):
    N_nodes = len(nodes) # Обчислює загальну кількість усіх вузлів сітки
    A = np.zeros((N_nodes, N_nodes))
    B = np.zeros(N_nodes)
    
    local_mats = {}

    # Матриця жорсткості Ke
    for e_idx, elem in enumerate(elements): # перебираємо кожен трикутник сітки
        p = nodes[elem] # координати трьох вершин конкретного трикутника
        x_i, y_i = p[0]
        x_j, y_j = p[1]
        x_m, y_m = p[2]

        # Геометричні коефіцієнти форми
        b = np.array([y_j - y_m, y_m - y_i, y_i - y_j])
        c = np.array([x_m - x_j, x_i - x_m, x_j - x_i])

        # Подвоєна площа трикутника, формула є розкриттям визначника матриці
        delta = abs(x_i*(y_j - y_m) + x_j*(y_m - y_i) + x_m*(y_i - y_j))
        
        K_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_e[i, j] = (a11 * b[i] * b[j] + a22 * c[i] * c[j]) / (2.0 * delta)
                
        M_e = d_c * (delta / 24.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        Q_e = f_c * (delta / 6.0) * np.array([1, 1, 1])
        
        local_mats[e_idx] = {'K': K_e, 'M': M_e, 'Q': Q_e}
        
        for i in range(3):
            B[elem[i]] += Q_e[i]
            for j in range(3):
                A[elem[i], elem[j]] += K_e[i, j] + M_e[i, j]

    bnd_mats = {}
    for b_idx, bedge in enumerate(boundary_edges):
        n1, n2 = bedge['nodes']
        btype = bedge['type']
        gamma = np.linalg.norm(nodes[n1] - nodes[n2])
        
        if btype == 3:
            R_e = (sigma / beta) * (gamma / 6.0) * np.array([[2, 1], [1, 2]])
            P_e = (psi / beta) * (gamma / 2.0) * np.array([1, 1])
            bnd_mats[b_idx] = {'R': R_e, 'P': P_e, 'nodes': [n1, n2], 'type': btype}
            
            A[n1, n1] += R_e[0, 0]
            A[n1, n2] += R_e[0, 1]
            A[n2, n1] += R_e[1, 0]
            A[n2, n2] += R_e[1, 1]
            B[n1] += P_e[0]
            B[n2] += P_e[1]
        else:
            bnd_mats[b_idx] = {'type': btype, 'nodes': [n1, n2]}

    dirichlet_nodes = set()
    for bedge in boundary_edges:
        if bedge['type'] == 1:
            dirichlet_nodes.update(bedge['nodes'])
            
    for n in dirichlet_nodes:
        A[n, :] = 0
        A[n, n] = 1.0
        B[n] = 0.0
        
    return A, B, local_mats, bnd_mats

st.set_page_config(page_title="МСЕ: Триангуляція та СЛАР", layout="wide")
st.title("Метод Скінченних Елементів (Варіант 12)")

st.sidebar.header("Параметри Триангуляції")
target_angle = st.sidebar.slider("Мінімальний кут (°)", 10.0, 50.0, 20.0, 1.0)
max_area = st.sidebar.number_input("Максимальна площа трикутника", 0.01, 5.0, 0.5, 0.05)
preset = st.sidebar.selectbox("Область:", ["Варіант 12 (4-кутник)", "Трикутник", "П'ятикутник"])

presets = {
    "Варіант 12 (4-кутник)": ([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 1.0]], [2, 1, 2, 3]),
    "Трикутник": ([[0.0, 0.0], [2.0, 0.0], [1.0, 1.73]], [1, 2, 3]),
    "П'ятикутник": ([[0.0, 0.0], [2.0, 0.0], [2.5, 1.5], [1.0, 2.5], [-0.5, 1.5]], [1, 2, 2, 3, 2])
}
poly_default, b_types = presets[preset]

verts = np.array([[float(x) for x in st.sidebar.text_input(f"V{i+1}:", f"{p[0]}, {p[1]}").replace(',', ' ').split()[:2]] 
                  for i, p in enumerate(poly_default)])

if not check_convexity(verts):
    st.sidebar.warning("Фігура неопукла!")

st.sidebar.header("Параметри ДФР (Варіант 12)")
a11 = st.sidebar.number_input("a11", value=8.0)
a22 = st.sidebar.number_input("a22", value=2.0)
d_c = st.sidebar.number_input("d (коеф. реакції)", value=0.0)
f_c = st.sidebar.number_input("f (джерело)", value=1.0)

st.sidebar.header("Крайова умова 3-го типу")
st.sidebar.markdown(r"$\beta \frac{\partial u}{\partial n} + \sigma u = \psi$")
sigma = st.sidebar.number_input("sigma", value=1.0)
beta = st.sidebar.number_input("beta", value=1.0)
psi = st.sidebar.number_input("psi", value=1.0)

nodes, elements, boundaries = generate_mesh(verts, b_types, target_angle, max_area)
boundary_edges = get_boundary_edges(elements, nodes, verts, b_types)

A, B, local_mats, bnd_mats = build_fem_matrices(nodes, elements, boundary_edges, a11, a22, d_c, f_c, sigma, beta, psi)

angles = [get_triangle_data(nodes[s])[0] for s in elements]
min_deg_found = min(angles) if len(angles) > 0 else 0.0

tab1, tab2, tab3 = st.tabs(["Триангуляція", "Локальні Матриці Елементів", "Глобальна СЛАР"])

with tab1:
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, (s, ang) in enumerate(zip(elements, angles)):
            pts_tri = np.vstack([nodes[s], nodes[s[0]]])
            if ang < target_angle:
                ax.fill(pts_tri[:,0], pts_tri[:,1], color='red', alpha=0.35)
            ax.plot(pts_tri[:,0], pts_tri[:,1], 'k-', lw=0.7)
            c = np.mean(nodes[s], axis=0)
            ax.text(c[0], c[1], f"E{idx}", color='blue', fontsize=7, ha='center', va='center')
            
        colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
        for idx, (p, b) in enumerate(zip(nodes, boundaries)):
            ax.plot(p[0], p[1], 'o', color=colors.get(b, 'black'), ms=4)
            ax.text(p[0]+0.02, p[1]+0.02, str(idx), fontsize=7)
            
        ax.set_aspect('equal')
        ax.grid(True, ls=':', alpha=0.5)
        ax.set_title(f"Фактичний мін. кут: {min_deg_found:.1f}° (Ціль: {target_angle:.0f}°)")
        st.pyplot(fig)
        
    with col2:
        st.metric("Мінімальний кут", f"{min_deg_found:.2f}°", delta=f"{min_deg_found - target_angle:.2f}°")
        st.write(f"**Вузлів:** {len(nodes)} | **Елементів:** {len(elements)}")
        
        st.write("**Таблиця зв'язності (Topology)**")
        st.dataframe(pd.DataFrame(elements, columns=['N1', 'N2', 'N3']).assign(**{"Мін. кут (°)": np.round(angles, 1)}), height=200, use_container_width=True)
        
        st.write("**Таблиця вузлів та крайових міток**")
        b_map = {0: "0 (Внутрішній)", 1: "1 (Діріхле)", 2: "2 (Неймана)", 3: "3 (Робіна)"}
        st.dataframe(pd.DataFrame(nodes, columns=['X', 'Y']).assign(Маркер=boundaries, Умова=[b_map.get(b, str(b)) for b in boundaries]), height=200, use_container_width=True)

with tab2:
    st.subheader("Локальні матриці по кожному скінченному елементу")
    st.info("Матриця $K_e$ змінюється від кутів нахилу елемента. Матриця $M_e$ та вектор $Q_e$ залежать від площі (і коефіцієнта $d$).")
    
    for e_idx, e_mats in local_mats.items():
        with st.expander(f"Елемент №{e_idx}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f" Матриця жорсткості **$K_{{{e_idx}}}$**")
            c1.dataframe(pd.DataFrame(e_mats['K']))
            
            c2.write(f"**$M_{{{e_idx}}}$**")
            c2.dataframe(pd.DataFrame(e_mats['M']))
            
            c3.write(f"**$Q_{{{e_idx}}}$**")
            c3.dataframe(pd.DataFrame(e_mats['Q'], columns=["Значення"]))
    
    st.divider()
    st.subheader("Граничні матриці для ребер 3-го типу (Робіна)")
    r3_edges = {k: v for k, v in bnd_mats.items() if v.get('type') == 3}
    
    if r3_edges:
        st.info("Значення $R_e$ та $P_e$ залежать від довжини відрізка на межі.")
        for b_idx, e_bnd in r3_edges.items():
            with st.expander(f"Граничне ребро №{b_idx} (Між вузлами {e_bnd['nodes']})", expanded=False):
                c1, c2 = st.columns(2)
                c1.write(f"**$R_{{{b_idx}}}$**")
                c1.dataframe(pd.DataFrame(e_bnd['R']))
                c2.write(f"**$P_{{{b_idx}}}$**")
                c2.dataframe(pd.DataFrame(e_bnd['P'], columns=["Значення"]))
    else:
        st.info("У даній конфігурації сітки немає ребер 3-го типу на межі.")

with tab3:
    st.subheader(f"Глобальна матриця СЛАР (Розмір: {A.shape[0]}x{A.shape[1]})")
    st.write("Повна матриця $A$:")
    st.dataframe(pd.DataFrame(A), use_container_width=True)
    
    st.write("Повний вектор $B$:")
    st.dataframe(pd.DataFrame(B, columns=["B"]), use_container_width=True)
