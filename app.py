import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

# ==========================================
# ЧАСТИНА 1: ГЕОМЕТРІЯ ТА ТРИАНГУЛЯЦІЯ
# ==========================================

def get_triangle_data(pts):
    A, B, C = pts[0], pts[1], pts[2]
    a, b, c = np.linalg.norm(B - C), np.linalg.norm(A - C), np.linalg.norm(A - B)
    cos_vals = [
        np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0),
        np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0),
        np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)
    ]
    min_deg = float(np.min(np.degrees(np.arccos(cos_vals))))

    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if abs(D) < 1e-9:
        return min_deg, np.mean(pts, axis=0)
    
    ux = ((A[0]**2 + A[1]**2)*(B[1] - C[1]) + (B[0]**2 + B[1]**2)*(C[1] - A[1]) + (C[0]**2 + C[1]**2)*(A[1] - B[1])) / D
    uy = ((A[0]**2 + A[1]**2)*(C[0] - B[0]) + (B[0]**2 + B[1]**2)*(A[0] - C[0]) + (C[0]**2 + C[1]**2)*(B[0] - A[0])) / D
    return min_deg, np.array([ux, uy])

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

def generate_mesh(poly, edge_types, target_angle, max_iter=150):
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
            deg, cc = get_triangle_data(pts[s])
            if deg < target_angle:
                bad_elems.append((target_angle - deg, cc, np.mean(pts[s], axis=0)))
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

def get_boundary_edges(elements, nodes, poly_default, edge_types):
    edges = {}
    for elem in elements:
        for i in range(3):
            n1, n2 = elem[i], elem[(i+1)%3]
            edge = tuple(sorted((n1, n2)))
            edges[edge] = edges.get(edge, 0) + 1

    bnd_edges = []
    n_v = len(poly_default)
    for edge, count in edges.items():
        if count == 1: # Граничне ребро
            n1, n2 = edge
            midpoint = (nodes[n1] + nodes[n2]) / 2.0
            edge_type = 0
            for i in range(n_v):
                p1, p2 = poly_default[i], poly_default[(i+1)%n_v]
                d1, d2, d3 = np.linalg.norm(midpoint - p1), np.linalg.norm(p2 - midpoint), np.linalg.norm(p2 - p1)
                if abs(d1 + d2 - d3) < 1e-4:
                    edge_type = edge_types[i]
                    break
            bnd_edges.append({'nodes': [n1, n2], 'type': edge_type})
    return bnd_edges

# ==========================================
# ЧАСТИНА 2: ПОБУДОВА МАТРИЦЬ МСЕ ТА СЛАР
# ==========================================
def build_fem_matrices(nodes, elements, boundary_edges, a11, a22, d_c, f_c, sigma, beta, psi):
    N_nodes = len(nodes)
    A = np.zeros((N_nodes, N_nodes))
    B = np.zeros(N_nodes)
    
    local_mats = {}
    
    # 1. Інтегрування по області (матриці Ke, Me, Qe)
    for e_idx, elem in enumerate(elements):
        p = nodes[elem]
        x_i, y_i = p[0]
        x_j, y_j = p[1]
        x_m, y_m = p[2]
        
        # Коефіцієнти b_i, c_i
        b = np.array([y_j - y_m, y_m - y_i, y_i - y_j])
        c = np.array([x_m - x_j, x_i - x_m, x_j - x_i])
        
        # Площа S = delta / 2
        delta = abs(x_i*(y_j - y_m) + x_j*(y_m - y_i) + x_m*(y_i - y_j))
        
        # Матриця жорсткості K_e
        K_e = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_e[i, j] = (a11 * b[i] * b[j] + a22 * c[i] * c[j]) / (2.0 * delta)
                
        # Матриця мас M_e
        M_e = d_c * (delta / 24.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        
        # Вектор навантаження Q_e
        Q_e = f_c * (delta / 6.0) * np.array([1, 1, 1])
        
        local_mats[e_idx] = {'K': K_e, 'M': M_e, 'Q': Q_e}
        
        # Ансамблювання
        for i in range(3):
            B[elem[i]] += Q_e[i]
            for j in range(3):
                A[elem[i], elem[j]] += K_e[i, j] + M_e[i, j]

    # 2. Інтегрування по границі (матриці Re, Pe)
    bnd_mats = {}
    for b_idx, bedge in enumerate(boundary_edges):
        n1, n2 = bedge['nodes']
        btype = bedge['type']
        gamma = np.linalg.norm(nodes[n1] - nodes[n2])
        
        if btype == 3: # Умова Робіна
            R_e = (sigma / beta) * (gamma / 6.0) * np.array([[2, 1], [1, 2]])
            P_e = (psi / beta) * (gamma / 2.0) * np.array([1, 1])
            bnd_mats[b_idx] = {'R': R_e, 'P': P_e, 'nodes': [n1, n2], 'type': btype}
            
            # Ансамблювання
            A[n1, n1] += R_e[0, 0]
            A[n1, n2] += R_e[0, 1]
            A[n2, n1] += R_e[1, 0]
            A[n2, n2] += R_e[1, 1]
            B[n1] += P_e[0]
            B[n2] += P_e[1]
        else:
            bnd_mats[b_idx] = {'type': btype, 'nodes': [n1, n2]}

    # 3. Застосування крайових умов Діріхле (тип 1)
    dirichlet_nodes = set()
    for bedge in boundary_edges:
        if bedge['type'] == 1:
            dirichlet_nodes.update(bedge['nodes'])
            
    for n in dirichlet_nodes:
        A[n, :] = 0
        A[n, n] = 1.0
        B[n] = 0.0 # Згідно з умовою u=0 на Г_1
        
    return A, B, local_mats, bnd_mats

# ==========================================
# ІНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="МСЕ: Триангуляція та СЛАР", layout="wide")
st.title("Метод Скінченних Елементів (Варіант 12)")

st.sidebar.header("Параметри Триангуляції")
target_angle = st.sidebar.slider("Мінімальний кут (°)", 10.0, 50.0, 20.0, 1.0)
preset = st.sidebar.selectbox("Область:", ["Варіант 12 (4-кутник)", "Трикутник", "П'ятикутник"])

presets = {
    "Варіант 12 (4-кутник)": ([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 1.0]], [2, 1, 2, 3]),
    "Трикутник": ([[0.0, 0.0], [2.0, 0.0], [1.0, 1.73]], [1, 2, 3]),
    "П'ятикутник": ([[0.0, 0.0], [2.0, 0.0], [2.5, 1.5], [1.0, 2.5], [-0.5, 1.5]], [1, 2, 2, 3, 2])
}
poly_default, b_types = presets[preset]
verts = np.array(poly_default)

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

# ГЕНЕРАЦІЯ СІТКИ
nodes, elements, boundaries = generate_mesh(verts, b_types, target_angle)
boundary_edges = get_boundary_edges(elements, nodes, verts, b_types)

# ФОРМУВАННЯ МАТРИЦЬ МСЕ
A, B, local_mats, bnd_mats = build_fem_matrices(nodes, elements, boundary_edges, a11, a22, d_c, f_c, sigma, beta, psi)

# ВІЗУАЛІЗАЦІЯ
tab1, tab2, tab3 = st.tabs(["Триангуляція", "Локальні Матриці (Елемент 0)", "Глобальна СЛАР"])

with tab1:
    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, s in enumerate(elements):
            pts_tri = np.vstack([nodes[s], nodes[s[0]]])
            ax.plot(pts_tri[:,0], pts_tri[:,1], 'k-', lw=0.7)
            c = np.mean(nodes[s], axis=0)
            ax.text(c[0], c[1], f"{idx}", color='blue', fontsize=7, ha='center', va='center')
            
        colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
        for idx, (p, b) in enumerate(zip(nodes, boundaries)):
            ax.plot(p[0], p[1], 'o', color=colors.get(b, 'black'), ms=4)
            ax.text(p[0]+0.02, p[1]+0.02, str(idx), fontsize=7)
            
        ax.set_aspect('equal')
        ax.grid(True, ls=':', alpha=0.5)
        st.pyplot(fig)
        
    with col2:
        st.write(f"**Вузлів:** {len(nodes)} | **Елементів:** {len(elements)}")
        b_map = {0: "Внутрішній", 1: "Діріхле (Г_1)", 2: "Неймана (Г_2)", 3: "Робіна (Г_3)"}
        st.dataframe(pd.DataFrame(nodes, columns=['X', 'Y']).assign(Умова=[b_map.get(b, str(b)) for b in boundaries]), height=300)

with tab2:
    st.subheader("Локальні матриці для Елемента №0")
    e0_mats = local_mats[0]
    c1, c2, c3 = st.columns(3)
    c1.write("**Матриця жорсткості $K_0$**")
    c1.dataframe(pd.DataFrame(e0_mats['K']))
    
    c2.write("**Матриця мас $M_0$**")
    c2.dataframe(pd.DataFrame(e0_mats['M']))
    
    c3.write("**Вектор навантаження $Q_0$**")
    c3.dataframe(pd.DataFrame(e0_mats['Q'], columns=["Значення"]))
    
    st.divider()
    st.subheader("Граничні матриці (для першого ребра 3-го типу)")
    r3_edges = [m for m in bnd_mats.values() if m['type'] == 3]
    if r3_edges:
        e_bnd = r3_edges[0]
        st.write(f"Ребро між вузлами: **{e_bnd['nodes']}**")
        c1, c2 = st.columns(2)
        c1.write("**Гранична матриця $R_e$**")
        c1.dataframe(pd.DataFrame(e_bnd['R']))
        c2.write("**Граничний вектор $P_e$**")
        c2.dataframe(pd.DataFrame(e_bnd['P'], columns=["Значення"]))
    else:
        st.info("У даній конфігурації сітки немає ребер 3-го типу на межі.")

with tab3:
    st.subheader(f"Глобальна матриця СЛАР (Розмір: {A.shape[0]}x{A.shape[1]})")
    st.write("Матриця $A$ (фрагмент 10x10):")
    st.dataframe(pd.DataFrame(A).head(10).iloc[:, :10])
    
    st.write("Вектор $B$ (перші 10 значень):")
    st.dataframe(pd.DataFrame(B, columns=["B"]).head(10))
