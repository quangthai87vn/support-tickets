# ===========================================
# ROUND-ROBIN SEATING — WEB APP (STREAMLIT)
# ===========================================
# pip install streamlit pandas matplotlib

import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Core logic (tái sử dụng từ notebook) ----------

@dataclass
class InitParams:
    n: int
    parity: str  # "odd" or "even"
    D: int       # số ngày

def init_params(n: int) -> InitParams:
    assert n >= 3
    if n % 2 == 1:
        return InitParams(n=n, parity="odd", D=(n - 1) // 2)
    else:
        return InitParams(n=n, parity="even", D=n // 2)

def wrap(x: int, N: int) -> int:
    return ((x - 1) % N) + 1

def edges_from_cycle(order: List[int]) -> Set[Tuple[int, int]]:
    n = len(order)
    E: Set[Tuple[int, int]] = set()
    for i in range(n):
        u = order[i]; v = order[(i + 1) % n]
        e = (u, v) if u < v else (v, u)
        E.add(e)
    return E

def generate_schedule_odd(n: int) -> List[List[int]]:
    assert n % 2 == 1
    days: List[List[int]] = []
    N = n - 1
    m = (n - 1) // 2
    for d in range(m):
        order = [0]
        for k in range(1, m + 1):
            order += [wrap(d + k, N), wrap(d - k, N)]
        days.append(order)
    return days

def generate_schedule_even(n: int) -> List[List[int]]:
    assert n % 2 == 0
    days: List[List[int]] = []
    N = n - 1
    m = n // 2
    for d in range(m - 1):
        order = [0]
        for k in range(1, m):
            order += [wrap(d + k, N), wrap(N - (k - 1) - d, N)]
        order.append(wrap(d + m, N))
        days.append(order)
    completion = []
    for i in range(m):
        completion += [i, i + m]
    days.append(completion)
    return days

def generate_schedule(n: int) -> List[List[int]]:
    return generate_schedule_odd(n) if n % 2 == 1 else generate_schedule_even(n)

def count_pair_coverage(days: List[List[int]]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for order in days:
        for e in edges_from_cycle(order):
            counts[e] += 1
    return counts

def verify_coverage(n: int, pair_counts: Dict[Tuple[int, int], int]) -> Tuple[bool, list[Tuple[int,int]]]:
    need_exact_one = (n % 2 == 1)
    missing = []
    for u in range(n):
        for v in range(u + 1, n):
            c = pair_counts.get((u, v), 0)
            if need_exact_one and c != 1:
                missing.append((u, v))
            if not need_exact_one and c < 1:
                missing.append((u, v))
    return (len(missing) == 0, missing)

def try_repair_last_day(days: List[List[int]], missing: List[Tuple[int,int]], max_tries: int = 200) -> List[List[int]]:
    if not missing or len(days) == 0:
        return days
    import random
    last = days[-1][:]
    n = len(last)
    need = set(tuple(sorted(e)) for e in missing)

    def score(order):
        E = edges_from_cycle(order)
        return sum(1 for e in E if e in need)

    best = last[:]
    best_score = score(best)
    for _ in range(max_tries):
        i, j = random.sample(range(n), 2)
        cand = best[:]
        cand[i], cand[j] = cand[j], cand[i]
        sc = score(cand)
        if sc > best_score:
            best, best_score = cand, sc
    return days[:-1] + [best]

# ---------- Helpers to build DataFrames & plot ----------

def df_days(days: List[List[int]]) -> pd.DataFrame:
    df = pd.DataFrame(days)
    df.index = [f"Day {i}" for i in range(1, len(days)+1)]
    df.columns = [f"Seat {j}" for j in range(len(days[0]))]
    return df

def df_pairs_by_day(days: List[List[int]]) -> pd.DataFrame:
    records = []
    for d, order in enumerate(days, 1):
        for (u, v) in sorted(edges_from_cycle(order)):
            records.append({"Day": d, "Pair": f"({u},{v})", "u": u, "v": v})
    return pd.DataFrame(records)

def df_pair_counts(n: int, pair_counts: Dict[Tuple[int,int], int]) -> pd.DataFrame:
    rows = []
    for u in range(n):
        for v in range(u+1, n):
            rows.append({"Pair": f"({u},{v})", "Count": pair_counts.get((u, v), 0), "u": u, "v": v})
    df = pd.DataFrame(rows).sort_values(["Count","u","v"], ascending=[False, True, True]).reset_index(drop=True)
    return df[["Pair","Count"]]

def plot_circle(order: List[int]):
    """Vẽ vòng tròn cho một ngày."""
    n = len(order)
    r = 1.0
    angles = [2*math.pi * i / n for i in range(n)]
    xs = [r*math.cos(a) for a in angles]
    ys = [r*math.sin(a) for a in angles]
    # Map: index position -> person
    fig, ax = plt.subplots(figsize=(5,5))
    # nodes
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.scatter(x, y)
        ax.text(x, y, str(order[i]), ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))
    # edges (neighbors)
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        j = (i+1) % n
        x2, y2 = xs[j], ys[j]
        ax.plot([x1, x2], [y1, y2], alpha=0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig

# ============== UI (Streamlit) ==============

st.set_page_config(page_title="Round-robin Seating", layout="wide")

st.title("🪑 Xếp chỗ vòng tròn – Round-robin (Hamilton cycles)")
st.caption("Nhập số người n, hệ thống sinh lịch ngồi theo ngày để mọi cặp người đều từng ngồi kề nhau (n lẻ: mỗi cặp đúng 1 lần; n chẵn: mỗi cặp ≥ 1 lần).")

with st.sidebar:
    st.header("Thiết lập")
    n = st.number_input("Số người (n ≥ 3)", min_value=3, value=10, step=1)
    auto_repair = st.checkbox("Thử Repair nếu chưa đạt (hoán vị nhẹ ngày cuối)", value=True)
    show_circle = st.checkbox("Vẽ vòng tròn minh hoạ cho 1 ngày", value=True)
    day_to_plot = st.number_input("Chọn ngày để vẽ", min_value=1, value=1, step=1)
    st.markdown("---")
    st.caption("Made with ❤️ Streamlit")

# A) Init
params = init_params(int(n))

# B) Schedule
days = generate_schedule(params.n)

# C) Verify
pair_counts = count_pair_coverage(days)
passed, missing = verify_coverage(params.n, pair_counts)

# D) Repair (optional)
if auto_repair and not passed:
    days2 = try_repair_last_day(days, missing, max_tries=300)
    pair_counts2 = count_pair_coverage(days2)
    passed2, missing2 = verify_coverage(params.n, pair_counts2)
    # dùng kết quả tốt hơn
    if passed2 or len(missing2) < len(missing):
        days, pair_counts, passed = days2, pair_counts2, passed2

# Kết quả tóm tắt
c1, c2, c3, c4 = st.columns(4)
c1.metric("n (người)", params.n)
c2.metric("Số ngày D", len(days), help="(n lẻ → (n−1)/2, n chẵn → n/2)")
c3.metric("Parity", params.parity.upper())
c4.metric("Bao phủ", "PASSED ✅" if passed else "NOT PASSED ❌")

# Bảng Days×Seats
st.subheader("📅 Lịch ngồi theo ngày (Days×Seats)")
st.dataframe(df_days(days), use_container_width=True)

# Cặp kề theo từng ngày
st.subheader("🤝 Các cặp kề theo từng ngày (Pairs_by_Day)")
st.dataframe(df_pairs_by_day(days), use_container_width=True, height=320)

# Bảng PairCounts
st.subheader("📈 Bảng đếm bao phủ (PairCounts)")
st.dataframe(df_pair_counts(params.n, pair_counts), use_container_width=True, height=360)

# Vẽ vòng tròn cho một ngày
if show_circle:
    d_idx = int(day_to_plot) - 1
    if 0 <= d_idx < len(days):
        st.subheader(f"🟢 Minh hoạ vòng tròn – Ngày {day_to_plot}")
        fig = plot_circle(days[d_idx])
        st.pyplot(fig)
    else:
        st.info("Chọn ngày hợp lệ để vẽ.")
