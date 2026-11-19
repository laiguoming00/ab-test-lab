"""A/B testing classroom demo: 'Longhorn Eats' checkout-banner experiment.

Run:  streamlit run app.py
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from posthog_demo import tab_posthog
from abtest_stats import (
    sample_size_per_arm,
    simulate_peeking,
    srm_check,
    two_proportion_test,
    welch_test,
)

st.set_page_config(page_title="A/B test lab: Longhorn Eats", page_icon=":material/science:", layout="wide")

EVENTS_FILE = Path(__file__).parent / "data" / "events.csv"
COLUMNS = ["ts", "visitor_id", "variant", "cart_total", "converted", "source"]

VARIANTS = {
    "A": {
        "name": "Control",
        "banner": "Free delivery on orders over $25",
        "icon": ":material/local_shipping:",
    },
    "B": {
        "name": "Treatment",
        "banner": "Arrives in 25 minutes or your next order is free",
        "icon": ":material/timer:",
    },
}


# ---------------------------------------------------------------- storage
@st.cache_resource
def _lock() -> threading.Lock:
    return threading.Lock()


def load_events() -> pd.DataFrame:
    if not EVENTS_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(EVENTS_FILE)
    df["converted"] = df["converted"].astype(int)
    return df


def append_events(rows: list[dict]) -> None:
    EVENTS_FILE.parent.mkdir(exist_ok=True)
    with _lock():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            EVENTS_FILE, mode="a", header=not EVENTS_FILE.exists(), index=False
        )


def reset_events() -> None:
    with _lock():
        EVENTS_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------- live experiment
def new_visitor() -> None:
    st.session_state.visitor = {
        "id": uuid.uuid4().hex[:8],
        "variant": random.choice(["A", "B"]),  # the randomization step
        "cart_total": round(random.uniform(14, 42), 2),
    }


def log_outcome(converted: bool) -> None:
    v = st.session_state.visitor
    if v is None:  # double-click: this visitor was already logged
        return
    append_events([{
        "ts": time.time(),
        "visitor_id": v["id"],
        "variant": v["variant"],
        "cart_total": v["cart_total"],
        "converted": int(converted),
        "source": "live",
    }])
    st.session_state.visitor = None


def add_simulated(n: int, p_a: float, p_b: float) -> None:
    rng = np.random.default_rng()
    variants = rng.choice(["A", "B"], size=n)
    rows = []
    t0 = time.time()
    for i, var in enumerate(variants):
        conv = rng.random() < (p_a if var == "A" else p_b)
        rows.append({
            "ts": t0 + i * 1e-3,
            "visitor_id": uuid.uuid4().hex[:8],
            "variant": var,
            "cart_total": round(float(rng.uniform(14, 42)), 2),
            "converted": int(conv),
            "source": "simulated",
        })
    append_events(rows)


def checkout_card(v: dict) -> None:
    var = VARIANTS[v["variant"]]
    with st.container(border=True):
        st.caption("Longhorn Eats  ·  Checkout")
        st.subheader("Your order")
        st.markdown("Breakfast tacos x2  \nHorchata x1  \nChips and queso x1")
        st.info(f"**{var['banner']}**", icon=var["icon"])
        st.metric("Cart total", f"${v['cart_total']:.2f}")
        c1, c2 = st.columns(2)
        c1.button("Place order", type="primary", width="stretch",
                  on_click=log_outcome, args=(True,), icon=":material/check:")
        c2.button("Leave cart", width="stretch",
                  on_click=log_outcome, args=(False,), icon=":material/close:")


def verdict(res) -> None:
    if res.p_value < 0.05:
        direction = "higher" if res.abs_diff > 0 else "lower"
        st.success(
            f"Statistically significant (p = {res.p_value:.3f}): B converts {direction} than A.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            f"Not significant (p = {res.p_value:.3f}): the data are consistent with no difference. "
            "That is not the same as proving there is none.",
            icon=":material/hourglass_top:",
        )


@st.fragment(run_every="3s")
def results_panel() -> None:
    df = load_events()
    st.subheader("Results so far")
    if df.empty:
        st.caption("No visitors yet. Send some traffic.")
        return

    g = df.groupby("variant").agg(n=("converted", "size"), conv=("converted", "sum"))
    n_a, c_a = (int(g.loc["A", "n"]), int(g.loc["A", "conv"])) if "A" in g.index else (0, 0)
    n_b, c_b = (int(g.loc["B", "n"]), int(g.loc["B", "conv"])) if "B" in g.index else (0, 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Visitors", len(df))
    m2.metric("A: free delivery over $25", f"{c_a / n_a:.1%}" if n_a else "n/a", f"{n_a} visitors", delta_color="off")
    m3.metric("B: 25 min guarantee", f"{c_b / n_b:.1%}" if n_b else "n/a", f"{n_b} visitors", delta_color="off")

    res = two_proportion_test(c_a, n_a, c_b, n_b)
    if res is None:
        st.caption("Need visitors in both variants to compare.")
        return

    st.metric(
        "Lift of B over A",
        f"{res.abs_diff * 100:+.1f} pts",
        f"{res.rel_lift:+.0%} relative" if res.rel_lift == res.rel_lift else None,
    )
    st.caption(
        f"95% confidence interval for the difference: "
        f"{res.ci_low * 100:+.1f} to {res.ci_high * 100:+.1f} percentage points"
    )
    verdict(res)

    # Running conversion rate per variant
    d = df.sort_values("ts").reset_index(drop=True)
    d["visitor_no"] = np.arange(1, len(d) + 1)
    run = pd.DataFrame({"visitor_no": d["visitor_no"]})
    for var in ("A", "B"):
        is_v = (d["variant"] == var).astype(int)
        run[f"{var}"] = (is_v * d["converted"]).cumsum() / is_v.cumsum().replace(0, np.nan)
    st.caption("Running conversion rate. Notice how noisy it is early on.")
    st.line_chart(run.set_index("visitor_no"), y_label="Conversion rate", x_label="Visitors so far")

    with st.expander("Health checks and a second metric"):
        p_srm = srm_check(n_a, n_b)
        if p_srm is not None:
            if p_srm < 0.01:
                st.error(f"Sample ratio mismatch (p = {p_srm:.4f}): the 50/50 split looks broken. Do not trust results.")
            else:
                st.caption(f"Sample ratio check: {n_a} vs {n_b} visitors (p = {p_srm:.2f}). Split looks fine.")
        rev = df.assign(revenue=df["cart_total"] * df["converted"])
        ra, rb = rev.loc[rev.variant == "A", "revenue"], rev.loc[rev.variant == "B", "revenue"]
        if len(ra) and len(rb):
            st.caption(f"Revenue per visitor: A ${ra.mean():.2f} vs B ${rb.mean():.2f}")
            w = welch_test(ra, rb)
            if w:
                st.caption(f"Welch t-test on revenue per visitor: p = {w[1]:.3f}")

    st.download_button("Download raw data (CSV)", df.to_csv(index=False), "ab_events.csv", "text/csv",
                       icon=":material/download:")


def tab_live() -> None:
    st.markdown(
        "**Scenario.** Longhorn Eats is a campus food-delivery app. Too many students "
        "abandon their cart at checkout. Product has two ideas for the banner above the "
        "*Place order* button. Which one gets more students to order?"
    )
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("A visitor arrives")
        if st.session_state.visitor is None:
            st.caption("Each click on the button below is one new student opening checkout. "
                       "The app flips a coin to assign A or B.")
            st.button("Next visitor", type="primary", on_click=new_visitor, icon=":material/person_add:")
        else:
            v = st.session_state.visitor
            st.caption(f"Visitor `{v['id']}` was randomly assigned to variant **{v['variant']}**.")
            checkout_card(v)

        with st.expander("Instructor panel"):
            st.caption("Fast-forward by adding simulated visitors drawn from a hidden 'true' effect.")
            p_a = st.slider("True conversion, A", 0.02, 0.40, 0.12, 0.01, format="%.2f")
            p_b = st.slider("True conversion, B", 0.02, 0.40, 0.14, 0.01, format="%.2f")
            c1, c2, c3 = st.columns(3)
            for col, n in zip((c1, c2, c3), (50, 200, 1000)):
                col.button(f"+{n}", key=f"sim{n}", width="stretch", on_click=add_simulated, args=(n, p_a, p_b))
            st.button("Reset all data", on_click=reset_events, icon=":material/delete:")
    with right:
        results_panel()


# ---------------------------------------------------------------- peeking
def tab_peeking() -> None:
    st.markdown(
        "**Why you cannot just stop when it looks significant.** In an *A/A test* both "
        "variants are identical, so any 'winner' is a false positive. A 5% significance "
        "level promises about 5% false positives if you look **once**. What if you check "
        "every day and stop as soon as p < 0.05?"
    )
    c1, c2, c3, c4 = st.columns(4)
    p = c1.slider("True conversion (same in A and B)", 0.02, 0.40, 0.12, 0.01)
    per_look = c2.slider("Visitors per arm per look", 50, 1000, 200, 50)
    looks = c3.slider("Number of looks", 2, 30, 14)
    sims = c4.select_slider("Simulated experiments", [200, 500, 1000, 2000, 5000], value=1000)

    if st.button("Run simulation", type="primary", icon=":material/play_arrow:"):
        st.session_state.peek = simulate_peeking(p, per_look, looks, sims)
        st.session_state.peek_looks = looks
    if "peek" in st.session_state:
        final_only, any_look, paths = st.session_state.peek
        m1, m2 = st.columns(2)
        m1.metric("False positives if you look once, at the end", f"{final_only:.1%}")
        m2.metric(f"False positives if you stop at the first p < 0.05 (up to {st.session_state.peek_looks} looks)",
                  f"{any_look:.1%}")
        st.caption("Each line is one A/A experiment. |z| above 1.96 means 'significant'. "
                   "Many lines wander across the threshold even though there is nothing to find.")
        st.line_chart(paths.assign(threshold=1.96), y_label="|z| statistic", x_label="Look number")


# ---------------------------------------------------------------- sample size
def tab_power() -> None:
    st.markdown(
        "**How long should we run it?** Decide the sample size *before* launching, from the "
        "smallest improvement worth detecting (the minimum detectable effect)."
    )
    c1, c2, c3, c4 = st.columns(4)
    base = c1.slider("Baseline conversion", 0.02, 0.50, 0.12, 0.01, key="pw_base")
    mde = c2.slider("Minimum lift worth detecting (relative)", 0.02, 0.50, 0.10, 0.01, format="%.2f")
    alpha = c3.select_slider("Significance level", [0.01, 0.05, 0.10], value=0.05)
    power = c4.select_slider("Power", [0.7, 0.8, 0.9, 0.95], value=0.8)
    n = sample_size_per_arm(base, mde, alpha, power)
    daily = st.number_input("Visitors reaching checkout per day (both arms)", 50, 100000, 400, 50)
    m1, m2, m3 = st.columns(3)
    m1.metric("Visitors per arm", f"{n:,}")
    m2.metric("Total visitors", f"{2 * n:,}")
    m3.metric("Days to run", f"{int(np.ceil(2 * n / daily))}")
    st.caption(f"To detect {base:.0%} → {base * (1 + mde):.1%} with {power:.0%} power at α = {alpha}.")

    mdes = np.arange(0.02, 0.51, 0.01)
    curve = pd.DataFrame(
        {"Visitors per arm": [sample_size_per_arm(base, m, alpha, power) for m in mdes]},
        index=pd.Index((mdes * 100).round(0), name="Relative lift to detect (%)"),
    )
    st.caption("Halving the lift you want to detect roughly quadruples the traffic you need.")
    st.line_chart(curve, x_label="Relative lift you want to detect (%)", y_label="Visitors needed per arm")


# ---------------------------------------------------------------- main
if "visitor" not in st.session_state:
    st.session_state.visitor = None

st.title("A/B test lab")
st.caption("OM 286 · Digital Product Process")
t1, t2, t3, t5, t4 = st.tabs([
    ":material/groups: Run the experiment",
    ":material/visibility: Peeking trap",
    ":material/calculate: Sample size",
    ":material/hub: Real platform: PostHog",
    ":material/menu_book: How it works",
])
with t1:
    tab_live()
with t2:
    tab_peeking()
with t3:
    tab_power()
with t5:
    tab_posthog(VARIANTS)
with t4:
    st.markdown(
        """
**The mechanics, in order**

1. **Hypothesis.** "Promising a 25-minute delivery will convince more students to place their order than offering free delivery over $25."
2. **Primary metric.** Checkout conversion = orders placed ÷ visitors who reached checkout. Pick it *before* the test.
3. **Randomization.** Each visitor gets A or B by a coin flip (`random.choice` in `new_visitor`). Randomness is what lets us attribute any difference to the banner and not to who happened to see it.
4. **Logging.** Every outcome is appended to `data/events.csv`: visitor, variant, cart total, converted or not.
5. **Analysis.** Two-proportion z-test on conversion; a 95% confidence interval for the difference; a Welch t-test for revenue per visitor.
6. **Guardrails.** A sample-ratio check flags a broken split. A higher conversion rate that lowers revenue per visitor may not be a win.
7. **Decision.** Significance says the difference is probably real. It does not say it is big enough to matter. Compare the confidence interval with the cost of shipping the change.

**Discussion questions**

- Our banner B promises something operations must deliver. What could go wrong if B "wins"?
- Why did the running conversion rate swing so much in the first 30 visitors?
- Is a +2 point lift with p = 0.04 worth shipping? What about +0.2 points with p = 0.001 on a million visitors?
- What else would you want to measure besides conversion? (Refunds, repeat orders, support tickets.)
        """
    )
