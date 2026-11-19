"""'Real platform' tab: run the Longhorn Eats experiment through PostHog.

PostHog decides who sees which banner (a feature flag) and receives the outcome
events; its Experiments page then does the statistics. Compare it with the
homemade test on the first tab.
"""
from __future__ import annotations

import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from abtest_stats import two_proportion_test

FLAG_KEY = "checkout-banner"
VARIANT_MAP = {"control": "A", "test": "B"}  # PostHog variant key -> our label
REGIONS = {
    "US cloud": ("https://us.i.posthog.com", "https://us.posthog.com"),
    "EU cloud": ("https://eu.i.posthog.com", "https://eu.posthog.com"),
}


def _secret_key() -> str:
    try:
        k = st.secrets.get("POSTHOG_API_KEY", "")
    except Exception:
        k = ""
    return k or os.environ.get("POSTHOG_API_KEY", "")


@st.cache_resource(show_spinner=False)
def _client(api_key: str, host: str):
    from posthog import Posthog

    # sync_mode sends each event immediately: slower, but simple and reliable for a demo.
    return Posthog(api_key, host=host, sync_mode=True, enable_local_evaluation=False)


def _assign(client, distinct_id: str) -> tuple[str | None, str | None]:
    """Ask PostHog which variant this visitor gets. Also sends the exposure event."""
    flag = client.get_feature_flag(FLAG_KEY, distinct_id)
    if not isinstance(flag, str):
        return None, None
    return VARIANT_MAP.get(flag), flag


def _send_outcome(client, distinct_id: str, flag: str, converted: bool, cart_total: float) -> None:
    client.capture(
        "order_placed" if converted else "cart_abandoned",
        distinct_id=distinct_id,
        properties={"cart_total": cart_total, f"$feature/{FLAG_KEY}": flag},
    )


def _tally(var: str, converted: bool) -> None:
    n, c = st.session_state.ph_tally[var]
    st.session_state.ph_tally[var] = [n + 1, c + int(converted)]


# -------------------------------------------------------------- callbacks
def _next_visitor(client) -> None:
    did = uuid.uuid4().hex
    var, flag = _assign(client, did)
    if var is None:
        st.session_state.ph_error = (
            f"PostHog returned no variant for the flag `{FLAG_KEY}`. Check that the experiment "
            "is created and launched, and that the flag key matches exactly."
        )
        return
    st.session_state.ph_error = None
    st.session_state.ph_visitor = {
        "id": did, "variant": var, "flag": flag,
        "cart_total": round(random.uniform(14, 42), 2),
    }


def _finish(client, converted: bool) -> None:
    v = st.session_state.ph_visitor
    if v is None:  # double-click guard
        return
    _send_outcome(client, v["id"], v["flag"], converted, v["cart_total"])
    _tally(v["variant"], converted)
    st.session_state.ph_visitor = None


def _simulate(client, n: int, p_a: float, p_b: float) -> None:
    def one(_):
        did = uuid.uuid4().hex
        var, flag = _assign(client, did)
        if var is None:
            return None
        converted = random.random() < (p_a if var == "A" else p_b)
        _send_outcome(client, did, flag, converted, round(random.uniform(14, 42), 2))
        return var, converted

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(n)))
    ok = [r for r in results if r]
    for var, converted in ok:
        _tally(var, converted)
    st.session_state.ph_error = None if ok else (
        f"PostHog returned no variants for `{FLAG_KEY}`. Is the experiment launched?"
    )


def _reset_tally() -> None:
    st.session_state.ph_tally = {"A": [0, 0], "B": [0, 0]}


# -------------------------------------------------------------- UI
def _setup_guide(app_host: str) -> None:
    st.markdown(
        f"""
1. **Create a free account** at [posthog.com](https://posthog.com) and a project.
2. Copy the **Project API key** (it starts with `phc_`; Settings → Project). It is designed to be public.
3. Go to **Experiments → New experiment** and set:
   - Name: `Checkout banner`
   - Feature flag key: **`{FLAG_KEY}`** (must match exactly)
   - Variants: keep `control` and `test`, split 50/50
   - Primary metric: event **`order_placed`**
4. Click **Launch**.
5. Give this app the key, either way:
   - paste it in the box below (this session only), or
   - put `POSTHOG_API_KEY = "phc_..."` in `.streamlit/secrets.toml` locally, or in the app's **Secrets** settings on Streamlit Community Cloud.

Open [{app_host}/experiments]({app_host}/experiments) to watch results arrive.
        """
    )


def tab_posthog(variants: dict) -> None:
    st.markdown(
        "**Same experiment, industry tool.** Here PostHog assigns each visitor to A or B "
        "(a *feature flag*), receives the outcome events, and computes the statistics for you. "
        "Run some visitors, then compare PostHog's Experiments page with the homemade test."
    )
    for key, default in (("ph_visitor", None), ("ph_error", None), ("ph_tally", {"A": [0, 0], "B": [0, 0]})):
        st.session_state.setdefault(key, default)

    region = st.segmented_control("PostHog region", list(REGIONS), default="US cloud", key="ph_region")
    region = region or "US cloud"
    host, app_host = REGIONS[region]

    api_key = _secret_key()
    if not api_key:
        api_key = st.text_input("PostHog project API key", type="password", placeholder="phc_...",
                                help="Not stored on disk. Lasts for this browser session.").strip()
    with st.expander("Setup guide (5 minutes)", expanded=not api_key):
        _setup_guide(app_host)
    if not api_key:
        st.info("Add a PostHog project API key to start.", icon=":material/key:")
        return

    client = _client(api_key, host)
    if st.session_state.ph_error:
        st.error(st.session_state.ph_error)

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("A visitor arrives")
        v = st.session_state.ph_visitor
        if v is None:
            st.button("Next visitor", type="primary", icon=":material/person_add:",
                      key="ph_next", on_click=_next_visitor, args=(client,))
        else:
            var = variants[v["variant"]]
            st.caption(f"PostHog assigned this visitor to variant **{v['variant']}** (`{v['flag']}`).")
            with st.container(border=True):
                st.caption("Longhorn Eats  ·  Checkout")
                st.info(f"**{var['banner']}**", icon=var["icon"])
                st.metric("Cart total", f"${v['cart_total']:.2f}")
                c1, c2 = st.columns(2)
                c1.button("Place order", type="primary", width="stretch", key="ph_buy",
                          on_click=_finish, args=(client, True), icon=":material/check:")
                c2.button("Leave cart", width="stretch", key="ph_leave",
                          on_click=_finish, args=(client, False), icon=":material/close:")

        with st.expander("Instructor panel"):
            st.caption("Send simulated visitors through PostHog (each one is a real API call, so 100 takes a few seconds).")
            p_a = st.slider("True conversion, A", 0.02, 0.40, 0.12, 0.01, format="%.2f", key="ph_pa")
            p_b = st.slider("True conversion, B", 0.02, 0.40, 0.14, 0.01, format="%.2f", key="ph_pb")
            c1, c2 = st.columns(2)
            for col, n in zip((c1, c2), (25, 100)):
                if col.button(f"+{n}", key=f"ph_sim{n}", width="stretch"):
                    with st.spinner(f"Sending {n} visitors to PostHog..."):
                        _simulate(client, n, p_a, p_b)
                    st.rerun()
            st.button("Clear local tally", on_click=_reset_tally, icon=":material/delete:")
            st.caption("This clears only the tally on this page. Data already sent stays in PostHog.")

    with right:
        st.subheader("What this app sent to PostHog")
        (n_a, c_a), (n_b, c_b) = st.session_state.ph_tally["A"], st.session_state.ph_tally["B"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Visitors sent", n_a + n_b)
        m2.metric("A: free delivery", f"{c_a / n_a:.1%}" if n_a else "n/a", f"{n_a} visitors", delta_color="off")
        m3.metric("B: 25 min guarantee", f"{c_b / n_b:.1%}" if n_b else "n/a", f"{n_b} visitors", delta_color="off")
        res = two_proportion_test(c_a, n_a, c_b, n_b)
        if res:
            st.caption(f"Homemade z-test on this tally: lift {res.abs_diff * 100:+.1f} pts, p = {res.p_value:.3f}")
        st.link_button("Open PostHog Experiments", f"{app_host}/experiments", icon=":material/open_in_new:")
        st.markdown(
            "**Compare with PostHog's results page**\n\n"
            "- Do the conversion rates match the tally above?\n"
            "- PostHog defaults to a **Bayesian** analysis (probability that B beats A) while we used a "
            "**frequentist** p-value. Do they tell the same story?\n"
            "- Where would you read the sample size, the exposure count and the confidence/credible interval?\n"
            "- PostHog only counts people who *saw* the flag. Why does that matter?"
        )
