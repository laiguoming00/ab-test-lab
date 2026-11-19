# A/B test lab (Longhorn Eats)

Classroom demo: a campus food-delivery app tests two checkout banners.

- **A (control):** "Free delivery on orders over $25"
- **B (treatment):** "Arrives in 25 minutes or your next order is free"

## Run

```powershell
python -m pip install streamlit scipy numpy pandas
streamlit run app.py
```

## Teaching flow (about 30 minutes)

1. **Run the experiment.** Students take turns clicking *Next visitor*, then *Place order* or *Leave cart* as a hungry student. To let students use their own phones or laptops, run `streamlit run app.py --server.address 0.0.0.0` and share your machine's IP and port. Everyone writes to the same `data/events.csv`, and the results panel refreshes every 3 seconds.
2. **Fast-forward.** In the *Instructor panel*, set a hidden true effect and add 50, 200 or 1000 simulated visitors. Show how the verdict changes with sample size.
3. **Peeking trap.** Run the A/A simulation. Looking once gives about 5% false positives; stopping at the first p < 0.05 over 14 looks gives roughly 20 to 25%.
4. **Sample size.** Compute how many visitors are needed before launch.
5. **Real tools.** Show the same experiment in GrowthBook or PostHog so students map the concepts to industry software.

## Files

- `app.py`: Streamlit UI (live experiment, peeking, sample size, explainer)
- `abtest_stats.py`: z-test, confidence interval, Welch t-test, sample-ratio check, power, peeking simulation
- `data/events.csv`: created at runtime; use *Reset all data* between class sections
