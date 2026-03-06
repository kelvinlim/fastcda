# Background Knowledge

FastCDA supports three types of prior knowledge constraints passed to GFCI via the `knowledge` dict. All types can be combined in the same dict.

---

## `addtemporal` — Temporal Tier Ordering

Assigns variables to tiers. Variables in lower-numbered tiers can only be *parents* of variables in higher-numbered tiers — they cannot be children. This is the standard constraint for time-lagged data.

```python
lag_stub = '_lag'
cols = df.columns

knowledge = {'addtemporal': {
    0: [col + lag_stub for col in cols],   # lag variables → tier 0 (earlier)
    1: [col for col in cols],              # current variables → tier 1 (later)
}}
```

**Helper method:** `create_lag_knowledge` generates this automatically from column names:

```python
knowledge = fc.create_lag_knowledge(df_lag.columns, lag_stub='_lag')
```

---

## `forbiddirect` — Block Specific Directed Edges

Prevents GFCI from producing a specific directed edge. Useful when domain theory rules out a direct causal path between two variables.

```python
knowledge = {
    'forbiddirect': [
        ['caffeine', 'stress'],    # forbid caffeine → stress
        ['alcohol', 'sleep'],      # forbid alcohol → sleep
    ]
}
```

Each item is a `[parent, child]` pair. The algorithm will not produce `parent --> child` in the output.

---

## `requiredirect` — Force Specific Directed Edges

Forces GFCI to include a specific directed edge. Useful when theory strongly asserts a direct causal relationship.

```python
knowledge = {
    'requiredirect': [
        ['exercise', 'productivity'],   # require exercise → productivity
    ]
}
```

Each item is a `[parent, child]` pair. The algorithm will always produce `parent --> child`.

---

## Combining All Three Types

```python
knowledge = {
    'addtemporal': {
        0: [col + '_lag' for col in cols],
        1: list(cols),
    },
    'forbiddirect': [
        ['caffeine', 'stress'],
    ],
    'requiredirect': [
        ['exercise', 'productivity'],
    ],
}

result, graph = fc.run_model_search(
    df_std,
    model='gfci',
    score={'sem_bic': {'penalty_discount': 1.0}},
    test={'fisher_z': {'alpha': 0.01}},
    knowledge=knowledge,
)
```

---

## Low-Level Methods

For fine-grained control, the individual Tetrad Knowledge methods are exposed directly:

```python
# Add a variable to a temporal tier
fc.add_to_tier(tier=0, node='stress_lag')

# Forbid a single directed edge
fc.set_forbidden('caffeine', 'stress')

# Require a single directed edge
fc.set_required('exercise', 'productivity')

# Reset all constraints
fc.clearKnowledge()
```

---

## Worked Example — Synthetic Wellness Data

The `fastcda_demo_short.ipynb` notebook includes a complete worked example using a synthetic 6-variable wellness dataset (`exercise`, `caffeine`, `stress`, `sleep`, `mood`, `productivity`) with a known ground-truth DAG. It demonstrates:

1. **Baseline** — unconstrained GFCI
2. **`forbiddirect`** — blocking `caffeine → stress`
3. **`requiredirect`** — forcing `exercise → productivity`
4. **Combined** — both constraints together, compared side-by-side with `show_n_graphs`
