# Lab 1 – Demand Modelling: notes and answers

Lecture source: Anders Johansson's demand slides (Parts 1 and 2). Lab sheet: `TMM202526Lab01.pdf`.

## 0. How to run the answers

All scripts live in this `answers/` folder. Run them **from this folder**. Every question has a Python script and a MATLAB script with the same name, and both give the same numbers.

| Question | Python | MATLAB | Runtime |
|---|---|---|---|
| Q1 geography | `q1_geography.py` | `q1_geography.m` | seconds |
| Q2 census stats | `q2_census_stats.py` | `q2_census_stats.m` | ~20 s |
| Q3 OD matrix (**run before Q4–Q8**) | `q3_od_matrix.py` | `q3_od_matrix.m` | ~15 s |
| Q4–Q5 gravity fits | `q4_q5_gravity_fit.py` | `q4_q5_gravity_fit.m` | ~15 s |
| Q6–Q7 constrained models | `q6_q7_constrained.py` | `q6_q7_constrained.m` | ~15 s |
| Q8 intervening opportunities / radiation | `q8_other_models.py` | `q8_other_models.m` | ~5 s |

Helper files:

- Python: `geo_utils.py` (distance, OS grid, Voronoi) and `gravity_utils.py` (load data, fitting, Furness, CPC).
- MATLAB: `haversine_matrix.m`, `latlon_to_osgb.m`, `ols_fit.m` (stands in for `fitlm`), `poisson_glm.m` (stands in for `glmfit`), `load_od_level.m`, `furness.m`, `cpc.m`, `fit_report.m`, `opportunities_matrix.m`.

Everything uses **base MATLAB only**, because your MATLAB install has no toolboxes. If you add the Statistics Toolbox, `fitlm(X, y)` and `glmfit(X, y, 'poisson')` give the same numbers as my helpers.

**Data notes (important):**

- `UK_census_msoa_stats.csv` is **2011** census data: 7,201 MSOAs.
- `ODWP01EW_MSOA.csv` is **2021** census data: 7,264 MSOAs. 7,080 codes are common to both, and their centroids agree to a median of 76 m.
- The coursework file `TMMcoursework2024Q1 (1).mat` stores 2021 MSOA eastings/northings and the MSOA → UTLA lookup. Its codes are MATLAB `categorical`s, which Python can't read, so I used MATLAB once to export them to `data/msoa2021_centroids.csv` and `data/utla2021.csv`.
- The coursework `.mat` demand data totals 27.67 M. That equals the 15.09 M actual commutes **plus** the 12.59 M "work mainly at home". So the coursework counts working from home as an intra-zonal trip. 2021 was during Covid, which is why that number is so large.
- `ODWP01EW_MSOA.csv.crdownload` is an identical duplicate of the CSV, left over from a browser download. You can delete it.

---

## 1. The big picture: the four-step model

Transport demand is a **derived demand**: people travel to *do* something somewhere else. The classic four-step model:

1. **Trip generation**: how many trips start in each zone ($O_i$) and end in each zone ($D_j$). This is estimated from zone attributes (population, households, jobs, car ownership...). → Q2
2. **Trip distribution**: who goes where. This is the OD matrix $T_{ij}$, given $O_i$, $D_j$ and the separation $d_{ij}$. → Q3–Q8
3. **Mode choice**: walk, cycle, bus, car...
4. **Assignment**: which route through the network. → Labs 2–3

Before any of this you need a **tessellation of space** into zones. Census geographies, from finest to coarsest:

- OA: 181k
- LSOA: 34.7k
- MSOA: 7.2k in 2011, 7.3k in 2021
- LAD (local authority district): ~350
- UTLA (upper-tier local authority): 174

Which level you use is a modelling choice with consequences (Q2(d), and the MAUP below).

---

## 2. Geography toolkit (Q1)

### Q1(a) Haversine distance

For points at latitude $\phi$ and longitude $\lambda$ (in radians), on a sphere of radius $R = 6371$ km:

$$a = \sin^2\!\frac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\!\frac{\Delta\lambda}{2}, \qquad d = 2R\arcsin\sqrt{a}$$

**Vectorising:** make set 1 a *column* and set 2 a *row*. Broadcasting (numpy) or implicit expansion (MATLAB) then gives the full $n_1 \times n_2$ matrix with no loops.

- Python: `p1[:, None] - p2[None, :]`
- MATLAB: `lat1(:) - lat2(:)'`

Check: London → Manchester = 262.4 km, which matches Anders's slide.

### Q1(b) lat/long → OS National Grid

There are two steps:

1. **Datum shift.** GPS uses the WGS84 ellipsoid, but the OS grid uses OSGB36 (the Airy 1830 ellipsoid). Convert to 3-D Cartesian coordinates, apply a 7-parameter **Helmert transformation** (3 shifts, 3 rotations, 1 scale), then convert back to lat/long. This is accurate to about 5 m.
2. **Transverse Mercator projection** using the OS series formulas. The true origin is 49°N, 2°W. False origin offsets are E₀ = 400 km and N₀ = −100 km, so all of GB has positive coordinates.

My implementation matches `pyproj` (EPSG:4326 → EPSG:27700) to within 1 mm.

A 12-digit grid reference is just `sprintf('%06d %06d', E, N)`, with units in metres. Once you are on the grid, **distance is just Pythagoras**. That is within 0.3% of Haversine across England & Wales, and much faster.

### Q1(c) Maps

- **MATLAB:** `geoscatter` + `geobasemap` (base MATLAB, but needs internet for the map tiles).
- **Python:** `contextily.add_basemap(ax, crs="EPSG:27700")` after `pip install contextily`.
- **Offline fallback:** scatter the 7,201 centroids. They trace out England and Wales well enough to recognise.

Always use `axis equal` / `ax.set_aspect("equal")` in grid coordinates.

### Q1(d) Voronoi tessellation and adjacency

- The Voronoi cell of a centroid is the region closer to it than to any other centroid. It is a cheap stand-in for real MSOA boundaries.
- **Key fact:** two Voronoi cells share an edge **if and only if** their centres are joined in the **Delaunay triangulation**. So the adjacency matrix comes straight from `delaunay`: each triangle (a, b, c) gives the edges ab, bc and ca.
- Build the triangulation in **grid metres, not degrees**. A degree of longitude is only about 0.62 of a degree of latitude at UK latitudes, so working in degrees distorts the cells.
- Mean degree ≈ 5.9. This is always just under 6 for a planar triangulation, by Euler's formula.
- **Caveat:** the convex hull joins zones across the sea (e.g. Cardiff–Somerset across the Bristol Channel, and the Isles of Scilly). I drop links longer than 20 km, which leaves Scilly with no neighbours.

### Q1(e) Shapefiles

A shapefile stores real boundary polygons, and "true" adjacency means the polygons touch.

- Download the MSOA (Dec 2021) boundaries from the ONS Open Geography Portal.
- **Python:** `geopandas.read_file(...)`.
- **MATLAB:** `shaperead` / `readgeotable`, which both need the Mapping Toolbox.

The script picks the file up automatically if you put it in `data/msoa_boundaries/`.

---

## 3. Census statistics and trip generation (Q2)

### Data types (MATLAB import tool / pandas)

- **Counts** (population, households, O, D) → `uint32`: they are never negative and use half the memory of `double`.
- **IDs** → row names (MATLAB `RowNames`) or the index (pandas). Then you can look rows up by MSOA code.
- **Categorical** = a variable with a small set of repeated values (country, local authority). It is stored once per category, with integer codes underneath. It is faster and enables grouping.
- ⚠️ **MATLAB gotcha (I hit this):** `T{:, {'pop','townsend'}}` concatenates a `uint32` column with a `double` column. **MATLAB converts the result to `uint32`**, silently rounding the Townsend scores. Use `table2array(varfun(@double, T(:, vars)))` instead. (In Python, pandas upcasts to float, so this doesn't happen.)

### Q2(a) O vs population

$O = 52 + 0.765\,\text{pop}$, with $R^2 = 0.67$ (r = 0.82). This matches Anders's slope of 0.76.

- **Interpretation:** each extra resident adds about 0.76 commuters, roughly the employment rate.
- O has a natural **upper limit**: it can't exceed the population.
- The intercept is not significant (p = 0.3), which makes sense: zero people means zero trips.

### Q2(b) O against everything else

| predictor | slope | p | R² alone |
|---|---|---|---|
| pop | 0.765 | ~0 | 0.668 |
| households | 1.77 | ~0 | 0.651 |
| area (ha) | −0.040 | 3e-33 | 0.020 |
| Townsend | −4.8 | 0.39 | 0.0001 |
| D | 0.019 | 2e-42 | 0.026 |

- **Significant ≠ important.** With n = 7,201, even tiny effects get tiny p-values (area: p = 1e-33, R² = 0.02). The p-value asks "is the slope non-zero?"; R² asks "how much does it explain?".
- **Multiple regression** O ~ pop + households + log(area) + Townsend: R² = 0.788. Every term is significant.
  - Deprivation has a **negative** effect once population is controlled for: deprived areas have lower employment.
  - Larger (rural) areas have fewer commuters per resident: more working from home, farming, and retired people.
- **Multicollinearity:** pop and households have r = 0.88. Their VIFs are about 4.4: high-ish, but under the usual 5–10 alarm level. Their individual coefficients are unstable, but the prediction is fine.
- Use **log(area)**: area spans 3 orders of magnitude, and a raw-area regression is driven entirely by a few huge rural MSOAs.
- **Standardised coefficients** (z-score everything) let you compare predictors measured in different units.

### Q2(c) D is not predictable from these variables

The best R² is 0.067 (0.19 on log scale). D measures where **jobs** are: the City of London has 7k residents and 713k incoming commuters. None of the residential variables measure jobs. Anders adds census employment-by-industry counts to improve this.

### Q2(d) Aggregation

- **(i) Name strings:** strip the trailing `" 001A"` with the regex `\s+\d{3}[A-Z]?$`. This gives 348 names, which are **Local Authority Districts** (lower tier), *not* UTLAs. "Barking and Dagenham" is a London borough, so its LAD is also its UTLA. "Cherwell" is a LAD inside the UTLA "Oxfordshire".
- **(ii) Clustering:**
  - (1) square grid cells, e.g. `floor(E/25000)`;
  - (2) Anders's algorithm: pick a random zone, merge it with its nearest neighbour, repeat until N zones remain.
- **Result (MAUP):** as zones merge, the D/pop histogram narrows towards the national average (~0.77) and starts to look like O/pop. A big zone contains both job centres and housing, which averages out. **Conclusions depend on the zone size.** This is the *Modifiable Areal Unit Problem*.
- **(iii) Lookup tables:** the official ONS MSOA → LAD → UTLA lookup. Here I used the one embedded in the coursework `.mat`. 7,080 of the 7,201 MSOAs (2011 codes) match; the others were split or merged in 2021.

---

## 4. The OD matrix (Q3)

### Building it

- **Filter** to place-of-work code 3 ("working in the UK, not at home"):
  - −8 = not working
  - 1 = works mainly from home: 12.6 M people!
  - 2 = offshore/abroad
- **Lookup:** map each MSOA code to a row/column index. In Python, `pd.Series(np.arange(n), index=codes)`; in MATLAB, `[tf, idx] = ismember(code, codes)`. Keep this table (Q3(b)).
- **Sparse matrix:** `sparse(i, j, count, n, n)` in MATLAB, `scipy.sparse.coo_matrix((v, (i, j)))` in Python. Duplicate (i, j) entries are **summed**. Here 3.45% of the entries are non-zero: sparse storage needs about 20 MB, dense `double` would need 422 MB.
- **Totals:** $O_i$ = row sums, $D_j$ = column sums. Both total 15.09 M trips, and 9.3% of trips are intra-zonal (i = j).

### Q3(d) Distance matrix

`hypot(E - E', N - N')` in MATLAB, or the broadcast version in numpy. Use `single`/`float32` to halve the memory (211 MB at MSOA level).

### Q3(e) Commute distance distribution

Weight each OD pair by the number of people, $T_{ij}$:

- mean straight-line commute is 12.9 km;
- median is 6.4 km.

| distance band (miles) | <1 | 1–2 | 2–5 | 5–10 | 10–25 | 25–50 | 50+ |
|---|---|---|---|---|---|---|---|
| % of commuters | 17.3 | 13.6 | 26.6 | 21.2 | 16.4 | 3.1 | 1.7 |

**Comparing with the NTS** (tables NTS0403/NTS0308 give commute trip lengths in mile bands):

- Ours are **centroid-to-centroid straight lines**. Road distances are about 1.2–1.4× longer.
- All intra-zonal trips are recorded as "short", so short trips are poorly represented.
- The NTS counts *trips* (two per commuting day), not people.

On a log-log plot, the tail falls off like a power law. My slope between 10 and 200 km is −2.4. Anders quotes about 1.75 using a different binning and normalisation. Expect the value to depend on the fitting range.

### Q3(f) Coarser geography

Use an aggregation matrix $P$ (zones × MSOAs, 0/1). Then $T^{UTLA} = P\,T\,P^\top$, one line in both languages. At UTLA level, **63% of trips are internal** and get "distance 0", which is why this level is useless for short trips.

### Q3(g) DataShine-style maps

For one zone, draw lines to/from every zone it exchanges commuters with. Make the line width proportional to the flow, as in `figures/q3g_flow_maps_bristol.png`.

---

## 5. Trip distribution models (Q4–Q8)

### Notation

- $T_{ij}$: observed trips from i to j
- $O_i = \sum_j T_{ij}$, $D_j = \sum_i T_{ij}$
- $f(d)$: the **deterrence function**, which decreases with distance

### Intra-zonal distance

$d_{ii} = 0$ makes $1/d^2$ infinite. The standard fix, used here, is $d_{ii}$ = half the distance to the nearest neighbouring centroid. Alternatives are to exclude the diagonal, or to use $\sqrt{\text{area}/\pi}$.

### Q4 Unconstrained gravity model

$$T_{ij} \approx c\,O_iD_j\,f(d_{ij}), \qquad f(d) = d^{-2}$$

Unravel the matrices into an $n^2 \times 2$ array $[X_{ij},\; T_{ij}]$ with $X = O_iD_j/d_{ij}^2$, then fit $T = cX$. **How you fit matters**, because T contains many zeros and a few huge values:

| method | formula for c | weakness |
|---|---|---|
| OLS through origin | $c = \sum XT / \sum X^2$ | dominated by the few biggest X (usually intra-zonal) |
| least squares on logs | $\log c = \text{mean}(\log T - \log X)$ over $T>0$ | ignores zeros, biased |
| **Poisson maximum likelihood** | $c = \sum T / \sum X$ | none serious: **use this** |

Why Poisson? If $T_{ij} \sim \text{Poisson}(cX_{ij})$, the log-likelihood is $\sum(T\log(cX) - cX)$. Setting the derivative with respect to c to zero gives $\sum T/c = \sum X$. The variance equals the mean, so large flows are down-weighted relative to OLS, and zeros are used properly. The fitted total also equals the observed total.

**Results (Bristol region, 368 MSOAs), 1/d²:**

| c fitted by | CPC |
|---|---|
| OLS | 0.11 |
| log | 0.23 |
| Poisson | 0.39 |

### Goodness of fit

- **CPC / Sørensen index:** $2\sum\min(T^m, T^e) / (\sum T^m + \sum T^e)$, the fraction of commuters put in the right place. It runs from 0 to 1. This is the standard measure in the mobility literature, and it is on Anders's slide.
- **Correlation r and RMSE:** both are dominated by the largest flows.

### Q5 Other inverse powers

Scan γ in $f = d^{-\gamma}$:

| level | best γ (by CPC) |
|---|---|
| Bristol region | 1.2 |
| UTLA level | 1.7 (Anders: 1.75) |

The classic γ = 2 is too steep at MSOA level. The exponential $e^{-\beta d}$ does better at MSOA scale: best β ≈ 0.16 /km, CPC = 0.62.

- **Empirical deterrence:** bin pairs by distance and plot $\sum T / \sum O_iD_j$ against d on log-log axes. The slope estimates −γ directly.
- **Generalised gravity** (Anders): $T = kO^\alpha D^\beta d^{-\gamma}$. Fit it as a **Poisson GLM**, $\log E[T] = \log k + \alpha\log O + \beta\log D - \gamma\log d$. Bristol region: α = 0.62, β = 0.83, γ = 1.21.

### Q6 Singly constrained

- **Unconstrained:** the row sums are off by 83% on average (range −98% to +1272%).
- **Origin-constrained:** multiply row i by $A_i = O_i / \sum_j cO_iD_jf_{ij}$. That gives

  $$T_{ij} = O_i \frac{D_j f_{ij}}{\sum_k D_k f_{ik}}$$

  The constant c cancels out: each origin shares out its $O_i$ trips in proportion to the attractiveness × deterrence of each destination. The rows are now exact, but the columns are still off by 34%.
- **Destination-constrained** is the same idea applied to the columns.

### Q7 Doubly constrained (Furness / IPF / RAS)

$$T_{ij} = a_iO_i\,b_jD_j\,f(d_{ij})$$

We need $\sum_j T_{ij} = O_i$ and $\sum_i T_{ij} = D_j$: 2n nonlinear equations in the 2n unknowns $a, b$.

**Furness iteration:** each equation is trivial to solve for one set of factors if the other set is fixed. Alternate:

$$a_i \leftarrow \frac{1}{\sum_j b_jD_jf_{ij}}, \qquad b_j \leftarrow \frac{1}{\sum_i a_iO_if_{ij}}$$

It converges quickly: 110 iterations to a relative error of 1e-8. In survey statistics the same algorithm is called **RIM weighting** or **raking**.

- **Non-uniqueness:** $(a, b) \to (ka, b/k)$ gives the same T, and one equation is redundant because $\sum O = \sum D$. So a general nonlinear solver needs least squares (`least_squares` / `lsqnonlin`), not a square system solver. The Python script shows this too.
- **Fit (Bristol):** doubly constrained 1/d² gives CPC 0.60. **Recalibrating γ** gives 1.4 by CPC (CPC 0.67), and 1.8 if you match the observed mean trip length (8.1 km). Matching the mean trip length (Hyman's method) is the standard calibration in practice.

### Q8 Opportunity-based models

Idea: distance only matters through **how many opportunities you pass**. $s_{ij}$ is the number of jobs in zones closer to i than j is (excluding i and j). To compute it, sort the zones by distance from i and take a cumulative sum.

**Intervening opportunities (Stouffer/Schneider).** Each opportunity is accepted with probability L:

$$P_{ij} \propto e^{-L V_{ij}} - e^{-L(V_{ij} + D_j)}, \qquad V_{ij} = D_i + s_{ij}$$

Rows are scaled to $O_i$. There is one parameter (best L = 2.5×10⁻⁵ per job). Anders's version writes $(1-\alpha)^V$ instead of $e^{-LV}$.

**Radiation model (Simini et al. 2012).** It has no parameters:

$$T_{ij} = O_i\frac{m_in_j}{(m_i+s_{ij})(m_i+n_j+s_{ij})}$$

Here I used $m = O$ (workers) and $n = D$ (jobs).

**Results (between-zone trips, Bristol region):**

| model | parameters | CPC |
|---|---|---|
| doubly constrained gravity, γ = 1.4 | 1 + 2n factors | 0.670 |
| intervening opportunities | 1 | **0.679** |
| radiation | 0 | 0.41 |

The radiation model puts far too many trips into the nearest zones. It does better at large scales and over heterogeneous areas, which matches the literature. A closed 50 km region with a hard boundary is a tough test for it.

### Which model is best? (Anders's last slide)

Weigh up:

- few parameters vs many;
- theory-based vs ad hoc;
- how much data the model needs;
- whether the parameters are observable;
- whether it transfers to other places or scales.

The doubly constrained gravity model fits well, but needs $O$, $D$ *and* 2n balancing factors. Intervening opportunities gets the same CPC with one parameter.

---

## 6. Patterns worth reusing in coursework

- **Vectorised pairwise operations:** column − row → matrix (`x(:) - y(:)'` or `x[:, None] - y[None, :]`).
- **Sparse from triplets:** `sparse(i, j, v, n, n)`; aggregate with `P*T*P'`.
- **Group sums:** `groupsummary` / `splitapply` / `accumarray` in MATLAB; `groupby().sum()` / `np.bincount` in Python.
- **Lookups:** `ismember(codes, keys)` in MATLAB; `pd.Series(index=keys).reindex(codes)` or `.map()` in Python.
- **Fitting counts:** Poisson ML, not OLS. Always report CPC alongside r.
- **Fixed-point iteration** (Furness) for "scale rows and columns to match totals".
- **Memory:** n² grows fast. Use sparse matrices, `single`, or a sub-region.

## 7. Comments on your own `lab1.py`

Your Q2(a) and the worked part of Q2(b) are correct, and they match my numbers (slope 0.7646, R² 0.668). The skeleton sections (simple regressions, multiple regression, VIF) are what `q2_census_stats.py` fills in. Two small differences:

- I used `log(area)` in the multiple regression.
- `area_ha` is cast to `uint32`, which is fine because the areas are whole hectares in this file.

`extract_lab.py` points to a PDF path that no longer exists, so it won't run. It isn't needed, though.
