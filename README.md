# TM Modelling: answers and notes index

## How each lab folder is organised

Every lab folder has its own `README.md` listing which files are which:

- **Provided content:** what you uploaded from Blackboard (lab sheet, slides, data, lecturer's code). It sits loose in the lab folder and is never edited by me.
- **Your own work:** scripts you wrote before (Labs 1, 2 and 4).
- **Answers:** everything I wrote, always inside the `answers/` subfolder, with a `NOTES.md` (theory, worked answers, results, coursework tips).

The provided files stay where they are because both your scripts and the answers load them from those paths.

| Lab | Notes | Code | Languages |
|---|---|---|---|
| 1 Demand modelling | [lab1 NOTES](lab1_demandModelling/answers/NOTES.md) | `q1`–`q8` | Python + MATLAB |
| 2 Network modelling | [lab2 NOTES](lab2_NetworkModelling/answers/NOTES.md) | `q1`–`q4`, `q5_q11_osm` | Python + MATLAB (OSM import in Python) |
| 3 Assignment modelling | [lab3 NOTES](lab3_assignmentModelling/answers/NOTES.md) | `q1`–`q14` | Python + MATLAB |
| 4 Microscopic modelling | [lab4 NOTES](lab4_microscopicM/answers/NOTES.md) | `p1`–`p4` | Python (as the lab) |
| 5 Data-driven modelling | [lab5 NOTES](lab5_DataDrivenModelling/answers/NOTES.md) | `s1`–`s3` | Python + MATLAB |
| 7 Practitioner studies (sheet says "Lab 06") | in progress, together; see [lab7 README](lab7_/README.md) | starter code: `lab7_data`, `q1_explore_data`, `q3_run_intersection` | Python |

## Things to know

- **Run each script from its `answers/` folder.** Each one also runs cell by cell in VS Code (`# %%` in Python, `%%` in MATLAB).
- **Your MATLAB R2026b has no toolboxes.** Everything here runs on base MATLAB, using hand-written stand-ins:

  | toolbox function | stand-in |
  |---|---|
  | `fitlm` | `ols_fit.m` |
  | `glmfit` | `poisson_glm.m` |
  | `quadprog` | `qp_small.m`, `gradient_projection.m`, `frank_wolfe.m` |
  | `arima` | least-squares AR |
  | `fitcnet` | kNN |

  UoB's campus licence lets you add the toolboxes free (Home → Add-Ons → Get Add-Ons: Statistics and Machine Learning, Optimization, Econometrics, Mapping, Deep Learning). The scripts then use them automatically where I've marked it.
- **Lab 5:** four Melbourne data files are missing (`pedestrian_counts.txt`, `lines_ys.txt`, `sensors_xs.txt`, `sensors_ys.txt`). Sections 1–2 run on synthetic stand-in data until you copy those files from Blackboard into `lab5_DataDrivenModelling/`.
- **Python packages used:** numpy, scipy, pandas, matplotlib, statsmodels, scikit-learn, networkx, geopandas/shapely, pyproj, osmium, Pillow. All are already installed.
# Y4_TMM
