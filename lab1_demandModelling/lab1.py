# TMM Lab 01 - Q2: Basics of the census data
# Run cell-by-cell in VS Code: each "# %%" is a cell (Shift+Enter to run).

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# %% Q2: Load the data
# pandas.read_csv is the Python equivalent of Matlab's import tool + readtable.
# A pandas DataFrame plays the same role as a Matlab table.
df = pd.read_csv("UK_census_msoa_stats.csv")

# Short column names are easier to type than "Townsend deprivation score"
df = df.rename(columns={
    "MSOA Code": "code",
    "MSOA Name": "name",
    "Longitude": "lon",
    "Latitude": "lat",
    "Population": "pop",
    "Area": "area_ha",
    "Households": "households",
    "Townsend deprivation score": "townsend",
    "O (trips)": "O",
    "D (trips)": "D",
})

# Check what types pandas guessed for each column
print(df.dtypes)
print(df.head())

# %% Q2: Sensible variable types
# Counts can never be negative, so an unsigned integer type is appropriate
# (same idea as uint32 in Matlab). int64 is pandas' default and wastes memory.
count_cols = ["pop", "area_ha", "households", "O", "D"]
df[count_cols] = df[count_cols].astype("uint32")

# The MSOA code is a unique ID for each row, so make it the index
# (like RowNames in a Matlab table). This lets you do df.loc["E02000001"].
df = df.set_index("code")

# A categorical variable takes one of a small set of values (e.g. country).
# The first letter of the code tells us England (E) or Wales (W).
df["country"] = df.index.str[0].map({"E": "England", "W": "Wales"}).astype("category")

print(df.dtypes)
print(df["country"].value_counts())
print(df.describe().T)

# %% Q2(a): Scatter plot of O(i) against population
fig, ax = plt.subplots()
ax.scatter(df["pop"], df["O"], s=3, alpha=0.4)
ax.set_xlabel("Population of MSOA")
ax.set_ylabel("Total origin trips O(i)")
ax.set_title("O(i) vs population")
plt.show()

# %% Q2(a): Correlation coefficient
# Pearson's r measures how close the relationship is to a straight line (-1 to 1)
r, p_value = stats.pearsonr(df["pop"], df["O"])
print(f"Pearson r = {r:.3f}, p-value = {p_value:.3g}")

# %% Q2(a): Linear regression  O = b0 + b1 * pop
# statsmodels OLS is roughly equivalent to Matlab's fitlm.
# add_constant adds the column of 1s that gives us the intercept b0.
X = sm.add_constant(df["pop"].astype(float))
y = df["O"].astype(float)
model_a = sm.OLS(y, X).fit()
print(model_a.summary())

# Draw the fitted line over the scatter
fig, ax = plt.subplots()
ax.scatter(df["pop"], df["O"], s=3, alpha=0.4, label="MSOAs")
x_line = np.linspace(df["pop"].min(), df["pop"].max(), 100)
b0, b1 = model_a.params
ax.plot(x_line, b0 + b1 * x_line, "r-", label=f"O = {b0:.0f} + {b1:.3f} * pop")
ax.set_xlabel("Population of MSOA")
ax.set_ylabel("Total origin trips O(i)")
ax.legend()
plt.show()

# =============================================================================
# Q2(b): Relationships between O(i) and the other variables
# =============================================================================

# %% Q2(b) WORKED: correlation matrix
# .corr() computes Pearson r for every pair of columns at once.
# Read along the "O" row to see how O relates to everything else, and look
# off-diagonal for variables that are strongly related to EACH OTHER
# (that matters later for multiple regression).
vars_b = ["O", "D", "pop", "households", "area_ha", "townsend"]
corr = df[vars_b].corr()
print(corr.round(3))

# A heatmap makes the matrix easier to read
fig, ax = plt.subplots()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(vars_b)), vars_b, rotation=45)
ax.set_yticks(range(len(vars_b)), vars_b)
for i in range(len(vars_b)):
    for j in range(len(vars_b)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, label="Pearson r")
plt.tight_layout()
plt.show()

# %% Q2(b) SKELETON 1: simple regression of O on each variable, one at a time
# Goal: a summary table with one row per predictor, showing
#   slope, its standard error, its p-value, and R^2.
#
# HINTS
# - Copy the pattern from part (a): X = sm.add_constant(...), sm.OLS(y, X).fit()
# - A fitted model has attributes: .params, .bse (std errors), .pvalues, .rsquared
#   Each of .params/.bse/.pvalues is indexed by column name, e.g. fit.params["pop"]
# - Build a list of dicts, then turn it into a DataFrame at the end.

predictors = ["pop", "households", "area_ha", "townsend", "D"]
y = df["O"].astype(float)

rows = []
for var in predictors:
    # TODO: build X for this single predictor (remember the constant)
    # TODO: fit the model
    # TODO: rows.append({...}) with keys "predictor", "slope", "std_err", "p_value", "R2"
    pass

simple_results = pd.DataFrame(rows)
print(simple_results)

# QUESTIONS to answer in a comment once you have the table:
# 1. Which single variable predicts O best? Is it better than population?
# 2. area_ha: is the slope positive or negative? Why might that be?
#    (Try plotting O against np.log(area_ha) too - does it look more linear?)
# 3. A p-value can be tiny while R^2 is small. What does each one tell you?

# %% Q2(b) SKELETON 2: multiple regression  O ~ pop + households + area_ha + townsend
# HINTS
# - Same as before, but X now has several columns: df[[...]].astype(float)
# - Compare the coefficients with the simple-regression slopes above.
#   Why does the pop coefficient change when households is also in the model?
#   (Look at the pop-households correlation in the heatmap.)
# - Check the "[0.025 0.975]" column in .summary(): a 95% confidence interval
#   that contains 0 means that variable isn't significant given the others.

multi_vars = ["pop", "households", "area_ha", "townsend"]
# TODO: X_multi = ...
# TODO: model_b = ...
# TODO: print(model_b.summary())

# QUESTIONS:
# 4. Has the adjusted R^2 improved much over pop alone (0.668)?
# 5. Try dropping one variable at a time. Which one hurts the fit most?

# %% Q2(b) OPTIONAL: multicollinearity check with VIF
# The variance inflation factor (VIF) measures how well each predictor can be
# predicted by the OTHER predictors. VIF > ~5-10 is a warning sign.
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# TODO: compute VIF for each column of X_multi (skip the constant)
