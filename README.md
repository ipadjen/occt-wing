# occt-wing

Playground for the OpenCascade geometry kernel

A simple tool for some basic parametric wing design.

![welcome_vid](example_vid.gif)

### What it does
- Airfoil options: NACA 4-series, import (.dat file), or directly fetch thousands of airfoils from the [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)
- Creates some typical wing shapes (straight, tapered, swept, elliptical)
- Wing twisting (as geometric washout) and/or different wingtip airfoil (as aerodynamic washout)
- Half-wing or full wing
- Finite-thickness trailing edge
- Export to STEP

### Wait it misses
- No dihedral angle
- Could benefit from better wingtip solutions

# Install
After installing conda, do
```
conda env create -f environment.yml
conda activate wing-designer
python wing-designer.py
```
