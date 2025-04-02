# 2DModel

To use this virtual, two-dimensional (2D) model system a user will first need to edit the input file, infile.py, to select the paremeters for the calculations they would like to run. Secondly, after the user is satisfied with their input file, the  user should run main.py.

## Input File

In the input file, infile.py, there are many variables one must set the value of (either by caning it or leaving it be). In this section, each variable of the input file will be described.

- **PrintOut**: This parameter tells the program wether to print results to the screen (PrintOut = 1) or not (PrintOut = 0)

### Ground State Calculation Parameters

- **c**: This parameter is the lattice constant for the model system, i.e. this is the size of the unit cell in real space.
- **A**, **B**: These parameters define the ground state potential of the system, each corresponding to the depth of one of two potential dips in the unit cell.
- **D**: This parameter defines the strength of a sinusoidal potential term. If this term is not desired, simply set D = 0.
- **Nel**: This parameter defines the number of electrons in each unit cell.
- **Kpoints**: This parameter defines the number of **k**-points considered along both the k<sub>x</sub> and k<sub>y</sub> directions.
- **Gnum**: This pareameter defines the number of **G**-vectors considered along both the positive G<sub>x</sub> and G<sub>y</sub> directions. Thus there will be a total of (2 * Gnum + 1) **G**-vectors considered along eack of the G<sub>x</sub> and G<sub>y</sub> directions.
- **Ntot**: This parameter defines the total number of electron energy bands which will be included in the calculations. *Note:* When calculating the dielectric function a larger value for Ntot is recommended.
- **HXC**: This parameter defines if the following elements will be included in the ground state Hamiltonian:
  - If HXC = 3, the ground state Hamiltonian will include the Hartree (H) energy and the exchange and correlation (XC) energy calculated using the local density approximation (LDA).
  - If HXC = 2, the ground state Hamiltonian will include the Hartree (H) energy and the exchange (X) energy calculated using the local density approximation (LDA). The correlation (C) energy from the LDA is excluded.
  - If HXC = 1, the ground state Hamiltonian will include the Hartree (H) energy. and the  energy calculated using the local density approximation (LDA). The exchange (X) and correlation (C) energy calculated using the local density approximation (LDA) is excluded.
  - If HXC = 0, the ground state Hamiltonian will exclude the Hartree (H) energy and the exchange and correlation (XC) energy calculated using the local density approximation (LDA).
- **TOL**: The parameter defines the threshold of numerical self-consistency desired when carrying out self-consitent calculations, i.e. the maximum difference allowed between a current and prior result for the calculation to be treated as having reached self-consitency.
- **MIX**: This parameter defines the weight of the previous result in a self-consistent calculation when calculating the weighted avaerage of the current and prior results (i.e. Final = MIX * Previous + (1 - MIX) * Current).
- **out**: This parameter determines what, if any, additional outputs will be calculated and/or plotted from the ground state of the system:
  - If out = 0, no additional output will be calculated.
  - If out = 1, a cross-section of the ground state density will be calculated along the x-axis and plotted.
  - If out = 2, the full two-dimensional ground state density will be calculated and plotted.
  - If out = 3, the ground state b and structure will be plotted.
- **restart**: This paremeter allows the user to read the Hartree (H) energy, as well as the exchange (X) and correlation (C) energy from the local density approximation (LDA) that will be included in the ground state Hamiltonian if it has the same parameters for the ground state calculation which produced the file containing said information following a self-consistency calculation, i.e. this allows the user to skip having to carry out the ground state self--consistent calculations for repeated calculations using the same ground state system.

### Linear Response Calculation Parameters

- 

### Time-Dependent Calculation Parameters

- 

## Outputs
