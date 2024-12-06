Python version 3.12
Packages:
pip install POT
conda install anaconda::pandas
conda install anaconda::pip
conda install anaconda::scikit-learn
conda install conda-forge::matplotlib
conda install plotly::plotly
conda install anaconda::seaborn
conda install conda-forge::nbformat

CLEAR NOTEBOOK OUTPUTS BEFORE COMMITTING
(Otherwise, if two people update code and generate different outputs, someone has to deal with merge conflicts for both the code *and* the output)

In Visual Studio, if you modify code in a file which gets imported by another file, you need to, for windows:
    ctrl+shift+p to open the command palette -> type "reload window" to find "Developer: Reload Window" -> click on that -> then rerun the importing script