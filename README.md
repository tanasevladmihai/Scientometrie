# Scientometrie v2

### What is Scientometrie?
Well, the word itself is Romanian for "scientometrics", a subfield of informetrics that studies quantitive aspects of scholarly literature. \
The program itself is very straight forward: it's a workflow for extracting and processing citations from indexed academic databases (such as Web of Science, Scopus, Google Scholar etc.). 

### About versions
[v2](https://github.com/Tudor230/Scientometrie.git) is built on the base program done by my [friend](https://github.com/Tudor230). This version now has an interactive GUI, options for greater flexibility and better error handling capabilities.
For v3 I am  planning on implementing PostgreSQL dabatase functionalities. \
Future versions will include wider support for more formats and sources, right now it's mostly tailored for  just Romanian papers.

## Instructions
1.  [x] Populate desired directory (default is *core_raw*) with exported .csv files from the [CORE Conference Rankings Portal](https://portal.core.edu.au/conf-ranks/). Search by year and export. For labelling, I suggest renaming them like this: CORE_XXXX.csv, where XXXX is the year.
2.  [x] Populate desired directory (default is *journal_raw*) with downloaded .xlsx files from the [uefiscfi platform](https://uefiscdi.gov.ro/scientometrie-reviste) which is a Romanian government website partaining to[cncsis](https://cncsis.gov.ro/). The downloads will be in .pdf format, you'll have to convert them to .xlsx yourself :) I also suggest labeling them AIS_XXXX. xslx or IF_XXXX.xslx repsectively. \
These files should contain tables populated by scientific journals and related information such as author,Q AIS and JIF rank etc. If you get your data from a different platform and their format is identical to the Romanian one, then the program should work unmodified.
3.  [x] Populate desired folder (default is *exports*) with exports from both WoS and Scopus of your paper's citations. They'll most likely be in both .csv and .xlsx formats. Just rename them so both have the same name.
## Technical aspects
### Technologies used:

- Python 3.13 with Tkinter Tcl/Tk toolkit for the graphic user interface.
- pandas, openpyxl, xlrd packages for handling .xlsx (Excel) files

### Installation guide for the necessary packages:

1. **Installing pandas (using pip)** \
`pip install pandas`

2. **Installing openpyxl (using pip)** \
`pip install openpyxl` \
**or if you prefer Conda:** \
`conda install openpyxl`
3. **Installing xlrd (using pip)** \
`pip install xlrd`

Tkinter does not require an installation as it is already present in every modern Python version installed.

***Note:** If you're using Jetbrains Pycharm IDE you can simply go into the Python Packages tab, search for the packages and download & install them from there.*
