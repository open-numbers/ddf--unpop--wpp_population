# UN World Population Prospects

This dataset contains all indicators from [UN World Population Prospects][1].

Check [the concepts file][2] for a list of all indicators.

[1]: https://population.un.org/wpp/
[2]: https://github.com/Gapminder/ddf--unpop--wpp_population2019/blob/master/ddf--concepts--continuous.csv

## Data sources summary

We use all standard projection xls files from the [download page][3]. Locations comes from the [Location Metadata File][4].
Note that there are indicators only available in a 5 year peroid, so we add a new dimension `freq` to these indicators, and
`year` for these indicators means the start year of the period.

[3]: https://population.un.org/wpp/Download/Standard/Population/
[4]: https://population.un.org/wpp/Download/Metadata/Documentation/

## How to run the script

The script was based on [@harpalshergill's work on wpp][5]. You should download all xls files from WPP and put them into
the `etl/source` directory. Check the `metadata.xlsx` file for details of file names and their related indicators. In
case of UN changed the file names, `metadata.xlsx` should be changed accordingly too.

[5]: https://github.com/harpalshergill/ddf--unpop--wpp_population/
