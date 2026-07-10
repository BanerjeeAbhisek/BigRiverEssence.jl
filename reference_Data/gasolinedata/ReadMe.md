# gasoline reference data (`generate_gasolinedata.R`)

The `generate_gasolinedata.R` script extracts the gasoline near-infrared (NIR) dataset
from the `pls` R package and saves it as CSV files. These files are read by
`plskern_tutorial.md` so the tutorial can load the data directly, without requiring R or
the `pls` package at build time. The script only needs to be run once; the resulting CSVs
are committed to the repository.

## It performs the following tasks:

- It loads the R package `pls` and the `gasoline` dataset.
- It extracts the octane numbers (the response) and the NIR spectra (the predictors) for
  60 gasoline samples.
- It writes the NIR spectra matrix (60 samples × 401 wavelengths) as `NIR.csv`.
- It writes the 60 octane numbers as `octane.csv`.

The data is stored as plain numeric CSVs (no headers or row names), matching how `readdlm`
reads them in Julia. The wavelength grid (900–1700 nm in 2 nm steps, giving 401 points) is
reconstructed directly in the tutorial and does not need to be stored.

## Where it is used

The CSV files are read by `plskern_tutorial.md`, which loads the NIR spectra and octane
numbers and applies `plskern` to build a partial-least-squares regression predicting
octane from the spectra. Storing the data as CSVs keeps the tutorial reproducible and
removes any dependency on R when the documentation is built.

## References

The gasoline dataset (near-infrared spectra and octane numbers for 60 gasoline samples) is
from Kalivas (1997) [1], obtained via the R `pls` package. The NIR spectra were measured
as log(1/R) from 900 to 1700 nm in 2 nm steps, giving 401 wavelengths.

[1] Kalivas, J. H. (1997). Two data sets of near infrared spectra. *Chemometrics and
    Intelligent Laboratory Systems*, 37, 255–259.




