# --- Setup ---------------------------------------------------------

# Load/install devtools for GitHub & Bitbucket installs
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("configr", quietly = TRUE)) install.packages("configr")

library(devtools)
library(configr)

# Install required packages (force GitHub/Bitbucket where needed)
if (!requireNamespace("rgdal", quietly = TRUE)) install_github("cran/rgdal")
if (!requireNamespace("rgeos", quietly = TRUE)) install_github("cran/rgeos")
if (!requireNamespace("iwi", quietly = TRUE)) install_bitbucket("hansekbrand/iwi")
if (!requireNamespace("DHSharmonisation", quietly = TRUE)) {
    install_bitbucket("hansekbrand/DHSharmonisation", ref = "memisc")
}
if (!requireNamespace("globallivingconditions", quietly = TRUE)) {
    # Assuming it comes with DHSharmonisation or available via CRAN
    library(globallivingconditions)
}

# Other required CRAN packages
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("countrycode", quietly = TRUE)) install.packages("countrycode")

library(dplyr)
library(countrycode)
library(globallivingconditions)

# --- Read Config ---------------------------------------------------

config <- read.config("../config.ini")
DATA_DIR <- config$PATHS$DATA_DIR
DHS_USER <- config$DHS$USERNAME
DHS_PASS <- config$DHS$PASSWORD

# --- Download and Harmonise DHS -----------------------------------

dt.all <- download.and.harmonise(
  dhs.user = DHS_USER,
  dhs.password = DHS_PASS,
  log.filename = "living-conditions.log",
  vars.to.keep = NULL,
  variable.packages = c("wealth"),
  file.types.to.download = c("PR", "GE", "IR", "KR", "MR"),
  make.pdf = FALSE,
  superclusters = TRUE
)

# --- Process Data --------------------------------------------------

df.sm <- dt.all %>%
  select(
    country.code.ISO.3166.alpha.3, RegionID, ClusterID, HouseholdID, source, lon, lat,
    year.of.interview, month.of.interview,
    rural, iwi, owns.tv, refridgerator, phone, bicycle, car,
    has.electricity, water, nr.of.sleeping.rooms,
    type.of.cooking.fuel, place.to.wash.hands, water.made.safe,
    cooking.on.stove.or.open.fire, motorcycle, animal.drawn.cart,
    boat.with.motor, owns.land, hectares.of.land, owns.livestock,
    has.bankaccount, time.to.water, has.computer, has.video,
    owns.radio, wall, roof, has.fan, has.table, has.oven,
    Nightlights_Composite
  ) %>%
  filter(!is.na(lon), !is.na(iwi)) %>%
  distinct()

# Add country and continent info
df.sm$country <- countrycode(df.sm$country.code.ISO.3166.alpha.3,
                             origin = 'iso3n', destination = 'country.name')
df.sm$continent <- countrycode(df.sm$country.code.ISO.3166.alpha.3,
                               origin = 'iso3n', destination = "continent")

# Filter to Africa
df.sm.africa <- df.sm %>%
  filter(continent == "Africa")

# Optional: rows with missing TV ownership
df.sm.africa.na <- df.sm.africa %>%
  filter(is.na(owns.tv))

# --- Save household data as CSV --------------------------------

output_file <- file.path(DATA_DIR, "raw_household_dhs_data.csv")
write.csv(df.sm.africa, file = output_file, row.names = FALSE)

cat("Data exported to:", output_file, "\n")
