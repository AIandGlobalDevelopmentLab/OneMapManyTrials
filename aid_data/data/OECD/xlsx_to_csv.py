import pandas as pd

oedc = pd.read_excel('DevFi_Classification.xlsx')

# 2. Rename columns
oedc = oedc.rename(
    columns={
        "Parent code": "parent_code",
        "code name e": "code_name_e",
        "code": "sect_code"
    }
)

# 3. Coerce & clean
oedc["sect_code"] = oedc["sect_code"].astype(str)
# Replace NBSPs in parent_code with actual NaN
oedc["parent_code"] = oedc["parent_code"].replace("\u00a0", pd.NA)

# Map specific subsectors up to their parent groups
reassign = {
    "311": "310",
    "312": "310",
    "321": "320",
    "322": "320",
    "331": "330",
}
oedc["parent_code"] = oedc.apply(
    lambda row: reassign.get(row["sect_code"], row["parent_code"]),
    axis=1
)

# 4. Strip leading numerals from the English name
oedc["sect_code_name"] = (
    oedc["code_name_e"]
    .str.replace(r"^.*\. (.*)", r"\1", regex=True)
)

# 5. Keep only the three key columns
oedc_sect_df = oedc[["sect_code", "sect_code_name", "parent_code"]].copy()

# 6. Append the missing “330” row
oedc_sect_df = pd.concat([
    oedc_sect_df,
    pd.DataFrame([{
        "sect_code":      "330",
        "sect_code_name": "Trade and Tourism",
        "parent_code":    pd.NA
    }])
], ignore_index=True)

oedc_sect_df.to_csv('DevFi_Classification.csv', index=False)