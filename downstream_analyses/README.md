# Downstream analyses

This directory contains all the code used for the downstream analyses in the paper.

## Region Means

This analysis can be reproduced with the `region_means.ipynb` notebook.

In our descriptive task, we estimate conditional means, averaging wealth by local second-level administrative region (ADM2), roughly corresponding to municipalities or counties. We group the clusters in the held-out downstream fold by ADM2 region and compare the mean observed IWI with the predicted IWI with and without the different methods of bias correction. These quantities are subject to the same shrinkage bias as causal estimates, and our framework applies equally well in this setting, enabling us to assess how effectively we correct biased group-level estimates.

## Empirical Analysis of Aid Interventions

This analysis can be reproduced with the `aid_analysis.ipynb` notebook.

To assess the practical utility of our correction methods in a setting where the true treatment effect is unknown, we construct an evaluation framework based on real-world development projects. Specifically, we leverage the same dataset of geo-referenced interventions funded by the World Bank and the Chinese government as was used in Malik et al. (2021); Conlin
(2024). Each project is tagged with an associated aid sector $s \in \mathcal{S}$ (e.g., *Health*, *Water Supply and Sanitation*, or *Women in Development*), which defines a downstream trial $D_{\text{Trial } s} = \{(X_i, \widehat{Y}_i, A_{is}): i\in \mathcal{I}_{\textrm{Trial } s}\}$. Here, $A_{is} \in \{0,1\}$ indicates exposure to a sector-$s$ intervention at location $i$, $X_i$ are satellite features, and $\widehat{Y}_i$ are IWI predictions from the upstream-trained model.

Since the treatment effects of these interventions are unknown, we deem this analysis a test of external validity by comparing EO-ML predicted wealth values with observed ones. Our goal is to assess whether our bias-corrected predictions can reflect causal estimates similar to those from observed data.

To avoid label leakage and emulate real-world constraints, we perform a two-stage evaluation. We begin by randomly partitioning the available DHS survey data into two disjoint sets: an upstream set for training and calibrating the EO-ML model, and a downstream set used solely for evaluation. The model is trained on the upstream data, without any exposure to the intervention data or knowledge of future evaluation criteria. Shrinkage corrections (e.g., LCC or Tweedie's correction) are also estimated on this upstream set. 

We match project locations to the corresponding ADM2-level administrative regions, roughly equivalent to districts or municipalities. For a given funder-sector pair (e.g., Health projects funded by the World Bank), we identify all ADM2 regions containing at least one relevant intervention site. We then consider villages surveyed in the downstream DHS data within those regions, 3 to 8 years after the intervention, as the treated group. Villages in the same surveys, but outside the treated ADM2 regions, serve as controls.

For each funder-sector combination, we estimate the average treatment effect as the difference in mean outcomes between treated/control villages. We compute this using both observed IWI values and the model-predicted (and correction-adjusted) IWI values.

## Miscellaneous

The three remaining notebooks contain the following:

- `make_maps.ipynb` recreates the maps in Figure 5, highlighting the correlation between over/underestimations and the Tweedie corrections.

- `model_performance.ipynb` plots some additional figures highlighting the model performance on held-out clusters.

- `ratledge_analysis.ipynb` makes plots related to the Ratledge models for the purpose of evaluating the best $\lambda_b$ parameter. The best of these models (with $\lambda_b=15$) are used in the other analyses.