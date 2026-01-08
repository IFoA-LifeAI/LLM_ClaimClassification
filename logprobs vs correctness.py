# %%
import pandas as pd
import numpy as np
from pathlib import Path
import great_tables as gt
from plotnine import *
np.random.seed(123)

# %%

# Human predictions
output_name_csv = "cod_human_labels"
output_csv_path = Path("./Data") / f"{output_name_csv}.csv"
hm_df = pd.read_csv(output_csv_path)

# %%
# AI predictions with logprobs
output_name_csv = "output_openai_gpt_4o_logprobs"
output_csv_path = Path("./Data") / f"{output_name_csv}.csv"
lp_df = pd.read_csv(output_csv_path)

cols_keep = ["cause_of_death", "category", "logprobs"]
lp_df = lp_df.loc[:, cols_keep]

# %%

rec = pd.merge(hm_df, lp_df, on="cause_of_death")
rec["match"] = rec["category_human"] == rec["category"]
rec["probs"] = np.exp(rec["logprobs"])
rec["probability"] = pd.cut(rec["probs"], bins=[0, 0.95, 1.0])

# %%
accuracy_df = rec.groupby("probability")["match"].aggregate(["count", "mean"])

accuracy_gt = gt.GT(accuracy_df.reset_index())
accuracy_gt = accuracy_gt.fmt_percent(["mean"], decimals=1)

(accuracy_gt.cols_label(
    {
        "probability": "Probability",
        "count": "Count",
        "mean": "Accuracy"
    }
)
)

# %%
(ggplot(rec, aes(y="probs", x="match", color="match")) +
    geom_jitter(width=0.3, alpha=0.35) +
    coord_flip() +
    scale_y_reverse() +
    xlab("LLM Guess Matches Human") +
    ylab("Probability of Guess") +
    scale_color_manual(values={True: "forestgreen", False: "blue"}) +
    theme_bw() +
    theme(
        legend_position="none",
        axis_line=element_blank(),      # removes axis lines (flip-safe)
        panel_border=element_blank(),   # removes plot outline
        panel_grid=element_blank()
)
)

# %%
