import sys
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# token_ids
x_intervals = [     3,      3,      3,      3,      3, 111757,    632,    660,   9437,  42932,    661,  10783,     15, 201573,   1002,    660,  54103,    861,  63808,    267,  20165,     17,  66828,    267,  12427,    861, 156788, 115739,    368,   8821,   6149, 105311,  47425,     29,    189,   7994,  18704,  19483,   4054,    661,   3291, 114054,    376,   1485,    368,    458,  16834,  28635,   5357,    660,    267,   5945,    386,    664,    602,  72632,   1900,  55671,     17,  66802,  54968,  73772,  98254,    376,  95708,   7553,   1485,   3808, 129915,    661,  10012,   2135,  11978,  31722,  15803, 226505,   2131,   3291,   3866,  35277,    427,  32048,     17,  17585,    267,  12442,  15226,  25754,     15,  97490,   1427, 131270,   2256,    368,  22779,    530,  38317,  78497,   3638,    361,    368,   4306,  31136,    386,  29853,   9322,   4644,     17,  18061,   1728,    861,    632,   9507,   1306,    368, 187353,   1492,  13133,  32639,    461,   3595,  37468,   3269,    267,  33792,    530,    368, 153796,  49986, 162502,    461,    267,   4733,  70470,  18210,   6149, 105311, 182924,     29,    189, 153772,    368,   9468,  42932,   1485,   7165,    427, 196427,    603, 105311,  66673,     29,  19732,    679,  12448, 227057,  10991,  17211, 139366,   2810,   4801,    283,    366,  92549,  78668,   1160,   7918,    447,   1742,   1750,   1394,    366, 108530,    283,    602,  72632,   1900,     17,   2695,  28139, 127469,  99259,  15179,    551,    772,   4801, 238065,  12742,  11300,   5258, 129915,  16155,   2783,  22600,   2135, 127935,   2586,    596,  61374,    283,  59634,  17211,  22157,   9783,    686,    413,  14259,  32032,     17,   3380,    393,    267,  12871,  16998,   8738,     15,    679, 159347,  36190,   4801,    808,  57678,    674,  48284,    948,    465, 160041,   4801,   1486,  40410, 187659, 232898, 172483,     17,  88017,     15,   1669,    795,  18105,    408,    679, 247358,  38792,   2483,    283,   2155,   1479,  99005,   1115,    693,    447,  33792,    674,    578, 107106,  61025,  49333,  25684,   6322,  65394, 122041,  46845,     17,      2]
y_intervals = x_intervals

attention_scores_file = sys.argv[1]
with open(attention_scores_file, "r", encoding="utf-8") as attn_f:
     attn_scores = attn_f.read()

attn_scores_list = ast.literal_eval(attn_scores)
data = np.array(attn_scores_list)

df = []
for i, y_range in enumerate(y_intervals):
    for j, x_range in enumerate(x_intervals):
        df.append({
            'x_range': f'{x_range}',
            'y_range': f'{y_range}',
            'value': data[i, j]
        })

df = pd.DataFrame(df)

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(data, annot=False, fmt="", cmap="YlOrRd")

heatmap.set_xticklabels([])
heatmap.set_yticklabels([])

title = "Self-Attention Heatmap for Post-Ins"
title_fontsize = 18
heatmap.set_title(title, fontsize=title_fontsize)

tick_fontsize = 12
heatmap.tick_params(axis='both', which='major', labelsize=tick_fontsize)

plt.show()
