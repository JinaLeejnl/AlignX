<div align="center">
  <h1 style="font-size: 40px;">AlignX</h1>
  <p>A large-scale dataset of over 1.3 million personalized preference examples</p>
</div>

# Links

- 📜 [Paper]()
- 🤗 [AlignX]()
- 🤗 [AlignX<sub>test</sub>]()
- 🤗 [AlignXpert<sub>ICA</sub> (Training with a 7% Subset)]()
- 🤗 [AlignXpert<sub>PBA</sub> (Training with a 7% Subset)]()
- 🤗 [AlignXpert<sub>ICA</sub> (Training with the Full Dataset)]()
- 🤗 [AlignXpert<sub>PBA</sub> (Training with the Full Dataset)]()


# Dataset Statistics

The table below summarizes the data sources and statistics for AlignX, involving both large-scale Reddit data and existing alignment datasets to maintain universal value alignment capabilities, with a total of 1,309,785 samples.

| **Source** | **Reddit** | **PKU-SafeRLHF** | **UltraFeedback** | **HelpSteer2** |
|------------|------------|------------------|-------------------|----------------|
| **Dimension** | The 90 self-defined preference dimensions | Safety | Helpfulness / Honesty / Instruction-Following / Truthfulness | Helpfulness / Correctness / Coherence / Complexity / Verbosity |
| **#Examples** | 1,224,151 | 10,714 | 11,629 / 16,809 / 36,169 / 7,219 | 2,255 / 144 / 26 / 33 / 636 |


# Dataset Overview

<img src="figures/dataset_overview.png" width="1200px">

# Dataset Format

```jsonc
{
    "prompt": "", // the post eliciting responses
    "chosen": "", // the user-preferred response
    "rejected": "", // the less preferred response relative to "chosen"
    "Preference Direction": [0/0.5/1] * 90, // a 90-element list: 1 = "Positive" (higher levels preferred), 0 = "Negative" (lower levels preferred), 0.5 = "Neutral" (no clear preference)
    "Demographic Information": "", // a comprehensive natural language description of the user
    "User-Generated Content": [ // comments written by the same user on other posts
        { // UGC 1
            "prompt": "",
            "comment": "",
            "Preference Direction": [0/0.5/1] * 90
        },
        { // UGC 2
            ...
        },
        { // UGC 3
            ...
        },
        { // UGC 4
            ...
        }
    ],
    "Pair-wise Comparative Feedback": [ // the preference pairs of the same user for comments under other posts
        { // PAIR 1
            "prompt": "",
            "chosen": "",
            "rejected": "",
            "Preference Direction": [0/0.5/1] * 90
        },
        { // PAIR 2
            ...
        },
        { // PAIR 3
            ...
        },
        { // PAIR 4
            ...
        }
    ]
}
```
