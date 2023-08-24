# Post-Instruction


## Contents

* [Overview](#overview)
* [Quick to Use](#quick-to-use)
* [Experiments](#experiments)
* 
* [Contact](#contact)

## Overview
<p align="center">
  <img src="https://github.com/Adaxry/Post-Instruction/blob/main/figures/examples.png" alt="examples" width="800"/>
</p>


Current instruct-following data generally put the task instruction before the input sentence (referred as "Pre-Ins") for sequqnce generation tasks (e.g., machine translation). We observe that LLMs may forget the frontmost task instruction when the input sentence is long, thus we propose to simply place the task instruction after the input sentence (referred as "Post-Ins"). Both our theoretical and experimental analyses show that Post-Ins pays larger attentions on the model's instruction-following capabilities, yielding consistent performance improvements across two common sequence generation tasks. For more details, please refer to our [technical report](https://arxiv.org/abs/2308.12097) (Instruction Position Matters in Sequence Generation with Large Language Models).


## Contact
Please feel free to contact us (yijinliu@tencent.com) for any further questions.
