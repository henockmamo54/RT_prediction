﻿# Transormer based retention time prediction model.

The model is trained on a murine liver dataset that includes samples from four tissues: liver, heart, kidney, and muscle, all extracted from the same animal. 

The retention time prediction model has two main components. First, it predicts the retention time for a given peptide sequence. However, this predicted value is not on the same scale as the required experiment. To align the predictions with the scale of the required experiments, a polynomial regression is coupled with a transformer-based prediction model. The regression model is trained using reference peptides' retention times from the desired experiment and the predicted retention time values for the same peptides.

![Picture1](https://github.com/user-attachments/assets/603c8525-0569-4dab-8d5f-f76feebd4a7f)
