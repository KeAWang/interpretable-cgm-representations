# Hybrid VAE for interpretable type-2 diabetes subtyping in-the-wild

![](assets/embedding.png)

> **Interpretable Mechanistic Representations for Glycemic Control**\
> Ke Alexander Wang, Emily B. Fox \
> Published at the *Machine Learning for Healthcare Symposium (ML4H), 2023*\
> [[Proceedings](https://proceedings.mlr.press/v225/wang23a.html)][[arXiv version](https://arxiv.org/abs/2312.03344)][[Poster](./assets/poster.pdf)]

## About

Our Hybrid VAE is a new machine learning model that can embed post-meal glucose responses in an interpretable space, using a combination of unsupervised learning and hybrid modeling. Given post-meal CGM data and optional meal logs and demographics data, we can summarize their glycemic control in our physiologically-meaningful latent space.

### For clinicians and biomedical engineers
Our model for the first time allows us to characterize Type-2 diabetes heterogeneity through various traditional metrics (glucose appearance time, glucose effectiveness, insulin sensitivity, etc.), as defined by *Bergman's minimal model*, using **only CGM and meal log data and no diagnosis information**. Each point in our embedding space represents a meal eaten by an individual. In the figure above, red represents T2D and blue represents pre-diabetes.

We do the above using in-the-wild data; up until now this would have needed an oral glucose tolerance test (OGTT) and glucose modeling expertise for each **meal and individual**.

![](assets/comparisons.png)

### For machine learning researchers
Our model is an autoregressive variational autoencoder with a mechanistic differential equation decoder.
Our mechanistic decoder uses our longstanding prior knowledge about glucose dynamics, based on Bergman's minimal model. By constraining the latent space to the ODE parameter space, we get both interpretability **and** improved performance.

![](assets/model_figure.png)

## Citation

If you use this codebase, or otherwise found our work valuable, please cite Mamba:
```
@InProceedings{hybridvae,
  title = 	 {Interpretable Mechanistic Representations for Meal-level Glycemic Control in the Wild},
  author =       {Wang, Ke Alexander and Fox, Emily B.},
  booktitle = 	 {Proceedings of the 3rd Machine Learning for Health Symposium},
  pages = 	 {607--622},
  year = 	 {2023},
  editor = 	 {Hegselmann, Stefan and Parziale, Antonio and Shanmugam, Divya and Tang, Shengpu and Asiedu, Mercy Nyamewaa and Chang, Serina and Hartvigsen, Tom and Singh, Harvineet},
  volume = 	 {225},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10 Dec},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v225/wang23a/wang23a.pdf},
  url = 	 {https://proceedings.mlr.press/v225/wang23a.html},
  abstract = 	 {Diabetes encompasses a complex landscape of glycemic control that varies widely among individuals. However, current methods do not faithfully capture this variability at the meal level. On the one hand, expert-crafted features lack the flexibility of data-driven methods; on the other hand, learned representations tend to be uninterpretable which hampers clinical adoption. In this paper, we propose a hybrid variational autoencoder to learn interpretable representations of CGM and meal data. Our method grounds the latent space to the inputs of a mechanistic differential equation, producing embeddings that reflect physiological quantities, such as insulin sensitivity, glucose effectiveness, and basal glucose levels. Moreover, we introduce a novel method to infer the glucose appearance rate, making the mechanistic model robust to unreliable meal logs. On a dataset of CGM and self-reported meals from individuals with type-2 diabetes and pre-diabetes, our unsupervised representation discovers a separation between individuals proportional to their disease severity. Our embeddings produce clusters that are up to 4x better than naive, expert, black-box, and pure mechanistic features. Our method provides a nuanced, yet interpretable, embedding space to compare glycemic control within and across individuals, directly learnable from in-the-wild data.}
}
```
