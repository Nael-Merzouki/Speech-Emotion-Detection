# Speech-Emotion-Detection

This is my working repository for exploring automatic emotion recognition from speech. The goal is to build, compare and analyze several machine learning and deep learning pipelines for classifying emotions from short speech recordings (features: waveforms, spectrograms, MFCCs). The project contains data pre-processing, augmentation, feature extraction, model definitions and evaluation code, plus saved results and diagnostic plots.

What I'm trying to do
- Build stable, reproducible baseline models for speech emotion recognition (SER).
- Compare classical and deep learning approaches (MLP, CNN, LSTM, CNN+LSTM, an attention-based model).
- Study the effect of preprocessing choices (normalization), feature types (spectrograms, MFCCs), and data augmentation on performance.
- Analyze typical confusion patterns between emotion classes and identify where models fail (to guide improvements).

Research questions
- Which model architectures are best suited to this dataset and feature set (frame-based MFCCs and spectrogram inputs)?
- How much do augmentation and class-balancing strategies help on under-represented emotions?
- Are temporal models (LSTM / CNN+LSTM / attention) better than purely convolutional or dense models for this task?
- Which emotion pairs are frequently confused, and what does that tell us about the representation or dataset?

Project contents (important files / directories)
- src/
  - train.py — main experiment / training driver
  - models.py — definitions for MLP, CNN, LSTM, CNN+LSTM and attention variants
  - preprocessing.py — waveform handling, trimming/padding, basic transforms
  - feature_extraction.py — MFCC and spectrogram extraction utilities
  - augmentation.py — audio augmentations (time shift, noise, pitch/tempo, etc.) and visualization helpers
  - class_balancing.py — utilities for sampling / balancing classes
  - data_loader.py — dataset and batch generator
  - evaluation.py — evaluation metrics, confusion matrix generation, plotting helpers
  - hyperparameter_tuning.py — script for automated parameter search
  - config.py — experiment configuration defaults
- results/
  - training histories (pickle): mlp_history.pkl, cnn_history.pkl, lstm_history.pkl, cnn_lstm_history.pkl, attention_history.pkl
  - predictions CSVs for each model (e.g. results/mlp_predictions.csv, results/cnn_predictions.csv, ...)
  - plots/ — diagnostic images (training histories, confusion matrices, augmentation visualizations and comparisons)

What I have done so far (summary of experiments and findings)
- Implemented a reproducible pipeline from raw waveforms to model evaluation (preprocessing → features → model → metrics).
- Trained and evaluated several baseline models saved under results/*. For each model I exported per-sample predictions (results/*_predictions.csv), training histories (results/*_history.pkl) and visual diagnostics (results/plots/*.png).
- Visualized how augmentation affects inputs (waveforms, spectrograms, MFCCs) and compared emotion-level performance with and without augmentation.

Key observations from the saved results
- Strong baselines: The convolutional (CNN) and the MLP models produce many correct classifications on the test set (their prediction CSVs show a large majority of correct rows). The CNN in particular shows consistent, strong per-sample accuracy across the test set (see results/cnn_predictions.csv and results/plots/cnn_confusion_matrix.png).
- Temporal models (LSTM, CNN+LSTM) are more brittle on this dataset: the LSTM-only model has many more misclassifications in the test predictions (results/lstm_predictions.csv) than the CNN/MLP baselines. The CNN+LSTM model improves on LSTM alone in some cases but still shows more confusion than the CNN in several emotion pairs (results/cnn_lstm_predictions.csv and results/plots/cnn_lstm_confusion_matrix.png).
- Attention model: the attention-based architecture (results/attention_predictions.csv and attention_confusion_matrix.png) currently performs worse than the CNN/MLP baselines on the same test set. Its predictions show a high rate of misclassifications and a tendency to predict a small subset of classes (this suggests the attention mechanism/hyperparameters need tuning or the input representation may not be ideal for the attention block used).
- Confusion patterns: across confusion matrices (results/plots/*_confusion_matrix.png) and raw prediction files, common confusions include:
  - calm <-> neutral <-> surprised (these classes often overlap acoustically)
  - happy <-> surprised (tempo/energy similarities)
  - anger <-> disgust (some acoustic overlap in aggressive voice quality)
  These patterns help prioritize where data augmentation, more discriminative features or targeted class-balanced sampling might help the most.
- Preprocessing & normalization: experiments comparing normalization strategies and feature representations are saved (results/plots/normalization_comparison.png, model_comparison.png and normalization plots). These indicate normalization and the choice of feature (MFCCs vs spectrograms) significantly affect performance; visualizations in results/plots/augmentation_mfccs.png and augmentation_spectrograms.png show how augmentations impact feature visuals.
- Augmentation: augmentation visualizations (results/plots/augmentation_waveforms.png, random_augmentations.png) and the emotion-level augmentation comparison (results/plots/emotion_augmentation_comparison.png) demonstrate that augmentation can increase intra-class variance and help reduce overfitting, but its effect on accuracy varies by emotion—augmentations help some classes more than others.

How the results are stored (where to look)
- Per-sample predictions (CSV): results/{mlp,cnn,lstm,cnn_lstm,attention}_predictions.csv — each row: true_label, predicted_label, correct
- Training histories (pickle): results/*_history.pkl and corresponding plots in results/plots/*_training_history.png (these show loss/accuracy curves for train & validation)
- Confusion matrices and comparison charts: results/plots/*_confusion_matrix.png and results/plots/model_comparison.png
- Augmentation visualizations: results/plots/augmentation_*.png (waveforms, spectrograms, MFCCs)

Where the project stands (current phase)
- Experimental / prototyping phase: I have implemented the full pipeline and established reasonable baselines (MLP, CNN). I have run systematic experiments to compare architectures, augmentations and normalization strategies and saved the resulting predictions and diagnostic plots.
- Key outcomes to date:
  - CNN and MLP are strong baselines on the current dataset and feature set.
  - LSTM and attention architectures need further tuning and/or different input representations to outperform the simpler models.
  - Augmentation and class-balancing show promise but need to be tailored per-emotion to maximize benefit.
- Not finished: hyperparameter tuning at scale, cross-validation (to produce robust aggregate metrics), clearer quantitative summaries (aggregate accuracies, per-class recall/precision in a single report), and experimentation with stronger attention/transformer-style architectures and/or pre-trained audio embeddings (e.g., wav2vec or other self-supervised features).

Next steps (planned)
- Run systematic cross-validation and automated hyperparameter searches (scripts available in src/hyperparameter_tuning.py).
- Improve attention / temporal models (tune attention heads, input dimensionality, and training regimen) or try pre-trained representations (fine-tune wav2vec-like models).
- Refine augmentation: tune augmentation policies per emotion (guided by the emotion_augmentation_comparison results).
- Produce a consolidated evaluation report that includes per-class precision/recall/F1 and overall accuracy across folds.
- Clean up and document example commands for reproducing experiments and creating the figures in results/plots.

Reproducing and running experiments
1. Install dependencies:
   - pip install -r requirements.txt
2. Inspect config options:
   - src/config.py contains defaults for model, feature, augmentation and training hyperparameters.
3. Training / evaluation:
   - Use src/train.py to run an experiment (it is the project's training driver).
   - Evaluation utilities are in src/evaluation.py — they generate confusion matrices and save per-sample predictions to results/.
4. Visualizations:
   - Plotting scripts and utilities in src/evaluation.py and augmentation.py recreate the images in results/plots/.

Notes and pointers
- The repository is purposely modular: feature extraction, augmentation, models, and evaluation are separated so you can easily swap features or models.
- The raw plots and CSVs in results/ are the best starting point to see concrete, per-sample outcomes and how different models behave.
- If you want to extend the project, I recommend either (a) trying pre-trained audio embeddings for feature extraction, or (b) focusing on model/augmentation combinations for the specific confusion pairs identified (calm/neutral/surprised and anger/disgust).

Thanks for checking out the project — I'm actively iterating on these models and welcome suggestions, issues or contributions.
