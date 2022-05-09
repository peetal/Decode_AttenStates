## Decode Attention States Using Whole-brain Functional Connectivity data
### About the repo
This repo stores all data and scripts from this project (as debried below). The first part of the project results in the preprint "whole-brain background functional connectivity patterns capture differences between perception and retrieval states", which is currently under preparation. All data and scripts for reproducing the main results shown in the manuscripts can be found in this repo. The second part of the project is currently in progress and the codes will be consistently updates. 
### Dependencies
- BrainIAK Conda enviroment (https://brainiak.org/): The first part of the project relies on the recently developed human fMRI data analysis method referred to as the full correaltion matrix analysis (FCMA; https://ntblab.yale.edu/wp-content/uploads/2015/06/Wang_JNM_2015.pdf), wrapped in BrainIAK. 
- Realtime fMRI Cloud Framework (https://github.com/brainiak/rt-cloud): The neurofeedback part of the project relies on the rtcloud framework provided by BrainIAK. 
---------------------------------------------------------------------------------------------------------------------------------------------------------
### Background: Psychological phenomenon
The same external stimuli can be both the target of perception and the trigger of episodic memory retrieval, depending on where attention points to. That is, if attention was deployed externally, the stimuli would lead to perception; if attention was deployed internally, the stimuli would lead to episodic memory retrieval.
<img width="720" alt="Screen Shot 2022-05-09 at 2 17 58 PM" src="https://user-images.githubusercontent.com/63365201/167500194-6170e4ba-792d-4e38-93bb-7bf7b1c1a0c1.png">
### Goal of the current project
- Identify the neural mechanism that deploys attention externally vs. internally.
- Using real-time neural feedback to allow for flexible attentional switch (this may help with rumination in depression, allowing patients to NOT focus their attention on depressive symptoms and the implications of these symptoms).
### Experimental design
- Experiment 1

The first experiment aims to examine the neural configuration differences between external and internal attention. Participants first learnt a set of associations between a face (male or female) and a scene (natural or man-made) image. In the external attention condition, participants were asked to make male/female decision for faces and natural/man-made decisions for scenes. In the internal attention condition, participants were asked to retrieve the cue-associated image and make male/female or natural/man-made decisions on the retrieved image. Neural activities were measured using fMRI during the two conditions and a model was trained to classify the ongoing attentional state based on neural signals.
- Experiment 2 (in progress)

The second experiment aims to perform real-time neuro-feedback training. While in the scanner, participants were presented with an external stimuli. The current attentional states were being quantified using the pre-trained model, and the classification confidence was used as neuro-feedback. Participants were asked to either sustain or switch between attention states.

<img width="712" alt="Screen Shot 2022-05-09 at 2 19 15 PM" src="https://user-images.githubusercontent.com/63365201/167500426-f33cb309-b20b-4818-b423-cf08a28fda72.png">

