# BlechCTA
Scripts used for Christina's Ph.D. thesis on neural mechanisms underlying learned and non-learned gaping.
Order of operations:

Initial set-up:
1. Run each session recording through Pytau, Blech_EMG_Classifier, and BlechClust packages before running this package.
2. Run create_tau_dict.py # combines tau, spike trains, and num cps from cp model into dictionary
3. Run combine_classifier_files.py # puts all Blech EMG Classifier segments files into one dataframe
4. Run initialize_dataframe.py # Adds important metrics to dataframe

Behavior-only analyses:
1. Run extract_emg_from_transition.py # calculates frequency of each behavior across trials, plots, and saves artifacts for each session
2. Run extract_emg_from_transition_aggregate.py # plots frequency of each behavior across trials, averages across all sessions
