# Research charter

Question: can multi-market causal microstructure pretraining improve calibrated, execution-aware market making under realistic friction and inventory constraints?

Primary comparisons are P1 M3T-SSL vs M3T-SUP, P2 M3T-EPT-CAL vs M3T-SUP, P3 common-controller M3T-EPT-CAL vs XGBoost, and P4 M3T controller vs AS/GLFT. The inference unit is instrument × trading day. TEST and FINAL_LOCKBOX are unavailable to selection, calibration, SSL, and controller tuning.
