# Surrogate Modeling Summary

Rows used: 3000
Features used (5): voltage, temperature, gate_length_nm, oxide_thickness_nm, doping_cm3

## Model Ranking (by MAE)
model,rmse,mae,r2,train_time_s,inference_total_s_test,inference_per_sample_s
linear_regression,0.031089365523509754,0.024582707081760972,0.997542575158661,0.0030055129900574684,0.0009019979916047305,1.5033299860078842e-06
mlp,0.04547580978106598,0.035863219770889646,0.9947420384176501,0.1941360060009174,0.0013610869937110692,2.268478322851782e-06
random_forest,0.05633237074743731,0.04444186802523921,0.9919318756355482,1.0821351219929056,0.01995873400301207,3.326455667168678e-05


Best model: **linear_regression**
Estimated speedup (best model): **47054394.01x**

## Where model may break down
Use `breakdown_mae_by_feature.csv` and corresponding plots to identify parameter ranges with elevated MAE.