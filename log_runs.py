import mlflow, random

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("king")

for i in range(5):
    with mlflow.start_run(run_name=f"upi_shield_v{i+1}"):
        mlflow.log_param("model_type", ["Transformer","GNN","Ensemble","PPO","Baseline"][i])
        mlflow.log_param("threshold_block", round(0.65+i*0.02, 2))
        mlflow.log_param("threshold_review", round(0.30+i*0.01, 2))
        mlflow.log_metric("fraud_recall",   round(0.91+random.uniform(0,.06), 4))
        mlflow.log_metric("false_pos_rate", round(0.003+random.uniform(0,.003), 4))
        mlflow.log_metric("f1_score",       round(0.88+random.uniform(0,.06), 4))
        mlflow.log_metric("latency_ms",     round(1.2+random.uniform(0,1.5), 2))
        print(f"Logged run {i+1}")
