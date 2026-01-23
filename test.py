
# 10 clients - Resnet18 - Todas as metricas menos a accuracia da ultima camada
import os

exp2_light = []

path_exp2_light = "/content/drive/MyDrive/results_tcc_2/result_lightnet_cifar/experiment_2"

for run_name in sorted(os.listdir(path_exp2_light)):
  run_path = os.path.join(path_exp2_light, run_name)
  run_dict = {
        "global": pd.read_csv(os.path.join(run_path, "metrics_global.csv")),
        "fid": pd.read_csv(os.path.join(run_path, "metrics_per_client_fid.csv")),
        "js": pd.read_csv(os.path.join(run_path, "metrics_per_client_js.csv")),
        "kl_divergence": pd.read_csv(os.path.join(run_path, "metrics_per_client_kl.csv")),
        "mmd": pd.read_csv(os.path.join(run_path, "metrics_per_client_mmd.csv")),
    }
  exp2_light.append(run_dict)


for run_dict in exp1_light:
    df_global = run_dict["global"]
    df_global = df_global[df_global["round"] != 0]
    run_dict["global"] = df_global.reset_index(drop=True)

ROUND_ALVO = 2

metricas = ["fid","js","kl_divergence","mmd"]
medias_por_run2_light = []

for i, run_dict in enumerate(exp2_light):
    medias_run = {}
    for metrica in metricas:
        df_metrica = run_dict[metrica]

        df_r2 = df_metrica[df_metrica["round"] == ROUND_ALVO]
        medias_run[metrica] = df_r2[metrica].mean()

    medias_por_run2_light.append(medias_run)

for i, run_dict in enumerate(exp2_light):
    df_global = run_dict["global"]

    # último round do global
    last_round = df_global["round"].max()

    global_acc = df_global.loc[
        df_global["round"] == last_round, "global_accuracy"
    ].iloc[0]

    # adiciona no dicionário de médias do run correspondente
    medias_por_run2_light[i]["global_accuracy"] = global_acc

medias_por_run_nomeadas2_light = {
    f"run_{i}": medias
    for i, medias in enumerate(medias_por_run2_light)
}

for run, medias in medias_por_run_nomeadas2_light.items():
    print(run, medias)


df_corr = pd.DataFrame(medias_por_run_nomeadas2_light).T
correlacoes_light2 = df_corr.corr(method="pearson")["global_accuracy"]



def corr_metric_globalacc(path:str, ):
    
    # Entrada o path da pasta 
    # Saida dicionário com os 5 dataframes em cada posição do vetor (cada run)
    experiment = []
    for run_name in sorted(os.listdir(path)):
        run_path = os.path.join(path,run_name)

        run_dict = {
            "global": pd.read_csv(os.path.join(run_path, "metrics_global.csv")),
            "fid": pd.read_csv(os.path.join(run_path, "metrics_per_client_fid.csv")),
            "js": pd.read_csv(os.path.join(run_path, "metrics_per_client_js.csv")),
            "kl_divergence": pd.read_csv(os.path.join(run_path, "metrics_per_client_kl.csv")),
            "mmd": pd.read_csv(os.path.join(run_path, "metrics_per_client_mmd.csv")),
        }

        experiment.append(run_dict)


    #dropamos o round 0 dos experimentos
    for run_dict in experiment:
        df_global = run_dict["global"]
        df_global = df_global[df_global["round"] != 0]
        run_dict["global"] = df_global.reset_index(drop=True)

    
    ROUND_ALVO = 2

    metricas = ["fid","js","kl_divergence","mmd"]
    medias_por_run = []

    # entrada vetor de dicts e saida vetor de dicts com as medias das metricas 
    for i, run_dict in enumerate(experiment):
        medias_run = {}
        for metrica in metricas:
            df_metrica = run_dict[metrica]

            df_r2 = df_metrica[df_metrica["round"] == ROUND_ALVO]
            medias_run[metrica] = df_r2[metrica].mean()

        medias_por_run.append(medias_run)

    for i, run_dict in enumerate(experiment):
        df_global = run_dict["global"]

        # último round do global
        last_round = df_global["round"].max()

        global_acc = df_global.loc[
            df_global["round"] == last_round, "global_accuracy"
        ].iloc[0]

        # adiciona no dicionário de médias do run correspondente
        medias_por_run[i]["global_accuracy"] = global_acc
        
    medias_por_run_nomeadas = {
    f"run_{i}": medias
    for i, medias in enumerate(medias_por_run)
    }

    for run, medias in medias_por_run_nomeadas.items():
        print(run, medias)

    df_corr = pd.DataFrame(medias_por_run_nomeadas).T
    correlacoes_pearson = df_corr.corr(method="pearson")["global_accuracy"]
    correlacoes_spearman = df_corr.corr(method="spearman")["global_accuracy"]

    df_result = pd.DataFrame({
    "pearson": correlacoes_pearson,
    "spearman": correlacoes_spearman
    })

    print(df_result)




