from my_imports_configs import *
import uci_dataset as datasets


def make_X_train(X_tra, y_tra, max_sample):
    if max_sample <= 1:
        return X_tra, y_tra
    elif 1 < max_sample < 2:
        return np.tile(X_tra, (2, 1)), np.tile(y_tra, 2)
    else:
        return np.tile(X_tra, (max_sample, 1)), np.tile(y_tra, max_sample)


def load_dataset(dataset_name):
    if dataset_name == "wine":
        return load_wine()
    if dataset_name == "cancer":
        return load_breast_cancer()
    if dataset_name == "digits":
        return load_digits()
    if dataset_name == "iris":
        return load_iris()
    if dataset_name == "parkinson":
        return datasets.load_parkinson()
    if dataset_name == "abalone":
        return datasets.load_abalone()
    if dataset_name == "arrhythmia":
        return datasets.load_arrhythmia()
    if dataset_name == "audiology":
        return datasets.load_audiology()
    if dataset_name == "diabetes":
        return pd.read_csv("diabetes.csv")
    if dataset_name == "adult":
        return pd.read_csv("adult.csv")
    if dataset_name == "australian":
        return fetch_ucirepo(id=143)
    if dataset_name == "balance":
        return fetch_ucirepo(id=12)
    if dataset_name == "breast_cancer":
        return fetch_ucirepo(id=15)
    if dataset_name == "echo":
        return pd.read_csv("echo.csv")
    if dataset_name == "ecoli":
        return fetch_ucirepo(id=39)
    if dataset_name == "german":
        return fetch_ucirepo(id=144)
    if dataset_name == "glass":
        return fetch_ucirepo(id=42)
    if dataset_name == "heart":
        return fetch_ucirepo(id=145)
    if dataset_name == "hepatitis":
        return fetch_ucirepo(id=46)
    if dataset_name == "horse_colic":
        return fetch_ucirepo(id=47)
    if dataset_name == "ionosphere":
        return fetch_ucirepo(id=52)
    if dataset_name == "labor":
        return pd.read_csv("labor.csv", na_values=['?'])
    if dataset_name == "liver":
        return fetch_ucirepo(id=60)
    if dataset_name == "thyroid":
        return pd.read_csv("new-thyroid.csv", names=["Class", "T3", "Thyroxin", "Triiodothyronine", "TSH", "TSH_diff"])
    if dataset_name == "segmentation":
        return fetch_ucirepo(id=147)
    if dataset_name == "sonar":
        return fetch_ucirepo(id=151)
    if dataset_name == "soybean":
        return pd.read_csv("soybean.csv", na_values=['?'])
    if dataset_name == "tic-tac-toe":
        return fetch_ucirepo(id=101)
    if dataset_name == "vehicle":
        return fetch_ucirepo(id=149)
    if dataset_name == "voting":
        return fetch_ucirepo(id=105)
    if dataset_name == "vowel":
        return pd.read_csv("vowel.csv", delimiter="\s+", header=None)
    if dataset_name == "twonorm":
        return generate_twonorm_dataset()
    if dataset_name == "threenorm":
        return generate_threenorm_dataset()
    if dataset_name == "ringnorm":
        return generate_ringnorm_dataset()
    if dataset_name == "waveform":
        return fetch_ucirepo(id=107)
    if dataset_name == "led24":
        return pd.read_csv("led24.csv")


def get_dataset_X_y(ds_name):
    dataset = load_dataset(ds_name)
    print("----- " + ds_name + " -----")

    if ds_name == "parkinson":
        return dataset.drop(columns=["name", "status"]).to_numpy(), dataset["status"].to_numpy()
    elif ds_name == "diabetes":
        return dataset.drop(columns='Outcome').to_numpy(), dataset['Outcome'].to_numpy()
    elif ds_name == "adult":
        dataset.drop(["education"], inplace=True, axis=1)
        dataset = pd.get_dummies(dataset, columns=["workclass", "marital-status", "occupation", "relationship", "race", "native-country"])
        dataset = pd.get_dummies(dataset, columns=["gender", "income"], drop_first=True)
        return dataset.drop(columns='income_>50K').to_numpy(), dataset['income_>50K'].to_numpy()
    elif ds_name == "abalone":
        dataset = pd.get_dummies(dataset, columns=["Sex"])
        dataset = dataset[~dataset["Rings"].isin([1, 2, 25, 26, 29])]   # removed, only one observation per class
        return dataset.drop(columns='Rings').to_numpy(), dataset['Rings'].to_numpy()
    elif ds_name == "arrhythmia":
        dataset.drop(["J"], inplace=True, axis=1)
        dataset.drop(["S'_Wave", 'AVL04', 'AVL06', 'AVF08', 'V408', 'V409', 'V504', 'V506', 'V508', 'V510', 'V604', 'V609', 'V610', 'S_Prime_Wave', 'AVL205', 'V5265', 'V6275'], inplace=True, axis=1)  # removed, only one value in column
        dataset.dropna(inplace=True)
        dataset.drop(["Rag_R_Nom", "AVL08", "AVF09"], inplace=True, axis=1)     # removed, only one value in column
        return dataset.drop(columns='diagnosis').to_numpy(), dataset['diagnosis'].to_numpy()
    elif ds_name == "audiology":
        dataset.fillna("missing", inplace=True)
        dataset = dataset[~dataset["Class"].isin(['mixed_cochlear_age_fixation', 'acoustic_neuroma', 'bells_palsy', 'cochlear_age_plus_poss_menieres', 'mixed_poss_central_om', 'poss_central'])]   # removed, only one observation per class
        dataset.drop(["indentifier"], inplace=True, axis=1)     # id removed
        dataset.drop(["late_wave_poor", "m_cond_lt_1k", "m_p_sn_gt_2k", "m_s_sn_gt_1k", "m_sn_gt_6k", "middle_wave_poor",
                 "mod_mixed", "mod_s_mixed", "mod_sn", "viith_nerve_signs"], inplace=True, axis=1)  # removed, only one value in column
        binary_columns = ["age_gt_60", "airBoneGap", "boneAbnormal", "bser", "history_buzzing", "history_dizziness",
                          "history_fluctuating", "history_fullness", "history_heredity", "history_nausea", "history_noise",
                          "history_recruitment", "history_ringing", "history_roaring", "history_vomiting", "m_at_2k", "m_gt_1k",
                          "m_m_gt_2k", "m_m_sn", "m_m_sn_gt_1k", "m_m_sn_gt_2k", "m_m_sn_gt_500", "m_s_gt_500", "m_s_sn",
                          "m_s_sn_gt_2k", "m_s_sn_gt_3k", "m_s_sn_gt_4k", "m_sn_2_3k", "m_sn_gt_1k", "m_sn_gt_2k", "m_sn_gt_3k",
                          "m_sn_gt_4k", "m_sn_gt_500", "m_sn_lt_1k", "m_sn_lt_2k", "m_sn_lt_3k", "mod_gt_4k", "mod_s_sn_gt_500",
                          "mod_sn_gt_1k", "mod_sn_gt_2k", "mod_sn_gt_3k", "mod_sn_gt_4k", "mod_sn_gt_500", "notch_4k",
                          "notch_at_4k", "s_sn_gt_1k", "s_sn_gt_2k", "s_sn_gt_4k", "static_normal", "wave_V_delayed",
                          "waveform_ItoV_prolonged"]
        categorical_columns = ["air", "ar_c", "ar_u", "bone", "o_ar_c", "o_ar_u", "speech", "tymp"]
        dataset = pd.get_dummies(dataset, columns=categorical_columns)
        dataset = pd.get_dummies(dataset, columns=binary_columns, drop_first=True)
        return dataset.drop(columns='Class').to_numpy(), dataset['Class'].to_numpy()
    elif ds_name == "digits":
        return np.delete(dataset.data, [0, 32, 39], axis=1), dataset.target     # removal of columns with sole zeros
    elif ds_name == "australian":
        X = dataset.data.features
        y = dataset.data.targets
        X = pd.get_dummies(X, columns=["A4", "A5", "A6", "A12"])
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "balance":
        X = dataset.data.features
        y = dataset.data.targets
        X = pd.get_dummies(X, columns=["right-distance", "right-weight", "left-distance", "left-weight"])
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "breast_cancer":
        X = dataset.data.features
        y = dataset.data.targets
        X["temp_target"] = y
        X.dropna(inplace=True)
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "echo":
        dataset.drop(["survival", "still-alive", "mult", "name", "group"], inplace=True, axis=1)
        dataset.dropna(subset=['alive-at-1'], inplace=True)
        dataset.dropna(inplace=True)
        return dataset.drop(columns='alive-at-1').to_numpy(), dataset['alive-at-1'].to_numpy()
    elif ds_name == "ecoli":
        X = dataset.data.features
        y = dataset.data.targets
        X.drop(["chg"], inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "german":
        X = dataset.data.features
        y = dataset.data.targets
        binary_columns = ["Attribute19", "Attribute20"]
        categorical_columns = ["Attribute1", "Attribute3", "Attribute4", "Attribute6", "Attribute7", "Attribute9",
                               "Attribute10", "Attribute12", "Attribute14", "Attribute15", "Attribute17"]
        X = pd.get_dummies(X, columns=categorical_columns)
        X = pd.get_dummies(X, columns=binary_columns, drop_first=True)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "glass":
        X = dataset.data.features
        y = dataset.data.targets
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "heart":
        X = dataset.data.features
        y = dataset.data.targets
        X = pd.get_dummies(X, columns=["chest-pain", "electrocardiographic", "thal"])
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "hepatitis":
        X = dataset.data.features
        y = dataset.data.targets
        X["temp_target"] = y
        X.dropna(subset=["Bilirubin", "Sgot"], inplace=True)
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)

        categorical_cols_with_na = ["Steroid", "Liver Big", "Liver Firm", "Spleen Palpable", "Spiders", "Ascites", "Varices"]
        binary_columns = ["Sex", "Antivirals", "Fatigue", "Malaise", "Anorexia", "Histology"]
        X[categorical_cols_with_na] = X[categorical_cols_with_na].fillna("missing", inplace=False)
        X = pd.get_dummies(X, columns=categorical_cols_with_na)
        X = pd.get_dummies(X, columns=binary_columns, drop_first=True)
        X[["Alk Phosphate", "Albumin", "Protime"]] = X[["Alk Phosphate", "Albumin", "Protime"]].fillna(X[["Alk Phosphate", "Albumin", "Protime"]].mean())
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "horse_colic":
        X = dataset.data.features
        y = dataset.data.targets

        X.drop(["hospital_number", "cp_data"], inplace=True, axis=1)
        binary_columns = ["age", "lesion_subtype"]
        categorical_cols_with_na = ["surgery", "temperature_of_extremities", "peripheral_pulse", "mucous_membranes", "capillary_refill_time", "pain", "peristalsis",
                    "abdominal_distension", "nasogastric_tube", "nasogastric_reflux", "rectal_examination_feces", "abdomen", "abdominocentesis_appearance", "outcome"]
        categorical_cols_no_na = ["lesion_site", "lesion_type"]
        continuous_with_na = ["rectal_temperature", "pulse", "respiratory_rate", "packed_cell_volume", "total_protein", "nasogastric_reflux_ph", "abdominocentesis_total_protein"]
        X[categorical_cols_with_na] = X[categorical_cols_with_na].fillna("missing", inplace=False)
        X = pd.get_dummies(X, columns=categorical_cols_with_na + categorical_cols_no_na)
        X = pd.get_dummies(X, columns=binary_columns, drop_first=True)
        X[continuous_with_na] = X[continuous_with_na].fillna(X[continuous_with_na].mean())
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "ionosphere":
        X = dataset.data.features
        y = dataset.data.targets
        X.drop(["Attribute2"], inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "labor":
        categorical_cols_with_na = ["cost-of-living-adjustment", "pension", "education-allowance", "vacation", "longterm-disability-assistance",
                                    "contribution-to-dental-plan", "bereavement-assistance", "contribution-to-health-plan"]
        continuous_with_na = ["duration", "wage-increase-first-year", "wage-increase-second-year", "wage-increase-third-year",
                              "working-hours", "standby-pay", "shift-differential", "statutory-holidays"]
        dataset[categorical_cols_with_na] = dataset[categorical_cols_with_na].fillna("missing", inplace=False)
        dataset = pd.get_dummies(dataset, columns=categorical_cols_with_na)
        dataset[continuous_with_na] = dataset[continuous_with_na].fillna(dataset[continuous_with_na].mean())
        return dataset.drop(columns='class').to_numpy(), dataset['class'].to_numpy()
    elif ds_name == "liver":
        X = dataset.data.features
        y = dataset.data.targets
        X["temp_target"] = y
        X['temp_target'] = (X['temp_target'] >= 3).astype(int)  # binarization based on the criterion provided in the reference paper
        X.drop_duplicates(inplace=True)
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "thyroid":
        return dataset.drop(columns='Class').to_numpy(), dataset['Class'].to_numpy()
    elif ds_name == "segmentation":
        X = dataset.data.features
        y = dataset.data.targets
        X["temp_target"] = y
        X.drop_duplicates(inplace=True)
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)
        X.drop("region-pixel-count", inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "sonar":
        X = dataset.data.features   # no duplicates, checked
        y = dataset.data.targets
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "soybean":
        dataset.drop_duplicates(inplace=True)
        categorical_cols_with_na = list(dataset.columns)
        categorical_cols_with_na.remove("class")
        categorical_cols_with_na.remove("leaves")
        dataset[categorical_cols_with_na] = dataset[categorical_cols_with_na].fillna("missing", inplace=False)
        dataset = pd.get_dummies(dataset, columns=categorical_cols_with_na)
        return dataset.drop(columns='class').to_numpy(), dataset['class'].to_numpy()
    elif ds_name == "tic-tac-toe":
        X = dataset.data.features   # no duplicates, checked
        y = dataset.data.targets
        X = pd.get_dummies(X)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "vehicle":
        X = dataset.data.features  # no duplicates, checked
        y = dataset.data.targets
        X["temp_target"] = y
        X.dropna(inplace=True)
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "voting":
        X = dataset.data.features
        y = dataset.data.targets
        X["temp_target"] = y
        X.drop_duplicates(inplace=True)
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)
        X.fillna("missing", inplace=True)
        X = pd.get_dummies(X)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "vowel":    # UCI name: Connectionist Bench (Vowel Recognition - Deterding Data)
        dataset.drop([0, 1, 2], inplace=True, axis=1)   # no duplicates, checked
        return dataset.drop(columns=13).to_numpy(), dataset[13].to_numpy()
    elif ds_name == "twonorm":
        return dataset.drop(columns='label').to_numpy(), dataset['label'].to_numpy()
    elif ds_name == "threenorm":
        return dataset.drop(columns='label').to_numpy(), dataset['label'].to_numpy()
    elif ds_name == "ringnorm":
        return dataset.drop(columns='label').to_numpy(), dataset['label'].to_numpy()
    elif ds_name == "waveform": # no duplicates, checked
        X = dataset.data.features
        y = dataset.data.targets
        X["temp_target"] = y
        np.random.seed(123)
        X_0 = X[X["temp_target"] == 0].sample(n=100)
        X_1 = X[X["temp_target"] == 1].sample(n=100)
        X_2 = X[X["temp_target"] == 2].sample(n=100)
        X = pd.concat([X_0, X_1, X_2])
        y = X["temp_target"]
        X.drop("temp_target", inplace=True, axis=1)
        return X.to_numpy(), y.to_numpy().ravel()
    elif ds_name == "led24":
        dataset.drop_duplicates(inplace=True)
        np.random.seed(123)
        sampled_ds = pd.concat([dataset[dataset["class"] == i].sample(n=20) for i in range(10)])
        return sampled_ds.drop(columns="class").to_numpy(), sampled_ds["class"].to_numpy().ravel()
    else:   # wine, cancer, iris
        return dataset.data, dataset.target

    X.info()
    print(X.describe())
    for c in X.columns:
        print(c, "\t", Counter(X[c]))
    print(Counter(y.to_numpy().ravel()))


def get_X_tr_y_tr_X_te_y_te(X, y, train, test):
    X_train, X_test = X[train, :], X[test, :]
    y_train, y_test = y[train], y[test]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, y_train, X_test, y_test


def save_plot_for_dataset(ds_name, clfs, results_per_ds):
    tab_20_colors = plt.get_cmap('tab20')
    results_per_clf = {clf[0]: [] for clf in clfs}
    for ms, ms_results in results_per_ds.items():
        for clf_name, clf_results in ms_results.items():
            results_per_clf[clf_name].append(np.mean(clf_results))
    for i, clf in enumerate(clfs):
        plt.plot(list(results_per_ds.keys()), results_per_clf[clf[0]], c=tab_20_colors(2 * i if i < 10 else 2 * (i - 10) + 1), linewidth=.8, marker="o", markersize=1)
    plt.xlim(0, list(results_per_ds.keys())[-1] + 0.1)
    plt.xticks([0] + list(results_per_ds.keys()))
    plt.grid(linewidth=0.2)
    plt.legend([clf[0].strip() for clf in clfs])
    plt.title(ds_name)
    plt.xlabel("max samples")
    plt.ylabel("accuracy")
    plt.savefig("figs//" + ds_name + "_" + str(cv_n_repeats) + "_" + str(list(results_per_ds.keys())[0]) + "-" + str(list(results_per_ds.keys())[-1]) + ".pdf")
    plt.clf()


def save_to_pickle(dict_to_save, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(dict_to_save, file)


def generate_twonorm_dataset(n_samples_per_class=150, d=20):
    np.random.seed(123)
    a = 2 / np.sqrt(d)
    cov_matrix = np.identity(d)

    samples_class_0 = np.random.multivariate_normal(np.full(d, a), cov_matrix, n_samples_per_class)
    labels_class_0 = np.zeros(n_samples_per_class)
    samples_class_1 = np.random.multivariate_normal(np.full(d, -a), cov_matrix, n_samples_per_class)
    labels_class_1 = np.ones(n_samples_per_class)
    samples = np.vstack((samples_class_0, samples_class_1))
    labels = np.hstack((labels_class_0, labels_class_1))

    columns = [f'feature_{i + 1}' for i in range(samples.shape[1])]
    df = pd.DataFrame(samples, columns=columns)
    df['label'] = labels
    return df


def generate_threenorm_dataset(n_samples_per_class=150, d=20):
    np.random.seed(123)
    a = 2 / np.sqrt(d)
    cov_matrix = np.identity(d)

    samples_class_0_a = np.random.multivariate_normal(np.full(d, a), cov_matrix, n_samples_per_class // 2)
    samples_class_0_b = np.random.multivariate_normal(np.full(d, -a), cov_matrix, n_samples_per_class // 2)
    samples_class_0 = np.vstack((samples_class_0_a, samples_class_0_b))
    labels_class_0 = np.zeros(n_samples_per_class)
    samples_class_1 = np.random.multivariate_normal([a if i % 2 == 0 else -a for i in range(d)], cov_matrix, n_samples_per_class)
    labels_class_1 = np.ones(n_samples_per_class)
    samples = np.vstack((samples_class_0, samples_class_1))
    labels = np.hstack((labels_class_0, labels_class_1))

    columns = [f'feature_{i + 1}' for i in range(samples.shape[1])]
    df = pd.DataFrame(samples, columns=columns)
    df['label'] = labels
    return df


def generate_ringnorm_dataset(n_samples_per_class=150, d=20):
    np.random.seed(123)
    a = 1 / np.sqrt(d)
    cov_matrix = np.identity(d)

    samples_class_0 = np.random.multivariate_normal(np.zeros(d), 4 * cov_matrix, n_samples_per_class)
    labels_class_0 = np.zeros(n_samples_per_class)
    samples_class_1 = np.random.multivariate_normal(np.full(d, a), cov_matrix, n_samples_per_class)
    labels_class_1 = np.ones(n_samples_per_class)
    samples = np.vstack((samples_class_0, samples_class_1))
    labels = np.hstack((labels_class_0, labels_class_1))

    columns = [f'feature_{i + 1}' for i in range(samples.shape[1])]
    df = pd.DataFrame(samples, columns=columns)
    df['label'] = labels
    return df




########################################### MODELS ##########################################
cpu_no = 12
rf_100 =    ('        RF', RandomForestClassifier(random_state=123, n_jobs=cpu_no, n_estimators=100))
rf_200 =    ('   RF[200]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, n_estimators=200))
rf_500 =    ('   RF[500]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, n_estimators=500))
rf_entr =   ('  RF[entr]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, criterion="entropy"))
rf_md_10 =  ('  RF[md10]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, max_depth=10))
rf_md_15 =  ('  RF[md15]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, max_depth=15))
rf_md_20 =  ('  RF[md20]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, max_depth=20))
rf_md_25 =  ('  RF[md25]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, max_depth=25))
rf_mss_3 =  ('  RF[mss3]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_split=3))
rf_mss_4 =  ('  RF[mss4]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_split=4))
rf_mss_6 =  ('  RF[mss6]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_split=6))
rf_mss_8 =  ('  RF[mss8]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_split=8))
rf_msl_2 =  ('  RF[msl2]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_leaf=2))
rf_msl_3 =  ('  RF[msl3]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_leaf=3))
rf_msl_4 =  ('  RF[msl4]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_leaf=4))
rf_msl_5 =  ('  RF[msl5]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, min_samples_leaf=5))
rf_mf_log = ('RF[mfLog2]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, max_features="log2"))
rf_mf_all = (' RF[mfAll]', RandomForestClassifier(random_state=123, n_jobs=cpu_no, max_features=None))
######################################## END OF MODELS ######################################

######################################### PARAMETERS ########################################
cv_n_splits = 2
cv_n_repeats = 200
max_samples = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 2, 3, 4, 5]
dataset_names = ["arrhythmia", "audiology", "parkinson", "wine", "cancer", "digits", "iris", "diabetes", "abalone", "adult",
                 "australian", "balance", "breast_cancer", "echo", "ecoli", "german", "glass", "heart", "hepatitis", "horse_colic",
                 "ionosphere", "labor", "liver", "thyroid", "segmentation", "sonar", "soybean", "tic-tac-toe", "vehicle", "voting",
                 "vowel", "twonorm", "threenorm", "ringnorm", "waveform", "led24"]
clfs = [rf_100, rf_200, rf_500, rf_entr, rf_md_10, rf_md_15, rf_md_20, rf_md_25, rf_mss_3, rf_mss_4, rf_mss_6, rf_mss_8,
        rf_msl_2, rf_msl_3, rf_msl_4, rf_msl_5, rf_mf_log, rf_mf_all]

##################################### END OF PARAMETERS #####################################

############################################ MAIN ###########################################
dataset_ratios = {}
all_results = {}
processed_dataset_list = ""

print("\t\t\t  ", end="")
for clf in clfs:
    print(clf[0], end="\t\t\t\t")
print()

for ds_name in dataset_names:
    results_per_dataset = {}
    X, y = get_dataset_X_y(ds_name)

    # duplicates removal
    unique_strings, encoded_integers = np.unique(y, return_inverse=True)
    text_to_int = {string: i for i, string in enumerate(unique_strings)}
    encoded_column = np.vectorize(text_to_int.get)(y)
    X_y = np.hstack([X, encoded_column.reshape(-1, 1)])
    X_y_unique = np.unique(X_y, axis=0)
    X = X_y_unique[:, :-1]
    y = X_y_unique[:, -1]
    # end of duplicates removal

    dataset_ratios[ds_name] = X.shape[0] * (cv_n_splits - 1) / cv_n_splits / X.shape[1]

    for ms in max_samples:
        results_per_ms = {clf[0]: [] for clf in clfs}

        kfold = RepeatedStratifiedKFold(n_splits=cv_n_splits, n_repeats=cv_n_repeats, random_state=123)
        for train, test in kfold.split(X, y):
            for clf in clfs:
                X_tr, y_tr, X_te, y_te = get_X_tr_y_tr_X_te_y_te(X, y, train, test)
                X_tr, y_tr = make_X_train(X_tr, y_tr, ms)
                clf[1].max_samples = None if ms >= 1 and ms != 1.2 else (0.6 if ms == 1.2 else ms)
                clf[1].fit(X_tr, y_tr)
                y_pred = clf[1].predict(X_te)
                results_per_ms[clf[0]].append(accuracy_score(y_te, y_pred))
        print("  {:.1f}:".format(ms), end="\t")
        for clf in clfs:
            print("{0:0.3f} +/- {1:0.3f}\t".format(100 * np.mean(results_per_ms[clf[0]]), 100 * np.std(results_per_ms[clf[0]])), end="\t")
        print()
        results_per_dataset[ms] = results_per_ms
        save_to_pickle(results_per_dataset, "pickles//" + ds_name + "_" + str(cv_n_repeats) + "_" + str(ms) + ".pkl")
    all_results[ds_name] = results_per_dataset
    processed_dataset_list += ("_" + ds_name)
    save_to_pickle(all_results, "pickles//ALL_" + str(cv_n_repeats) + processed_dataset_list + ".pkl")
    save_plot_for_dataset(ds_name, clfs, results_per_dataset)
