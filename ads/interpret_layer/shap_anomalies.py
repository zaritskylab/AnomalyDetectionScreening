import torch
from sklearn import preprocessing
import numpy as np
import pandas as pd
import shap
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
# from scripts.run_ad import test_autoencoder
from scripts.classify_moa import get_moa_dirname, remove_classes_with_few_moa_samples, remove_multi_label_moa
from scripts.calc_reproducibility import calc_percent_replicating
from utils.general import add_exp_suffix
from utils.global_variables import DS_INFO_DICT, TOP_MOAS_DICT, NEW_MOAS_DICT
from anomaly_pipeline import train_autoencoder, load_checkpoint
from data_layer.data_preprocess import pre_process, construct_dataloaders
from data_layer.data_utils import load_data
import logging

logging.getLogger('shap').setLevel(logging.WARNING) # turns off the "shap INFO" logs

def run_anomaly_shap(configs, model=None, filter_non_reproducible=True):

    data , __ = load_data(configs.general.base_dir,configs.general.dataset,configs.data.profile_type, modality=configs.data.modality)
    data_preprocess,features =  pre_process(data,configs)
    dataloaders = construct_dataloaders(data_preprocess,configs.model.batch_size,features)

    if model is None:
        model = load_checkpoint(configs.model.ckpt_dir)

    if model is None:
        model = train_autoencoder(configs, data, features)

    X_control =  data_preprocess[data_preprocess['Metadata_set'] == 'test_ctrl'].reset_index(drop=True)
    X_test = data_preprocess[data_preprocess['Metadata_set'] == 'test_treat']
    
    # if group_by_col =='moa_col':
    cpd_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality]['cpd_col']
    group_by = 'moa'
    # elif group_by == 'cpd':
    if not DS_INFO_DICT[configs.general.dataset]['has_moa']:
        group_by = 'cpd'
    group_by_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality][f'{group_by}_col']
    
    # set all moa cols with lowercase
    X_test[group_by_col] = X_test[group_by_col].str.lower()
    X_test_processed, _, _ = remove_classes_with_few_moa_samples(X_test, min_samples = 5)  # remove classes with less than 5 samples
    X_test_processed, _ = remove_multi_label_moa(X_test_processed)
    X_test_processed = X_test_processed.reset_index(drop=True)
    
    exp_suffix = add_exp_suffix(configs.data.profile_type,configs.eval.by_dose,configs.eval.normalize_by_all)
    corr_path = f'{configs.general.res_dir}reproducible_cpds{exp_suffix}.csv'

    if filter_non_reproducible:

        if not os.path.exists(corr_path):
            calc_percent_replicating(configs,data_reps=['ae_diff','baseline'])
            # sys.exit()
        repcorr_df = pd.read_csv(corr_path)

        if 'reproducible_acl'in repcorr_df.columns:
            reproduce_col = 'reproducible_acl'
        elif 'reproducible_cl' in repcorr_df.columns:
            reproduce_col = 'reproducible_cl'
        elif 'reproducible_raw' in repcorr_df.columns:
            reproduce_col = 'reproducible_raw'
        elif 'reproducible_anomaly' in repcorr_df.columns:
            reproduce_col = 'reproducible_anomaly'
        else:
            raise ValueError('reproduce_col not found in repcorr_df')
            
        if configs.general.dataset == 'TAORF':
            only_reproducible = list(repcorr_df[repcorr_df[reproduce_col] == True]['cpd'].str.lower())
        else:
            only_reproducible = list(repcorr_df[repcorr_df[reproduce_col] == True]['cpd'])
        # only_reproducible = list(repcorr_df[repcorr_df[reproduce_col] == True]['cpd'].str.lower())
        X_test_reproducible = X_test_processed[X_test_processed[cpd_col].isin(only_reproducible)]
        X_test = X_test_reproducible.reset_index(drop=True)
        filter_str = 'reproducible'
    else:
        filter_str = 'all'
    

    if DS_INFO_DICT[configs.general.dataset]['has_moa']:
        
        # if configs.general.dataset == 'LINCS':
        top_moas = TOP_MOAS_DICT[configs.general.dataset]
        # else:
            # top_moas = ['atpase inhibitor']
        X_test = X_test.dropna(subset=['Metadata_moa'])
        only_top_moa_data = X_test[X_test[group_by_col].isin(top_moas)]
        only_top_moa_data = only_top_moa_data.reset_index(drop=True)

        save_dir = os.path.join(configs.general.res_dir, f'shap_vis_{filter_str}_moa_top')
        os.makedirs(save_dir, exist_ok=True)
        features_to_run = ['Cells_AreaShape_FormFactor', 'Cells_Texture_DifferenceVariance_RNA_10_0','Cytoplasm_Granularity_4_AGP', 'Nuclei_Granularity_2_DNA',
       'Cells_Texture_Contrast_ER_10_0',
       'Cytoplasm_Texture_DifferenceVariance_RNA_10_0',
       'Cells_Texture_DifferenceEntropy_RNA_10_0',
       'Cytoplasm_Granularity_3_ER',
       'Cytoplasm_Texture_DifferenceVariance_ER_5_0',
       'Cells_Texture_DifferenceVariance_Mito_10_0',
       'Nuclei_Texture_DifferenceVariance_Mito_10_0',
       'Cytoplasm_RadialDistribution_MeanFrac_AGP_1of4']
        exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=3, max_features_per_sample=10, reconstruction_error_percent=0.8,features_to_run=features_to_run)
        
        
        # logging.getLogger('matplotlib').setLevel(logging.WARNING)
        all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                    x_explain=only_top_moa_data,
                                                                    autoencoder=model,
                                                                    return_shap_values=True,
                                                                    group_by=group_by_col,
                                                                    save_dir=save_dir)
        
        new_moas = NEW_MOAS_DICT[configs.general.dataset]
        only_new_moa_data = X_test[X_test[group_by_col].isin(new_moas)]
        only_new_moa_data = only_new_moa_data.reset_index(drop=True)
        save_dir = os.path.join(configs.general.res_dir, f'shap_vis_{filter_str}_moa_new')
        os.makedirs(save_dir, exist_ok=True)

        exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=2, max_features_per_sample=5)
        all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                    x_explain=only_new_moa_data,
                                                                    autoencoder=model,
                                                                    return_shap_values=True,
                                                                    group_by=group_by_col,
                                                                    save_dir=save_dir)
    
    # run shap by compounds
    group_by = 'cpd'
    group_by_col = DS_INFO_DICT[configs.general.dataset][configs.data.modality][f'{group_by}_col']
    save_dir = os.path.join(configs.general.res_dir, f'shap_vis_{filter_str}_cpd')
    os.makedirs(save_dir, exist_ok=True)

    # x_explain = only_top_moa_data.copy()
    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=3, max_features_per_sample=5)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=X_test,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                group_by=group_by_col,
                                                                save_dir=save_dir)
    
    # run shap by replicates
    save_dir = os.path.join(configs.general.res_dir, f'shap_vis_{filter_str}_replicates')
    os.makedirs(save_dir, exist_ok=True)

    # x_explain = only_top_moa_data.copy()
    exp_model = ExplainAnomaliesUsingSHAP(model,features=features, num_anomalies_to_explain=3, max_features_per_sample=5)
    all_sets_explaining_features = exp_model.explain_unsupervised_data(x_train=X_control, 
                                                                x_explain=X_test,
                                                                autoencoder=model,
                                                                return_shap_values=True,
                                                                # group_by=group_by_col,
                                                                save_dir=save_dir)
    





#### from https://github.com/ronniemi/explainAnomaliesUsingSHAP/blob/master/ExplainAnomaliesUsingSHAP.py 
class ExplainAnomaliesUsingSHAP:
    '''
    This class implements method described in 'Explaining Anomalies Detected by Autoencoders Using SHAP' to explain
    anomalies revealed by an unsupervised Autoencoder model using SHAP.
    '''

    autoencoder = None
    num_anomalies_to_explain = None
    reconstruction_error_percent = None
    shap_values_selection = None
    counter = None
    features=None
    max_features_per_sample = None
    features_to_run = None

    def __init__(self, model,num_anomalies_to_explain=100, reconstruction_error_percent=0.5, shap_values_selection='mean',features=None,max_features_per_sample = 4,features_to_run=None):
        """
        Args:
            num_anomalies_to_explain (int): number of top ranked anomalies (ranked by anomaly score that is the mse) to
                                            explain.
            reconstruction_error_percent (float): Number between 0 to 1- see explanation to this parameter in
                                                  'Explaining Anomalies Detected by Autoencoders Using SHAP' under
                                                  ReconstructionErrorPercent.
            shap_values_selection (str): One of the possible methods to choose explaining features by their SHAP values.
                                         Can be: 'mean', 'median', 'constant'. See explanation to this parameter in
                                         'Explaining Anomalies Detected by Autoencoders Using SHAP' under
                                         SHAPvaluesSelection.
        """

        self.num_anomalies_to_explain = num_anomalies_to_explain
        self.reconstruction_error_percent = reconstruction_error_percent
        self.shap_values_selection = shap_values_selection
        self.autoencoder = model
        self.features = features
        self.max_features_per_sample = max_features_per_sample
        self.features_to_run = features_to_run

    # def train_model(self, x_train, nb_epoch=1000, batch_size=64):
    #     """
    #     Train 6-layer Autoencoder model on the given x_train data.

    #     Args:
    #         x_train (data frame): The data to train the Autoencoder model on
    #         nb_epoch (int): Number of epoch the model will perform
    #         batch_size (int): Size of each batch of data enter to the model

    #     Returns:
    #         model: Trained autoencoder
    #     """
        
    #     input_dim = x_train.shape[1]

    #     input_layer = Input(shape=(input_dim,))

    #     encoder = Dense(int(input_dim / 2), activation="relu", activity_regularizer=regularizers.l1(10e-7))(
    #         input_layer)

    #     encoder = Dense(int(input_dim / 4), activation="relu", kernel_regularizer=regularizers.l2(10e-7))(encoder)

    #     decoder = Dense(int(input_dim / 2), activation='relu', kernel_regularizer=regularizers.l2(10e-7))(encoder)

    #     decoder = Dense(input_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(10e-7))(decoder)

    #     self.autoencoder = Model(inputs=input_layer, outputs=decoder)

    #     self.autoencoder.summary()

    #     self.autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    #     earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    #     self.autoencoder.fit(x_train, x_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True,
    #                          validation_split=0.1, verbose=2, callbacks=[earlystopper])

    #     return self.autoencoder

    def get_top_anomaly_to_explain(self, x_explain, group_by=None, x_train=None):
        """
        Sort all records in x_explain by their MSE calculated according to their prediction by the trained Autoencoder
        and return the top num_anomalies_to_explain (its value given by the user at class initialization) records.

        Args:
            x_explain (data frame): Set of records we want to explain the most anomalous ones from it.

        Returns:
            list: List of index of the top num_anomalies_to_explain records with highest MSE that will be explained.
        """
        
        x_explain_meta = x_explain.loc[:,~x_explain.columns.isin(self.features)]
        x_explain_tensor = torch.tensor(x_explain[self.features].values, dtype=torch.float32).to(self.autoencoder.device)
        explain_predictions = self.autoencoder.predict(x_explain_tensor).cpu().detach().numpy()
        
        # explain_predictions = test_autoencoder(self.autoencoder,x_explain)
        
        square_errors = np.power(x_explain[self.features] - explain_predictions, 2)
        square_errors = x_explain[self.features] - explain_predictions
        # mse_series = pd.Series(np.mean(square_errors, axis=1))

        # normalize each feature by the feature  predictions of the training set
        if x_train is not None:
            x_train_tensor = torch.tensor(x_train[self.features].values, dtype=torch.float32).to(self.autoencoder.device)
            train_predictions = self.autoencoder.predict(x_train_tensor).detach().cpu().numpy()

            # train_predictions = test_autoencoder(self.autoencoder,x_train, features=self.features)
            train_square_errors = np.power(x_train[self.features] - train_predictions, 2)
            train_square_errors = x_train[self.features] - train_predictions
            scaler = preprocessing.StandardScaler()
            # scaler.fit(control_errors)
            scaler.fit(train_square_errors)
            square_errors = pd.DataFrame(scaler.transform(square_errors),columns = square_errors.columns, index=square_errors.index)
            
        mse_series = pd.Series(np.mean(square_errors, axis=1))
            # square_errors = square_errors / train_square_errors.mean(axis=0)


        most_anomal_trx = mse_series.sort_values(ascending=False)
        columns = ["id", "mse_all_columns"]
        columns.extend(["squared_error_" + x for x in list(x_explain[self.features].columns)])
        items = []
        for x in most_anomal_trx.items():
            item = [x[0], x[1]]
            item.extend(square_errors.loc[x[0]])
            items.append(item)

        df_anomalies = pd.DataFrame(items, columns=columns)
        
        df_anomalies.set_index('id', inplace=True)
        if group_by is not None:
            df_anomalies_with_meta = pd.concat([x_explain_meta,df_anomalies],axis=1)

            group_by_mses = df_anomalies_with_meta.groupby(group_by)['mse_all_columns'].mean()
            group_by_mses = group_by_mses.sort_values(ascending=False)
            top_anomalies_to_explain = list(group_by_mses.head(self.num_anomalies_to_explain).index)
        else:
            top_anomalies_to_explain = df_anomalies.head(self.num_anomalies_to_explain).index
        return top_anomalies_to_explain

    def get_num_features_with_highest_reconstruction_error(self, total_squared_error, errors_df):
        """
        Calculate the number of features whose reconstruction errors sum to reconstruction_error_percent of the
        total_squared_error of the records that selected to be explained at the moment. This is the number of the
        top reconstructed errors features that going to be explained and eventually this features together with their
        explanation will build up the features explanation set of this record.

        Args:
            total_squared_error (int): MSE of the records selected to be explained
            errors_df (data frame): The reconstruction error of each feature- this is the first output output of
                                    get_errors_df_per_record function

        Returns:
            int: Number of features whose reconstruction errors sum to reconstruction_error_percent of the
                 total_squared_error of the records that selected to be explained at the moment
        """

        error = 0
        for num_of_features, index in enumerate(errors_df.index):
            error += errors_df.loc[index, 'err']
            if error >= self.reconstruction_error_percent * total_squared_error:
                break
            if num_of_features >= (self.max_features_per_sample-1):
                break
        return num_of_features + 1

    def get_background_set(self, x_train, background_size=200):
        """
        Get the first background_size records from x_train data and return it. Used for SHAP explanation process.

        Args:
            x_train (data frame): the data we will get the background set from
            background_size (int): The number of records to select from x_train. Default value is 200.

        Returns:
            data frame: Records from x_train that will be the background set of the explanation of the record that we
                        explain at that moment using SHAP.
        """

        background_set = x_train.head(background_size)
        return background_set

    def get_errors_df_per_record(self, record, scaler = None,run_diff = True):
        """
        Create data frame of the reconstruction errors of each features of the given record. Eventually we get data
        frame so each row contain the index of feature, its name, and its reconstruction error based on the record
        prediction provided by the trained autoencoder. This data frame is sorted by the reconstruction error of the
        features

        Args:
            record (pandas series): The record we explain at the moment; values of all its features.

        Returns:
            data frame: Data frame of all features reconstruction error sorted by the reconstruction error.
        """

        record_tensor = torch.tensor(record.values.astype(np.float32), dtype=torch.float32).to(self.autoencoder.device)
        # prediction = self.autoencoder(np.array([[record]])[0])[0]
        prediction = self.autoencoder.predict(record_tensor).detach().cpu().numpy()
        # square_errors = np.power(record - prediction, 2)
        if run_diff:
            square_errors = record - prediction
        else:
            square_errors = np.power(record - prediction, 2)

        if scaler is not None:
            # square_errors = scaler.transform(square_errors)
            square_errors = square_errors.copy()
            square_errors.loc[:,:] = scaler.transform(square_errors.values)
            
            # prediction_transformed = scaler.transform(prediction)
        
        # square_errors_transformed = np.power(record - prediction_transformed, 2)
        

        if len(square_errors.shape) > 1:
            square_errors = square_errors.median(axis=0)
            # square_errors = pd.DataFrame(square_errors, columns=record.index)

            errors_df = pd.DataFrame({'col_name': square_errors.index, 'err': square_errors}).reset_index(drop=True)
        # if len(square_errors.shape) > 1:
            if run_diff:
                total_mse = np.sum(np.abs(square_errors))
            else:
                total_mse = np.sum(square_errors)

        else:
            errors_df = pd.DataFrame({'col_name': record.index, 'err': square_errors}).reset_index(drop=True)
            total_mse = np.mean(square_errors)

        errors_df.sort_values(by='err', ascending=False, inplace=True, key=abs)
        return errors_df, total_mse

    def get_highest_shap_values(self, shap_values_df):
        """
        Choosing explaining features based on their SHAP values by shap_values_selection method (mean, median, constant)
        i.e. remove all features with SHAP values that do not meet the method requirements as described in 'Explaining
        Anomalies Detected by Autoencoders Using SHAP' under SHAPvaluesSelection.

        Args:
            shap_values_df (data frame): Data frame with all existing features and their SHAP values.

        Returns:
            data frame: Data frame that contain for each feature we explain (features with high reconstruction error)
                        its explaining features that selected by the shap_values_selection method and their SHAP values.
        """

        all_explaining_features_df = pd.DataFrame()

        for i in range(shap_values_df.shape[0]):
            shap_values = shap_values_df.iloc[i]

            if self.shap_values_selection == 'mean':
                treshold_val = np.mean(shap_values)

            elif self.shap_values_selection == 'median':
                treshold_val = np.median(shap_values)

            elif self.shap_values_selection == 'constant':
                num_explaining_features = 5
                explaining_features = shap_values_df[i:i + 1].stack().nlargest(num_explaining_features)
                all_explaining_features_df = pd.concat([all_explaining_features_df, explaining_features], axis=0)
                continue

            else:
                raise ValueError('unknown SHAP value selection method')

            num_explaining_features = 0
            for j in range(len(shap_values)):
                if shap_values[j] > treshold_val:
                    num_explaining_features += 1
            explaining_features = shap_values_df[i:i + 1].stack().nlargest(num_explaining_features)
            all_explaining_features_df = pd.concat([all_explaining_features_df, explaining_features], axis=0)
        return all_explaining_features_df

    def func_predict_feature(self, record):
        """
        Predict the value of specific feature (with 'counter' index) using the trained autoencoder

        Args:
            record (pandas series): The record we explain at the moment; values of all its features.

        Returns:
            list: List the size of the number of features, contain the value of the predicted features with 'counter'
                  index (the feature we explain at the moment)
        """

        weights = self.autoencoder.get_weights(module='encoder', layer_num=1)
        weights_feature = weights.data.clone()
        weights_feature[:,self.counter] = 0
    
        # model weights are updated
        model_feature = copy.deepcopy(self.autoencoder)        
        model_feature.update_weights(weights_feature,module='encoder', layer_num=1)

        record_tensor = torch.tensor(record.astype(np.float32), dtype=torch.float32).to(self.autoencoder.device)
        record_prediction = model_feature.predict(record_tensor).detach().cpu().numpy()[:, self.counter]

        return record_prediction

    def explain_unsupervised_data(self, x_train, x_explain, autoencoder=None, return_shap_values=False,save_dir=None,group_by=None,scale=True):
        """
        First, if Autoencoder model not provided ('autoencoder' is None) train Autoencoder model on given x_train data.
        Then, for each record in 'top_records_to_explain' selected from given 'x_explain' as described in
        'get_top_anomaly_to_explain' function, we use SHAP to explain the features with the highest reconstruction
        error based on the output of 'get_num_features_with_highest_reconstruction_error' function described above.
        Then, after we got the SHAP value of each feature in the explanation of the high reconstructed error feature,
        we select the explaining features using 'highest_contributing_features' function described above. Eventually,
        when we got the explaining features for each one of the features with highest reconstruction error, we build the
        explaining features set so the feature with the highest reconstruction error and its explaining features enter
        first to the explaining features set, then the next feature with highest reconstruction error and its explaining
        features enter to the explaining features set only if they don't already exist in the explaining features set
        and so on (full explanation + example exist in 'Explaining Anomalies Detected by Autoencoders Using SHAP')

        Args:
            x_train (data frame): The data to train the autoencoder model on and to select the background set from (for
                                  SHAP explanation process)
            x_explain (data frame): The data from which the top 'num_anomalies_to_explain' records are selected by their
                                    MSE to be explained.
            autoencoder (model): Trained Autoencoder model that will be used to explain x_explain data. If None (model 
                                 not provided) then we will build and train from scratch a Autoencoder model as described 
                                 in train_model function.
            return_shap_values (bool): If False, the resulting explnation featues set for each record will include only 
                                       the names of the explaining features. If True, in addition to explaining feature name,
                                       the explnation featues set will include the SHAP value of each feature in the explnation
                                       featues set so the explnation featues set will be composed of tupels of (str, float)
                                       when str will be the name of the explaining feature and float will be its SHAP value.
                                       Note that for the explained features (features with high reconstraction error), if they 
                                       did not appear in previuse feature explanation (explnation of feature with higher 
                                       recustraction error), they will not have any SHAP values. Therefore they get unique
                                       value of -1.
                                       
        Returns:
            dict: Return all_sets_explaining_features dictionary that contain the explanation for
                  'top_records_to_explain' records so that the keys are int; the records indexes and the values are
                  lists; the explanation features sets.
        """
        # if self.features is not None:
            # x_train_with_metadata = x_train.copy()
            # x_explain_with_metadata = x_explain.copy()
            # x_train = x_train[self.features]
            # x_explain = x_explain[self.features]

        # self.autoencoder = autoencoder
        # if self.autoencoder is None:
            # self.train_model(x_train)   
        if scale:
            
            scaler_diffs = preprocessing.StandardScaler()
            x_train_tensor = torch.tensor(x_train[self.features].values, dtype=torch.float32).to(self.autoencoder.device)
            x_train_preds = self.autoencoder.predict(x_train_tensor).detach().cpu().numpy()
            errors_train = (x_train[self.features].values - x_train_preds) 
            scaler_diffs.fit(errors_train)


        meta_features = [c for c in x_train.columns if c not in self.features]

        top_records_to_explain = self.get_top_anomaly_to_explain(x_explain, group_by=group_by,x_train=x_train)
        # top_records_to_explain = self.get_top_anomaly_to_explain(x_explain, group_by=group_by)

        if group_by is None:
            x_explain_top_records = x_explain.loc[top_records_to_explain]

        all_sets_explaining_features = {}
        shap_values_all_records = {}
        for j,record_idx in tqdm(enumerate(top_records_to_explain)):
            print(record_idx)
            if group_by is None:
                record_to_explain = x_explain.loc[record_idx]
            else:
                record_to_explain = x_explain.loc[x_explain[group_by] == record_idx]
            cpd_name = record_to_explain['Metadata_broad_sample']

            df_err, total_mse = self.get_errors_df_per_record(record_to_explain[self.features],scaler=None)
            # num_of_features = self.get_num_features_with_highest_reconstruction_error(total_mse * df_err.shape[0],
                                                                                    #   df_err)
            if self.features_to_run is not None:
                num_of_features = len(self.features_to_run)
                df_top_err = df_err[df_err['col_name'].isin(self.features_to_run)]
            else:
                num_of_features = self.get_num_features_with_highest_reconstruction_error(total_mse,
                                                                                      df_err)
                df_top_err = df_err.head(num_of_features)
            
            all_sets_explaining_features[record_idx] = []
            shap_values_all_features = {}

            backgroungd_set = self.get_background_set(x_train[self.features], 400).values
            backgroungd_set = self.get_background_set(x_train[self.features], 200).values
            # backgroungd_set = shap.kmeans(backgroungd_set, 10)
            for i in range(num_of_features):

                self.counter = df_top_err.index[i]
                feature_name = df_top_err.values[i][0]
                error = df_top_err.values[i][1]
                explainer = shap.KernelExplainer(self.func_predict_feature, backgroungd_set)
                shap_values = explainer.shap_values(record_to_explain[self.features], nsamples='auto')
                # if len(shap_values.shape) == 1:
                    # shap_values = np.reshape(shap_values, (1,len(shap_values)))

                if save_dir is not None:
                    if group_by is not None:
                            
                        shap.summary_plot(shap_values,
                                          plot_type = 'dot', 
                                          feature_names=self.features, 
                                          class_names=self.features,max_display=5,
                                        #   title=feature_name,
                                          plot_size=(6,2.5))  #max_display=10,
                        
                        # plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=80, ha='right')
                        plt.gca().set_yticklabels(plt.gca().get_yticklabels(),fontsize=12)
                        plt.gca().set_xlabel('SHAP Values', fontsize=12)
                        # plt.gca().set_ylabel('Top features', fontsize=12)
                        # plt.gca().set_title(feature_name, fontsize=12)

                        # ax.set_xticklabels(ax.get_xticklabels(), rotation=80, ha='right')
                        save_path = os.path.join(save_dir,f'shap_summary_plot_{j}_{record_idx}_{i}_{feature_name}.png')
                        # shap_figure = shap.dependence_plot(i, shap_values, X.values, feature_names=X.columns, dot_size=10, interaction_index=None, x_jitter=-0.5)
                        
                        # plt.title(feature_name)
                        plt.tight_layout()
                        plt.show()
                        plt.savefig(save_path)
                        plt.close('all')
                    else:
                        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, 
                                                               shap_values, 
                                                               feature_names=self.features,
                                                               max_display=5)


                        save_path = os.path.join(save_dir,f'shap_waterfall_plot_{j}_{record_idx}_{cpd_name}_{i}_{feature_name}.png')
                        plt.gca().set_yticklabels(plt.gca().get_yticklabels(), rotation=10,fontsize=8, ha='right')
                        plt.gca().set_xlabel('SHAP Values', fontsize=12)
                        plt.gca().set_ylabel('Top features', fontsize=12)
                        # plt.title(feature_name)
                        plt.tight_layout()
                        plt.show()
                        plt.savefig(save_path)
                        plt.close('all')
                shap_values_all_features[feature_name] = {
                    'shap_values':shap_values,
                    'error': error
                }

            # shap_values_all_features = np.fabs(shap_values_all_features)
            shap_values_all_records[record_idx] = shap_values_all_features

            # shap_values_all_features = pd.DataFrame(data=shap_values_all_features, columns=self.features)
            # highest_contributing_features = self.get_highest_shap_values(shap_values_all_features)
            
            # for idx_explained_feature in range(num_of_features):
            #     set_explaining_features =[]
            #     for idx, row in highest_contributing_features.iterrows():
            #         if idx[0] == idx_explained_feature:
            #             set_explaining_features.append((idx[1], row[0]))
            #     explained_feature_index = df_top_err.index[idx_explained_feature]
            #     set_explaining_features.insert(0, (x_train.columns[explained_feature_index], -1))

            #     all_sets_explaining_features[record_idx].append(set_explaining_features)

            # final_set_features = []
            # final_set_items = []
            # for item in sum(all_sets_explaining_features[record_idx], []):
            #     if item[0] not in final_set_features:
            #         final_set_features.append(item[0])
            #         final_set_items.append(item)
                    
            # if return_shap_values:
            #     all_sets_explaining_features[record_idx] = final_set_items
            # else:
            #     all_sets_explaining_features[record_idx] = final_set_features

        return shap_values_all_records
