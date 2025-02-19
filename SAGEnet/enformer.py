import tensorflow_hub as hub
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TFHUB_URL = 'https://tfhub.dev/deepmind/enformer/1'
TRACK_IDX = 4980 # CAGE:brain, adult
CENTER_BINS = [447,448,449] # three center bins of enformer output
NUM_TRACKS=5313

class Enformer:
    def __init__(self, finetuned_weights_dir='/homes/gws/aspiro17/seqtoexp/PersonalGenomeExpression-dev/input_data/enformer_finetuned_weights/'):
        """
        Load Enformer model from TFHUB_URL and set finetuned weights and intercept, if provided. 
        Parameters: 
        - finetuned_weights_dir: String path to directory containing 'coef.npy' and 'intercept.npy' to fine-tune Enformer predictions. 
        """
        self._model = hub.load(TFHUB_URL).model
        if finetuned_weights_dir: 
            self.weights = np.load(f'{finetuned_weights_dir}coef.npy')
            self.intercept = np.load(f'{finetuned_weights_dir}intercept.npy')
        else: 
            self.weights=None
  
    def predict_on_batch(self, inputs,save_mode='finetuned'):
        """
        Get Enformer predictions on a batch of input data. 
        
        Parameters: 
        - inputs: Tensor of one-hot-encoded sequence of shape [batch_size, 4, input_len] 
        - save_mode: String indicating how Enformer's outputs should be processed. If save_mode=='finetuned', log(predictions+1) from the 3 center bins will be transformed using weights from finetuned_weights_dir and summed to yield a single prediction value. If save_mode=='only_brain', log(predictions+1) from the 3 center bins, track 4980 ('CAGE:brain, adult') will be summed to yield a single prediction value. If save_mode=='all_tracks', prediction values from the center 3 bins from all tracks (shape 3,5313) will be saved. 
        
        Returns: Enformer predictions for a given batch (single float value for save_mode=='finetuned' and save_mode==only_brain, array of shape (3,5313) for save_mode=='all_tracks'). 
        """
        # to work with our datasets: our models takes [batch_size, 4, input_len]  vs. Enformer takes [batch_size, 4, input_len] 
        inputs = inputs.permute(0, 2, 1) 
        
        predictions = self._model.predict_on_batch(inputs)
        predictions={k: v.numpy() for k, v in predictions.items()}
        
        if save_mode == 'only_brain': 
            predictions=predictions['human'][0][CENTER_BINS,TRACK_IDX] 
            predictions = np.sum(np.log(predictions+1)) # single float value 
        elif save_mode == 'all_tracks': 
            predictions=predictions['human'][0][CENTER_BINS,:] # shape (3,5313) 
        elif save_mode == 'finetuned': 
            if self.weights is None: 
                raise ValueError("if save_mode=='finetuned', finetuned_weights_dir must be provided in model initialization")
            predictions=predictions['human'][0][CENTER_BINS,:]  
            predictions = np.sum(np.sum(self.weights * np.log(predictions + 1),axis=0)) + self.intercept # single float value
        else: 
            raise ValueError("save_mode must be one of {only_brain,all_tracks,finetuned}")        
        return predictions
   

    def predict_on_dataset(self,dataset,save_mode='finetuned',using_personal_dataset=True,predict_from_personal=True):        
        """
        Get Enformer predictions on ReferenceGenomeDataset or PersonalGenomeDataset
        
        Parameters: 
        - dataset: ReferenceGenomeDataset or PersonalGenomeDataeset. 
        - save_mode: String indicating how Enformer's outputs should be processed. If save_mode=='finetuned', log(predictions+1) from the 3 center bins will be transformed using weights from finetuned_weights_dir and summed to yield a single prediction value. If save_mode=='only_brain', log(predictions+1) from the 3 center bins, track 4980 ('CAGE:brain, adult') will be summed to yield a single prediction value. If save_mode=='all_tracks', prediction values from the center 3 bins from all tracks (shape 3,5313) will be saved. 
        - using_personal_dataset: Boolean indicating if PersonalGenomeDataset is being used. If False, assume ReferenceGenomeDataset is being used. 
        - predict_from_personal: Boolean indicating if Enformer should predict from the "personal" sequence input of PersonalGenomeDataset. If not, Enformer predicts from the "reference" sequence input. Only applicable if predict_from_personal==True. 
        
        Returns: Enformer predictions for the given datset (array of len(dataset) for save_mode=='finetuned' and save_mode=='only_brain', array of shape (len(dataset), 3,5313) array for save_mode=='all_tracks'). 
        """
        
        num_datapoints = len(dataset) 
        
        if save_mode=='all_tracks': 
            dataset_res = np.zeros((num_datapoints, len(CENTER_BINS),NUM_TRACKS))
        elif save_mode=='only_brain': 
            dataset_res = np.zeros((num_datapoints)) 
        elif save_mode=='finetuned': 
            dataset_res = np.zeros((num_datapoints)) 
        else: 
            raise ValueError("save_mode must be one of {only_brain,all_tracks,finetuned}")
        
        if using_personal_dataset or predict_from_personal: # predict using PersonalGenomeDataset 
            for i, (x, y,gene_idx,sample_idx) in enumerate(dataset):
                print(i)
                if predict_from_personal: 
                    mat_seq = x[[1], :4, :]
                    pat_seq = x[[1], 4:, :]
                    mat_out = self.predict_on_batch(mat_seq,save_mode=save_mode)
                    pat_out = self.predict_on_batch(pat_seq,save_mode=save_mode)
                    pred = (mat_out+pat_out)/2 # take mean 
                else: 
                    ref_seq = x[[0], :4, :]
                    pred = self.predict_on_batch(ref_seq,save_mode=save_mode)

                if save_mode=='finetuned' or save_mode=='only_brain': 
                    dataset_res[i] = pred
                else: 
                    dataset_res[i,:] = pred
        
        else: # predict using ReferenceGenomeDataset  
            for i, (x, y) in enumerate(dataset):
                print(i)
                pred = self.predict_on_batch(x.reshape(1,4,-1),save_mode=save_mode)
                if save_mode=='finetuned' or save_mode=='only_brain': 
                    dataset_res[i] = pred
                else: 
                    dataset_res[i,:] = pred
        return dataset_res

            

        
        




