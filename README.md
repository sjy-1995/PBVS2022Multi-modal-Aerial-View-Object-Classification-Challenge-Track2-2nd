# PBVS2022Multi-modal-Aerial-View-Object-Classification-Challenge-Track2-2nd
some requirements: Python 3.7.0, PyTorch 1.7.1
some other libraries can be easily installed by 'pip install XXX'


#########################################################################################################
######### For testing the trained model and obtaining the final predictions of the test samples #########
#########################################################################################################
The testing process can be directly implemented in this dir by running the following bash codes.

The pretrained models and weights can be downloaded from:

https://drive.google.com/file/d/1vpzuxRMidbLFnyrR7LLLGAlQSJLcfVdS/view?usp=sharing

https://drive.google.com/file/d/1z9jRCSBcFmpKtqD3_1Ux07L_h9AHYS6N/view?usp=sharing

https://drive.google.com/file/d/1ogHyydnsT76wwNwGPdRCi_NoRlcvuPId/view?usp=sharing

https://drive.google.com/file/d/1NV0QuhO92m8PXhN43e6MJX5k5j0OpHLH/view?usp=sharing

 
Step1: Run 'test_swin_Track2_EO_transductive_fortest.py' and the 'test_swin_Track2_EOSAR_transductive_fortest.py' respectively for obtaining the predictions and the corresponding probabilities of the test samples.
-------------------------------------------------------------------------------
```bash
python test_swin_Track2_EO_transductive_fortest.py
```
```bash
python test_swin_Track2_EOSAR_transductive_fortest.py
```
--------------------------------------------------------------------------------
Step2: Run 'copredict.py' for co-predicting the final results according to the two models.
--------------------------------------------------------------------------------
```bash
python copredict.py
```
--------------------------------------------------------------------------------
The obtained 'results_final.csv' contains the final predictions reported in our solution.


#################################################################################################################################
######################################### For training all the models in our solution ###########################################
#################################################################################################################################
Due to the size limit of the file, the training data and some intermediate trained models are stored in Google Drive, which can be downloaded from the links.

The training data can be downloaded from https://codalab.lisn.upsaclay.fr/my/datasets/download/642745ec-31ca-4158-b228-070f253b247f. And the '.DS_Store' file in it should be removed. 
Please download the pretrained weights of Swin-B on ImageNet-22K (swin_base_patch4_window7_224_22k.pth, which can also be found by google search) from the link: https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiE4uu4pMj2AhUIA94KHbTSChAQFnoECAMQAQ&url=https%3A%2F%2Fgithub.com%2FSwinTransformer%2Fstorage%2Freleases%2Fdownload%2Fv1.0.0%2Fswin_base_patch4_window7_224_22k.pth&usg=AOvVaw2Mr0PHPCaEquYWE0RtPDHr

```bash
cd for_training 
```
--------------------------------------------------------------------------------
Step1: Train the baseline models on the EO images and EO-SAR images respectively.
--------------------------------------------------------------------------------
```bash
python train_swin_EO_baseline.py
```
```bash
python train_swin_EOSAR_baseline.py
```
Thus we can obtain the intermediate trained models 'swinB_F_bestacc_Track2_EO.pth', 'swinB_C_bestacc_Track2_EO.pth', 'swinB_F_bestacc_Track2_EOSAR_fortest.pth', and 'swinB_C_bestacc_Track2_EOSAR_fortest.pth', which can also be downloaded from the link: https://drive.google.com/file/d/1reTuwpIXBU7yQd24yCe0fdvfdfgMyKps/view?usp=sharing; https://drive.google.com/file/d/1txhBS_x5GscABtqA0ZntYXi1vX9MXZoW/view?usp=sharing; https://drive.google.com/file/d/14U-u8tNUoRx9T-9dSB07mfyUbRjWPTtc/view?usp=sharing; https://drive.google.com/file/d/1qoOGwRFMu4YqPfwYZN987IB21UwidMaC/view?usp=sharing.
--------------------------------------------------------------------------------
Step2: Train the models on the EO images and EO-SAR images transductively and respectively.
--------------------------------------------------------------------------------
```bash
python train_swin_EO_transductive_iter1_fortest.py
```
```bash
python train_swin_EOSAR_transductive_iter1_fortest.py
```
Thus we can obtain the intermediate trained models 'swinB_F_bestacc_Track2_EO_iteration_num1_fortest.pth', 'swinB_C_bestacc_Track2_EO_iteration_num1_fortest.pth', 'swinB_F_bestacc_Track2_EOSAR_iteration_num1_fortest.pth', and 'swinB_C_bestacc_Track2_EOSAR_iteration_num1_fortest.pth', which can also be downloaded from the link: https://drive.google.com/file/d/15w4lR4irq02CJqQSSCIDTbFEkBc82--h/view?usp=sharing; https://drive.google.com/file/d/1mW7uCOZ8vMMrGhVGb1AWw5GO89GA9qfq/view?usp=sharing; https://drive.google.com/file/d/1EmV3zgbpK8ZvFtlCOaoyVjTFDH4iVwjc/view?usp=sharing; https://drive.google.com/file/d/1dpKctyFdGTzGy9Ps-gksf1ES6HuBf7uY/view?usp=sharing.
--------------------------------------------------------------------------------
Step3: Train the models on the EO images transductively again.
--------------------------------------------------------------------------------
```bash
python train_swin_EO_transductive_iter2_fortest.py
```
The obtained 'swinB_F_bestacc_Track2_EO_iteration_num2_fortest.pth' and 'swinB_C_bestacc_Track2_EO_iteration_num2_fortest.pth' are the finally trained model of EO images.

Then we can use the finally obtained models for new testing.





