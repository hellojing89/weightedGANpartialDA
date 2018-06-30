This is a caffe repository for "Importance Weighted Adversarial Nets for Partial Domain Adaptation".

The source models are pre-trained using bvlc_alexnet.

When the source model is pre-trained, it is stored into two duplications. The first represents the source model which remains fixed during adaptation, the second represents the target model (initialized by source model parameters) which will be trained for partial domain adaptation.

We provide the example proto files, pre-trained models, and running script in 'models/partialDAoffice31' folder.
The example pre-trained 'dslr' source model can be downloaded from:
https://drive.google.com/open?id=1MYfLAg9fDvKaXeJnL4WxWisE88U3E8QR


usage:

Put the pre-trained source model into the 
'weightedGANpartialDA/models/partialDAoffice31/dslr_to_amazonC10/snapshots_dc' folder, 
and run the script 
'weightedGANpartialDA/models/partialDAoffice31/dslr_to_amazonC10/scripts/train_dc.sh' 
from the caffe root.
