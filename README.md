# AttentionModule
这次，我复现了se,cbam和bam这三篇有关于卷积注意力机制的论文并将其insert到resnet18网络之中。之后，我使用猫狗数据训练了resnet18,resnet18_se, resnet18_cbam, resnet18_bam四个模型，并从模型的收敛速度以及power of feature represent来探讨其优缺点。
# Environment
python 3.7    
pytorch 1.5    
torchvision 0.6    
opencv 3.4  
# learning_cure
四个模型的学习曲线分别如下所示，分别为resnet,resnet_se,resnet_cbam, resnet_bam  
![image](images/resnet18_learning_cure.PNG)  
![image](images/se_learning_cure.PNG)  
![image](images/cbam_learning_cure.PNG)  
![image](images/bam_learning_cure.PNG)  
从这四条曲线可知，四个模型的收敛速度和学习能力相差无几，resnet_bam收敛速度偏慢。在前30个batch中，resnet18的收敛曲线趋于平缓，没有任何明显的抖动。可能是由于注意力机制的存在，使得其它嵌入注意力机制的模型的泛化能力较差，模型方差大。同时，这也有可能是注意力机制中引入的参数而导致，所以在训练注意力机制模型的时候，我们可以考虑采用正则化或dropout来提高模型的泛化能力，降低模型的方差。
# visualization

