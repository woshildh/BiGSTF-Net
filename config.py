class ModelConfig:
    def __init__(self):
        self.use_single_branch = False
        self.single_branch_name = "fnirs" # eegor fnirs
        
        self.use_same_conv = False
        self.use_channel_conv  = True
        
        self.no_cross_modal_fusion =  False
        self.no_gate_fusion = False
        
        self.no_transformer = False

class LossConfig:
    def __init__(self):
        self.only_softmax = False
        self.normal_label_smooth = False
        self.use_asoftmax = True

class DataAugConfig:
    def __init__(self):
        self.use_data_aug = True
        
        self.use_generic = True
        
        self.use_eeg = True
        
        self.use_fnirs = True

model_config = ModelConfig()
loss_config = LossConfig()
data_aug_config = DataAugConfig()