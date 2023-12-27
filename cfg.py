import torchvision.transforms as transforms
import albumentations as A

class CFG():
    # model 
    batch_size = 32 # for now, batch size is fixed to 1
    img_transform_size_W = img_transform_size_H = 224
    # num_classes = 10 # automatically calculated by train_dataset_dir
    label_smoothing = 0.0
    
    lr = 1e-4
    optim_betas = (0.9,0.98)
    optim_eps = 1e-8
    optim_weight_decay = 0.05
    
    # dataset
    test_size = 0.1
    train_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
    val_transforms = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
    
    
    
    train_dataset_dir = "/zz1236zz/workspace/Dataset/train_data"
    test_dataset_dir = "/zz1236zz/workspace/Dataset/test_data"
    num_classes_of_tooth_num = 32
    num_classes_of_tooth_position = 5
    

