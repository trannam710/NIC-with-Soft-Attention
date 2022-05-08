from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='uitviic',
                       train_path ='./UIT-ViIC/train_vn.json',
                       val_path ='./UIT-ViIC/val_vn.json',
                       test_path ='./UIT-ViIC/test_vn.json',
                       image_folder='./UIT-ViIC/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./caption data word/',
                       max_len=50,
                       use_pre_train=True)
