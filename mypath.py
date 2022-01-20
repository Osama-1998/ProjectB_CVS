class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'C:\\Users\oabboud\Miniconda3\envs\ProjectB\dataloaders\datasets\Pascal_dataset'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return 'C:\\Users\oabboud\Miniconda3\envs\ProjectB\dataloaders\datasets\Pascal_dataset'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return 'C:\\Users\oabboud\Miniconda3\envs\ProjectB\dataloaders\datasets\Pascal_dataset'  # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return 'C:\\Users\oabboud\Miniconda3\envs\ProjectB\dataloaders\datasets\Pascal_dataset'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
