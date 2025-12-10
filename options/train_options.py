from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--img_file', type=str, default='/media/work/qinfeng/celeba-1024/',
        #                     help='training and testing dataset')

        self.isTrain = True

        return parser
