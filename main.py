import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Main entrance for the application.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Create parser for the "train" command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--accumulate', type=int, default=4, help='Batches to accumulate before optimizing')
    train_parser.add_argument('--cfg', type=str, default='cfg/mde.cfg', help='Path to configuration file')
    train_parser.add_argument('--data', type=str, default='data/coco2017.data', help='Path to data file')
    train_parser.add_argument('--multi-scale', action='store_true', help='Adjust image size every 10 batches')
    train_parser.add_argument('--img-size', nargs='+', type=int, default=[512], help='Image sizes (min_train, max_train, test)')
    train_parser.add_argument('--rect', action='store_true', help='Rectangular training')
    train_parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    train_parser.add_argument('--nosave', action='store_true', help='Only save final checkpoint')
    train_parser.add_argument('--notest', action='store_true', help='Only test final epoch')
    train_parser.add_argument('--evolve', action='store_true', help='Evolve hyperparameters')
    train_parser.add_argument('--bucket', type=str, default='', help='GSutil bucket')
    train_parser.add_argument('--cache-images', action='store_true', help='Cache images for faster training')
    train_parser.add_argument('--weights', type=str, default='weights/best.pt', help='Initial weights path')
    train_parser.add_argument('--name', default='', help='Rename results file if supplied')
    train_parser.add_argument('--device', default='', help='Device ID (e.g., 0, 0,1, or cpu)')
    train_parser.add_argument('--adam', action='store_true', help='Use Adam optimizer')
    train_parser.add_argument('--single-cls', action='store_true', help='Train as single-class dataset')

    # Create parser for the "detect" command
    detect_parser = subparsers.add_parser('detect', help='Detect objects')
    detect_parser.add_argument('--cfg', type=str, default='cfg/yolov3-custom.cfg', help='*.cfg path')
    detect_parser.add_argument('--names', type=str, default='data/customdata/custom.names', help='*.names path')
    detect_parser.add_argument('--weights', type=str, default='weights/last.pt', help='weights path')
    detect_parser.add_argument('--source', type=str, default='data/customdata/images', help='source')  # input file/folder, 0 for webcam
    detect_parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    detect_parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    detect_parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    detect_parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    detect_parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    detect_parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    detect_parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    detect_parser.add_argument('--view-img', action='store_true', help='display results')
    detect_parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    detect_parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    detect_parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    detect_parser.add_argument('--augment', action='store_true', help='augmented inference')

    args = parser.parse_args()

    if args.command == 'train':
        import train
        train.main()
    elif args.command == 'detect':
        import detect
        detect.main()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
